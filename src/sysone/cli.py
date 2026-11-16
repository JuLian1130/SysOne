"""命令行入口：`sysone inspect` / `sysone run` / `sysone doctor`。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from .devices.apple import ApplePlatform, measured
from .devices.reference import CharEncoder, ReferencePlatform, ReferenceRunner
from .domain.attention import MaskMode
from .domain.decision import DecisionRequest, Question
from .engine.engine import Engine
from .engine.strategies.assembly import BACKENDS as STRATEGY_BACKENDS
from .engine.strategies.assembly import assemble
from .env import read as read_env
from .errors import ConfigError, SysOneError
from .models import families as _family_modules  # noqa: F401  导入即注册
from .models.descriptor import ModelDescriptor
from .models.registry import families, family
from .service import app as service_app
from .service.app import EngineDecider
from .service.protocol import KEY_VARIABLE
from .weights.store import TensorStore

_INPUT_KEYS = {"state", "questions", "attention_mode", "request_id"}
# split 链的缓冲宽度默认取调度上限：链在装载时就按它分配 GPU 侧的面，
# 而超过它的 prompt 在 `SplitRunner` 里就是 422。
_SPLIT_WIDTH = SchedulerConfig().max_len
# 待命常驻窗口的默认值 = 一个块（4 层）。出货形态（T 16884 / P 16384）上换窗的代价 ≈ +1 %
# （三轮暖配对 +0.8 / +1.3 / +0.6 %），代价落在短请求上（200 token 2.52×、1k +139 %）；出货形态
# 是长请求，所以默认取窗口。0 保留给"全装"的对照形态（换算与对账见 docs/sysone/plan.md 第 10/11 片）。
# h2o 同口径（票据 15 §25）：待命 3406 → 426 MiB（8×），142 token 上窗口慢 2.2×，长 prompt 上
# 这笔 0.41 s 的固定开销摊薄（按 16k 外推 ≈ +3%）。
_RESIDENT_W = 4
# 后端不给 `--model` 时的默认家族。mude 维持原样；pooled 只有 h2o。
_DEFAULT_FAMILY = {"split": "mude", "pooled": "h2o"}


def _model_flags() -> argparse.ArgumentParser:
    """`run` 与 `serve` 共用的模型/后端选择（`add_help=False` 让它们挂在子命令下）。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        default="reference",
        choices=("reference",) + STRATEGY_BACKENDS,
        help="执行路径：reference（M0 语义裁判，只做全注意力层）/ split（码表读出，mude）"
             "/ pooled（标签行读出，h2o）",
    )
    common.add_argument(
        "--model", default=None, choices=families(),
        help="家族（默认按后端取：split → mude、pooled → h2o）",
    )
    common.add_argument("--descriptor", type=Path, default=None, help="descriptor JSON；reference 路径必填")
    common.add_argument("--weights", type=Path, required=True, help="模型目录或 safetensors")
    common.add_argument("--attention-mode", default=None, help="causal / noncausal_full / prefix_noncausal")
    common.add_argument("--debug-timings", action="store_true", help="返回分阶段耗时")
    common.add_argument("--bucket", type=int, default=0,
                        help="split / pooled：ANE program 的列宽（默认 1024）")
    common.add_argument("--width", type=int, default=0,
                        help=f"split / pooled：GPU 缓冲的列数（默认 {_SPLIT_WIDTH}）")
    common.add_argument(
        "--W", type=int, default=_RESIDENT_W, dest="resident",
        help=f"待命常驻的层数（默认 {_RESIDENT_W} = 一个块；0 = 全部常驻）。"
             "实测代价：h2o 32 层在 142 token 上窗口比全装慢 2.2×（票据 15 §25），"
             "长 prompt 上这笔固定开销摊薄",
    )
    return common


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysone", description="SysOne 决策推理引擎")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="解析模型配置并打印描述")
    inspect.add_argument("--model", required=True, choices=families())
    inspect.add_argument("--weights", type=Path, required=True, help="模型目录")

    run = sub.add_parser("run", parents=[_model_flags()], help="跑一次请求")
    run.add_argument("--input", type=Path, required=True, help="请求 JSON")

    serve = sub.add_parser("serve", parents=[_model_flags()], help="起 /v1/systemone 服务")
    serve.add_argument("--host", default="127.0.0.1", help="监听地址（默认只回环）")
    serve.add_argument("--port", type=int, default=8000, help="监听端口（0 表示随机空闲端口）")
    serve.add_argument("--queue", type=int, default=None, help="准入等待位（默认取 scheduler.queue）")
    serve.add_argument("--admit-timeout", type=float, default=120.0, help="排队超过这么多秒就 429")
    serve.add_argument("--name", default=None, help="回包里的 model 名（默认模型 id）")

    doctor = sub.add_parser("doctor", help="打印平台能力与内存预算")
    doctor.add_argument("--platform", default="auto", choices=("auto", "reference", "apple"))
    doctor.add_argument(
        "--model", default=None, choices=families(),
        help="顺带检查这个模型：它的几何在不在 Apple 的实测白名单里（不装载权重）",
    )
    doctor.add_argument("--weights", type=Path, default=None, help="配合 --model 的模型目录")
    return parser


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise ConfigError("文件不存在", path=str(path))
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("JSON 解析失败", path=str(path), error=str(exc)) from exc


def _parse_mode(value: str | None) -> MaskMode | None:
    if value is None:
        return None
    try:
        return MaskMode(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in MaskMode)
        raise ConfigError("未知注意力模式", mode=value, allowed=allowed) from exc


def _load_request(path: Path) -> DecisionRequest:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ConfigError("请求 JSON 必须是对象", path=str(path))
    unknown = set(payload) - _INPUT_KEYS
    if unknown:
        raise ConfigError("请求 JSON 有未知字段", fields=sorted(unknown))
    state = payload.get("state")
    if not isinstance(state, str) or not state:
        raise ConfigError("缺少 state 文本")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ConfigError("questions 必须是非空数组")
    questions = []
    for index, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise ConfigError("问题必须是对象", question=index)
        text = item.get("text")
        if not isinstance(text, str) or not text:
            raise ConfigError("问题缺少 text", question=index)
        options = item.get("options") or ()
        kind = item.get("kind") or "choice"
        if not isinstance(options, (list, tuple)) or not all(isinstance(o, str) for o in options):
            raise ConfigError("options 必须是字符串数组", question=index)
        questions.append(Question(text=text, options=tuple(options), kind=kind))
    return DecisionRequest(
        state=state,
        questions=tuple(questions),
        attention_mode=_parse_mode(payload.get("attention_mode")),
        request_id=payload.get("request_id") or "",
    )


def _tensor_files(weights: Path) -> list[Path]:
    if weights.is_dir():
        files = sorted(weights.glob("*.safetensors"))
        if not files:
            raise ConfigError("目录里没有 safetensors", dir=str(weights))
        return files
    if weights.is_file():
        return [weights]
    raise ConfigError("权重路径不存在", path=str(weights))


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _platform_report(target: str) -> dict[str, object]:
    """报告一个平台的能力与预算；不可用写成数据，不抛异常（doctor 的职责是诊断）。"""
    if target == "reference":
        caps = ReferencePlatform().capabilities()
        return {
            "platform": "reference",
            "available": True,
            "dtypes": sorted(dtype.value for dtype in caps.dtypes),
            "quant": sorted(scheme.value for scheme in caps.quant),
            "modes": sorted(mode.value for mode in caps.modes),
            "note": "M0 的语义裁判，只做全注意力层",
        }
    try:
        report = {"available": True, **ApplePlatform().probe()}
    except SysOneError as exc:
        return {
            "platform": "apple",
            "available": False,
            "reason": exc.message,
            "context": exc.context,
        }
    return report


def _geometry_report(model: str, weights: Path | None) -> dict[str, object]:
    """一个模型的几何是否在 Apple 的**实测白名单**里；解析失败也写成数据。

    判据只有 `(hidden, ffn)`（`devices/apple/platform.measured`），不判层数、头数，也**不判
    量化形态** —— 检查点存 per-row int8、ANE 吃装载时派生的 per-channel blob，拿检查点形态去
    验会当场把 h2o 拒掉。`measured=False` 不是"不能跑"：ANE 会照常编译，只是数值没对拍过。
    """
    if model is None:
        raise ConfigError("doctor --weights 需要 --model", weights=str(weights))
    if weights is None:
        raise ConfigError("doctor --model 需要 --weights", model=model)
    descriptor = family(model).parse_config(weights)
    report: dict[str, object] = {
        "family": descriptor.family,
        "model_id": descriptor.model_id,
        "hidden": descriptor.hidden,
        "ffn": descriptor.ffn,
        "num_layers": descriptor.num_layers,
        "measured": measured(descriptor),
    }
    if not report["measured"]:
        report["note"] = "这套几何没有做过端到端数值对拍：ANE 会照常编译跑出数字，结果只能当实验读数"
    return report


def _build_engine(args: argparse.Namespace, mode: MaskMode | None) -> Engine:
    """按 `--backend` 装配引擎：参考后端（M0 语义裁判）或 mude 的 ANE+GPU 链。"""
    if args.backend in STRATEGY_BACKENDS:
        name = args.model or _DEFAULT_FAMILY[args.backend]
        width = int(args.width or _SPLIT_WIDTH)
        # 形状与执行器都由注册表装配（`engine.strategies.assembly`）：CLI 不认识任何家族名，
        # 也不认识哪条链该配哪套题面 —— 那是家族自己声明的策略。
        options: dict[str, int | None] = {
            "bucket": args.bucket or None, "width": width,
            "resident": args.resident or None,
        }
        assembled = assemble(args.weights, name, backend=args.backend, **options)
        descriptor = assembled.descriptor
        config = EngineConfig(
            model=ModelSelection(name=args.weights.name, dir=args.weights),
            scheduler=SchedulerConfig(attention_mode=mode, max_len=width),
            service=ServiceConfig(debug_timings=bool(args.debug_timings)),
        )
        config.validate()
        # 编码器只是 Engine 的必填位：这两条策略都不排 MaskPlan，题面由家族自己的 Prompter
        # 或厂商答面渲染。
        return Engine(
            descriptor=descriptor,
            runner=assembled.runner,
            encoder=CharEncoder(vocab=descriptor.vocab),
            config=config,
        )
    if args.descriptor is None:
        raise ConfigError("reference 后端需要 --descriptor")
    descriptor = ModelDescriptor.from_mapping(_load_json(args.descriptor))
    config = EngineConfig(
        model=ModelSelection(name=args.weights.name, dir=args.weights),
        scheduler=SchedulerConfig(attention_mode=mode),
        service=ServiceConfig(debug_timings=bool(args.debug_timings)),
    )
    config.validate()
    return Engine(
        descriptor=descriptor,
        runner=ReferenceRunner(descriptor, TensorStore(_tensor_files(args.weights))),
        encoder=CharEncoder(vocab=descriptor.vocab),
        config=config,
    )


def _serve(args: argparse.Namespace) -> int:
    """前台服务：先绑口、再在后台线程里装载引擎，装载期间探针拿到 503。"""

    def load() -> tuple[EngineDecider, object]:
        engine = _build_engine(args, _parse_mode(args.attention_mode))
        startup = engine.start()
        return (
            EngineDecider(
                engine,
                model_name=args.name or engine.descriptor.model_id,
                max_input_tokens=engine.config.scheduler.max_len,
                debug_timings=bool(args.debug_timings),
            ),
            startup,
        )

    queue = SchedulerConfig().queue if args.queue is None else int(args.queue)
    return service_app.run(
        load=load,
        host=args.host,
        port=args.port,
        api_key=read_env()[KEY_VARIABLE] or None,
        queue=queue,
        admit_timeout=float(args.admit_timeout),
        debug_timings=bool(args.debug_timings),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            geometry = (
                None
                if args.model is None and args.weights is None
                else _geometry_report(args.model, args.weights)
            )
            if args.platform == "auto":
                payload: dict[str, object] = {
                    name: _platform_report(name) for name in ("reference", "apple")
                }
            else:
                payload = _platform_report(args.platform)
            if geometry is not None:
                # 挂在 apple 那一支下面：白名单是**这台机器**（ANE）的事实，不是模型的属性。
                payload["geometry"] = geometry
            _print(payload)
            return 0
        if args.command == "inspect":
            descriptor = family(args.model).parse_config(args.weights)
            _print(descriptor.summary())
            return 0
        if args.command == "serve":
            return _serve(args)
        engine = _build_engine(args, _parse_mode(args.attention_mode))
        engine.start()
        result = engine.run(_load_request(args.input))
        _print(dataclasses.asdict(result))
        return 0
    except SysOneError as exc:
        payload = {"error": exc.message, "code": exc.code, "status": exc.status, "context": exc.context}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
