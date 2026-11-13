"""命令行入口：`sysone inspect` / `sysone run` / `sysone doctor`。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from .devices.apple import ApplePlatform
from .devices.reference import CharEncoder, ReferencePlatform, ReferenceRunner
from .domain.attention import MaskMode
from .domain.decision import DecisionRequest, Question
from .engine.engine import Engine
from .engine.strategies.split import mude_runner
from .errors import ConfigError, SysOneError
from .models import families as _family_modules  # noqa: F401  导入即注册
from .models.descriptor import ModelDescriptor
from .models.registry import families, family
from .weights.store import TensorStore

_INPUT_KEYS = {"state", "questions", "attention_mode", "request_id"}
# split 链的缓冲宽度默认取调度上限：链在装载时就按它分配 GPU 侧的面，
# 而超过它的 prompt 在 `SplitRunner` 里就是 422。
_SPLIT_WIDTH = SchedulerConfig().max_len


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysone", description="SysOne 决策推理引擎")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="解析模型配置并打印描述")
    inspect.add_argument("--model", required=True, choices=families())
    inspect.add_argument("--weights", type=Path, required=True, help="模型目录")

    run = sub.add_parser("run", help="跑一次请求：参考后端，或 mude 的 ANE+GPU 链")
    run.add_argument(
        "--backend",
        default="reference",
        choices=("reference", "split"),
        help="执行路径：reference（M0 语义裁判，只做全注意力层）/ split（mude 整链）",
    )
    run.add_argument("--model", default=None, choices=families(), help="split 路径的家族（默认 mude）")
    run.add_argument("--descriptor", type=Path, default=None, help="descriptor JSON；reference 路径必填")
    run.add_argument("--weights", type=Path, required=True, help="模型目录或 safetensors")
    run.add_argument("--input", type=Path, required=True, help="请求 JSON")
    run.add_argument("--attention-mode", default=None, help="causal / noncausal_full / prefix_noncausal")
    run.add_argument("--debug-timings", action="store_true", help="返回分阶段耗时")
    run.add_argument("--bucket", type=int, default=0, help="split：ANE program 的列宽（默认 1024）")
    run.add_argument("--width", type=int, default=0, help=f"split：GPU 缓冲的列数（默认 {_SPLIT_WIDTH}）")

    doctor = sub.add_parser("doctor", help="打印平台能力与内存预算")
    doctor.add_argument("--platform", default="auto", choices=("auto", "reference", "apple"))
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
        return {"available": True, **ApplePlatform().probe()}
    except SysOneError as exc:
        return {
            "platform": "apple",
            "available": False,
            "reason": exc.message,
            "context": exc.context,
        }


def _build_engine(args: argparse.Namespace, mode: MaskMode | None) -> Engine:
    """按 `--backend` 装配引擎：参考后端（M0 语义裁判）或 mude 的 ANE+GPU 链。"""
    if args.backend == "split":
        name = args.model or "mude"
        width = int(args.width or _SPLIT_WIDTH)
        descriptor = family(name).parse_config(args.weights)
        config = EngineConfig(
            model=ModelSelection(name=args.weights.name, dir=args.weights),
            scheduler=SchedulerConfig(attention_mode=mode, max_len=width),
            service=ServiceConfig(debug_timings=bool(args.debug_timings)),
        )
        config.validate()
        # 编码器只是 Engine 的必填位：split 不排 MaskPlan，真正渲染在 `mude_runner` 的题面里。
        return Engine(
            descriptor=descriptor,
            runner=mude_runner(args.weights, bucket=args.bucket or None, width=width),
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            if args.platform == "auto":
                _print({name: _platform_report(name) for name in ("reference", "apple")})
            else:
                _print(_platform_report(args.platform))
            return 0
        if args.command == "inspect":
            descriptor = family(args.model).parse_config(args.weights)
            _print(descriptor.summary())
            return 0
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
