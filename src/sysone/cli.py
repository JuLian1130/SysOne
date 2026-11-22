"""命令行入口：`sysone inspect` / `sysone run` / `sysone serve` / `sysone compile` / `sysone doctor`。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from .config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig, StorageConfig
from .config.storage import (
    ARTIFACTS_VARIABLE,
    WEIGHTS_VARIABLE,
    default_artifacts_dir,
    default_weights_dir,
)
from .devices.apple import ApplePlatform, measured
from .devices.reference import CharEncoder, ReferencePlatform, ReferenceRunner
from .domain.attention import MaskMode
from .domain.decision import DecisionRequest, Question
from .domain.work import CompilationTimes
from .engine.engine import Engine
from .engine.strategies.assembly import BACKENDS as STRATEGY_BACKENDS
from .engine.strategies.assembly import QUANT_MODES, assemble
from .env import read as read_env
from .errors import ConfigError, SysOneError
from .models import families as _family_modules  # noqa: F401  导入即注册
from .models.descriptor import ModelDescriptor
from .models.registry import families, family
from .service import app as service_app
from .service.app import EngineDecider, startup_report
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
# 后端不给 `--model` 时的默认家族。mude 维持原样；pooled 默认 h2o（torchcast 也在这条路上，
# 但它的目录带两个厂商标记件、认不出唯一家族，所以一直要显式 `--model`）。`vendor` 今天没有
# 出货家族 —— 这条路仍在注册表里，取它就要显式给家族。
_DEFAULT_FAMILY = {"split": "mude", "pooled": "h2o"}
# 视觉塔的空闲卸载窗口（票据 07 §6 未定项 2 的默认值）。60 s 是"连续几张图算同一批"的量级：
# 装一次 0.6–2 s、卸一次同样，窗口短于这个数就等于每张图都付两遍。
_VISION_IDLE = 60.0
# run / serve 只列有 CLI 装配入口的家族。`inspect` / `doctor` 仍列全部家族（它们只看配置）。
_CLI_FAMILIES = tuple(name for name in families() if family(name).strategy in STRATEGY_BACKENDS)

# 没有 --model 时按包身份认家族：每个家族列它那份发布包里**必需**的一组文件，全在才算命中。
# 这不是按权重形状猜家族 —— 猜错不会崩，只会拿错模板算出看似正常的答案；认发布包的必需件比
# 认形状或目录名可靠。
#
# 必须逐个列成一组：mude 与 torchcast 都带 `decision_config.json`（两份厂商题面格式同名），
# 单看它两家都命中。mude 多一份码表 `readout.safetensors`，torchcast 多一份 int8 清单而没有
# 码表，于是各命中各的。torchcast 的厂商包 `torchcast_decision/` **不**是标记件：改走 pooled
# 之后出货路径不读它，把它当标记会把"没带厂商包"的目录推到别的家族上。
#
# d3 / mude / pplx12 / pplx13 **共用同一组必需件**：它们都是"255 个回答码 + 独立读出表"的
# 检查点，目录里能看到的文件一模一样（pplx 还多一份 `joint_config.json`，但两版 pplx 的它
# 只是内容不同）。所以这几家一律要求显式 `--model`：猜成 mude 会拿到 `prefix_noncausal`
# 这个 d3 / pplx 都没训练过的默认掩码，概率看起来仍然像模像样。
_FAMILY_MARKERS = {
    "d3": ("decision_config.json", "readout.safetensors"),
    "mude": ("decision_config.json", "readout.safetensors"),
    "pplx12": ("decision_config.json", "readout.safetensors", "joint_config.json"),
    "pplx13": ("decision_config.json", "readout.safetensors", "joint_config.json"),
    "h2o": ("serve_config.json",),
    "torchcast": ("decision_config.json", "int8_manifest.json"),
}


def _infer_family(weights: Path) -> str:
    """从模型目录的必需件推断家族；认不出**唯一**一家就要求显式 `--model`。"""
    if weights.is_dir():
        found = sorted(name for name, markers in _FAMILY_MARKERS.items()
                       if all((weights / marker).is_file() for marker in markers))
        if len(found) == 1:
            return found[0]
        if found:
            raise ConfigError(
                "模型目录认不出唯一家族",
                weights=str(weights),
                matches=found,
                hint="加 --model 显式指定（d3 / h2o / mude / pplx12 / pplx13 / torchcast）；"
                     "d3、mude 与两版 pplx 都是同形的检查点，必需件本来就一样",
            )
    raise ConfigError(
        "需要 --model 或 --backend",
        weights=str(weights),
        available=list(_CLI_FAMILIES),
        hint="--model 取 d3 / h2o / mude / pplx12 / pplx13 / torchcast；"
             "reference 后端用 --backend reference --descriptor",
    )


def _resolve_backend(args: argparse.Namespace) -> str:
    """`--backend` 未给时按 `--model`（或权重包身份）推断；reference 仍必须显式给。"""
    if args.backend is not None:
        return args.backend
    if args.model is None:
        if args.weights is None:
            raise ConfigError(
                "需要 --model 或 --backend",
                hint="省略 --weights 时没有权重包可认家族；"
                     "先给 --model（d3 / h2o / mude / pplx12 / pplx13 / torchcast）",
            )
        args.model = _infer_family(args.weights)
    strategy = family(args.model).strategy
    if strategy not in STRATEGY_BACKENDS:
        raise ConfigError(
            "这个家族没有 CLI 装配入口",
            family=args.model,
            strategy=strategy,
            available=list(_CLI_FAMILIES),
            hint="给 --backend split（mude / d3）或 pooled（h2o / torchcast）",
        )
    return strategy


def _model_flags() -> argparse.ArgumentParser:
    """`run` / `serve` / `compile` 共用的模型/后端选择（`add_help=False` 让它们挂在子命令下）。"""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--backend",
        default=None,
        choices=("reference",) + STRATEGY_BACKENDS,
        help="执行路径：不给时按 --model 的家族推断（mude / d3 / pplx12 / pplx13 → split、"
             "h2o 与 torchcast → pooled，没给 --model 时按权重包身份认）；reference（M0 语义裁判，"
             "只做全注意力层）必须显式给，且只配合 --descriptor",
    )
    common.add_argument(
        "--model", default=None, choices=_CLI_FAMILIES,
        help="家族（默认按后端取：split → mude、pooled → h2o；也可由权重包身份推断）",
    )
    common.add_argument("--descriptor", type=Path, default=None, help="descriptor JSON；reference 路径必填")
    common.add_argument(
        "--weights", type=Path, default=None,
        help="模型目录或 safetensors；省略时按发布来源首启下载（需网络，--offline 可关）",
    )
    common.add_argument(
        "--offline", action="store_true",
        help="不从网络下载权重；本地没有时报错并给出手动 `hf download` 命令",
    )
    common.add_argument(
        "--quant", default="int8", choices=QUANT_MODES,
        help="ANE 权重的存储形态（默认 int8：per-channel 码 + fp16 标度；fp16 则整块 16 bit）。"
             "计算始终 fp16 —— ANE 没有 int8 MAC",
    )
    common.add_argument("--attention-mode", default=None, help="causal / noncausal_full / prefix_noncausal")
    common.add_argument(
        "--batch", type=int, default=None, metavar="N",
        help="一个 pass 最多装几条请求（默认取 scheduler.batch=4；0/1 = 不攒批）。"
             "它只是策略上限：形状放不放得下由执行器判，装不下静默退回逐条",
    )
    common.add_argument("--debug-timings", action="store_true", help="返回分阶段耗时")
    common.add_argument("--bucket", type=int, default=0,
                        help="split / pooled：ANE program 的列宽（默认 1024）")
    common.add_argument("--width", type=int, default=0,
                        help=f"split / pooled：GPU 缓冲的列数（默认 {_SPLIT_WIDTH}）")
    common.add_argument(
        "--W", type=int, default=_RESIDENT_W, dest="resident",
        help=f"待命常驻的层数（默认 {_RESIDENT_W} = 一个块；0 = 全部常驻）。"
             "实测代价：h2o 32 层在 142 token 上窗口比全装慢 2.2×，"
             "长 prompt 上这笔固定开销摊薄",
    )
    common.add_argument(
        "--artifacts", type=Path, default=None,
        help="编译产物的落点（默认平台缓存目录；SYS1_ARTIFACTS_DIR 也会被读到）。"
             "清掉它等于下次启动重编一遍",
    )
    common.add_argument(
        "--vision", default="auto", choices=("auto", "off"),
        help="auto（默认）：模型有塔就收图；启动时把缺失的塔产物编好，编完归还（0 常驻）；"
             "off：纯文本服务，不建塔、不读塔权重，收图时 422 并说明原因",
    )
    common.add_argument(
        "--vision-idle", type=float, default=_VISION_IDLE, dest="vision_idle",
        help=f"视觉塔空闲多少秒后真卸载（默认 {_VISION_IDLE:.0f}；0 = 每请求后立刻卸）。"
             "卸载是把 26 个 program 卸掉、面释放，不是挂起。代价已实测：卸一次 31 ms，"
             "装回来 80 ms（其中 64 是 ANE 装载），端到端每图 +119 ms（1024²）/ +78 ms（512²）；"
             "换来空闲时释放 629–690 MiB（随桶宽）",
    )
    common.add_argument(
        "--vision-bucket", type=int, default=None, dest="vision_bucket", metavar="COLS",
        help="视觉塔自己的列宽（默认跟随链的桶）。它决定每层切几个瓦片：越窄瓦片越多、"
             "GPU 越藏得住，但 ANE 每 token 更贵、表面更小。1024² 上实测 256/512/1024/2048/4096 "
             "= 959/902/781/794/849 ms，768² 上 512 最好（427 vs 1024 的 492）。必须是 4 的倍数",
    )
    return common


def _chain_knobs_given(args: argparse.Namespace) -> list[str]:
    """用户**显式**给过的链形状参数；与默认值相同的一律当作没给。

    `assemble` 只看"值是不是 None"，认不出默认值与显式值；vendor 路径又不接这些旋钮，
    所以这一层负责把"默认值"和"用户真给了"分开，好让后者按既有的规矩报错。
    """
    given: list[str] = []
    if args.bucket:
        given.append("--bucket")
    if args.width:
        given.append("--width")
    if args.resident != _RESIDENT_W:
        given.append("--W")
    if args.quant != "int8":
        given.append("--quant")
    if args.vision != "auto":
        given.append("--vision")
    if args.vision_idle != _VISION_IDLE:
        given.append("--vision-idle")
    if args.vision_bucket is not None:
        given.append("--vision-bucket")
    return given


def _artifacts_dir(explicit: Path | None = None) -> Path:
    """产物落点：显式给的（`--artifacts`）> `SYS1_ARTIFACTS_DIR` > 平台缓存目录。"""
    if explicit is not None:
        return Path(explicit)
    configured = str(read_env()[ARTIFACTS_VARIABLE])
    return Path(configured) if configured else default_artifacts_dir()


def _weights_dir() -> Path:
    """自动下载的权重落点：`SYS1_WEIGHTS_DIR` > 平台缓存目录。

    显式 `--weights` 是一个完整路径，不经过这里：它表达"用这一份"，而不是"下到哪"。
    """
    configured = str(read_env()[WEIGHTS_VARIABLE])
    return Path(configured) if configured else default_weights_dir()


def _resolve_weights(
    explicit: Path | None, family_name: str, *, offline: bool, root: Path
) -> Path:
    """显式 `--weights` 只读；省略时才按发布来源下载到 `root`。

    显式路径不存在时**不**退化成下载：拼错一个目录名不该触发多 GB 的网络请求，也不该
    悄悄写到用户没指定的位置。
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise ConfigError(
                "权重路径不存在",
                path=str(path),
                hint="去掉 --weights 让引擎按发布来源自动下载；或给一个已有的模型目录",
            )
        return path
    from .models.sources import source_for
    from .weights.fetch import ensure_weights

    return ensure_weights(source_for(family_name), root, offline=offline)


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

    sub.add_parser(
        "compile", parents=[_model_flags()],
        help="按给定形状预编译产物，不起服务（部署时跑一次，之后起服务不必再编）",
    )
    doctor = sub.add_parser("doctor", help="打印平台能力与内存预算")
    doctor.add_argument("--platform", default="auto", choices=("auto", "reference", "apple"))
    doctor.add_argument(
        "--model", default=None, choices=families(),
        help="顺带检查这个模型：它的几何在不在 Apple 的实测白名单里（不装载权重）",
    )
    doctor.add_argument("--weights", type=Path, default=None, help="配合 --model 的模型目录")
    doctor.add_argument(
        "--artifacts", type=Path, default=None,
        help="配合 --model/--weights 顺带查产物：多少层可复用、缺哪些（默认取平台缓存目录）",
    )
    doctor.add_argument(
        "--bucket", type=int, default=0,
        help="配合 --artifacts 的桶宽（默认 1024；产物按桶宽分家，装错桶就全不命中）",
    )
    doctor.add_argument(
        "--quant", default="int8", choices=QUANT_MODES,
        help="配合 --artifacts 的量化档；它必须与装载时给的一致，否则全部不命中",
    )
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
    from .devices.apple.compile import ARTIFACT_VERSION

    try:
        report = {
            "available": True,
            "artifact_version": ARTIFACT_VERSION,
            **ApplePlatform().probe(),
        }
    except SysOneError as exc:
        return {
            "platform": "apple",
            "available": False,
            "artifact_version": ARTIFACT_VERSION,
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


def _artifact_report(
    model: str | None, weights: Path | None, *, bucket: int,
    artifacts: Path | None, quant: str = "int8",
) -> dict[str, object]:
    """这份权重的编译产物能不能被复用：逐层判据与装载完全相同。

    两个用途：部署前回答"这台机器热不热"（全命中 = 起服务时编译段是 0 秒），以及排查
    "为什么又要重编"（缺哪些层、产物落在哪个目录）。只读磁盘、不碰 ANE —— 判据
    （`_artifact_plan` + `cached_program`）是从装载那边抽出来的，所以这里不会报一个装载
    不认的"命中"。
    """
    if model is None or weights is None:
        raise ConfigError(
            "doctor --artifacts 需要 --model 与 --weights 一起给",
            model=model, weights=None if weights is None else str(weights),
        )
    from .devices.apple.chain import artifact_state
    from .devices.apple.compile import S_NATIVE

    entry = family(model)
    if entry.weight_plan is None:
        raise ConfigError(
            "这个家族没有交给设备层的权重计划，也就没有产物可查",
            family=model, strategy=entry.strategy,
            hint="vendor 策略走厂商桥，不产生产物",
        )
    if quant not in QUANT_MODES:
        raise ConfigError("未知的量化档", quant=quant, available=list(QUANT_MODES))
    s = int(bucket or S_NATIVE)
    plan = entry.weight_plan(Path(weights))
    # 落点与 serve 同一个解析（`--artifacts` > 环境变量 > 平台缓存目录）：口径不同的话，
    # doctor 说"全命中"、起服务照样重编，这个报告就没有意义了。
    state = artifact_state(weights, weights=plan, s=s, quantize=quant == "int8",
                          artifacts_dir=_artifacts_dir(artifacts))
    report: dict[str, object] = {
        "family": plan.descriptor.family,
        "s": s,
        "quant": quant,
        "directory": str(state.directory),
        "layers": {"reusable": state.hits, "total": state.total},
    }
    tower = _vision_report(weights, descriptor=plan.descriptor, bucket=s, artifacts=artifacts)
    if tower is not None:
        report["vision"] = tower
    if state.missing:
        report["missing_layers"] = list(state.missing)
        report["note"] = (
            f"起服务时要重编这 {state.total - state.hits} 层；"
            "`sysone compile` 可以把这一步提前做掉"
        )
    else:
        report["note"] = "全部可复用：起服务时编译段是 0 秒"
    return report


def _vision_report(
    weights: Path, *, descriptor: ModelDescriptor, bucket: int, artifacts: Path | None
) -> dict[str, object] | None:
    """视觉塔在这一份权重上的账：能力表（四条预算）+ 26 份产物命中几个 + 常驻多少字节。

    只读磁盘：`artifact_state` 与装载共用判据，所以这里不会报一个装载不认的"命中"。
    `bytes` 是**装起来之后**的量（26 份 blob + 一套面），不是现在占着的 —— 塔默认不装。
    """
    from .devices.apple import vision as vision_device
    from .models.vision.contract import image_contract
    from .weights.checkpoint import open_checkpoint

    if descriptor.vision is None:
        return {"available": False, "unavailable": "这个模型没有视觉塔"}
    store = open_checkpoint(weights)
    try:
        state = vision_device.artifact_state(store, descriptor=descriptor, s=bucket,
                                             artifacts_dir=_artifacts_dir(artifacts))
        # 常驻字节要真装一次才算得准，而 doctor 不该把塔装上（那是"选了常驻"）。所以报的是
        # 26 份 blob 的和，与 `VisionTower.bytes` 的差值就是那套复用的面（几十 MB 量级）。
        blobs = sum(
            (state.directory / name / "weights.bin").stat().st_size
            for name in ["patch", "merger"] + [f"block{index:02d}" for index in range(int(descriptor.vision.depth))]
            if (state.directory / name / "weights.bin").is_file()
        )
    finally:
        store.close()
    report: dict[str, object] = {
        "available": True,
        "directory": str(state.directory),
        "programs": {"reusable": state.hits, "total": state.total},
        "blob_bytes": blobs,
        "contract": image_contract(),
    }
    if state.missing:
        report["note"] = (
            f"{state.missing} 份塔产物缺失；下一次 `sysone compile` 或首启预处理会补建，"
            "补建后塔本身仍按第一次带图请求才常驻"
        )
    return report


def _scheduler(
    args: argparse.Namespace, mode: MaskMode | None, *, max_len: int | None = None
) -> SchedulerConfig:
    """把 CLI 的调度旋钮折成 `SchedulerConfig`：默认值仍只有 dataclass 那一个来源。"""
    default = SchedulerConfig()
    return SchedulerConfig(
        attention_mode=mode,
        max_len=default.max_len if max_len is None else int(max_len),
        batch=default.batch if args.batch is None else int(args.batch),
    )


def _build_engine(args: argparse.Namespace, mode: MaskMode | None) -> Engine:
    """按后端装配引擎：参考后端（M0 语义裁判）或 mude/h2o 的 Apple 链。"""
    backend = _resolve_backend(args)
    artifacts = _artifacts_dir(args.artifacts)
    weights_root = _weights_dir()
    if backend in STRATEGY_BACKENDS:
        name = args.model or _DEFAULT_FAMILY.get(backend)
        if name is None:
            raise ConfigError(
                "这个后端没有默认家族",
                backend=backend,
                hint="torchcast 已改走 pooled；厂商直通不再有出货家族",
            )
        weights = _resolve_weights(
            args.weights, name, offline=bool(args.offline), root=weights_root
        )
        width = int(args.width or _SPLIT_WIDTH)
        # 形状与执行器都由注册表装配（`engine.strategies.assembly`）：CLI 不认识任何家族名，
        # 也不认识哪条链该配哪套题面 —— 那是家族自己声明的策略。
        #
        # 链形状参数（桶宽、缓冲宽、待命窗口、视觉旋钮、ANE 量化档）只对 split / pooled 有意义：
        # vendor 的前向、缓冲与模板都在厂商引擎里。显式给了就报错 —— 与 `assemble` 的拒绝同一条
        # 规矩（静默丢一个旋钮会让"我设了它"和"它没生效"同时成立），只是这里要先分清默认值与
        # 用户真给的。
        if backend == "vendor":
            refused = _chain_knobs_given(args)
            if refused:
                raise ConfigError(
                    "这条后端不认链形状参数",
                    backend=backend,
                    refused=refused,
                    hint="厂商后端的前向与缓冲在厂商引擎里：去掉这些参数",
                )
            options: dict[str, object] = {}
        else:
            options = {
                "bucket": args.bucket or None, "width": width,
                "resident": args.resident or None,
                "vision": (args.vision != "off"),
                "vision_idle": float(args.vision_idle),
                "vision_bucket": args.vision_bucket,
            }
        assembled = assemble(
            weights, name, backend=backend, artifacts_dir=artifacts,
            quant=None if backend == "vendor" else args.quant, **options,
        )
        descriptor = assembled.descriptor
        config = EngineConfig(
            model=ModelSelection(name=weights.name, dir=weights),
            scheduler=_scheduler(args, mode, max_len=width),
            storage=StorageConfig(artifacts_dir=artifacts, weights_dir=weights_root),
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
    if args.weights is None:
        raise ConfigError(
            "reference 后端需要 --weights",
            hint="reference 只做语义裁判，不读发布来源、也不下载权重",
        )
    if args.descriptor is None:
        raise ConfigError("reference 后端需要 --descriptor",
                          hint="reference 只用于语义裁判：--backend reference --descriptor …")
    descriptor = ModelDescriptor.from_mapping(_load_json(args.descriptor))
    config = EngineConfig(
        model=ModelSelection(name=args.weights.name, dir=args.weights),
        scheduler=_scheduler(args, mode),
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

    def load() -> tuple[EngineDecider, CompilationTimes]:
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


def _compile(args: argparse.Namespace) -> int:
    """预编译：走 `serve` 完全相同的装载闭包，只把服务层换掉（不绑端口、不建就绪门）。

    R6 里"首启编译慢"的缓解手段就是它：部署时先跑一次，起服务时编译那一段就是 0 秒。
    幂等由产物缓存本身保证 —— 全部命中时它只花装载与预热的时间。
    """
    backend = _resolve_backend(args)
    if backend not in STRATEGY_BACKENDS:
        raise ConfigError(
            "compile 只对按需编译的路径有意义",
            backend=backend,
            available=list(STRATEGY_BACKENDS),
            hint="给 --backend split（mude）或 pooled（h2o）；reference 后端不产生产物",
        )
    engine = _build_engine(args, _parse_mode(args.attention_mode))
    vision = None
    try:
        times = engine.start()
        # 预处理已在 `compile_or_warm_up` 里做完：这里只读一遍磁盘账，不重新装塔。
        runner = getattr(engine, "_runner", None)
        query = None if runner is None else getattr(runner, "vision_artifacts", None)
        vision = query() if callable(query) else None
    finally:
        # 这条命令比服务寿命短：退出前把 ANE 程序与表面还回去（不还也只是进程退出）。
        engine.shutdown()
    print(f"sysone compile: {times.total_s:.2f}s{startup_report(times)}")
    print(f"artifacts: {engine.config.storage.resolved_artifacts_dir()}")
    if vision is not None:
        print(f"vision: {vision['reusable']}/{vision['total']} programs")
    return 0


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
                payload["artifacts"] = _artifact_report(
                    args.model, args.weights,
                    bucket=int(args.bucket), artifacts=args.artifacts, quant=args.quant,
                )
            _print(payload)
            return 0
        if args.command == "inspect":
            descriptor = family(args.model).parse_config(args.weights)
            _print(descriptor.summary())
            return 0
        if args.command == "serve":
            return _serve(args)
        if args.command == "compile":
            return _compile(args)
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
