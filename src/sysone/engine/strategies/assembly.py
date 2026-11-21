"""按注册表装配执行器：CLI 不认识任何具体家族，家族也不 import 引擎。

一次装配要决定两件事，今天都由家族自己声明：

1. **形状**：`parse_config(model_dir)` → `ModelDescriptor`（引擎的 `SchedulerConfig` 与
   内存预算都吃它）；
2. **谁来做前向**：家族声明 `strategy`，这里查"策略 → 执行器工厂"。`split` 是 mude 的
   码表读出（整链 ANE + Metal 注意力），`pooled` 是 h2o 的标签行读出，`vendor` 是
   torchcast 的厂商引擎（它的模板、前向与读出，我们只提供权重）。

不在这里做"猜家族"：`--model` 显式给，猜错家族不会崩，只会拿错模板算出看似正常的答案。
后端与家族声明的策略不符时**报错并说出两边**，而不是悄悄按后端跑 —— 那样
`--backend split --model h2o` 会走一条根本没为它验证过的路。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ...errors import ConfigError
from ...models import families as _family_modules  # noqa: F401  导入即注册
from ...models.descriptor import ModelDescriptor
from ...models.registry import family
from .pooled import h2o_runner
from .split import mude_runner
from .vendor import torchcast_runner

# 可选的 `--backend`（`reference` 不走这里：它是 M0 的语义裁判，不读家族的权重）。
BACKENDS = ("split", "pooled", "vendor")
# ANE 权重的存储形态：int8（默认，per-channel + fp16 scale）或 fp16。计算始终 fp16 ——
# ANE 没有 int8 MAC，int8 只省权重字节与 DMA。
QUANT_MODES = ("int8", "fp16")

# 策略名 → 工厂。`bucket` / `width` 是链的形状（两条链策略都有）。
_BUILDERS: dict[str, Callable[..., object]] = {
    "split": mude_runner,
    "pooled": h2o_runner,
    "vendor": torchcast_runner,
}

# 每条策略**认**哪些参数（`resident` 是 Metal 流水线的待命窗口，两条链策略都有 Metal 形态）。
# 给了不认的参数要报错 —— 静默丢掉一个用户显式给的旋钮，会让"我设了窗口"和"它没生效"
# 同时成立。
_OPTIONS: dict[str, frozenset[str]] = {
    "split": frozenset({"bucket", "width", "resident", "quant", "vision", "vision_idle",
                        "vision_bucket"}),
    "pooled": frozenset({"bucket", "width", "resident", "quant", "vision", "vision_idle",
                         "vision_bucket"}),
    # vendor 的前向、缓冲与模板都在厂商引擎里：桶宽、待命窗口、视觉塔一个也不接。
    "vendor": frozenset(),
}

# 三条策略各有执行器，所以"装配不出来"只剩一种来源：家族声明了一个这里没有工厂的策略名。
# 那时报错要说清缺的是哪一步（工厂），而不是说"没有执行器" —— 执行器可能存在，缺的是
# 「用哪个桥、怎么构造」。


@dataclass(frozen=True)
class Assembly:
    """一次装配的结果：描述符、执行器，以及实际选中的策略名。"""

    descriptor: ModelDescriptor
    runner: object
    strategy: str


def assemble(
    model_dir: str | Path,
    name: str,
    *,
    backend: str,
    bucket: int | None = None,
    width: int | None = None,
    resident: int | None = None,
    quant: str | None = None,
    vision: bool | None = None,
    vision_idle: float | None = None,
    vision_bucket: int | None = None,
    artifacts_dir: str | Path | None = None,
) -> Assembly:
    """按家族声明装配一条执行器；后端与声明不符、或该策略还没有执行器时抛出 `ConfigError`。

    `artifacts_dir` 是编译产物的落点（部署概念）；执行器那一层叫它 `scratch_dir`，因为对链
    来说那只是一处"把 MIL/blob 放哪"的目录 —— 只有 CLI 与配置层知道它应当持久。

    `quant` 是 ANE 权重的存储形态（`int8` / `fp16`）；装配层把它翻译成执行器认的布尔
    `quantize`，因为设备层只关心"现量化还是用容器自带的码"，不认识档位名字。产物目录也按
    它分家（`s<s>` 对 `s<s>-fp16`），两档不会互相顶掉 sidecar。

    `vision` / `vision_idle` / `vision_bucket` 是视觉那条路的三个旋钮（`False` = 纯文本服务；
    `vision_idle` 是塔的空闲卸载秒数；`vision_bucket` 是塔自己的列宽，`None` = 跟随链的桶）。
    它们**不进 `cache_key`**：塔的产物身份由权重与形状决定，与"这次要不要收图"无关 —— 关掉
    视觉再打开，26 份产物照样命中。**但 `vision_bucket` 会进塔自己的产物身份**（`_identity`
    里带 `s`），不同列宽的塔各有各的产物目录，不会互相认领。
    """
    entry = family(name)
    # 先问"这个家族的策略装配得起来吗"，再问"你选的这个后端对不对"：反过来的话，一个策略
    # 还没有工厂的家族会先撞上"后端不符"，然后建议一个 `--backend` 里选不出来的值。
    builder = _BUILDERS.get(entry.strategy)
    if builder is None:
        raise ConfigError(
            "这个家族声明的策略装配不出来",
            family=name,
            strategy=entry.strategy,
            ready=sorted(_BUILDERS),
            hint="由 --backend 直接选的是 split（码表读出）、pooled（标签行读出）"
                 "与 vendor（厂商自带读出）",
        )
    if backend != entry.strategy:
        raise ConfigError(
            "后端的执行策略与家族声明不符",
            family=name,
            backend=backend,
            strategy=entry.strategy,
            available=list(BACKENDS),
            hint=f"{name} 的答案是按 {entry.strategy} 读出的，用 --backend {entry.strategy}",
        )
    given: dict[str, object] = {
        "bucket": bucket, "width": width, "resident": resident, "quant": quant,
        "vision": vision, "vision_idle": vision_idle, "vision_bucket": vision_bucket,
    }
    accepted = _OPTIONS[entry.strategy]
    refused = sorted(key for key, value in given.items() if value is not None and key not in accepted)
    if refused:
        raise ConfigError(
            "这条策略不认这些参数",
            strategy=entry.strategy,
            refused=refused,
            accepted=sorted(accepted),
        )
    if quant is not None and quant not in QUANT_MODES:
        raise ConfigError("未知的量化档", quant=quant, available=list(QUANT_MODES))
    root = Path(model_dir)
    descriptor = entry.parse_config(root)
    # `quant` 是装配层的名字，执行器只认 `quantize: bool`：别把两套名字同时递给工厂，
    # 加一个参数就会在 `mude_runner` 的签名上撞成 TypeError。
    options = {key: value for key, value in given.items()
               if key in accepted and key != "quant" and value is not None}
    if quant is not None:
        options["quantize"] = quant == "int8"
    runner = builder(root, **options, scratch_dir=artifacts_dir)
    return Assembly(descriptor=descriptor, runner=runner, strategy=entry.strategy)


__all__ = ["BACKENDS", "QUANT_MODES", "Assembly", "assemble"]
