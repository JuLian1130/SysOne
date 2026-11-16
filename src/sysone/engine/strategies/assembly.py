"""按注册表装配执行器：CLI 不认识任何具体家族，家族也不 import 引擎。

一次装配要决定两件事，今天都由家族自己声明：

1. **形状**：`parse_config(model_dir)` → `ModelDescriptor`（引擎的 `SchedulerConfig` 与
   内存预算都吃它）；
2. **谁来做前向**：家族声明 `strategy`，这里查"策略 → 执行器工厂"。`split` 是 mude 的
   码表读出（整链 ANE + Metal 注意力），`pooled` 是 h2o 的标签行读出。

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

# 可选的 `--backend`（`reference` 不走这里：它是 M0 的语义裁判，不读家族的权重）。
BACKENDS = ("split", "pooled")

# 策略名 → 工厂。`bucket` / `width` 是链的形状（两种策略都有）。
_BUILDERS: dict[str, Callable[..., object]] = {
    "split": mude_runner,
    "pooled": h2o_runner,
}

# 每条策略**认**哪些参数。`resident` 是 Metal 流水线的待命窗口：pooled 今天走 numpy 那侧，
# 没有换窗这回事，所以不认它。给了不认的参数要报错 —— 静默丢掉一个用户显式给的旋钮，
# 会让"我设了窗口"和"它没生效"同时成立。
_OPTIONS: dict[str, frozenset[str]] = {
    "split": frozenset({"bucket", "width", "resident"}),
    "pooled": frozenset({"bucket", "width"}),
}


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
) -> Assembly:
    """按家族声明装配一条执行器；后端与声明不符、或该策略还没有执行器时抛出 `ConfigError`。"""
    entry = family(name)
    # 先问"这个家族的策略有没有执行器"，再问"你选的这个后端对不对"：反过来的话，一个只有
    # `vendor` 执行器的家族（torchcast 是这种）会先撞上"后端不符"，然后建议一个 `--backend`
    # 根本选不出来的值。
    builder = _BUILDERS.get(entry.strategy)
    if builder is None:
        raise ConfigError(
            "这个家族声明的策略还没有执行器",
            family=name,
            strategy=entry.strategy,
            ready=sorted(_BUILDERS),
            hint="今天只有 split（码表读出）与 pooled（标签行读出）两条；vendor 那条走厂商桥",
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
    given = {"bucket": bucket, "width": width, "resident": resident}
    accepted = _OPTIONS[entry.strategy]
    refused = sorted(key for key, value in given.items() if value is not None and key not in accepted)
    if refused:
        raise ConfigError(
            "这条策略不认这些参数",
            strategy=entry.strategy,
            refused=refused,
            accepted=sorted(accepted),
        )
    root = Path(model_dir)
    descriptor = entry.parse_config(root)
    options = {key: value for key, value in given.items() if key in accepted}
    runner = builder(root, **options)
    return Assembly(descriptor=descriptor, runner=runner, strategy=entry.strategy)


__all__ = ["BACKENDS", "Assembly", "assemble"]
