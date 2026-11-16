"""模型家族注册表：每个家族提供两个纯函数，外加一条"它的答案怎么读"的声明。

旧实现靠 `named_modules()` 里的形状去猜 text backbone，并且把厂商代码用
monkeypatch 挂进来。这里把"读懂 config"（`parse_config`）与"切层"（`build_layer_plan`）
固定成注册项，厂商权重装载放到 `sysone.models.vendors`（M1）。

`strategy` 是**声明**不是实现：家族只说"我的答案从这个形态读出"（固定码表 = `split`、
标签行 = `pooled`、厂商自带读出 = `vendor`），具体哪个执行器接它由
`engine.strategies.assembly` 查。这样 CLI 不必认识任何家族名，家族也不必 import 引擎。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import EngineConfig
from ..errors import RegistrationError, UnknownModelError
from .descriptor import ModelDescriptor
from .layer_plan import LayerPlan

ParseConfig = Callable[[Path], ModelDescriptor]
BuildLayerPlan = Callable[[ModelDescriptor, EngineConfig], LayerPlan]


@dataclass(frozen=True)
class Family:
    """一个已注册的模型家族。"""

    name: str
    parse_config: ParseConfig
    build_layer_plan: BuildLayerPlan
    strategy: str
    source: str = ""


_REGISTRY: dict[str, Family] = {}


def register_family(
    name: str,
    parse_config: ParseConfig,
    build_layer_plan: BuildLayerPlan,
    *,
    strategy: str,
    source: str = "",
) -> None:
    """注册家族；重复注册直接报错（同名两份实现是配置事故，不该静默覆盖）。"""
    if not strategy:
        raise RegistrationError("家族必须声明读出策略", family=name, source=source)
    if name in _REGISTRY:
        raise RegistrationError("模型家族重复注册", family=name, source=source or _REGISTRY[name].source)
    _REGISTRY[name] = Family(
        name=name,
        parse_config=parse_config,
        build_layer_plan=build_layer_plan,
        strategy=strategy,
        source=source,
    )


def family(name: str) -> Family:
    """按名字取家族；未知 → `UnknownModelError`（服务层回 400）。"""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownModelError(
            "模型家族未注册", family=name, available=sorted(_REGISTRY)
        ) from None


def families() -> tuple[str, ...]:
    """已注册家族（排序后返回，便于稳定输出）。"""
    return tuple(sorted(_REGISTRY))
