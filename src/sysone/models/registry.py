"""模型家族注册表：每个家族提供两个纯函数，外加一条"它的答案怎么读"的声明。

`weight_plan` 是可选的第三项：把"这份检查点怎么存"（前缀、norm 偏移、RoPE 角基）交给
设备层的那四项纯数据。它进注册表是为了让**不装配执行器**的调用方也能拿到它（`sysone
doctor` 查产物命中就是这种：装执行器要先起 ANE，而查缓存不该有这个前提）；`vendor` 策略
不经过设备链，所以可以不给。

旧实现靠 `named_modules()` 里的形状去猜 text backbone，并且把厂商代码用
monkeypatch 挂进来。这里把"读懂 config"（`parse_config`）与"切层"（`build_layer_plan`）
固定成注册项，厂商权重装载放到 `sysone.models.vendors`（M1）。

`strategy` 是**声明**不是实现：家族只说"我的答案从这个形态读出"（固定码表 = `split`、
标签行 = `pooled`、厂商自带读出 = `vendor`），具体哪个执行器接它由
`engine.strategies.assembly` 查。这样 CLI 不必认识任何家族名，家族也不必 import 引擎。

`recipe` 是同一份声明的另一半：策略挑执行器，配方给**这一家**的答案码、题面与判答口径
（`models.recipe`）。两者成对才能装配 —— 只有策略名时，新家族会静默借走前一家的题面。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import EngineConfig
from ..errors import RegistrationError, UnknownModelError
from .descriptor import ModelDescriptor
from .layer_plan import LayerPlan, WeightPlan

ParseConfig = Callable[[Path], ModelDescriptor]
BuildLayerPlan = Callable[[ModelDescriptor, EngineConfig], LayerPlan]
WeightPlanFn = Callable[[Path], WeightPlan]


@dataclass(frozen=True)
class Family:
    """一个已注册的模型家族。

    `recipe` 是这一家交给策略的接入件（`models.recipe` 的三种之一）。它与 `strategy` 成对
    才有意义：策略挑执行器，配方给这家自己的题面与读出。缺配方时装配**必须报错** ——
    早先的写法按策略名直接查工厂，新家族声明同一个策略就会静默借走前一家的题面。

    `params` 是这一家**对应哪一档参数量**，形如 `"4.54B"`：一个家族名固定一份发布检查点
    （同一个模型可能发 2B / 4B / 8B 几档，家族名只说"哪一家"、不说"多大"），所以这个数写在
    注册项里，由 `--model` 的帮助、`sysone inspect` 与文档一起念出来。它是**实测值**：按
    safetensors 头里的形状乘积求和，不是厂商卡片上的约数（torchcast 因为有解绑的 `lm_head`
    比同一个基座的其它 4B 多 0.64B）。读不出来（比如自定义 `--weights` 换了一档）时它仍然
    是这家的发布档位 —— 那是"我们这个家族名承诺什么"，不是"你手上这份有多大"。
    """

    name: str
    parse_config: ParseConfig
    build_layer_plan: BuildLayerPlan
    strategy: str
    weight_plan: WeightPlanFn | None = None
    recipe: object | None = None
    params: str = ""
    source: str = ""


_REGISTRY: dict[str, Family] = {}


def register_family(
    name: str,
    parse_config: ParseConfig,
    build_layer_plan: BuildLayerPlan,
    *,
    strategy: str,
    weight_plan: WeightPlanFn | None = None,
    recipe: object | None = None,
    params: str = "",
    source: str = "",
) -> None:
    """注册家族；重复注册直接报错（同名两份实现是配置事故，不该静默覆盖）。

    `recipe` 与 `strategy` 配套：策略是共享实现，配方是这一家的差异（见 `models.recipe`）。
    注册时**不**强制有配方 —— 装配才是"能不能接上"的判据，报错也就在那一步说清楚缺什么。
    """
    if not strategy:
        raise RegistrationError("家族必须声明读出策略", family=name, source=source)
    if name in _REGISTRY:
        raise RegistrationError("模型家族重复注册", family=name, source=source or _REGISTRY[name].source)
    _REGISTRY[name] = Family(
        name=name,
        parse_config=parse_config,
        build_layer_plan=build_layer_plan,
        strategy=strategy,
        weight_plan=weight_plan,
        recipe=recipe,
        params=params,
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
