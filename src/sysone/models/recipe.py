"""接入配方：家族交给某条策略的那几件东西。

`strategy` 声明的是"答案从哪种形态读出"——固定码表（`split`）、标签行（`pooled`）、厂商自带
（`vendor`）；执行器（`engine/strategies/*`）是**共享实现**，掩码计划、前缀、合批、产物、常驻
窗口都在那一条实现里。配方装的是这一家与那条实现的差异：怎么取答案码、题面怎么渲染、概率按
什么口径给置信度。

为什么要有这一层：在"策略名 → 执行器工厂"直接查表的时候，一个新家族只要声明
`strategy="split"` 就会静默拿到 mude 的 `MudePrompter` 与 255 行答案码表 —— 它算得出形状
正常的答案，错得没有声音。现在缺配方是装配错误（`engine/strategies/assembly`）。

配方只装可调用对象，不 import 引擎：方向仍是 `models ← engine`，家族模块照旧不必认识
执行器；反过来，执行器也不必认识任何具体家族。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class SplitRecipe:
    """`split` 的接入件：答案码表 + 题面渲染 + 置信度口径。

    `labels(model_dir)` 取这一家的答案码，顺序即读出表的行序（**随 checkpoint 走**，不从
    分词器重算：重算出来的行序错了，概率看起来仍然像模像样）。
    `prompter(tokenizer, labels)` 渲染题面，返回的对象要实现
    `engine.strategies.split.Prompter`（`encode` → 整条 prompt / 前缀长度 / 标签 / 选项序号）。
    `confidence(shares, kind)` 是这一家报给协议层的置信度口径，与"最大概率"不是一回事；
    `kind` 是题型：pplx 的厂商答面对 choice 与 score 用的是两个公式，一个钩子要能分开。
    """

    labels: Callable[[Path], tuple[str, ...]]
    prompter: Callable[..., object]
    confidence: Callable[[Sequence[float], str], float]


@dataclass(frozen=True)
class PooledRecipe:
    """`pooled` 的接入件：厂商答面（我们算 hidden，它判答案）。

    `serve(model_dir, tokenizer)` 交出厂商的答面对象：题面渲染、按题型换温度、noul 的地板与
    阈值、score 的期望值都在它里面。只有一项，是因为 pooled 的差异**全部**在答面 ——
    读出用的那几行词表由策略按标签 id 现取，不需要家族再说一遍。
    """

    serve: Callable[..., object]


@dataclass(frozen=True)
class VendorRecipe:
    """`vendor` 的接入件：厂商桥。

    `bridge(model_dir)` 建一个厂商桥（模板、前向与读出全归厂商）。它通常 import torch，
    所以家族那边要写成惰性函数：启动路径不该因为"注册了这个家族"就把 torch 拖进来。
    """

    bridge: Callable[[Path], object]


__all__ = ["PooledRecipe", "SplitRecipe", "VendorRecipe"]
