"""缓冲声明与内存预算：内存计划从声明推导，而不是在驱动里手算。

旧实现把 `wired_gb` / 池大小散在 `e8_a1m.py` 的常量与 env 里；这里每个缓冲区声明
自己的"每 token 字节"与"每层固定字节"，`bytes_for()` 是唯一算法，预算比较集中在
`MemoryPlan`。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping

from .quant import DType


class BufferName(StrEnum):
    """引擎会声明的缓冲区种类。"""

    H = "h"
    GDN_STATE = "gdn_state"
    KV = "kv"
    RING = "ring"
    SCRATCH = "scratch"


@dataclass(frozen=True)
class BufferSpec:
    """一个缓冲区的空间声明（单位：字节）。

    `per_token` 是每 token 的字节数（已含 dtype 宽度），`per_layer` 是每层固定字节数。
    """

    name: BufferName
    dtype: DType
    per_token: int = 0
    per_layer: int = 0
    shareable: bool = False

    def bytes_for(self, *, seq_len: int, prefix_len: int = 0, layers: int = 1) -> int:
        """给定本次工作负载所需的字节数。"""
        return self.per_token * (seq_len + prefix_len) + self.per_layer * layers


@dataclass(frozen=True)
class MemoryPlan:
    """把若干缓冲声明换算成总量并与预算比较。"""

    buffers: Mapping[BufferName, int]
    artifacts: int
    budget: int

    @property
    def total(self) -> int:
        return sum(self.buffers.values()) + self.artifacts

    def fits(self) -> bool:
        return self.budget <= 0 or self.total <= self.budget

    def report(self) -> str:
        detail = " + ".join(f"{name.value}={size}" for name, size in self.buffers.items())
        return f"{detail} + artifacts={self.artifacts} / budget={self.budget or 'unlimited'}"
