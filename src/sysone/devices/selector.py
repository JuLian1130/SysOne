"""算子实现选择器：形状 + 角色 + 模式 + 精度 → 一个 `KernelBackend`。

选择规则只有一条：平台给出的实现必须声明支持这组参数；否则 `NoKernelError`。
不做静默回退（静默换精度/换模式正是旧实现最难查的坑）。
"""

from __future__ import annotations

from ..domain.attention import AttentionRole, MaskMode
from ..domain.quant import DType, QuantSpec
from ..domain.work import StageOp
from ..errors import NoKernelError
from .base import KernelBackend, Platform, Shape


def select(
    platform: Platform,
    op: StageOp,
    shape: Shape,
    *,
    role: AttentionRole,
    mode: MaskMode,
    dtype: DType,
    quant: QuantSpec,
) -> KernelBackend:
    """选出唯一实现；找不到或能力不匹配都在这里报错。"""
    backend = platform.backend(op, shape)
    if backend is None:
        raise NoKernelError(
            "平台没有该算子的实现",
            platform=platform.name,
            op=op.value,
            shape=f"s={shape.s},hidden={shape.hidden}",
        )
    if not backend.supports(role=role, mode=mode, dtype=dtype, quant=quant):
        raise NoKernelError(
            "算子实现不支持该参数组合",
            platform=platform.name,
            op=op.value,
            role=role.value,
            mode=mode.value,
            dtype=dtype.value,
            quant=str(quant),
        )
    return backend
