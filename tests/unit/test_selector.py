"""内核选择：不支持就报错，不静默回退。"""

from __future__ import annotations

import pytest

from helpers import load_tiny_descriptor
from sysone.devices.base import Capabilities, MemoryInfo, Placement, Shape
from sysone.devices.selector import select
from sysone.domain.attention import AttentionRole, MaskMode
from sysone.domain.quant import DType, QuantSpec
from sysone.domain.work import StageOp
from sysone.errors import NoKernelError


class _Backend:
    op = StageOp.ATTENTION

    def supports(self, *, role, mode, dtype, quant) -> bool:
        return True

    def prepare(self, shape, views):
        return None

    def run(self, compiled, mask, stage):
        return None

    def verify(self) -> None:
        return None


class _Platform:
    name = "fake"

    def __init__(self, backend) -> None:
        self._backend = backend

    def capabilities(self) -> Capabilities:
        return Capabilities(
            dtypes=frozenset({DType.FP32}),
            quant=frozenset({None}),
            modes=frozenset(MaskMode),
            max_seq=0,
            max_tile=0,
        )

    def verify_model(self, descriptor) -> None:
        return None

    def verify_quant(self, quant) -> None:
        return None

    def memory(self) -> MemoryInfo:
        return MemoryInfo(total=0, current=0, usable=0)

    def default_placement(self, descriptor) -> Placement:
        return Placement("cpu")

    def backend(self, op, shape):
        return self._backend

    def compile(self, programs):
        return None

    def sync(self) -> None:
        return None


def test_no_backend_is_an_error() -> None:
    descriptor = load_tiny_descriptor()
    shape = Shape(4, descriptor.hidden, descriptor.ffn, descriptor.heads, descriptor.kv_heads, descriptor.head_dim)
    with pytest.raises(NoKernelError):
        select(
            _Platform(None),
            StageOp.ATTENTION,
            shape,
            role=AttentionRole.FULL,
            mode=MaskMode.CAUSAL,
            dtype=DType.FP32,
            quant=QuantSpec(),
        )


def test_selected_backend_is_returned() -> None:
    descriptor = load_tiny_descriptor()
    shape = Shape(4, descriptor.hidden, descriptor.ffn, descriptor.heads, descriptor.kv_heads, descriptor.head_dim)
    backend = _Backend()
    assert select(
        _Platform(backend),
        StageOp.ATTENTION,
        shape,
        role=AttentionRole.FULL,
        mode=MaskMode.PREFIX_NONCAUSAL,
        dtype=DType.FP32,
        quant=QuantSpec(),
    ) is backend
