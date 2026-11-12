"""domain 层的不变量：量化组合、缓冲区尺寸、计划一致性。"""

from __future__ import annotations

import dataclasses

import pytest

from helpers import load_tiny_descriptor
from sysone.domain.attention import MaskMode
from sysone.domain.buffers import BufferName, BufferSpec, MemoryPlan
from sysone.domain.quant import DType, QuantSpec, Scales, Scheme, Storage
from sysone.domain.work import ResidencyPlan, WorkPlan
from sysone.engine.planner import build_mask_plan
from sysone.errors import ConfigError, MaskPlanError


def test_quant_spec_allows_three_combinations() -> None:
    QuantSpec(Scheme.NONE, Scales.NONE, Storage.TENSOR).validate()
    QuantSpec(Scheme.PER_ROW_SYM_INT8, Scales.FP32, Storage.TENSOR).validate()
    QuantSpec(Scheme.PER_CHANNEL_SYM_INT8, Scales.FP16, Storage.MIXED_BLOB).validate()


def test_quant_spec_rejects_mismatch() -> None:
    with pytest.raises(ConfigError):
        QuantSpec(Scheme.PER_ROW_SYM_INT8, Scales.FP16, Storage.TENSOR).validate()


def test_buffer_bytes() -> None:
    spec = BufferSpec(BufferName.H, DType.FP32, per_token=64, per_layer=1024, shareable=True)
    assert spec.bytes_for(seq_len=4, prefix_len=2, layers=2) == 64 * 6 + 1024 * 2


def test_memory_plan_budget() -> None:
    plan = MemoryPlan(buffers={BufferName.H: 100}, artifacts=0, budget=0)
    assert plan.total == 100
    assert plan.fits() is True
    assert MemoryPlan(buffers={BufferName.H: 100}, artifacts=0, budget=50).fits() is False


def test_work_plan_rejects_length_mismatch() -> None:
    mask = build_mask_plan(MaskMode.CAUSAL, 4, 0, 2)
    with pytest.raises(MaskPlanError):
        WorkPlan(
            lane="x",
            tokens=(1, 2, 3),
            prefix_len=0,
            questions=(),
            mask=mask,
            residency=ResidencyPlan(),
        )


def test_descriptor_default_mode_must_be_supported() -> None:
    descriptor = load_tiny_descriptor()
    broken = dataclasses.replace(
        descriptor,
        supported_modes=frozenset({MaskMode.NONCAUSAL_FULL}),
        default_mode=MaskMode.PREFIX_NONCAUSAL,
    )
    with pytest.raises(ConfigError):
        broken.validate()


def test_descriptor_roundtrip() -> None:
    descriptor = load_tiny_descriptor()
    again = type(descriptor).from_mapping(descriptor.to_mapping())
    assert again == descriptor
