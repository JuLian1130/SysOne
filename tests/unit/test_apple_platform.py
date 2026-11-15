"""Apple 平台骨架的加载期门禁：几何白名单、量化形态、放置（不需要跑 ANE）。"""

import dataclasses

import pytest

from sysone.devices.apple import (
    MEASURED_GEOMETRIES,
    MAX_SEQ,
    MAX_TILE,
    ApplePlatform,
    available,
)
from sysone.devices.base import Shape
from sysone.domain.attention import MaskMode
from sysone.domain.quant import DType, QuantSpec, Scales, Scheme, Storage
from sysone.domain.work import StageOp
from sysone.errors import ConfigError, UnsupportedPrecisionError
from sysone.models.families import mude

from helpers import FIXTURES, load_tiny_descriptor

PER_CHANNEL = QuantSpec(Scheme.PER_CHANNEL_SYM_INT8, Scales.FP16, Storage.MIXED_BLOB)
PER_ROW = QuantSpec(Scheme.PER_ROW_SYM_INT8, Scales.FP32, Storage.TENSOR)

needs_apple = pytest.mark.skipif(not available(), reason="没有可用的 ANE dylib（先跑 tools/build_ane.sh）")


def test_capabilities_are_the_apple_ones():
    caps = ApplePlatform().capabilities()
    assert caps.dtypes == frozenset({DType.FP16})
    assert caps.quant == frozenset({Scheme.NONE, Scheme.PER_CHANNEL_SYM_INT8})
    assert caps.modes == frozenset(MaskMode)
    assert caps.zero_copy is True
    assert (caps.max_seq, caps.max_tile) == (MAX_SEQ, MAX_TILE)


def test_verify_model_accepts_a_measured_geometry():
    """白名单里唯一有端到端证据的几何：mude 的 (2048, 6144)。"""
    ApplePlatform().verify_model(mude.parse_config(FIXTURES / "mude"))


def test_verify_model_rejects_a_geometry_outside_the_whitelist():
    """白名单是**精确集合**，不是尺寸上限：小几何没实测过也一样拒绝。"""
    descriptor = dataclasses.replace(load_tiny_descriptor(), hidden=2560, ffn=9216)
    with pytest.raises(ConfigError) as excinfo:
        ApplePlatform().verify_model(descriptor)
    context = excinfo.value.context
    assert (context["hidden"], context["ffn"]) == (2560, 9216)
    assert context["measured"] == [
        {"hidden": hidden, "ffn": ffn} for hidden, ffn in sorted(MEASURED_GEOMETRIES)
    ]


def test_a_long_advertised_context_is_the_capability_ceiling_not_a_rejection():
    """模型自报的 `max_position` 大于平台的 `MAX_SEQ` 不是拒绝理由：mude 自报 262144、出货 16k，
    而 16k 就是 `MAX_SEQ`。平台只在 `capabilities().max_seq` 里声明自己服务到哪。"""
    descriptor = dataclasses.replace(
        mude.parse_config(FIXTURES / "mude"), max_position=MAX_SEQ * 16
    )
    ApplePlatform().verify_model(descriptor)
    assert ApplePlatform().capabilities().max_seq == MAX_SEQ


@pytest.mark.parametrize("quant", [QuantSpec(), PER_CHANNEL])
def test_verify_quant_accepts_ane_forms(quant):
    ApplePlatform().verify_quant(quant)


def test_verify_quant_rejects_row_int8():
    with pytest.raises(UnsupportedPrecisionError) as excinfo:
        ApplePlatform().verify_quant(PER_ROW)
    assert "per-channel" in excinfo.value.message


def test_placement_and_backend_are_m2_0_stubs():
    platform = ApplePlatform()
    assert str(platform.default_placement(load_tiny_descriptor())) == "ane"
    shape = Shape(s=64, hidden=16, ffn=32, heads=2, kv_heads=1, head_dim=8)
    assert platform.backend(StageOp.NORM_TO, shape) is None
    with pytest.raises(ConfigError):
        platform.compile(object())


@needs_apple
def test_memory_and_probe_report_real_numbers():
    platform = ApplePlatform()
    info = platform.memory()
    assert info.total > 0
    assert info.usable > 0
    report = platform.probe()
    assert str(report["dylib"]).endswith("libe8ane.dylib")
    assert report["programs"] == 0
    assert report["dtypes"] == ["fp16"]
