"""MIL 文本的生成测试：形状参数化、权重引用、int8 形式、三段计算的拼接。"""

import pytest

from sysone.devices.apple.compile import (
    BUILD_INFO,
    ffn_mil,
    out_mil,
    projection_mil,
)
from sysone.errors import NotReadyError
from sysone.weights.packing.ane_blob import BlobEntry


def _entry(offset: int = 128, scale_offset: int | None = None, shape: tuple[int, int] = (16, 8)) -> BlobEntry:
    return BlobEntry(
        name="w",
        dtype="int8" if scale_offset is not None else "fp16",
        offset=offset,
        nbytes=64,
        shape=shape,
        scale_offset=scale_offset,
    )


def test_fp16_mil_shape_and_blob_reference():
    mil = projection_mil(hidden=8, cout=16, s=32, entry=_entry(128))
    assert mil.startswith("program(1.3)\n")
    assert BUILD_INFO in mil
    assert "func main<ios18>(tensor<fp16, [1, 8, 1, 32]> x)" in mil
    assert "tensor<fp16, [1, 16, 1, 32]> y = conv(" in mil
    assert "tensor<fp16, [16, 8, 1, 1]> p_w = const()" in mil
    assert 'BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(128))' in mil
    assert "constexpr_affine_dequantize" not in mil
    # 一个 program 只有一个 func：M2.1 还没有 FFN / attention 的 proc。
    assert mil.count("func ") == 1


def test_int8_mil_uses_dequantize_form():
    mil = projection_mil(hidden=8, cout=16, s=32, entry=_entry(128, scale_offset=400))
    assert "constexpr_affine_dequantize()[axis = int32(0)" in mil
    assert "quantized_data = tensor<int8, [16, 8, 1, 1]>(BLOBFILE(" in mil
    assert "scale = tensor<fp16, [16]>(BLOBFILE(" in mil
    assert "offset = uint64(128)" in mil and "offset = uint64(400)" in mil
    assert "zero_point = int8(0)" in mil


def test_mil_geometry_is_parameterized():
    """同一份生成器要能出不同几何 —— 4B 复用与 16k 分块的前提。"""
    small = projection_mil(hidden=8, cout=16, s=32, entry=_entry())
    large = projection_mil(hidden=2048, cout=6144, s=1024, entry=_entry(0))
    assert "[1, 8, 1, 32]" in small and "[1, 2048, 1, 1024]" in large
    assert "[16, 8, 1, 1]" in small and "[6144, 2048, 1, 1]" in large


@pytest.mark.parametrize("kwargs", [
    {"hidden": 0, "cout": 16, "s": 32},
    {"hidden": 8, "cout": 0, "s": 32},
    {"hidden": 8, "cout": 16, "s": 0},
])
def test_bad_shapes_are_rejected(kwargs):
    with pytest.raises(NotReadyError):
        projection_mil(entry=_entry(), **kwargs)


def test_out_mil_slices_input_and_adds_residual():
    """out 投影的输入是 [attn_out | residual] 两段拼接，输出要再加回残差。"""
    mil = out_mil(hidden=8, s=32, entry=_entry(offset=64, shape=(8, 8)))
    assert "func main<ios18>(tensor<fp16, [1, 16, 1, 32]> x)" in mil
    assert "tensor<fp16, [1, 8, 1, 32]> v = slice_by_size(" in mil
    assert "tensor<fp16, [1, 8, 1, 32]> r = slice_by_size(" in mil
    assert "val = tensor<int32, [4]>([0, 0, 0, 0])" in mil
    assert "val = tensor<int32, [4]>([0, 8, 0, 0])" in mil
    assert "tensor<fp16, [8, 8, 1, 1]> o_w = const()" in mil
    assert "tensor<fp16, [1, 8, 1, 32]> y = add(x = ov, y = r)" in mil
    assert mil.count("conv(") == 1
    assert mil.count("func ") == 1


def test_ffn_mil_splits_k_axis_and_fuses_activation():
    """融合 FFN：gate/up 两个 conv → silu×up → down 按 ffn/hidden 切段求和 → 残差。"""
    ffn, hidden = 24, 8
    entries = {
        "gate": _entry(offset=100, shape=(ffn, hidden)),
        "up": _entry(offset=200, shape=(ffn, hidden)),
        "down0": _entry(offset=300, shape=(hidden, hidden)),
        "down1": _entry(offset=400, shape=(hidden, hidden)),
        "down2": _entry(offset=500, shape=(hidden, hidden)),
    }
    mil = ffn_mil(hidden=hidden, ffn=ffn, s=32, entries=entries)
    assert "func main<ios18>(tensor<fp16, [1, 16, 1, 32]> x)" in mil
    assert "tensor<fp16, [1, 24, 1, 32]> gt = conv(" in mil
    assert "tensor<fp16, [1, 24, 1, 32]> gs = silu(x = gt)" in mil
    assert "mul(x = gs, y = up)" in mil
    # 三个 down 段各自切 K 轴，偏移是 0 / 8 / 16
    for begin in (0, 8, 16):
        assert f"val = tensor<int32, [4]>([0, {begin}, 0, 0])" in mil
    assert "tensor<fp16, [8, 8, 1, 1]> k0_w = const()" in mil
    assert "tensor<fp16, [8, 8, 1, 1]> k1_w = const()" in mil
    assert "tensor<fp16, [8, 8, 1, 1]> k2_w = const()" in mil
    assert mil.count("conv(") == 5, "gate/up 各一，down 切成 3 段各一"
    # 两两求和，再加残差
    assert "tensor<fp16, [1, 8, 1, 32]> s1 = add(x = y0, y = y1)" in mil
    assert "tensor<fp16, [1, 8, 1, 32]> s = add(x = s1, y = y2)" in mil
    assert "tensor<fp16, [1, 8, 1, 32]> y = add(x = s, y = r)" in mil
    assert mil.count("func ") == 1


def test_ffn_mil_requires_every_down_chunk():
    entries = {
        "gate": _entry(shape=(24, 8)),
        "up": _entry(shape=(24, 8)),
        "down0": _entry(shape=(8, 8)),
        "down1": _entry(shape=(8, 8)),
    }
    with pytest.raises(NotReadyError) as excinfo:
        ffn_mil(hidden=8, ffn=24, s=32, entries=entries)
    assert excinfo.value.context["missing"] == ["down2"]
