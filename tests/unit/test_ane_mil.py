"""MIL 文本的生成测试：形状参数化、权重引用、int8 形式。"""

import pytest

from sysone.devices.apple.compile import BUILD_INFO, projection_mil
from sysone.errors import NotReadyError
from sysone.weights.packing.ane_blob import BlobEntry


def _entry(offset: int = 128, scale_offset: int | None = None) -> BlobEntry:
    return BlobEntry(
        name="w",
        dtype="int8" if scale_offset is not None else "fp16",
        offset=offset,
        nbytes=64,
        shape=(16, 8),
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
