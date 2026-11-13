"""ANE blob 容器的纯 numpy 侧测试：布局、偏移、量化误差。"""

import struct
from pathlib import Path

import numpy as np
import pytest

from sysone.errors import NotReadyError
from sysone.weights.packing import ane_blob


def _weights():
    rng = np.random.default_rng(3)
    big = (rng.standard_normal((8, 1024)) * 0.05).astype(np.float16)
    small = (rng.standard_normal((16,)) * 0.5).astype(np.float16)
    return [("big", big), ("bias", small)]


def test_build_layout_matches_container():
    blob, entries = ane_blob.build(_weights(), quantize=True)
    count, version = struct.unpack_from("<II", blob, 0)
    # big -> (codes, scale) 两条；bias 是一维，保持 fp16 一条。
    assert (count, version) == (3, 2)

    for name, entry in entries.items():
        magic, dtype, nbytes, data_at = struct.unpack_from("<IIQQ", blob, entry.offset)
        assert magic == ane_blob.MAGIC
        assert dtype == ane_blob._DTYPE_CODE[entry.dtype]
        assert nbytes == entry.nbytes
        assert data_at == entry.offset + ane_blob.ENTRY_BYTES
        assert data_at + nbytes <= len(blob)

    codes = entries["big"]
    assert codes.quantized and codes.scale_offset is not None
    assert codes.scale_offset == codes.offset + ane_blob.ENTRY_BYTES + codes.nbytes
    assert entries["bias"].dtype == "fp16" and entries["bias"].scale_offset is None


def test_quantized_payload_roundtrip():
    blob, entries = ane_blob.build([("w", (np.random.default_rng(4).standard_normal((6, 32)) * 0.1))],
                                   quantize=True)
    entry = entries["w"]
    codes = np.frombuffer(blob, dtype=np.int8, count=entry.nbytes, offset=entry.offset + 64)
    scale = np.frombuffer(blob, dtype=np.float16, count=6, offset=entry.scale_offset + 64)
    w = (np.random.default_rng(4).standard_normal((6, 32)) * 0.1).astype(np.float16)
    q = ane_blob.quant_per_channel(w)
    np.testing.assert_array_equal(codes.reshape(6, 32), q.codes)
    np.testing.assert_array_equal(scale, q.scale)
    # 反量化结果与写入前的一致（fp16 存储，误差只有 scale 本身）。
    np.testing.assert_array_equal(ane_blob.dequantize(codes.reshape(6, 32), scale), q.dequant)


def test_quantization_error_bound():
    rng = np.random.default_rng(5)
    w = (rng.standard_normal((4, 64)) * 0.1).astype(np.float16)
    q = ane_blob.quant_per_channel(w)
    # 上界是半个 fp16 步长，再加一次 fp16 落盘舍入（`dequant` 是 fp16，乘积也要存成 fp16）。
    half = q.scale.astype(np.float32) / 2.0
    slack = np.spacing(q.dequant).astype(np.float32).max(axis=1)
    err = np.abs(q.dequant.astype(np.float32) - w.astype(np.float32)).max(axis=1)
    assert np.all(err <= half + slack)
    # 相对误差也要在量级上：最大值的那一行不该超过 1 % 的步长。
    assert np.all(err <= half * 1.01 + 1e-9)


def test_zero_row_keeps_zero_codes():
    w = np.zeros((3, 8), dtype=np.float16)
    w[1] = 1.0
    q = ane_blob.quant_per_channel(w)
    assert np.all(q.codes[0] == 0) and np.all(q.codes[2] == 0)
    assert q.scale[0] == 0.0 and q.scale[2] == 0.0
    np.testing.assert_array_equal(ane_blob.dequantize(q.codes, q.scale), q.dequant)
    assert np.all(q.dequant[0] == 0) and np.all(q.dequant[2] == 0)


def test_fp16_path_keeps_bytes():
    w = (np.random.default_rng(6).standard_normal((3, 5)) * 0.2).astype(np.float16)
    blob, entries = ane_blob.build([("w", w)], quantize=False)
    entry = entries["w"]
    assert entry.dtype == "fp16" and not entry.quantized
    raw = np.frombuffer(blob, dtype=np.float16, count=15, offset=entry.offset + 64)
    np.testing.assert_array_equal(raw.reshape(3, 5), w)


def test_build_is_deterministic():
    first = ane_blob.build(_weights(), quantize=True)
    second = ane_blob.build(_weights(), quantize=True)
    assert first[0] == second[0]
    assert first[1].keys() == second[1].keys()


def test_write_creates_file(tmp_path: Path):
    target = tmp_path / "nested" / "weights.bin"
    entries = ane_blob.write(target, _weights())
    assert target.is_file()
    assert target.read_bytes()[:8] == struct.pack("<II", 3, 2)
    assert set(entries) == {"big", "bias"}


def test_quant_per_channel_rejects_vectors():
    with pytest.raises(ValueError):
        ane_blob.quant_per_channel(np.zeros((4,), dtype=np.float16))


def test_int8_entry_without_scale_is_unusable():
    """手工造一个"int8 但没有 scale 条目"的位置表，MIL 生成应拒绝，而不是写成空引用。"""
    entry = ane_blob.BlobEntry(
        name="w", dtype="int8", offset=0, nbytes=4, shape=(2, 2), scale_offset=None
    )
    from sysone.devices.apple.compile import _conv

    with pytest.raises(NotReadyError):
        _conv("p", cin=2, cout=2, s=4, entry=entry, src="x", out="y")
