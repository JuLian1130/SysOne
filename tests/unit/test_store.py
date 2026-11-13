"""safetensors 读取：F32 往返、BF16 解码、缺张量、坏文件、按行读取。"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from helpers import write_safetensors
from sysone.errors import MissingTensorError
from sysone.weights.store import SafeTensorFile, TensorStore, read_file


def test_f32_roundtrip(tmp_path) -> None:
    tensors = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones((1,), dtype=np.float32)}
    path = write_safetensors(tmp_path / "t.safetensors", tensors)
    loaded = read_file(path)
    assert set(loaded) == {"a", "b"}
    assert np.array_equal(loaded["a"], tensors["a"])
    assert loaded["a"].dtype == np.float32


def test_missing_tensor(tmp_path) -> None:
    path = write_safetensors(tmp_path / "t.safetensors", {"a": np.zeros(2, dtype=np.float32)})
    with pytest.raises(MissingTensorError):
        TensorStore([path]).read("b")
    assert TensorStore([path]).missing({"a", "b"}) == {"b"}


def test_bf16_decode(tmp_path) -> None:
    truncated = np.array([1.5, -2.25], dtype=np.float32)
    raw = (truncated.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = {"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, len(raw)]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    assert np.array_equal(read_file(path)["w"], truncated)


def test_corrupt_header(tmp_path) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x00" * 8 + b"not-json")
    with pytest.raises(MissingTensorError):
        read_file(path)


def test_read_rows_f32(tmp_path) -> None:
    table = np.arange(20, dtype=np.float32).reshape(5, 4)
    path = write_safetensors(tmp_path / "t.safetensors", {"table": table})
    handle = SafeTensorFile(path)
    picked = handle.read_rows("table", [3, 1, 3])
    handle.close()
    assert picked.shape == (3, 4)
    assert np.array_equal(picked, table[[3, 1, 3]])


def test_read_rows_bf16(tmp_path) -> None:
    """embedding 表是 BF16，行读取必须走同一条解码路径。"""
    table = np.array([[1.5, -2.25], [3.0, 0.5], [-1.0, 7.0]], dtype=np.float32)
    raw = (table.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = {"w": {"dtype": "BF16", "shape": [3, 2], "data_offsets": [0, len(raw)]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    handle = SafeTensorFile(path)
    picked = handle.read_rows("w", [2, 0])
    handle.close()
    assert np.array_equal(picked, table[[2, 0]])


def test_read_rows_out_of_range(tmp_path) -> None:
    path = write_safetensors(tmp_path / "t.safetensors", {"table": np.zeros((2, 3), dtype=np.float32)})
    with pytest.raises(MissingTensorError):
        SafeTensorFile(path).read_rows("table", [2])
