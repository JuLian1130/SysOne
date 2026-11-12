"""safetensors 读取：F32 往返、BF16 解码、缺张量、坏文件。"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from helpers import write_safetensors
from sysone.errors import MissingTensorError
from sysone.weights.store import TensorStore, read_file


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
