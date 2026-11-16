"""容器形态的选择：同一份权重按哪种布局存放，以及"容器自带的 int8 码"怎么取。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import write_row_int8_checkpoint, write_safetensors
from sysone.domain.quant import Scheme
from sysone.errors import MissingTensorError
from sysone.weights.checkpoint import int8_parts, open_checkpoint, stored_quant
from sysone.weights.packing import RowInt8Store
from sysone.weights.store import SafeTensorFile


def _plain(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "plain"
    directory.mkdir()
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


def _row_int8(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "row-int8"
    directory.mkdir()
    return write_row_int8_checkpoint(directory, tensors)


def test_a_manifest_means_row_int8_and_its_absence_means_a_tensor_file(tmp_path: Path) -> None:
    """判据只有 `int8_manifest.json` 一个：它就是 row-int8 的索引本身。"""
    tensors = {"w": np.full((8, 8), 0.5, dtype=np.float32)}
    plain = open_checkpoint(_plain(tmp_path, tensors))
    packed = open_checkpoint(_row_int8(tmp_path, tensors))
    try:
        assert isinstance(plain, SafeTensorFile)
        assert isinstance(packed, RowInt8Store)
        # 两种容器读出来是同一件事：float32 数值。
        np.testing.assert_allclose(packed.read("w"), plain.read("w"), atol=1e-2)
    finally:
        plain.close()
        packed.close()


def test_stored_quant_follows_the_container(tmp_path: Path) -> None:
    """同一份数值、两种容器 ⇒ 两种量化描述：报的是"怎么存的"，不是"算了什么"。"""
    tensors = {"w": np.full((8, 8), 0.5, dtype=np.float32)}
    assert stored_quant(_plain(tmp_path, tensors)).scheme is Scheme.NONE
    assert stored_quant(_row_int8(tmp_path, tensors)).scheme is Scheme.PER_ROW_SYM_INT8


def test_parts_are_only_available_where_the_container_stored_codes(tmp_path: Path) -> None:
    """row-int8 容器只量化够大的二维张量；其余保持 16 bit，没有码可拿。"""
    tensors = {
        "big": np.full((32, 32), 0.5, dtype=np.float32),
        "tiny": np.arange(4, dtype=np.float32),
    }
    directory = _row_int8(tmp_path, tensors)
    store = open_checkpoint(directory)
    try:
        codes, scale = int8_parts(store, "big")  # type: ignore[misc]
        assert codes.dtype == np.int8 and codes.shape == (32, 32)
        assert scale.shape == (32,) and scale.dtype == np.float32
        assert int8_parts(store, "tiny") is None
        with pytest.raises(MissingTensorError):
            int8_parts(store, "absent")
    finally:
        store.close()


def test_a_tensor_file_has_no_codes_to_lend(tmp_path: Path) -> None:
    """单文件张量存的是原样权重 —— 装载侧只能自己量化。"""
    store = open_checkpoint(_plain(tmp_path, {"w": np.ones((4, 4), dtype=np.float32)}))
    try:
        assert int8_parts(store, "w") is None
    finally:
        store.close()


def test_the_fingerprint_names_the_container(tmp_path: Path) -> None:
    """产物缓存的身份里必须带容器形态：同样的数值换成另一种容器，产物就不是同一份。"""
    tensors = {"w": np.full((32, 32), 0.5, dtype=np.float32)}
    plain = open_checkpoint(_plain(tmp_path, tensors))
    packed = open_checkpoint(_row_int8(tmp_path, {"w": tensors["w"]}))
    try:
        assert plain.fingerprint()["container"] == "tensor"
        assert packed.fingerprint()["container"] == "row-int8"
        # 每份容器都列出自己全部的文件（row-int8 是 manifest + 分片），大小与 mtime 都在里面。
        for record in plain.fingerprint()["files"] + packed.fingerprint()["files"]:
            assert set(record) == {"path", "bytes", "mtime_ns"}
    finally:
        plain.close()
        packed.close()
