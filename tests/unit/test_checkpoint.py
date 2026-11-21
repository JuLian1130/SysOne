"""容器形态的选择：同一份权重按哪种布局存放，以及"容器自带的 int8 码"怎么取。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import write_row_int8_checkpoint, write_safetensors, write_sharded_checkpoint
from sysone.domain.quant import Scheme
from sysone.errors import MissingTensorError
from sysone.weights.checkpoint import int8_parts, open_checkpoint, stored_quant
from sysone.weights.packing import RowInt8Store
from sysone.weights.store import SafeTensorFile, ShardedTensorStore


def _plain(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "plain"
    directory.mkdir()
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


def _row_int8(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "row-int8"
    directory.mkdir()
    return write_row_int8_checkpoint(directory, tensors)


_SHARD_A = "model-00001-of-00002.safetensors"
_SHARD_B = "model-00002-of-00002.safetensors"
_EMBED = "language_model.embed_tokens.weight"
_Q_PROJ = "language_model.layers.0.self_attn.q_proj.weight"
_EMBED_TABLE = np.arange(16, dtype=np.float32).reshape(4, 4)


def _sharded(tmp_path: Path) -> Path:
    """两份分片的检查点：embedding 与 q_proj 刻意落在**不同的分片**里。"""
    directory = tmp_path / "sharded"
    directory.mkdir()
    return write_sharded_checkpoint(directory, {
        _SHARD_A: {_EMBED: _EMBED_TABLE, "language_model.norm.weight": np.ones(8, dtype=np.float32)},
        _SHARD_B: {_Q_PROJ: np.full((8, 8), 0.25, dtype=np.float32)},
    })


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


# ---------------------------------------------------------------------- 分片（d3 / pplx / torchcast）


def test_an_index_means_shards_and_reads_span_them(tmp_path: Path) -> None:
    """发布的三份 4B 权重按分片发：`names()` 来自索引，`read` 只打开命中那个分片。"""
    store = open_checkpoint(_sharded(tmp_path))
    try:
        assert isinstance(store, ShardedTensorStore)
        assert store.names() == frozenset({_EMBED, "language_model.norm.weight", _Q_PROJ})
        np.testing.assert_array_equal(store.read(_EMBED), _EMBED_TABLE)
        np.testing.assert_allclose(store.read(_Q_PROJ), 0.25)
        # 按行读也跨分片，行序与调用方给的顺序一致（embedding 取词表就这样用）。
        np.testing.assert_array_equal(store.read_rows(_EMBED, [2, 0]), _EMBED_TABLE[[2, 0]])
    finally:
        store.close()


def test_the_fingerprint_covers_every_shard(tmp_path: Path) -> None:
    """产物身份要带索引与**每个**分片：任一分片换了，产物就不是同一份。"""
    directory = _sharded(tmp_path)
    store = open_checkpoint(directory)
    try:
        identity = store.fingerprint()
    finally:
        store.close()
    assert identity["container"] == "sharded"
    assert {Path(record["path"]).name for record in identity["files"]} == {
        "model.safetensors.index.json", _SHARD_A, _SHARD_B,
    }
    # 只动一份分片（大小不变、mtime 变），身份也必须跟着变。
    (directory / _SHARD_B).write_bytes((directory / _SHARD_B).read_bytes())
    touched = open_checkpoint(directory)
    try:
        assert touched.fingerprint()["files"] != identity["files"]
    finally:
        touched.close()


def test_a_shard_file_that_is_gone_fails_where_it_is_read(tmp_path: Path) -> None:
    """索引说有这个张量不等于文件在了：缺的那份分片在取数时按路径报错。"""
    directory = _sharded(tmp_path)
    (directory / _SHARD_B).unlink()
    store = open_checkpoint(directory)
    try:
        assert _Q_PROJ in store.names()
        with pytest.raises(MissingTensorError) as caught:
            store.read(_Q_PROJ)
        assert _SHARD_B in str(caught.value.context["path"])
    finally:
        store.close()


def test_a_name_the_index_does_not_have_is_reported_with_the_index(tmp_path: Path) -> None:
    """索引本身是容器唯一的张量名单，缺名字时指出的就是它。"""
    store = open_checkpoint(_sharded(tmp_path))
    try:
        with pytest.raises(MissingTensorError) as caught:
            store.read("absent")
        assert "model.safetensors.index.json" in str(caught.value.context["index"])
    finally:
        store.close()


def test_a_corrupt_index_is_a_container_error(tmp_path: Path) -> None:
    """索引不是合法 JSON 时不能静默退回单文件（那会读到一半就报缺张量）。"""
    directory = _sharded(tmp_path)
    (directory / "model.safetensors.index.json").write_text("{", encoding="utf-8")
    with pytest.raises(MissingTensorError):
        open_checkpoint(directory)


def test_row_int8_still_wins_over_a_leftover_index(tmp_path: Path) -> None:
    """打过包的目录可能还留着发布包的索引：manifest 是更具体的容器，先认它。"""
    directory = _row_int8(tmp_path, {"w": np.full((8, 8), 0.5, dtype=np.float32)})
    (directory / "model.safetensors.index.json").write_text(
        '{"weight_map": {"w": "model-00001-of-00001.safetensors"}}', encoding="utf-8"
    )
    store = open_checkpoint(directory)
    try:
        assert isinstance(store, RowInt8Store)
    finally:
        store.close()


def test_shards_store_plain_weights_and_lend_no_codes(tmp_path: Path) -> None:
    """发布包是 bf16 原样权重：没有可复用的 int8 码，量化由编译期现做。"""
    directory = _sharded(tmp_path)
    store = open_checkpoint(directory)
    try:
        assert int8_parts(store, _Q_PROJ) is None
    finally:
        store.close()
    assert stored_quant(directory).scheme is Scheme.NONE
