"""blob 里的 int8 码从哪来：容器自带的码，还是装载时现量化。

跑的是 `_build_layer_program` 的真路径（读权重 -> 装 blob -> 生成 MIL），只是不碰 ANE：
产物落在临时目录，比的是 MIL 文本与 blob 字节。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from helpers import write_row_int8_checkpoint
from sysone.devices.apple.chain import _artifact_source, _build_layer_program
from sysone.domain.attention import AttentionRole
from sysone.devices.apple.blobplan import ane_layer_segments
from sysone.weights.checkpoint import open_checkpoint
from sysone.weights.packing import QUANT_SUFFIX, SCALE_SUFFIX
from sysone.weights.packing.ane_blob import quant_per_channel
from sysone.weights.store import write_file

HIDDEN, FFN, CORE, S = 64, 128, 16, 256
COUT = 48 + 16 + 2 + 2        # qkv 48 + z 16 + b 2 + a 2
ENV_PREFIX = "language_model"
BASE = f"{ENV_PREFIX}.layers.0"


def _tensors(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, 0.35, (rows, cols)).astype(np.float32)

    return {
        f"{BASE}.linear_attn.in_proj_qkv.weight": matrix(48, HIDDEN),
        f"{BASE}.linear_attn.in_proj_z.weight": matrix(16, HIDDEN),
        f"{BASE}.linear_attn.in_proj_b.weight": matrix(2, HIDDEN),
        f"{BASE}.linear_attn.in_proj_a.weight": matrix(2, HIDDEN),
        f"{BASE}.linear_attn.out_proj.weight": matrix(HIDDEN, CORE),
        f"{BASE}.mlp.gate_proj.weight": matrix(FFN, HIDDEN),
        f"{BASE}.mlp.up_proj.weight": matrix(FFN, HIDDEN),
        f"{BASE}.mlp.down_proj.weight": matrix(HIDDEN, FFN),
    }


def _build(directory: Path, output: Path):
    store = open_checkpoint(directory)
    try:
        source = _artifact_source(store, prefix=ENV_PREFIX, layer=0,
                                  role=AttentionRole.LINEAR, cout=COUT, core_out=CORE,
                                  hidden=HIDDEN, ffn=FFN, s=S, quantize=True)
        return _build_layer_program(
            store, prefix=ENV_PREFIX, directory=output, index=0,
            role=AttentionRole.LINEAR, cout=COUT, core_out=CORE, hidden=HIDDEN, ffn=FFN,
            s=S, quantize=True, source=source,
        )
    finally:
        store.close()


def _plain(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "plain"
    directory.mkdir()
    write_file(directory / "model.safetensors", tensors)
    return directory


def _row_int8(tmp_path: Path, tensors: dict[str, np.ndarray]) -> Path:
    directory = tmp_path / "row-int8"
    directory.mkdir()
    return write_row_int8_checkpoint(directory, tensors)


def _handmade_codes(tmp_path: Path, codes: dict[str, np.ndarray], scale: float) -> Path:
    """直接写 `__q` / `__s`：码与标度就是**厂商给的**，不经过我们的 amax 规则。

    真实检查点的码也不是按我们的打包器算出来的，这条路径与它同形。
    """
    directory = tmp_path / "vendor"
    directory.mkdir()
    stored, manifest = {}, {}
    for name, matrix in codes.items():
        stored[name + QUANT_SUFFIX] = matrix.astype(np.int8)
        stored[name + SCALE_SUFFIX] = np.full(matrix.shape[0], scale, dtype=np.float32)
        manifest[name] = {"kind": "int8", "shape": list(matrix.shape), "dtype": "bfloat16"}
    write_file(directory / "model.safetensors", stored)
    (directory / "int8_manifest.json").write_text(
        json.dumps({"tensors": manifest, "source_bf16_bytes": 0, "int8_bytes": 0}), "utf-8"
    )
    return directory


def _blob_arrays(blob: bytes) -> list[tuple[str, np.ndarray]]:
    """按物理顺序读回 blob 的每个条目（`(dtype, 一维数组)`）。

    量化条目成对写：先 codes 后 scale，所以 `2 * i` 是第 i 个条目的码。
    """
    count, version = struct.unpack_from("<II", blob, 0)
    assert version == 2
    offset, out = 64, []
    for _ in range(count):
        _magic, dtype, nbytes, data_at = struct.unpack_from("<IIQQ", blob, offset)
        assert data_at == offset + 64
        out.append(({1: "fp16", 4: "int8"}[dtype],
                    np.frombuffer(blob[data_at:data_at + nbytes],
                                  dtype={1: np.float16, 4: np.int8}[dtype])))
        offset += 64 + nbytes
    return out


def _entry_at(arrays: list[tuple[str, np.ndarray]], name: str) -> tuple[str, np.ndarray, np.ndarray]:
    """取某个条目的 `(dtype, 码, 标度)`；条目顺序就是装配表的顺序。"""
    names = list(ane_layer_segments(layer=0, layer_type=AttentionRole.LINEAR,
                                    hidden=HIDDEN, ffn=FFN, prefix=ENV_PREFIX))
    at = 2 * names.index(name)
    return arrays[at][0], arrays[at][1], arrays[at + 1][1]


def test_the_same_numbers_in_either_container_compile_to_the_same_mil(tmp_path: Path) -> None:
    """容器只决定 blob 里的码，不决定程序：MIL 必须与容器的量化形态无关。"""
    tensors = _tensors()
    plain = _build(_plain(tmp_path, tensors), tmp_path / "program-plain")
    packed = _build(_row_int8(tmp_path, tensors), tmp_path / "program-row-int8")
    assert plain.mil == packed.mil


def test_the_blob_keeps_the_stored_codes_instead_of_re_quantizing(tmp_path: Path) -> None:
    """厂商给的码是 ±1、整行标度 0.5；现量化会把它摊到 -127..127 并换成更细的标度。

    down 这一条尤其能分辨：容器只有**整行**标度，切片后再现量化会按切片 amax 重新算 ——
    那正是被去掉的第二次量化。
    """
    rng = np.random.default_rng(11)
    codes = {
        name: np.where(rng.random(matrix.shape) < 0.5, 1, -1).astype(np.int8)
        for name, matrix in _tensors().items()
    }
    program = _build(_handmade_codes(tmp_path, codes, 0.5), tmp_path / "program")
    arrays = _blob_arrays(program.blob_path.read_bytes())

    for name in ("proj", "o", "down0", "down1"):
        kind, flat_codes, flat_scale = _entry_at(arrays, name)
        assert kind == "int8"
        assert np.array_equal(np.abs(flat_codes), np.ones_like(flat_codes)), name
        np.testing.assert_array_equal(flat_scale, np.full(flat_scale.size, 0.5, dtype=np.float16))

    # down0 的码就是容器里那一块（列切不动标度），不是重新量化出来的。
    _, flat_codes, _ = _entry_at(arrays, "down0")
    np.testing.assert_array_equal(flat_codes.reshape(HIDDEN, HIDDEN),
                                  codes[f"{BASE}.mlp.down_proj.weight"][:, :HIDDEN])
    # 现量化会得到另一套码：这就是"再量化一次"会多出来的损失。
    dequant = codes[f"{BASE}.mlp.down_proj.weight"][:, :HIDDEN].astype(np.float32) * np.float32(0.5)
    assert not np.array_equal(quant_per_channel(dequant).codes,
                              codes[f"{BASE}.mlp.down_proj.weight"][:, :HIDDEN])
