"""row-int8 装载：每输出行对称 int8 + fp32 scale。

检查点形态（与旧实现 `int8store` 一致）：`name.__q`（int8，`[rows, cols]`）与
`name.__s`（F32，`[rows]`），另有一份 `int8_manifest.json` 记录每个张量的
kind / shape 与总字节数。装载时按张量惰性取数，不把 5–7 GB 的检查点整载进内存。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ...errors import ConfigError, MissingTensorError
from ..store import SafeTensorFile, file_identity

MANIFEST_NAME = "int8_manifest.json"
QUANT_SUFFIX = ".__q"
SCALE_SUFFIX = ".__s"
# 少于这个元素数的张量保持 16 bit：embeddings / norms / 读出行的量化收益抵不上精度损失。
MIN_QUANT_ELEMS = 1 << 20


@dataclass(frozen=True)
class TensorEntry:
    name: str
    kind: str               # "keep" | "int8"
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RowInt8Manifest:
    entries: Mapping[str, TensorEntry]
    source_bf16_bytes: int = 0
    int8_bytes: int = 0

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RowInt8Manifest:
        raw = payload.get("tensors")
        if not isinstance(raw, dict):
            raise ConfigError("int8_manifest 缺少 tensors 字段")
        entries: dict[str, TensorEntry] = {}
        for name, meta in raw.items():
            if not isinstance(meta, dict):
                raise ConfigError("int8_manifest 条目不合法", tensor=str(name))
            kind = str(meta.get("kind"))
            if kind not in ("keep", "int8"):
                raise ConfigError("int8_manifest 的 kind 未知", tensor=str(name), kind=kind)
            shape = tuple(int(value) for value in (meta.get("shape") or ()))
            entries[str(name)] = TensorEntry(str(name), kind, shape)
        return cls(
            entries,
            int(payload.get("source_bf16_bytes") or 0),
            int(payload.get("int8_bytes") or 0),
        )

    @classmethod
    def load(cls, model_dir: Path | str) -> RowInt8Manifest:
        path = Path(model_dir) / MANIFEST_NAME
        if not path.exists():
            raise MissingTensorError("没有 int8_manifest.json，无法按 row-int8 装载", path=str(path))
        try:
            payload = json.loads(path.read_text("utf-8"))
        except json.JSONDecodeError:
            raise ConfigError("int8_manifest.json 不是合法 JSON", path=str(path)) from None
        if not isinstance(payload, dict):
            raise ConfigError("int8_manifest.json 顶层不是对象", path=str(path))
        return cls.from_dict(payload)

    def names(self) -> frozenset[str]:
        return frozenset(self.entries)

    def quantized(self) -> frozenset[str]:
        return frozenset(name for name, entry in self.entries.items() if entry.kind == "int8")

    def shape(self, name: str) -> tuple[int, ...]:
        try:
            return self.entries[name].shape
        except KeyError:
            raise MissingTensorError("int8_manifest 缺张量", tensor=name) from None


def pack_row(
    weight: np.ndarray, *, min_elems: int = MIN_QUANT_ELEMS
) -> tuple[np.ndarray, np.ndarray] | None:
    """把 2D 权重压成 `(int8 codes, per-row fp32 scale)`；太小则返回 None（保持 16 bit）。"""
    if weight.ndim != 2:
        raise ConfigError("row-int8 只处理 2D 权重", shape=list(weight.shape))
    if weight.size < min_elems:
        return None
    row_max = np.abs(weight.astype(np.float32)).max(axis=1)
    scale = np.where(row_max > 0, row_max / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(weight.astype(np.float32) / scale[:, None]), -127, 127).astype(np.int8)
    return codes, scale


def dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """`codes * scale[:, None]`；全零行的 scale 取 1.0，反量化仍是 0。"""
    return codes.astype(np.float32) * scale[:, None]


def pack_tensors(
    tensors: Mapping[str, np.ndarray], *, min_elems: int = MIN_QUANT_ELEMS
) -> tuple[dict[str, np.ndarray], RowInt8Manifest]:
    """把一组张量打包成 `__q` / `__s` 形态并生成 manifest（写盘前的纯计算部分）。"""
    packed: dict[str, np.ndarray] = {}
    entries: dict[str, TensorEntry] = {}
    source_bytes = stored_bytes = 0
    for name, array in tensors.items():
        source_bytes += int(array.nbytes)
        quantized = pack_row(array, min_elems=min_elems) if array.ndim == 2 else None
        if quantized is not None:
            codes, scale = quantized
            packed[name + QUANT_SUFFIX] = codes
            packed[name + SCALE_SUFFIX] = scale
            entries[name] = TensorEntry(name, "int8", tuple(array.shape))
            stored_bytes += int(codes.nbytes) + int(scale.nbytes)
            continue
        packed[name] = array
        entries[name] = TensorEntry(name, "keep", tuple(array.shape))
        stored_bytes += int(array.nbytes)
    return packed, RowInt8Manifest(entries, source_bytes, stored_bytes)


class RowInt8Store:
    """row-int8 检查点的惰性只读视图（`read` 返回已反量化的 float32）。"""

    def __init__(self, model_dir: Path | str) -> None:
        self.dir = Path(model_dir)
        self.manifest = RowInt8Manifest.load(self.dir)
        files = sorted(self.dir.glob("*.safetensors"))
        if not files:
            raise MissingTensorError("row-int8 目录里没有 safetensors", path=str(self.dir))
        self._files = [SafeTensorFile(path) for path in files]

    def read(self, name: str) -> np.ndarray:
        entry = self.manifest.entries.get(name)
        if entry is None:
            raise MissingTensorError("row-int8 检查点缺少张量", tensor=name, dir=str(self.dir))
        if entry.kind == "keep":
            return self.read_raw(name)
        codes = self.read_raw(name + QUANT_SUFFIX)
        scale = self.read_raw(name + SCALE_SUFFIX)
        return dequantize(codes, scale)

    def read_parts(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 int8 条目的 (codes, scale) 原始两片，供 torch 侧常驻 int8。"""
        entry = self.manifest.entries.get(name)
        if entry is None:
            raise MissingTensorError("row-int8 检查点缺少张量", tensor=name, dir=str(self.dir))
        if entry.kind != "int8":
            raise MissingTensorError("该张量没有被量化", tensor=name, kind=entry.kind)
        codes = self.read_raw(name + QUANT_SUFFIX)
        scale = self.read_raw(name + SCALE_SUFFIX)
        return codes, scale

    def read_raw(self, name: str) -> np.ndarray:
        for handle in self._files:
            if name in handle.names():
                return handle.read(name)
        raise MissingTensorError("row-int8 分片里找不到张量", tensor=name, dir=str(self.dir))

    def read_rows(self, name: str, rows: Sequence[int]) -> np.ndarray:
        """只读二维张量的若干行（词表这类整表用不到、按行够用的张量走这条）。

        底下就是 `SafeTensorFile.read_rows` —— mude 的 `SplitChain` 一直用它取 embedding 行，
        这里把它透出来，vendor 侧才用得上（整表 248 320 × 2 560 上设备是 1.18 GiB 白占）。
        """
        for handle in self._files:
            if name in handle.names():
                return handle.read_rows(name, rows)
        raise MissingTensorError("row-int8 分片里找不到张量", tensor=name, dir=str(self.dir))

    def names(self) -> frozenset[str]:
        return self.manifest.names()

    def missing(self, expected: Iterable[str]) -> set[str]:
        return set(expected) - set(self.manifest.names())

    def fingerprint(self) -> dict:
        """这份权重容器的身份：manifest 加上每个分片（换一个文件就是另一份权重）。"""
        paths = [self.dir / MANIFEST_NAME, *(handle.path for handle in self._files)]
        return {"container": "row-int8", "files": [file_identity(path) for path in paths]}

    @property
    def artifact_bytes(self) -> int:
        return self.manifest.int8_bytes

    def close(self) -> None:
        for handle in self._files:
            handle.close()
        self._files.clear()
