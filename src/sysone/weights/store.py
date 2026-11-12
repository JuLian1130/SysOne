"""最小 safetensors 读取器：只用 numpy，不依赖 torch / safetensors。

CI 与本地默认解释器都没有 torch，因此权重读取不能把 torch 拖进核心路径。
格式按官方 spec：8 字节 little-endian header 长度 + JSON header + 连续数据区。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..errors import MissingTensorError

_NP_DTYPES: dict[str, type] = {
    "F32": np.float32,
    "F16": np.float16,
    "F64": np.float64,
    "I8": np.int8,
    "U8": np.uint8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
}


def read_file(path: Path) -> dict[str, np.ndarray]:
    """读一个 safetensors 文件，返回 `名字 -> ndarray`（整文件载入，M0 只用于小模型）。"""
    try:
        with open(path, "rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise MissingTensorError("权重文件已损坏（header 长度缺失）", path=str(path))
            header_length = struct.unpack("<Q", raw_length)[0]
            header = json.loads(handle.read(header_length))
            data_start = 8 + header_length
            handle.seek(data_start)
            payload = handle.read()
    except FileNotFoundError:
        raise MissingTensorError("权重文件不存在", path=str(path)) from None
    except (struct.error, json.JSONDecodeError):
        raise MissingTensorError("权重文件不是合法 safetensors", path=str(path)) from None

    tensors: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise MissingTensorError("权重文件 header 不合法", path=str(path), tensor=name)
        dtype = str(meta.get("dtype"))
        start, end = (int(value) for value in meta.get("data_offsets", (0, 0)))
        shape = tuple(int(value) for value in meta.get("shape", ()))
        if end > len(payload):
            raise MissingTensorError("张量数据越界", path=str(path), tensor=name)
        raw = payload[start:end]
        tensors[name] = _decode(raw, dtype, shape, path, name)
    return tensors


def _decode(raw: bytes, dtype: str, shape: tuple[int, ...], path: Path, name: str) -> np.ndarray:
    if dtype == "BF16":
        codes = np.frombuffer(raw, dtype="<u2")
        bits = (codes.astype(np.uint32) << 16).view(np.float32)
        return bits.reshape(shape)
    numpy_dtype = _NP_DTYPES.get(dtype)
    if numpy_dtype is None:
        raise MissingTensorError("不支持的 safetensors dtype", path=str(path), tensor=name, dtype=dtype)
    return np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)


class TensorStore:
    """跨多个 safetensors 文件的只读张量视图。"""

    def __init__(self, paths: Sequence[Path | str]) -> None:
        self._tensors: dict[str, np.ndarray] = {}
        self._files: list[Path] = []
        for path in paths:
            resolved = Path(path)
            self._files.append(resolved)
            self._tensors.update(read_file(resolved))

    def read(self, name: str) -> np.ndarray:
        try:
            return self._tensors[name]
        except KeyError:
            raise MissingTensorError("权重缺少张量", tensor=name, files=[str(item) for item in self._files]) from None

    def names(self) -> frozenset[str]:
        return frozenset(self._tensors)

    def missing(self, expected: Iterable[str]) -> set[str]:
        """返回 `expected` 中缺的键（用于一次性报全所有缺件）。"""
        wanted = set(expected)
        return wanted - set(self._tensors)

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)
