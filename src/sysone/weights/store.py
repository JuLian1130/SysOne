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


_ENCODE_DTYPES: dict[type, str] = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.float64): "F64",
    np.dtype(np.int8): "I8",
    np.dtype(np.uint8): "U8",
}


def write_file(path: Path | str, tensors: dict[str, np.ndarray]) -> Path:
    """写一个 safetensors 文件（F32/F16/F64/I8/U8）。

    M1 的打包路径与测试造小权重都走这里，因此它只实现够用的 dtype 集合，
    遇到别的 dtype 直接报错而不是悄悄转换。
    """
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    header: dict[str, dict] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        encoded = _ENCODE_DTYPES.get(array.dtype)
        if encoded is None:
            raise MissingTensorError("无法写入该 dtype", tensor=name, dtype=str(array.dtype))
        data = array.astype(array.dtype, copy=False).tobytes(order="C")
        header[name] = {
            "dtype": encoded,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        payloads.append(data)
        offset += len(data)
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(resolved, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded_header)))
        handle.write(encoded_header)
        for data in payloads:
            handle.write(data)
    return resolved


class SafeTensorFile:
    """单文件惰性读取：只解析 header，数据区按需 mmap 切片。

    4B 的 int8 检查点动辄 5–7 GB，整文件载入既慢又会把内存打满；M1 的
    `RowInt8Store` 因此只按张量取数，取到的切片再按行反量化。
    """

    def __init__(self, path: Path | str) -> None:
        import mmap

        self.path = Path(path)
        try:
            self._handle = open(self.path, "rb")
        except FileNotFoundError:
            raise MissingTensorError("权重文件不存在", path=str(self.path)) from None
        try:
            raw_length = self._handle.read(8)
            if len(raw_length) != 8:
                raise MissingTensorError("权重文件已损坏（header 长度缺失）", path=str(self.path))
            header_length = struct.unpack("<Q", raw_length)[0]
            header = json.loads(self._handle.read(header_length))
        except (struct.error, json.JSONDecodeError):
            self._handle.close()
            raise MissingTensorError("权重文件不是合法 safetensors", path=str(self.path)) from None
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._data_start = 8 + header_length
        self._header: dict[str, dict] = {}
        for name, meta in header.items():
            if name == "__metadata__" or not isinstance(meta, dict):
                continue
            self._header[name] = meta

    def names(self) -> frozenset[str]:
        return frozenset(self._header)

    def read(self, name: str) -> np.ndarray:
        try:
            meta = self._header[name]
        except KeyError:
            raise MissingTensorError("权重缺少张量", tensor=name, path=str(self.path)) from None
        dtype = str(meta.get("dtype"))
        start, end = (int(value) for value in meta.get("data_offsets", (0, 0)))
        shape = tuple(int(value) for value in meta.get("shape", ()))
        absolute = self._data_start + start
        raw = self._mmap[absolute : self._data_start + end]
        return _decode(raw, dtype, shape, self.path, name)

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()


class TensorStore:
    """跨多个 safetensors 文件的只读张量视图。

    `lazy=True` 时文件只在第一次取某个张量时 mmap 切片；默认整载，因为参考后端
    与小测试模型需要反复随机访问同一批张量。
    """

    def __init__(self, paths: Sequence[Path | str], *, lazy: bool = False) -> None:
        self._tensors: dict[str, np.ndarray] = {}
        self._lazy: list[SafeTensorFile] = []
        self._lazy_cache: dict[str, np.ndarray] = {}
        self._files: list[Path] = []
        for path in paths:
            resolved = Path(path)
            self._files.append(resolved)
            if lazy:
                self._lazy.append(SafeTensorFile(resolved))
            else:
                self._tensors.update(read_file(resolved))

    def read(self, name: str) -> np.ndarray:
        if not self._lazy:
            try:
                return self._tensors[name]
            except KeyError:
                raise MissingTensorError("权重缺少张量", tensor=name, files=[str(item) for item in self._files]) from None
        cached = self._lazy_cache.get(name)
        if cached is not None:
            return cached
        for handle in self._lazy:
            if name in handle.names():
                tensor = handle.read(name)
                self._lazy_cache[name] = tensor
                return tensor
        raise MissingTensorError("权重缺少张量", tensor=name, files=[str(item) for item in self._files])

    def names(self) -> frozenset[str]:
        if self._lazy:
            merged: set[str] = set()
            for handle in self._lazy:
                merged.update(handle.names())
            return frozenset(merged)
        return frozenset(self._tensors)

    def missing(self, expected: Iterable[str]) -> set[str]:
        """返回 `expected` 中缺的键（用于一次性报全所有缺件）。"""
        return set(expected) - set(self.names())

    def close(self) -> None:
        for handle in self._lazy:
            handle.close()
        self._lazy.clear()
        self._lazy_cache.clear()

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)
