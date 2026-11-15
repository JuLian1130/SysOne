"""ANE 权重 blob：一个文件装多张权重，每张带 64 字节自描述头。

容器格式与旧引擎（`docs/ANE/anebench/e8_ane.py` 的 `build_blob_mixed`）逐字节一致：

- 文件头 64 字节：`<II` = (条目数, 2)，其余补零；
- 每个条目 64 字节描述 `<IIQQ` = (MAGIC, dtype, nbytes, data_offset)，其后紧跟数据；
- MIL 用 `BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(N))` 引用
  第 N 字节处的条目 —— **offset 指向描述头，不是数据起点**，dylib 按这套自描述容器解析。

量化是 per-output-channel 对称 int8（与 `e5-int8-results.md` 的出货形态一致）：
`W ≈ codes * scale[:, None]`，scale 存 fp16，所以一个量化权重占两个条目（codes + scale）。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

MAGIC = 0xDEADBEEF
DTYPE_FP16 = 1
DTYPE_INT8 = 4

HEADER_BYTES = 64
ENTRY_BYTES = 64

_DTYPE_NAME = {DTYPE_FP16: "fp16", DTYPE_INT8: "int8"}
_DTYPE_CODE = {v: k for k, v in _DTYPE_NAME.items()}


@dataclass(frozen=True)
class BlobEntry:
    """一张权重在 blob 里的位置。

    `offset` 直接用于 MIL 的 `BLOBFILE(offset=...)`；`scale_offset` 只有 int8 条目才有，
    指向紧随其后的 fp16 per-channel 标度条目。
    """

    name: str
    dtype: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    scale_offset: int | None = None

    @property
    def quantized(self) -> bool:
        return self.dtype == "int8"


@dataclass(frozen=True)
class Quantized:
    """一个权重的 int8 视图：codes、fp16 标度、以及反量化回 fp16 的结果。"""

    codes: np.ndarray
    scale: np.ndarray
    dequant: np.ndarray
    max_abs_error: float


def quant_per_channel(weight: np.ndarray) -> Quantized:
    """对称 per-output-channel int8，和 `e8_ane.quant_per_channel` 同式。

    scale 用 fp16 存，所以反量化走 fp32 再落回 fp16 —— 与 ANE 上 `scale` 的精度一致。
    全零行在除零保护下 codes 全 0、scale 记 0；反量化得到的仍是 0，不影响读出。
    """
    w32 = np.ascontiguousarray(weight, dtype=np.float32)
    if w32.ndim != 2:
        raise ValueError(f"per-channel 量化只接受二维权重，得到 {w32.shape}")
    amax = np.abs(w32).max(axis=1)
    scale = (amax / 127.0).astype(np.float16)
    s32 = scale.astype(np.float32)
    s32[s32 == 0] = 1.0
    codes = np.clip(np.round(w32 / s32[:, None]), -127, 127).astype(np.int8)
    dequant = (codes.astype(np.float32) * s32[:, None]).astype(np.float16)
    return Quantized(
        codes=codes,
        scale=scale,
        dequant=dequant,
        max_abs_error=float(np.abs(dequant.astype(np.float32) - w32).max()),
    )


def dequantize(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """int8 codes + fp16 标度 -> fp16 权重（参考路径用它对齐 ANE 的读法）。"""
    s32 = np.asarray(scale, dtype=np.float32)
    s32 = np.where(s32 == 0, 1.0, s32)
    return (np.asarray(codes, dtype=np.float32) * s32[:, None]).astype(np.float16)


def build(
    weights: Sequence[tuple[str, np.ndarray]] | Mapping[str, np.ndarray],
    *,
    quantize: bool = True,
) -> tuple[bytes, dict[str, BlobEntry]]:
    """把若干 fp16 权重装成一个 blob，返回字节与每张权重的位置表。

    顺序按入参顺序（Mapping 用插入序），量化时每张权重占 codes + scale 两个条目。
    权重**已经是 int8** 时走 `build_quantized`：那条路不再按 amax 重算标度。
    """
    items = list(weights.items()) if isinstance(weights, Mapping) else list(weights)
    # 交错写：量化权重的 codes 与它的 scale 条目相邻，读侧不需要额外索引。
    flat: list[tuple[str, str, bytes, tuple[int, ...], bool]] = []
    for name, weight in items:
        w16 = np.ascontiguousarray(weight, dtype=np.float16)
        if quantize and w16.ndim == 2:
            q = quant_per_channel(w16)
            flat.append((name, "int8", q.codes.tobytes(), q.codes.shape, True))
            flat.append((name, "fp16", q.scale.tobytes(), q.scale.shape, False))
        else:
            flat.append((name, "fp16", w16.tobytes(), w16.shape, True))
    return _assemble(flat)


def build_quantized(
    weights: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[bytes, dict[str, BlobEntry]]:
    """已经是 `(int8 码, 标度)` 的权重直接装进 blob，**不再量化一次**。

    检查点自带 int8 码时必须走这条：按 amax 重算标度等于把已经落过格的数值再取整一次，
    误差会叠加。按输出行切片（`down0`/`down1`/…）时更明显 —— 切片的 amax 比整行小，
    标度更细，重算出来的码与检查点的码不一致，等于给这一层多加半次量化误差
    （票据 15 §14 的实测：2B 的 logits 偏移由此从 1.1e-3 涨到 7.8e-3）。

    标度按 fp16 落盘（blob 的容器约定），调用方给的标度是 fp32 时会被舍入一次。
    """
    flat: list[tuple[str, str, bytes, tuple[int, ...], bool]] = []
    for name, (codes, scale) in weights.items():
        c8 = np.ascontiguousarray(codes, dtype=np.int8)
        s16 = np.ascontiguousarray(scale, dtype=np.float16)
        if c8.ndim != 2:
            raise ValueError(f"{name} 的 int8 码不是二维的：{c8.shape}")
        if s16.shape != (c8.shape[0],):
            raise ValueError(
                f"{name} 的标度形状 {s16.shape} 与码 {c8.shape} 不配对（每行一个标度）"
            )
        flat.append((name, "int8", c8.tobytes(), c8.shape, True))
        flat.append((name, "fp16", s16.tobytes(), s16.shape, False))
    return _assemble(flat)


def _assemble(
    flat: Sequence[tuple[str, str, bytes, tuple[int, ...], bool]],
) -> tuple[bytes, dict[str, BlobEntry]]:
    """把 `(名字, dtype, 字节, 形状, 是不是主条目)` 的顺序表写成容器字节与位置表。"""
    total = HEADER_BYTES + sum(ENTRY_BYTES + len(payload) for _n, _t, payload, _s, _k in flat)
    buf = bytearray(total)
    struct.pack_into("<II", buf, 0, len(flat), 2)

    entries: dict[str, BlobEntry] = {}
    offset = HEADER_BYTES
    for name, dtype, payload, shape, is_main in flat:
        nbytes = len(payload)
        struct.pack_into("<IIQQ", buf, offset, MAGIC, _DTYPE_CODE[dtype], nbytes, offset + ENTRY_BYTES)
        buf[offset + ENTRY_BYTES : offset + ENTRY_BYTES + nbytes] = payload
        if is_main:
            entries[name] = BlobEntry(
                name=name,
                dtype=dtype,
                offset=offset,
                nbytes=nbytes,
                shape=tuple(int(d) for d in shape),
            )
        else:
            prev = entries[name]
            entries[name] = BlobEntry(
                name=prev.name,
                dtype=prev.dtype,
                offset=prev.offset,
                nbytes=prev.nbytes,
                shape=prev.shape,
                scale_offset=offset,
            )
        offset += ENTRY_BYTES + nbytes
    assert offset == total
    return bytes(buf), entries


def write(
    path: str | Path,
    weights: Sequence[tuple[str, np.ndarray]] | Mapping[str, np.ndarray],
    *,
    quantize: bool = True,
) -> dict[str, BlobEntry]:
    """落盘一个 blob，返回位置表。父目录不存在就建。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    blob, entries = build(weights, quantize=quantize)
    target.write_bytes(blob)
    return entries
