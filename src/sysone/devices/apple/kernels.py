"""Metal kernel 的参数结构体与启动契约。

参数不是逐个 `setBytes` 传的：`e8_mtl.m` 把结构体整块写进 buffer index 30（`E8_ARGS_INDEX`，
`e8_common.h:50`），所以下面每个 `pack` 的格式串就是与 C 侧的契约本身——字段增删必须两边同时
改，`tests/unit/test_metal_kernels.py` 会逐字节钉住它。

启动网格的几何取自旧引擎 `e8_a1m.py` 的 `cgrid` / `attn_launch` / `gdn_launch`（那是实测过的
那份）：参考核一格一个 (query, head) 或一格一个 head，快核按 AT_BM=128 查询分块。

缓冲区的布局全篇统一：**通道在前、token 在后**（`[C, T]`，fp16），即 ANE surface 的
`[1, C, 1, T]` 展平之后的样子。这样 ANE 与 GPU 之间不需要任何转置或拷贝。
"""

from __future__ import annotations

import struct

from ...errors import NotReadyError
from .metal import MetalBuffer, MetalRuntime

# ---- 几何（e8_common.h:52-67；M2 只做 mude，4B 是 M4 的事）------------------------------
HID = 2048
FFN = 6144
COUT_FULL = 5120
COUT_GDN = 8224
Q_HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64  # head_dim * partial_rotary_factor(0.25)
QK_DIM = Q_HEADS * HEAD_DIM  # 2048
KV_PROJ = KV_HEADS * HEAD_DIM  # 512
ZDIM = 2048
L_HEADS = 16
L_KD = 128
L_VD = 128
CONV_K = 4
CONV_DIM = 6144  # 2*16*128 + 16*128
HEAD_SCALE = HEAD_DIM**-0.5  # attention 的 1/sqrt(256)

# ---- 参数结构体（与 e8_common.h 一一对应，单位是字节数）-------------------------------
_NRM = struct.Struct("<IIIfIIII")  # struct NormArgs，32 B
_COPY = struct.Struct("<IIIIIIII")  # struct CopyArgs，32 B
_ATTN = struct.Struct("<IIIfIIIIIIII")  # struct AttnArgs，48 B
_GDN = struct.Struct("<IIIIfII")  # struct GdnArgs，28 B


def nrm_args(
    *,
    stile: int,
    s0: int,
    src_stride: int,
    eps: float,
    nka: int = 0,
    segw: int = 0,
    dst_stride: int = 0,
    dst_s0: int = 0,
) -> bytes:
    """`k_norm_to` / `k_norm_resid` / `k_q_prep` / `k_kv_prep` 共用（`e8_common.h:73`）。"""
    return _NRM.pack(stile, s0, src_stride, eps, nka, segw, dst_stride, dst_s0)


def copy_args(
    *,
    nchan: int,
    src_c0: int,
    dst_c0: int,
    stile: int,
    s0: int,
    src_stride: int,
    dst_stride: int,
    src_s0: int = 0,
) -> bytes:
    """`k_copy`（`e8_common.h:97`）。"""
    return _COPY.pack(nchan, src_c0, dst_c0, stile, s0, src_stride, dst_stride, src_s0)


def attn_args(
    *,
    T: int,
    stile: int,
    s0: int,
    scale: float,
    nk: int,
    nq: int,
    nka: int = 0,
    segw: int = 0,
    nkw: int = 0,
    qgT: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
) -> bytes:
    """`k_attn` / `k_attn_ref`（`e8_common.h:108`）。

    `nk` 是这一窗**可见的 key 上界**——注意力模式就是它，kernel 里没有任何掩码项：
    causal 传 `s0+j` 对应的位置、`noncausal_full` 传全长、`prefix_noncausal` 的前缀窗传 P。
    `nka`/`segw`/`nkw` 是"一批 tail 共享一个 prefix"的槽位（S6/S7 的批量形态），单请求传 0。
    """
    return _ATTN.pack(
        T, stile, s0, scale, nk, nq, nka, segw, nkw, qgT, out_stride, out_s0
    )


def gdn_args(
    *, stile: int, s0: int, T: int, first: bool, eps: float, out_stride: int = 0, out_s0: int = 0
) -> bytes:
    """`k_gdn` 家族（`e8_common.h:137`）。`first=True` 表示这是这一层的第一个 tile（卷积尾巴无效）。"""
    return _GDN.pack(stile, s0, T, 1 if first else 0, eps, out_stride, out_s0)


# ---- 网格 -----------------------------------------------------------------------------
def _copy_grid(nchan: int, stile: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_copy`：一格 64 列 × 8 通道（`e8_a1m.py:62`）。"""
    return ((stile + 63) // 64, (nchan + 7) // 8, 1), (64, 8, 1)


def _norm_grid(stile: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_norm_to`：一列一个线程、256 个一组（`e8_a1m.py:67`）。"""
    return ((stile + 255) // 256, 1, 1), (256, 1, 1)


def _prep_grid(tokens: int, heads: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_q_prep` / `k_kv_prep`：一格一个 (token, head)，线程 = 通道（`e8_kernels.metal:216`）。"""
    return (tokens, heads, 1), (HEAD_DIM, 1, 1)


def _attn_grid(nq: int, *, fast: bool) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """fast 核：一格 (head, 128 个查询)，`8*ceil(n/128)` 格、512 线程（`e8_a1m.py:74`）。"""
    if not fast:
        return (nq, Q_HEADS, 1), (256, 1, 1)
    bm = 128
    return (Q_HEADS * ((nq + bm - 1) // bm), 1, 1), (bm // 8 * 32, 1, 1)


def _gdn_ref_grid() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (L_HEADS, 1, 1), (L_KD, 1, 1)


def _gdn_fast_grids(
    stile: int,
) -> list[tuple[str, tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """快核是 4 个 dispatch（`e8_a1m.py:96`），顺序不能换：prep 的 halo 读要先于 tail 写。"""
    return [
        ("k_gdn_prep", ((L_HEADS, (stile + 7) // 8, 1), (128, 1, 1))),
        ("k_gdn_tail", ((L_HEADS, 1, 1), (128, 1, 1))),
        ("k_gdn", ((L_HEADS, 4, 1), (128, 1, 1))),
        ("k_gdn_norm", (((stile + 3) // 4, L_HEADS, 1), (128, 1, 1))),
    ]


# ---- 启动 -----------------------------------------------------------------------------
def copy_columns(
    runtime: MetalRuntime,
    src: MetalBuffer,
    dst: MetalBuffer,
    *,
    nchan: int,
    src_c0: int,
    dst_c0: int,
    stile: int,
    s0: int,
    src_stride: int,
    dst_stride: int,
    src_s0: int = 0,
) -> None:
    grid, tg = _copy_grid(nchan, stile)
    runtime.dispatch(
        "k_copy",
        [src, dst],
        args=copy_args(
            nchan=nchan,
            src_c0=src_c0,
            dst_c0=dst_c0,
            stile=stile,
            s0=s0,
            src_stride=src_stride,
            dst_stride=dst_stride,
            src_s0=src_s0,
        ),
        grid=grid,
        tg=tg,
    )


def norm_to(
    runtime: MetalRuntime,
    src: MetalBuffer,
    dst: MetalBuffer,
    *,
    stile: int,
    src_stride: int,
    eps: float,
    s0: int = 0,
    nka: int = 0,
    segw: int = 0,
    dst_stride: int = 0,
    dst_s0: int = 0,
) -> None:
    """`k_norm_to`：对 `src` 的每一列做（零中心 +1 之后的）RMSNorm，写进 `dst`。"""
    grid, tg = _norm_grid(stile)
    runtime.dispatch(
        "k_norm_to",
        [src, dst],
        args=nrm_args(
            stile=stile,
            s0=s0,
            src_stride=src_stride,
            eps=eps,
            nka=nka,
            segw=segw,
            dst_stride=dst_stride,
            dst_s0=dst_s0,
        ),
        grid=grid,
        tg=tg,
    )


def norm_residual(
    runtime: MetalRuntime,
    src: MetalBuffer,
    dst: MetalBuffer,
    *,
    stile: int,
    src_stride: int,
    eps: float,
    s0: int = 0,
    dst_stride: int = 0,
    dst_s0: int = 0,
) -> None:
    """`k_norm_resid`：`dst = [rmsnorm(src) | src]`（两半各 HID 通道，正是 ANE 的 ffn 入口）。"""
    grid, tg = _norm_grid(stile)
    runtime.dispatch(
        "k_norm_resid",
        [src, dst],
        args=nrm_args(
            stile=stile,
            s0=s0,
            src_stride=src_stride,
            eps=eps,
            dst_stride=dst_stride,
            dst_s0=dst_s0,
        ),
        grid=grid,
        tg=tg,
    )


def q_prep(
    runtime: MetalRuntime,
    qg: MetalBuffer,
    q_norm: MetalBuffer,
    cos: MetalBuffer,
    sin: MetalBuffer,
    *,
    stile: int,
    src_stride: int,
    eps: float,
    s0: int = 0,
    nka: int = 0,
    segw: int = 0,
) -> None:
    """`k_q_prep`（原地）：q 半区做 RMSNorm+部分 RoPE，gate 半区做 sigmoid。"""
    grid, tg = _prep_grid(stile, Q_HEADS)
    runtime.dispatch(
        "k_q_prep",
        [qg, q_norm, cos, sin],
        args=nrm_args(
            stile=stile,
            s0=s0,
            src_stride=src_stride,
            eps=eps,
            nka=nka,
            segw=segw,
            dst_stride=0,
            dst_s0=0,
        ),
        grid=grid,
        tg=tg,
    )


def kv_prep(
    runtime: MetalRuntime,
    kv: MetalBuffer,
    k_norm: MetalBuffer,
    cos: MetalBuffer,
    sin: MetalBuffer,
    *,
    stile: int,
    src_stride: int,
    eps: float,
    s0: int = 0,
    nka: int = 0,
    segw: int = 0,
) -> None:
    """`k_kv_prep`（原地）：只动 k 半区（v 不需要归一化或旋转）。"""
    grid, tg = _prep_grid(stile, KV_HEADS)
    runtime.dispatch(
        "k_kv_prep",
        [kv, k_norm, cos, sin],
        args=nrm_args(
            stile=stile,
            s0=s0,
            src_stride=src_stride,
            eps=eps,
            nka=nka,
            segw=segw,
            dst_stride=0,
            dst_s0=0,
        ),
        grid=grid,
        tg=tg,
    )


def attention(
    runtime: MetalRuntime,
    qg: MetalBuffer,
    kv: MetalBuffer,
    out: MetalBuffer,
    scratch: MetalBuffer | None,
    *,
    T: int,
    stile: int,
    nk: int,
    nq: int | None = None,
    s0: int = 0,
    nka: int = 0,
    segw: int = 0,
    nkw: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
    fast: bool = True,
) -> None:
    """注意力：`qg`/`kv` 必须已经过 prep；结果写进 `out` 的第 `out_s0` 起的 HID 行（含 gate）。

    `fast=True` 走寄存器内 score 分块（不需要 scratch）；`fast=False` 走参考核，需要
    `8*Stile*T` 个 float 的 scratch（只为不含快核的机器或对拍留一条路）。
    """
    if scratch is None:
        raise NotReadyError(
            "注意力核需要一个 scratch 缓冲（快核不读它，但 buffer 必须绑定）",
            fast=fast,
            bytes=8 * stile * T * 4,
        )
    n_eff = stile if nq is None else nq
    if n_eff <= 0:
        return
    grid, tg = _attn_grid(n_eff, fast=fast)
    runtime.dispatch(
        "k_attn" if fast else "k_attn_ref",
        [qg, kv, out, scratch],
        args=attn_args(
            T=T,
            stile=stile,
            s0=s0,
            scale=HEAD_SCALE,
            nk=nk,
            nq=n_eff,
            nka=nka,
            segw=segw,
            nkw=nkw,
            qgT=0,
            out_stride=out_stride,
            out_s0=out_s0,
        ),
        grid=grid,
        tg=tg,
    )


def gdn(
    runtime: MetalRuntime,
    proj: MetalBuffer,
    state: MetalBuffer,
    tail: MetalBuffer,
    wconv: MetalBuffer,
    a_log: MetalBuffer,
    dt_bias: MetalBuffer,
    norm: MetalBuffer,
    out: MetalBuffer,
    *,
    stile: int,
    eps: float,
    first: bool = True,
    s0: int = 0,
    T: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
    fast: bool = True,
) -> None:
    """GDN：`proj` 是 ANE 的 8224 行投影（通道在前），结果写进 `out` 的前 HID 行。

    快核是 4 个 dispatch（prep → tail → delta 递推 → gated RMSNorm），参考核是一个。
    两者都把 fp32 的 `state`（16×128×128）与 3 列的 `tail` 原地推进，所以同一序列的多次
    tile 调用必须按 s0 递增顺序来，`first=True` 只给第一个 tile。
    """
    args = gdn_args(
        stile=stile, s0=s0, T=T, first=first, eps=eps, out_stride=out_stride, out_s0=out_s0
    )
    buffers_list = [proj, state, tail, wconv, a_log, dt_bias, norm, out]
    if not fast:
        grid, tg = _gdn_ref_grid()
        runtime.dispatch("k_gdn_ref", buffers_list, args=args, grid=grid, tg=tg)
        return
    for name, (grid, tg) in _gdn_fast_grids(stile):
        runtime.dispatch(name, buffers_list, args=args, grid=grid, tg=tg)
