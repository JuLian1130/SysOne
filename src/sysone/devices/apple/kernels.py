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
from collections.abc import Sequence

from ...errors import NotReadyError
from .geometry import Geometry, mude_geometry
from .metal import LANE_LONG, LANE_SMALL, MetalBuffer, MetalRuntime

__all__ = [
    "LANE_LONG",
    "LANE_SMALL",
    "attention",
    "copy_columns",
    "gdn",
    "kv_prep",
    "norm_residual",
    "norm_to",
    "q_prep",
    "vattn",
    "vrope",
]

# ---- 几何（默认 = mude 2B，与 e8_common.h:52-67 的默认宏一致）------------------------------
# 运行时路径（`pipeline.py`）一律显式传 `geometry=`；下面的模块常量只为测试与默认几何保留，
# 它是"这份源码默认编译出来的那个几何"的别名，不是引擎的假设（规矩见 architecture.md §13 R1）。
GEOMETRY = mude_geometry()
HID = GEOMETRY.hid
FFN = GEOMETRY.ffn
COUT_FULL = GEOMETRY.cout_full
COUT_GDN = GEOMETRY.cout_gdn
Q_HEADS = GEOMETRY.q_heads
KV_HEADS = GEOMETRY.kv_heads
HEAD_DIM = GEOMETRY.head_dim
ROTARY_DIM = GEOMETRY.rotary_dim
QK_DIM = GEOMETRY.qk_dim
KV_PROJ = GEOMETRY.kv_proj
ZDIM = GEOMETRY.z_dim
L_HEADS = GEOMETRY.l_v_heads
L_KD = GEOMETRY.l_k_dim
L_VD = GEOMETRY.l_v_dim
CONV_K = GEOMETRY.conv_k
CONV_DIM = GEOMETRY.conv_dim
HEAD_SCALE = GEOMETRY.head_scale  # attention 的 1/sqrt(head_dim)

# ---- 参数结构体（与 e8_common.h 一一对应，单位是字节数）-------------------------------
_NRM = struct.Struct("<IIIfIIIII")  # struct NormArgs，36 B
_COPY = struct.Struct("<IIIIIIII")  # struct CopyArgs，32 B
_ATTN = struct.Struct("<IIIfIIIIIIIIIIIIII")  # struct AttnArgs，72 B
_GDN = struct.Struct("<IIIIfIII")  # struct GdnArgs，32 B
_VATTN = struct.Struct("<IIIIIf")  # struct VAttnArgs，24 B
_VROPE = struct.Struct("<III")  # struct VRopeArgs，12 B


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
    pos0: int = 0,
) -> bytes:
    """`k_norm_to` / `k_norm_resid` / `k_q_prep` / `k_kv_prep` 共用（`e8_common.h:73`）。

    `pos0` 只有两条 prep 核读：`qg` 成环之后，列号不再是 rope 位置（见 `e8_common.h`）。
    """
    return _NRM.pack(stile, s0, src_stride, eps, nka, segw, dst_stride, dst_s0, pos0)


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
    nkw_slots: Sequence[int] | None = None,
    qseq: int = 0,
    qgT: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
    caus: int = 0,
) -> bytes:
    """`k_attn` / `k_attn_ref`（`e8_common.h:108`）。

    `nk` 是这一窗**可见的 key 上界**：`noncausal_full` 传全长、`prefix_noncausal` 的前缀窗
    传 P、`causal` 传本段末行的位置（末行自己的那一列，`caus` 再从它往下逐行收）。`caus`
    是**加在 `nk` 上的一个项**，不是又一种模式：进了槽位规则之后，一行仍看 `[0, nka)` 加自己
    那一槽，只是槽内也收到自己那一列为止。
    `nka`/`segw`/`nkw` 是"一批 tail 共享一个 prefix"的槽位（S6/S7 的批量形态），单请求传 0。
    """
    # 批尾巴（票据 10）：8 个槽位的可见行数，每项 16 bit。`nkw_slots` 缺省时表全 0，
    # 核退回"一个 `nkw` 管整片"的老规则。
    words = [0, 0, 0, 0]
    for index, count in enumerate(nkw_slots or ()):
        if index >= 8:
            break
        words[index >> 1] |= (min(int(count), 0xFFFF) & 0xFFFF) << ((index & 1) * 16)
    return _ATTN.pack(
        T, stile, s0, scale, nk, nq, nka, segw, nkw, qgT, out_stride, out_s0,
        qseq, words[0], words[1], words[2], words[3], caus,
    )


def vattn_args(
    *, T: int, nq: int, s0: int, hidden: int, dim: int, scale: float
) -> bytes:
    """`k_vattn`（`shaders/e8_vattn.metal`）。

    视觉塔的注意力参数比 `k_attn` 少一大半，因为视觉没有 GQA、没有 mask、没有 gate、没有环、
    没有槽位：`T`/`nq`/`s0` 是 qkv 与 out 的列布局，`hidden`/`dim` 是通道布局，`scale` 是 1/√d。
    """
    return _VATTN.pack(T, nq, s0, hidden, dim, scale)


def vrope_args(*, T: int, hidden: int, dim: int) -> bytes:
    """`k_vrope`（`shaders/e8_vattn.metal`）：列数/stride、q 的通道数、head_dim（偶数）。"""
    return _VROPE.pack(T, hidden, dim)


def gdn_args(
    *, stile: int, src_s0: int, T: int, first: bool, eps: float, out_stride: int = 0,
    out_s0: int = 0, cols: int = 0,
) -> bytes:
    """`k_gdn` 家族（`e8_common.h:137`）。`first=True` 表示这是这一层的第一个 tile（卷积尾巴无效）。

    `stile` 是**表面步长**（`proj[c*stile + j]`），`cols` 是本瓦片真实列数（0 = `stile`）。
    两者分开之后，部分瓦片（共享前缀的尾巴、prompt 的最后一段）才能给出正确的卷积尾巴。
    """
    return _GDN.pack(stile, src_s0, T, 1 if first else 0, eps, out_stride, out_s0, cols)


# ---- 网格 -----------------------------------------------------------------------------
def _copy_grid(nchan: int, stile: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_copy`：一格 64 列 × 8 通道（`e8_a1m.py:62`）。"""
    return ((stile + 63) // 64, (nchan + 7) // 8, 1), (64, 8, 1)


def _norm_grid(stile: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_norm_to`：一列一个线程、256 个一组（`e8_a1m.py:67`）。"""
    return ((stile + 255) // 256, 1, 1), (256, 1, 1)


def _prep_grid(
    tokens: int, heads: int, head_dim: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """`k_q_prep` / `k_kv_prep`：一格一个 (token, head)，线程 = 通道（`e8_kernels.metal:216`）。"""
    return (tokens, heads, 1), (head_dim, 1, 1)


def _attn_grid(
    nq: int, *, fast: bool, q_heads: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """fast 核：一格 (head, 128 个查询)，`q_heads*ceil(n/128)` 格、512 线程（`e8_a1m.py:74`）。"""
    if not fast:
        return (nq, q_heads, 1), (256, 1, 1)
    bm = 128
    return (q_heads * ((nq + bm - 1) // bm), 1, 1), (bm // 8 * 32, 1, 1)


def _gdn_ref_grid(
    l_heads: int, l_kd: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return (l_heads, 1, 1), (l_kd, 1, 1)


def _gdn_fast_grids(
    stile: int, l_heads: int, l_v_dim: int,
) -> list[tuple[str, tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """快核是 4 个 dispatch（`e8_a1m.py:96`），顺序不能换：prep 的 halo 读要先于 tail 写。"""
    tg = (l_v_dim, 1, 1)
    return [
        ("k_gdn_prep", ((l_heads, (stile + 7) // 8, 1), tg)),
        ("k_gdn_tail", ((l_heads, 1, 1), tg)),
        ("k_gdn", ((l_heads, 4, 1), tg)),
        ("k_gdn_norm", (((stile + 3) // 4, l_heads, 1), tg)),
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
    lane: int = LANE_SMALL,
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
        lane=lane,
    )


def norm_to(
    runtime: MetalRuntime,
    src: MetalBuffer,
    weight: MetalBuffer,
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
    lane: int = LANE_SMALL,
) -> None:
    """`k_norm_to`：对 `src` 的每一列做 RMSNorm（`weight` 已在装配时补过 +1），写进 `dst`。

    核是三个缓冲 `[src, weight, dst]` —— 少了 `weight` 这一格 ANE 会读到空 buffer。
    """
    grid, tg = _norm_grid(stile)
    runtime.dispatch(
        "k_norm_to",
        [src, weight, dst],
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
        lane=lane,
    )


def norm_residual(
    runtime: MetalRuntime,
    src: MetalBuffer,
    weight: MetalBuffer,
    dst: MetalBuffer,
    *,
    stile: int,
    src_stride: int,
    eps: float,
    s0: int = 0,
    dst_stride: int = 0,
    dst_s0: int = 0,
    lane: int = LANE_SMALL,
) -> None:
    """`k_norm_resid`：`dst = [rmsnorm(src) | src]`（两半各 HID 通道，正是 ANE 的 ffn 入口）。

    与 `k_norm_to` 一样是三个缓冲 `[src, weight, dst]`：漏掉中间那格 ANE 会读到空 buffer。
    """
    grid, tg = _norm_grid(stile)
    runtime.dispatch(
        "k_norm_resid",
        [src, weight, dst],
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
        lane=lane,
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
    pos0: int = 0,
    nka: int = 0,
    segw: int = 0,
    geometry: Geometry | None = None,
    lane: int = LANE_LONG,
) -> None:
    """`k_q_prep`（原地）：q 半区做 RMSNorm+部分 RoPE，gate 半区做 sigmoid。

    `pos0 + s0 + j` 是第 j 列的 rope 位置：`qg` 成环时列落在槽位里，位置仍按整条序列算。
    """
    geom = GEOMETRY if geometry is None else geometry
    grid, tg = _prep_grid(stile, geom.q_heads, geom.head_dim)
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
            pos0=pos0,
        ),
        grid=grid,
        tg=tg,
        lane=lane,
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
    pos0: int = 0,
    nka: int = 0,
    segw: int = 0,
    geometry: Geometry | None = None,
    lane: int = LANE_LONG,
) -> None:
    """`k_kv_prep`（原地）：只动 k 半区（v 不需要归一化或旋转）。"""
    geom = GEOMETRY if geometry is None else geometry
    grid, tg = _prep_grid(stile, geom.kv_heads, geom.head_dim)
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
            pos0=pos0,
        ),
        grid=grid,
        tg=tg,
        lane=lane,
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
    qgT: int = 0,
    nka: int = 0,
    segw: int = 0,
    nkw: int = 0,
    nkw_slots: Sequence[int] | None = None,
    qseq: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
    caus: int = 0,
    fast: bool = True,
    geometry: Geometry | None = None,
    lane: int = LANE_LONG,
) -> None:
    """注意力：`qg`/`kv` 必须已经过 prep；结果写进 `out` 的第 `out_s0` 起的 HID 行（含 gate）。

    `fast=True` 走寄存器内 score 分块（不需要 scratch）；`fast=False` 走参考核，需要
    `8*Stile*T` 个 float 的 scratch（只为不含快核的机器或对拍留一条路）。

    `T` 是 **kv** 的列数（整窗缓存）；`qgT` 是 **qg** 的列数，0 表示两者同一张缓冲。`qg`
    成环时 `s0` 是环槽位的起始列（q 与 gate 都从那里读），`nk` 仍是整条序列的可见上界。
    `caus=1` 时再按行收到自己那一列，行的序列列号取自 `qseq`（见 `attn_args`）。
    """
    geom = GEOMETRY if geometry is None else geometry
    if scratch is None:
        raise NotReadyError(
            "注意力核需要一个 scratch 缓冲（快核不读它，但 buffer 必须绑定）",
            fast=fast,
            bytes=8 * stile * T * 4,
        )
    n_eff = stile if nq is None else nq
    if n_eff <= 0:
        return
    grid, tg = _attn_grid(n_eff, fast=fast, q_heads=geom.q_heads)
    runtime.dispatch(
        "k_attn" if fast else "k_attn_ref",
        [qg, kv, out, scratch],
        args=attn_args(
            T=T,
            stile=stile,
            s0=s0,
            scale=geom.head_scale,
            nk=nk,
            nq=n_eff,
            nka=nka,
            segw=segw,
            nkw=nkw,
            nkw_slots=nkw_slots,
            qseq=qseq,
            qgT=qgT,
            out_stride=out_stride,
            out_s0=out_s0,
            caus=caus,
        ),
        grid=grid,
        tg=tg,
        lane=lane,
    )


_VROPE_TG = 128      # `k_vrope` 的线程数
_VROPE_ITER = 4      # 一格扫过的列数 = `_VROPE_TG * _VROPE_ITER`（核里的 `VR_ITER`）


def vrope(
    runtime: MetalRuntime,
    qkv: MetalBuffer,
    cos: MetalBuffer,
    sin: MetalBuffer,
    *,
    T: int,
    hidden: int,
    dim: int,
    lane: int = LANE_LONG,
) -> None:
    """视觉塔的轴向 RoPE：把 `qkv` 前 `2·hidden` 行（q、k）按整图**原地**转过去。

    `cos`/`sin` 是**转置**过的 `(dim/2, T)` fp16：沿 token 连续，一格里的 32 个 lane 才读得成
    一条合并访问。核看不出这个轴，所以形状是调用方的约定（`vision.py::encode` 负责转置）。

    数值与 `vision.py::rope_qk` 逐位相同：四个乘各舍入一次，核里再用
    `#pragma clang fp contract(off)` 把"不折 fma"写死（不靠编译器的默认值）。
    """
    if dim < 2 or dim % 2:
        raise NotReadyError("视觉 rope 的 head_dim 必须是正偶数", dim=dim)
    if hidden % dim:
        raise NotReadyError("视觉 rope 的通道数必须是 head_dim 的整数倍", hidden=hidden, dim=dim)
    span = _VROPE_TG * _VROPE_ITER
    runtime.dispatch(
        "k_vrope",
        [qkv, cos, sin],
        args=vrope_args(T=T, hidden=hidden, dim=dim),
        grid=((T + span - 1) // span, hidden, 1),
        tg=(_VROPE_TG, 1, 1),
        lane=lane,
    )


def vattn(
    runtime: MetalRuntime,
    qkv: MetalBuffer,
    out: MetalBuffer,
    *,
    T: int,
    nq: int,
    hidden: int,
    dim: int,
    scale: float,
    s0: int = 0,
    lane: int = LANE_LONG,
) -> None:
    """视觉塔的非因果注意力：`qkv` 依次是 `[q | k | v]` 各 `hidden` 行，结果写 `out` 的前 `hidden` 行。

    q/k 的轴向 RoPE 由调用方先做完（`vrope`，见它的 docstring）：核不读 cos/sin，所以它只是
    softmax(QKᵀ)·V —— 给定 q/k 之后，注意力本身没有别的约定。
    一格管 **64 列 query × 一个头**：`grid = (ceil(nq/64), hidden//dim)`、`tg = (256)`，键按
    32 一档。`nq` 不是 64 的倍数时，最后一格只写前 `nq % 64` 列；`s0 + nq <= T` 是调用方的
    约定（`out` 里只有 `[s0, s0+nq)` 这些列会被写）。
    """
    if dim != 64:
        # 核里"8 档通道 × 8 个分片"是按 head_dim == 64 写死的。`k_attn` 对 `E8_HEAD_DIM` 用
        # 编译期 `#error` 拦同一件事，视觉的 head_dim 是运行期参数（来自描述符），所以拦在这里。
        raise NotReadyError("视觉注意力核只实现了 head_dim = 64", dim=dim, hidden=hidden)
    runtime.dispatch(
        "k_vattn",
        [qkv, out],
        args=vattn_args(T=T, nq=nq, s0=s0, hidden=hidden, dim=dim, scale=scale),
        grid=((nq + 63) // 64, hidden // dim, 1),
        tg=(256, 1, 1),
        lane=lane,
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
    scratch: MetalBuffer,
    *,
    stile: int,
    eps: float,
    first: bool = True,
    src_s0: int = 0,
    T: int = 0,
    out_stride: int = 0,
    out_s0: int = 0,
    cols: int = 0,
    fast: bool = True,
    geometry: Geometry | None = None,
    lane: int = LANE_SMALL,
) -> None:
    """GDN：`proj` 是 ANE 的 8224 行投影（通道在前），结果写进 `out` 的前 HID 行。

    快核是 4 个 dispatch（prep → tail → delta 递推 → gated RMSNorm），参考核是一个。
    两者都把 fp32 的 `state`（16×128×128）与 3 列的 `tail` 原地推进，所以同一序列的多次
    tile 调用必须按 src_s0 递增顺序来，`first=True` 只给第一个 tile。`scratch` 是第 9 个缓冲
    （`8192*stile` 个 fp16，旧引擎 `d.alloc(8192 * tile * 2)`），参考核与快核都会绑它。
    """
    geom = GEOMETRY if geometry is None else geometry
    args = gdn_args(
        stile=stile, src_s0=src_s0, T=T, first=first, eps=eps, out_stride=out_stride, out_s0=out_s0,
        cols=cols,
    )
    buffers_list = [proj, state, tail, wconv, a_log, dt_bias, norm, out, scratch]
    if not fast:
        grid, tg = _gdn_ref_grid(geom.l_v_heads, geom.l_k_dim)
        runtime.dispatch("k_gdn_ref", buffers_list, args=args, grid=grid, tg=tg, lane=lane)
        return
    for name, (grid, tg) in _gdn_fast_grids(stile, geom.l_v_heads, geom.l_v_dim):
        runtime.dispatch(name, buffers_list, args=args, grid=grid, tg=tg, lane=lane)
