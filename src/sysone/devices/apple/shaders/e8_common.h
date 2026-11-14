// E8b shared contract: tensor layouts, arg structs, and the formulas both GPU cores must
// agree on.  Included by e8_kernels.metal (glue + reference cores), e8_attn.metal (fast
// attention) and e8_gdn.metal (fast GDN core).
//
// ---------------------------------------------------------------------------------------
// Memory layout
// ---------------------------------------------------------------------------------------
// Every ANE surface and every device buffer holding an activation is CHANNEL-MAJOR:
//
//     x[c, s]  at  element index  c * stride + s          (stride = columns per channel)
//
// That is the layout an ANE `conv` with weight (COUT, CIN, 1, 1) consumes and produces, so
// no transposes happen anywhere in the pipeline.  Kernels take the stride and the absolute
// column offset s0, therefore the same kernel addresses a tile of a full-length buffer or a
// standalone tile buffer.
//
// Per-layer tensors (HID = 2048):
//
//   io_b      (HID, T)        half   layer input h; also the destination of `out` and `ffn`
//   in_proj   (HID, tile)     half   rmsnormed layer input     <- k_norm_to
//   out_proj  (COUT, tile)    half   proj procedure output
//   io_a      (2*HID, tile)   half   [core_out | h] then [xn | y]
//   qg        (4096, T)       half   full-attn [q|gate] per head, i.e. head h owns
//                                    channels [h*512, h*512+512) = q(256) then gate(256)
//   kv        (1024, T)       half   full-attn [k(512) | v(512)], k/v head j owns
//                                    channels [j*256, j*256+256)
//
// COUT = 5120 for a full-attention layer ([q_proj 4096 | k_proj 512 | v_proj 512]) and
// 8224 for a GDN layer ([in_proj_qkv 6144 | in_proj_z 2048 | in_proj_b 16 | in_proj_a 16]).
//
// ---------------------------------------------------------------------------------------
// Directions (which engine reads/writes what) -- this is what makes the handover free
// ---------------------------------------------------------------------------------------
//   rmsnorm:   io_b[0:HID]        -> in_proj                     (k_norm_to)
//   ANE proj:  in_proj            -> out_proj
//   gather:    out_proj           -> qg / kv                     (k_copy)
//   core:      out_proj | qg,kv   -> io_a[0:HID]                 (k_attn / k_gdn)
//   residual:  io_b[0:HID]        -> io_a[HID:2HID]              (k_copy)
//   ANE out:   io_a               -> io_b            (o_proj + residual)
//   rmsnorm:   io_b[0:HID]        -> io_a[0:HID] + io_a[HID:]    (k_norm_resid)
//   ANE ffn:   io_a               -> io_b            (fused FFN + residual)
//
// io_b holds h across the whole layer: it is the residual source for `out` and the proj
// input source (via in_proj), so nothing between layers is copied through the host.

#pragma once
#include <metal_stdlib>
using namespace metal;

#define E8_ARGS_INDEX 30

// 几何宏（模型形状）。运行时会先生成一段 `#define E8_GEOMETRY_SET 1` 的头文件、拼在本文件之前，
// 所以这里的默认值只在"没有注入几何"时生效（例如单独跑源码编译的测试）。见 architecture §13 R1。
#ifndef E8_GEOMETRY_SET
#define E8_HID        2048
#define E8_FFN        6144
#define E8_COUT_FULL  5120
#define E8_COUT_GDN   8224
#define E8_Q_HEADS    8
#define E8_KV_HEADS   2
#define E8_HEAD_DIM   256
#define E8_ROTARY_DIM 64          // head_dim * partial_rotary_factor(0.25)
#define E8_QK_DIM     2048        // Q_HEADS * HEAD_DIM
#define E8_KV_PROJ    512         // KV_HEADS * HEAD_DIM
#define E8_L_HEADS    16
#define E8_L_KD       128
#define E8_L_VD       128
#define E8_CONV_K     4
#define E8_CONV_DIM   6144        // 2*16*128 + 16*128
#define E8_Z_DIM      2048
#endif

// ---------------------------------------------------------------------------------------
// Argument blocks.  Field order is part of the ABI: e8_mtl.py packs these with struct.pack
// using the format string noted next to each struct.  Never reorder a field.
// ---------------------------------------------------------------------------------------
struct NormArgs {          // e8_mtl.py: "<IIIfIIIII" -> 36 bytes
    uint  Stile;           // 0  destination columns
    uint  s0;              // 1  source column offset in the (C, src_stride) source
    uint  src_stride;      // 2  columns per channel in the source (= T)
    float eps;             // 3
    // S6 (batched tails): the *position* a column is normalised and rotated at is not the column
    // itself once K tails share a tile -- a tail's rows live in slot columns, but every tail is
    // the continuation of the same prefix, so all of them are positions [nka, nka+cnt).  The two
    // kernels that do rope (`k_q_prep`, `k_kv_prep`) fold the column back with `e8_fold`; the
    // plain norms ignore both fields.  segw = 0 (every caller before S6) keeps column == position.
    uint  nka;             // 4  first column of the slot region (the prefix length)
    uint  segw;            // 5  slot width in columns, 0 = no slots
    // Appended for the short pack's direct write: a packed program surface is wider than one
    // request's slot, so the destination is no longer "(channels, Stile) at column j".  0 means
    // the old layout (dst_stride = Stile, dst_s0 = 0), which is every pre-pack caller.
    uint  dst_stride;      // 6  columns per channel in dst; 0 = Stile
    uint  dst_s0;          // 7  destination column offset inside dst
    // A1：`qg` 成环之后，rope 位置不再等于列号。环里的 q 按槽位摆放（列 = `s0 + j`），而它的
    // 位置是 `pos0 + s0 + j`（槽位 NCH 个一循环，所以 `pos0 = 本瓦片起始位置 - 槽位起始列`）。
    // 整窗缓存（`qg` 未成环、`kv`）的 `pos0` 恒为 0，加 0 就是原来的行为。
    uint  pos0;            // 8  rope 位置基准；0 = 列号即位置
};

// The rope position of data column `c`: inside a slot, the row's position within its own request.
inline uint e8_fold(uint nka, uint segw, uint c) {
    return (segw && c >= nka) ? nka + (c - nka) % segw : c;
}

struct CopyArgs {          // e8_mtl.py: "<IIIIIIII" -> 32 bytes
    uint  nchan;           // 0  channels to copy
    uint  src_c0;          // 1  first source channel
    uint  dst_c0;          // 2  first destination channel
    uint  Stile;           // 3  columns to copy
    uint  s0;              // 4  destination column offset
    uint  src_stride;      // 5
    uint  dst_stride;      // 6
    uint  src_s0;          // 7  source column offset (tile surfaces pass 0)
};

struct AttnArgs {          // e8_mtl.py: "<IIIfIIIIIIII" -> 48 bytes
    uint  T;               // 0  sequence length (columns of qg / kv)
    uint  Stile;           // 1  queries in this tile
    uint  s0;              // 2  first query position of this tile
    float scale;           // 3  HEAD_DIM ** -0.5
    // Appended, never inserted: the offsets above are the ABI.  `nk` is what makes a padded last
    // tile a *correctness* question rather than a cost one -- the zero rows are keys for every
    // real row, so until they are masked away the answer depends on how many of them there are
    // (measured: h differs by 1.14-1.98 at T = 16 884 depending on the tile, 0 at an aligned T).
    uint  nk;              // 4  keys actually attended; keys >= nk are masked out
    uint  nq;              // 5  query rows actually produced (the launch grid uses it)
    // S6 (batched tails): `segw` != 0 turns the single bound `nk` into a *slot* rule -- every
    // query row sees the first `nka` keys plus its own slot of `nkw` keys.  A batched tail pass
    // runs K tails as K slots of one tile, which is exactly that rule.  With `segw` = 0 the three
    // fields are unused and the mask is `key < nk`, i.e. every call made before S6.
    uint  nka;             // 6  keys [0, nka) are visible to every query row (the shared prefix)
    uint  segw;            // 7  slot width in columns; a multiple of AT_BM, 0 = no slots
    uint  nkw;             // 8  real (visible) rows per slot
    // S7 (window-sized qg): a *phase* engine's qg buffer only holds its own tile window, while
    // kv is still the whole-sequence cache -- so the two tensors no longer share a column count.
    // `qgT` is qg's; `T` stays kv's (and is what every offset/bound above is expressed in).
    uint  qgT;             // 9  columns per channel in qg; 0 = the same as T (single buffer)
    // Appended for the short pack: the attention output can be written straight into a packed
    // `io_a` surface at this request's slot instead of a per-slot buffer.  0 / 0 is the old
    // layout (stride = Stile, offset = 0).
    uint  out_stride;      // 10 columns per channel in out; 0 = Stile
    uint  out_s0;          // 11 output column offset inside out
};

struct GdnArgs {           // e8_mtl.py: "<IIIIfII" -> 28 bytes
    uint  Stile;           // 0  tokens in this tile
    uint  s0;              // 1  first token position (for rope-free absolutes / debug)
    uint  T;               // 2  total sequence length (unused by the A1 cores, kept for A2)
    uint  first;           // 3  1 = first tile of the layer (conv tail is invalid)
    float eps;             // 4
    // Appended for the short pack: write the core's pre-norm y straight into a packed `io_a`
    // surface at this request's slot.  0 / 0 is the old layout (stride = Stile, offset = 0).
    uint  out_stride;      // 5  columns per channel in out; 0 = Stile
    uint  out_s0;          // 6  output column offset inside out
};

// ---------------------------------------------------------------------------------------
// Numerics both cores must match exactly (mlx-lm Qwen3.5 GDN path)
// ---------------------------------------------------------------------------------------

// mlx-lm rms_norm with weight=None: x * rsqrt(mean(x^2) + eps)
inline float e8_rms_scale(float sumsq, int n, float eps) {
    return rsqrt(sumsq / float(n) + eps);
}

// softplus computed as logaddexp(x, 0), the way mx.logaddexp(a + dt_bias, 0) does it:
// numerically safe for large |x| (a plain log(1+exp(x)) overflows past ~88).
inline float e8_softplus0(float x) {
    float m = max(x, 0.0f);
    return m + log(1.0f + exp(-fabs(x)));
}

// GDN gate: g = exp(-exp(A_log) * softplus(a + dt_bias))
inline float e8_gdn_gate(float a, float A_log, float dt_bias) {
    return exp(-exp(A_log) * e8_softplus0(a + dt_bias));
}

// The rotation itself; one call per (i, i+32) pair, passing the two table bases.
inline void e8_rot(thread float &a, thread float &b,
                   device const float *ctab, device const float *stab,
                   uint t, uint i) {
    float c = ctab[t * 32u + i];
    float s = stab[t * 32u + i];
    float a2 = a * c - b * s;
    float b2 = b * c + a * s;
    a = a2; b = b2;
}
