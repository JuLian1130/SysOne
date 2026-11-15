// E8b fast GDN core (linear-attention layers): the 15/20 of Qwen3.5-2B that are Gated
// DeltaNet.  Three kernels per proj tile, one command buffer, dispatched in this order:
//
//   k_gdn_prep   conv1d(K=4)+SiLU, L2norm(q,k), gate(g,beta)  -> scratch     (parallel)
//   k_gdn        per-token delta rule over the fp32 state     -> out (pre-norm y, fp16)
//   k_gdn_norm   gated RMSNorm over the head's 128 dv         -> out (final, fp16)
//
// ---------------------------------------------------------------------------------------
// Launch contract (9 buffers, identical list for all three kernels; args at buffer 30)
// ---------------------------------------------------------------------------------------
//   0 proj    (COUT_GDN, Stile) half  ANE surface, channel-major, stride Stile (read only)
//   1 state   (L_HEADS*L_KD*L_VD) float  carried across tiles, indexed [v head][dv][dk]
//   2 tail    (CONV_DIM*3) half    pre-conv hangover, carried across tiles
//   3 wconv   (CONV_DIM*4) half
//   4 A_log   (L_HEADS) float
//   5 dt_bias (L_HEADS) float
//   6 norm    (L_VD) half
//   7 out     (2*HID, Stile) half  only the first L_HEADS*L_VD rows are touched
//   8 scratch (E8G_S_ROW*Stile) half   internal layout below
//
//   k_gdn_prep   grid = (L_HEADS, ceil(Stile/E8G_TB))   tg = (128,)   one group per v head
//   k_gdn_tail   grid = (L_HEADS, 1)                    tg = (128,)
//   k_gdn        grid = (L_HEADS, 4)                    tg = (128,)  (4 x 32 value rows)
//   k_gdn_norm   grid = (ceil(Stile/4), L_HEADS)        tg = (128,)  (4 warps, warp = 1 token)
//
//   2B draws L_HEADS == L_K_HEADS == 16 and the numbers above are the familiar
//   2048/4096/6144/8192.  4B keeps 16 k heads but has 32 v heads, so q and k are addressed by
//   the k head `h / E8G_R` while v, g and beta are addressed by the v head `h`; that is the
//   `repeat_interleave(q, E8G_R, dim=heads)` of the reference implementation, done by indexing
//   instead of by materialising an expanded vector.
//
//   driver: one `e8_mtl.a_gdn(Stile, s0, T, first)` for all of them, one commit+wait per tile.
//   `e8_gdn_check.launch()` is the helper I timed.
//
// `scratch` layout: one E8G_S_ROW-half row per token, so every access the GPU kernels make is
// coalesced along the row (that is the point of the redesign):
//
//   row = t * E8G_S_ROW        (QK_DIM/CONV_DIM/Z_DIM are the geometry macros)
//   [           0,        QK_DIM)  qn[kh*128 + dk]  fp16-rounded L2-normalized q (inv^2 in)
//   [       QK_DIM,      2*QK_DIM)  kn[kh*128 + dk]  fp16-rounded L2-normalized k (inv in)
//   [     2*QK_DIM, 2*QK_DIM+Z_DIM)  v[h*128 + dv]   SiLU(conv) of the v part
//   [2*QK_DIM+Z_DIM,     +L_HEADS)  g[h]           per-token decay
//   [   the next,        +L_HEADS)  beta[h]        per-token beta
//   rest                          padding; the stride rounds up and never goes below 8192
//
// ---------------------------------------------------------------------------------------
// Why this split, and what it measures (all numbers from `python e8_gdn_check.py`)
// ---------------------------------------------------------------------------------------
// The fp32 state is 1 MB and must stay in registers across the token loop.  A per-thread fp32
// state array of N floats costs, for the bare 4-ops-per-element-per-token recurrence at
// Stile = 4096 (1.5 ms = ALU floor for 16 cores x 128 lanes x 1.4 GHz), `--probe`:
//
//     N     lanes/row  threads   max_tg   ms/tile   vs floor
//     16        8       16384      1024     1.20      0.80x
//     32        4        8192      1024     1.02      0.68x   <- used here
//     64        2        4096       448    21.26     14.20x   <- spilled
//    128        1        2048      1024    61.97     41.41x   <- spilled
//
// So 32 state floats/thread: 4 lanes share a value row and the dk reductions are two
// `simd_shuffle_xor` steps.  Fusing prep+delta+norm into one kernel measured 16.6 ms/tile
// because each state element needs kn twice and qn once per token, and the only cross-thread
// broadcast inside a kernel is threadgroup memory (replacing those per-token reads with
// constants took that kernel from 12.7 to 3.3 ms/tile).  With the prep in its own kernel over
// `scratch`, the sequential kernel needs no barrier at all and reads coalesced:
//
//     k_gdn_prep  1.53 ms/tile    k_gdn  5.77 ms/tile    k_gdn_norm  1.82 ms/tile
//     one tile (3 kernels, 1 command buffer) 8.65 ms  ->  34.6 ms/layer at 16k
//     (fused single-kernel version was 18.4 ms/tile = 73.6 ms/layer)
//
// Inside k_gdn's 5.77 ms: 1.02 ms is the recurrence skeleton (`--probe`), ~1.8 ms is the
// half->float conversion of the two vectors plus latency, and ~2.9 ms is the load of qn/kn/v
// (measured by replacing them with constants: 2.84 ms).  The next wins would be storing
// qn/kn as fp32 in scratch (6144 floats/token fits exactly in the 8192-half row, but then v,
// g and beta no longer fit) and software-pipelining the next token's vectors.

// ---------------------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------------------
// Exactly mlx-lm's `gated_delta_update(..., use_kernel=True)` + `gated_rmsnorm`, i.e. what
// e8_gpu.gdn_core computes.  Per token, per head:
//
//   conv   pre[c] = sum_{k<4} qkv_raw[c][t-3+k] * wconv[c*4+k]        (fp32 accumulate)
//          conv[c] = half( SiLU( half(pre[c]) ) )                     (the model runs fp16)
//   norm   qn = inv^2 * conv_q * rsqrt(mean(conv_q^2) + eps*inv^2)   (fp16-rounded)
//          kn = inv   * conv_k * rsqrt(mean(conv_k^2) + eps*inv^2)   (fp16-rounded)
//   gate   g = exp(-exp(A_log[h]) * softplus(a_h + dt_bias[h])) ; beta = sigmoid(b_h)
//          with a_h = proj[COUT_GDN - L_HEADS + h][s] and
//               b_h = proj[COUT_GDN - 2*L_HEADS + h][s]        (one channel per head)
//   delta  S[dv][:] *= g ;  kv_mem = S[dv] . kn ;  delta = (v[dv] - kv_mem) * beta
//          S[dv][:] += kn * delta ;  y[dv] = S[dv] . qn
//   out    yh = fp16(y) ; o = yh * rsqrt(mean_dv(yh^2) + eps) * SiLU(z) * norm[dv]
//
// `k_gdn_ref` in e8_kernels.metal is the independent oracle; e8_gdn_check.py checks all of
// this against MLX, including the multi-tile state + conv-tail handover.

#include "e8_common.h"

#define E8G_D      128           // E8_L_KD == E8_L_VD
#define E8G_L      4             // lanes per value row
#define E8G_M      32            // state elements per thread (= E8G_D / E8G_L)
#define E8G_RPW    8             // value rows per warp (= 32 / E8G_L)
#define E8G_ROWS   32            // value rows per threadgroup
#define E8G_TG     128           // threads per threadgroup (= E8G_ROWS * E8G_L)
#define E8G_TPH    4             // threadgroups per head (= E8G_D / E8G_ROWS)
#define E8G_TB     8             // tokens per prep threadgroup

// `k_gdn` 的线程几何是按 `E8G_D = 128` 排出来的（32 个 state 元素/线程、4 个 lane 共享一行、
// 4 个 threadgroup 覆盖一个头）。换头维就要换一套线程几何，不在这里悄悄将就。
#if (E8_L_KD != 128) || (E8_L_VD != 128)
#error "k_gdn: 线程几何只对 q/k 与 v 头维都等于 128 的模型排过（E8G_L/E8G_M/E8G_ROWS）"
#endif

// v 头数可以是 k 头数的整数倍：4B 是 32 对 16，同一个 k 头的 qn/kn 由它下面 E8G_R 个 v 头
// 共享——这就是参考实现里 `repeat_interleave(q, E8G_R, dim=heads)` 的展开，在这里由寻址
// 隐含完成，不另存一份展开后的向量。
#define E8G_R      (E8_L_HEADS / E8_L_K_HEADS)
#if ((E8_L_HEADS % E8_L_K_HEADS) != 0) || (E8_L_HEADS < E8_L_K_HEADS)
#error "k_gdn: v 头数必须是 k 头数的整数倍（E8G_R 是 repeat_interleave 的步长）"
#endif

// scratch row layout, in halves: [qn | kn | v | g | beta]。q/k 两段按 k 头编号，v/g/beta 按
// v 头编号，所以 4B 的行比 2B 宽（8256 对 6176）。步长向上取整到 1024 且不低于 8192，必须与
// `Geometry.gdn_scratch_halves` 算出同一个数——宿主按它分配 buffer。
#define E8G_S_NEED  (2 * E8_QK_DIM + E8_Z_DIM + 2 * E8_L_HEADS)
#define E8G_S_ROUND ((E8G_S_NEED + 1023) / 1024 * 1024)
#define E8G_S_ROW   (E8G_S_ROUND < 8192 ? 8192 : E8G_S_ROUND)
#define E8G_S_QN    0
#define E8G_S_KN    (E8_QK_DIM)
#define E8G_S_V     (2 * E8_QK_DIM)
#define E8G_S_G     (2 * E8_QK_DIM + E8_Z_DIM)
#define E8G_S_BETA  (2 * E8_QK_DIM + E8_Z_DIM + E8_L_HEADS)

// `proj` 的通道起点（共 COUT_GDN 行）：qkv 在 [0, conv_dim)，z 紧贴其后。**不要拿
// E8_HID / E8_FFN 顶替它们**：2B 恰好 hidden == qk_dim（2048）、ffn == conv_dim（6144），
// 4B 不是（2560 对 2048、9216 对 8192），顶错了就是默默读到别的通道。
#define E8G_CQ(kh, d)  ((kh) * E8G_D + (d))
#define E8G_CK(kh, d)  (E8_QK_DIM + (kh) * E8G_D + (d))
#define E8G_CV(h, dv)  (2u * E8_QK_DIM + (h) * E8G_D + (dv))
#define E8G_CZ(h, dv)  (E8_CONV_DIM + (h) * E8G_D + (dv))
#define E8G_CA(h)      (E8_COUT_GDN - E8_L_HEADS + (h))       // in_proj_a
#define E8G_CB(h)      (E8_COUT_GDN - 2 * E8_L_HEADS + (h))   // in_proj_b（在 a 之前）
// 这个 v 头 h 属于哪个 k 头
#define E8G_KH(h)      ((h) / E8G_R)

// Sum across the E8G_L lanes that share one value row (they are consecutive lanes).
inline float e8g_rowsum(float v) {
    v += simd_shuffle_xor(v, 1u);
    v += simd_shuffle_xor(v, 2u);
    return v;
}

// Depthwise conv tap (raw t-3..t) + SiLU, fp16-rounded at both steps like the fp16 model.
inline float e8g_conv4(float x0, float x1, float x2, float x3, float4 w) {
    float pre = x0 * w.x + x1 * w.y + x2 * w.z + x3 * w.w;
    float h = float(half(pre));
    return float(half(h / (1.0f + exp(-h))));
}

// ---------------------------------------------------------------------------------------
// k_gdn_prep: everything in the GDN core that does not depend on the recurrent state.
// One threadgroup per (head, block of E8G_TB tokens); thread = dk index, so it owns the q
// channel h*128+dk, the k channel HID+h*128+dk and the v channel 2*HID+h*128+dk; the L2-norm
// sums over the head's 128 dk are threadgroup reductions.  grid=(16, ceil(Stile/TB)), tg=(128,)
// ---------------------------------------------------------------------------------------
kernel void k_gdn_prep(device const half  *proj   [[buffer(0)]],
                       device float       *state  [[buffer(1)]],
                       device half        *tail   [[buffer(2)]],
                       device const half  *wconv  [[buffer(3)]],
                       device const float *A_log  [[buffer(4)]],
                       device const float *dtb    [[buffer(5)]],
                       device const half  *norm   [[buffer(6)]],
                       device half        *out    [[buffer(7)]],
                       device half        *scr    [[buffer(8)]],
                       constant GdnArgs   &a      [[buffer(E8_ARGS_INDEX)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]]) {
    (void)state; (void)norm; (void)out;
    const uint  tid = tid2.x;
    const uint  h   = tgid.x;
    const ulong sp  = a.Stile;
    const int   St  = int(a.cols ? a.cols : a.Stile);
    const int   t0  = int(tgid.y) * E8G_TB;
    // Ticket 10: a batched tail pass runs one slot per dispatch, so every read of the source
    // surface is at `a.src_s0 + <tile-local column>`; 0 is the old single-state call.
    const uint  so  = a.src_s0;
    if (t0 >= St) return;

    threadgroup float red[2 * E8G_TB][8];      // per-token sumq (rows 0..TB-1), sumk (TB..)
    threadgroup float sc[2 * E8G_TB];          // q scales then k scales
    threadgroup float tg_gt[E8G_TB], tg_bt[E8G_TB];

    // 本组 v 头 h 对应的 k 头 kh。q/k 只由每个 k 头的"代表组"（h % E8G_R == 0）写进
    // scratch；同一 k 头下其余组算出的 qn/kn 一律丢弃，qn/kn 两段因此被 E8G_R 个 v 头共享。
    // 2B 的 E8G_R 是 1，"代表组"恒真、kh == h，这一层判断被编译器整段消掉。
    const uint kh  = E8G_KH(h);
    const bool rep = ((h % E8G_R) == 0u);
    const uint cqk = E8G_CQ(kh, tid);            // q 通道 + qn 槽（按 k 头编号）
    const uint cq  = h * E8G_D + tid;            // v 通道 + v 槽（按 v 头编号）
    const uint ck  = E8G_CK(kh, tid);
    const uint cv  = E8G_CV(h, tid);
    float4 wq = float4(float(wconv[(ulong)cqk * 4u + 0u]), float(wconv[(ulong)cqk * 4u + 1u]),
                       float(wconv[(ulong)cqk * 4u + 2u]), float(wconv[(ulong)cqk * 4u + 3u]));
    float4 wk = float4(float(wconv[(ulong)ck * 4u + 0u]), float(wconv[(ulong)ck * 4u + 1u]),
                       float(wconv[(ulong)ck * 4u + 2u]), float(wconv[(ulong)ck * 4u + 3u]));
    float4 wv = float4(float(wconv[(ulong)cv * 4u + 0u]), float(wconv[(ulong)cv * 4u + 1u]),
                       float(wconv[(ulong)cv * 4u + 2u]), float(wconv[(ulong)cv * 4u + 3u]));

    // ---- raw columns t0-3 .. t0+E8G_TB-1 of this thread's three channels -----------------
    // the 3-column halo predates the tile (tail, or zero on the first tile); the block itself
    // is one coalesced 16-byte load per channel when proj's stride is 8-half aligned.
    float rq[E8G_TB + 3], rk[E8G_TB + 3], rv[E8G_TB + 3];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        int tc = t0 - 3 + i;
        if (tc >= 0) {
            rq[i] = float(proj[(ulong)cqk * sp + (ulong)so + (ulong)tc]);
            rk[i] = float(proj[(ulong)ck * sp + (ulong)so + (ulong)tc]);
            rv[i] = float(proj[(ulong)cv * sp + (ulong)so + (ulong)tc]);
        } else if (a.first) {
            rq[i] = 0.0f; rk[i] = 0.0f; rv[i] = 0.0f;
        } else {
            uint ti = (uint)(tc + 3);
            rq[i] = float(tail[(ulong)cqk * 3u + ti]);
            rk[i] = float(tail[(ulong)ck * 3u + ti]);
            rv[i] = float(tail[(ulong)cv * 3u + ti]);
        }
    }
    if ((sp & 7ul) == 0ul && (so & 7u) == 0u && t0 + E8G_TB <= St) {
        vec<half, 8> hq = *(device const vec<half, 8> *)(proj + (ulong)cqk * sp + (ulong)so
                                                         + (ulong)t0);
        vec<half, 8> hk = *(device const vec<half, 8> *)(proj + (ulong)ck * sp + (ulong)so
                                                        + (ulong)t0);
        vec<half, 8> hv = *(device const vec<half, 8> *)(proj + (ulong)cv * sp + (ulong)so
                                                        + (ulong)t0);
        #pragma unroll
        for (int i = 0; i < E8G_TB; i++) {
            rq[i + 3] = float(hq[i]); rk[i + 3] = float(hk[i]); rv[i + 3] = float(hv[i]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < E8G_TB; i++) {
            int tc = min(t0 + i, St - 1);
            rq[i + 3] = float(proj[(ulong)cqk * sp + (ulong)so + (ulong)tc]);
            rk[i + 3] = float(proj[(ulong)ck * sp + (ulong)so + (ulong)tc]);
            rv[i + 3] = float(proj[(ulong)cv * sp + (ulong)so + (ulong)tc]);
        }
    }

    // ---- conv + SiLU for the block, plus the per-token sums of squares ------------------
    half cq8[E8G_TB], ck8[E8G_TB], cv8[E8G_TB];
    #pragma unroll
    for (int i = 0; i < E8G_TB; i++) {
        cq8[i] = half(e8g_conv4(rq[i + 0], rq[i + 1], rq[i + 2], rq[i + 3], wq));
        ck8[i] = half(e8g_conv4(rk[i + 0], rk[i + 1], rk[i + 2], rk[i + 3], wk));
        cv8[i] = half(e8g_conv4(rv[i + 0], rv[i + 1], rv[i + 2], rv[i + 3], wv));
    }
    #pragma unroll
    for (int i = 0; i < E8G_TB; i++) {
        float sq = float(cq8[i]) * float(cq8[i]);
        float sk = float(ck8[i]) * float(ck8[i]);
        sq = simd_sum(sq);
        sk = simd_sum(sk);
        if ((tid & 31u) == 0u) {
            red[i][tid >> 5]           = sq;
            red[E8G_TB + i][tid >> 5]  = sk;
        }
        if (tid >= 64u && tid < 64u + (uint)E8G_TB) {          // one thread per token: gate
            uint tokn = tid - 64u;
            int  tc = min(t0 + int(tokn), St - 1);
            float av = float(proj[(ulong)E8G_CA(h) * sp + (ulong)so + (ulong)tc]);
            float bv = float(proj[(ulong)E8G_CB(h) * sp + (ulong)so + (ulong)tc]);
            tg_gt[tokn] = exp(-exp(A_log[h]) * e8_softplus0(av + dtb[h]));
            tg_bt[tokn] = 1.0f / (1.0f + exp(-bv));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv     = rsqrt(float(E8G_D));
    const float inv2    = inv * inv;
    const float rms_eps = a.eps * inv * inv;
    if (tid < 2u * E8G_TB) {
        uint tokn = tid % (uint)E8G_TB, which = tid / (uint)E8G_TB;
        float s = 0.0f;
        for (uint w = 0; w < (ntg2.x >> 5); w++) s += red[which * E8G_TB + tokn][w];
        sc[tid] = (which == 0u) ? inv2 * rsqrt(s / float(E8G_D) + rms_eps)
                                : inv  * rsqrt(s / float(E8G_D) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- publish qn / kn / v / g / beta -------------------------------------------------
    #pragma unroll
    for (int i = 0; i < E8G_TB; i++) {
        int t = t0 + i;
        if (t < St) {
            ulong row = (ulong)(so + t) * E8G_S_ROW;
            if (rep) {   // 同一 k 头的其余 v 头不算 qn/kn，因此也不写（避免同址互写）
                scr[row + E8G_S_QN + cqk] = half(float(cq8[i]) * sc[i]);
                scr[row + E8G_S_KN + cqk] = half(float(ck8[i]) * sc[E8G_TB + i]);
            }
            scr[row + E8G_S_V + cq] = cv8[i];
        }
    }
    if (tid < (uint)E8G_TB) {
        int t = t0 + int(tid);
        if (t < St) {
            ulong row = (ulong)(so + t) * E8G_S_ROW;
            scr[row + E8G_S_G + h]    = half(tg_gt[tid]);
            scr[row + E8G_S_BETA + h] = half(tg_bt[tid]);
        }
    }

    // The tail write used to be here, and it was a bug: this dispatch *reads* the carried tail for
    // the three tokens before the tile (the halo at t0 == 0) and *writes* the tile's last three raw
    // columns (from the threadgroup whose block reaches St), and Metal orders nothing between
    // threadgroups of one dispatch.  When the writing threadgroup won, the first three tokens of
    // the tile took the *new* halo, and the delta-rule state carried that wrong input for ~10
    // columns before it decayed -- a `max|d|` of 1-4 that grew layer by layer and appeared in about
    // one run in ten.  The write is now its own dispatch (`k_gdn_tail`), which the encoder orders
    // after every read in here.
}

// ---------------------------------------------------------------------------------------
// k_gdn_tail: hand the last 3 pre-conv raw columns of the tile to the next tile.
//
// Its own dispatch on purpose.  It used to be the tail of `k_gdn_prep`, which also *reads* the
// carried tail for the first 3 tokens -- and nothing orders those two against each other inside
// one dispatch, so the reader could see the writer's value.  Splitting it out is free (36 KB of
// stores, ~3 us) and makes the handoff deterministic.  Same arguments and buffers as
// `k_gdn_prep`, so the launcher change is one entry in `gdn_launch`.
// grid = (E8_L_HEADS, 1, 1), tg = (E8G_D, 1, 1)
// ---------------------------------------------------------------------------------------
kernel void k_gdn_tail(device const half  *proj   [[buffer(0)]],
                       device float       *state  [[buffer(1)]],
                       device half        *tail   [[buffer(2)]],
                       device const half  *wconv  [[buffer(3)]],
                       device const float *A_log  [[buffer(4)]],
                       device const float *dtb    [[buffer(5)]],
                       device const half  *norm   [[buffer(6)]],
                       device half        *out    [[buffer(7)]],
                       device half        *scr    [[buffer(8)]],
                       constant GdnArgs   &a      [[buffer(E8_ARGS_INDEX)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]]) {
    (void)state; (void)wconv; (void)A_log; (void)dtb; (void)norm; (void)out; (void)scr;
    const uint tid = tid2.x;
    const uint h   = tgid.x;
    const ulong sp = a.Stile;
    const uint  ncol = a.cols ? a.cols : a.Stile;
    const uint  so   = a.src_s0;
    if (ncol < 3u || h >= (uint)E8_L_HEADS || tid >= (uint)E8G_D) return;
    const uint kh   = E8G_KH(h);
    const bool rep  = ((h % E8G_R) == 0u);
    const uint cqk  = E8G_CQ(kh, tid);
    const uint ck   = E8G_CK(kh, tid);
    const uint cv   = E8G_CV(h, tid);
    const ulong c0 = (ulong)(ncol - 3u);
    #pragma unroll
    for (int k = 0; k < 3; k++) {
        const uint kk = (uint)k;
        if (rep) {   // q/k 的尾巴按 k 头只存一份，同一 k 头的 E8G_R 个 v 头共用
            tail[(ulong)cqk * 3u + kk] = proj[(ulong)cqk * sp + so + c0 + (ulong)kk];
            tail[(ulong)ck * 3u + kk]  = proj[(ulong)ck * sp + so + c0 + (ulong)kk];
        }
        tail[(ulong)cv * 3u + kk] = proj[(ulong)cv * sp + so + c0 + (ulong)kk];
    }
}

// ---------------------------------------------------------------------------------------
// k_gdn: the sequential recurrence.  One threadgroup per (head, 32 value rows); each thread
// owns 32 state elements of one value row.  Reads qn/kn/v/g/beta coalesced from `scratch`
// (one 8192-half row per token), so there is no threadgroup barrier in the token loop.
// grid = (E8_L_HEADS, E8G_TPH), tg = (E8G_TG,)
// ---------------------------------------------------------------------------------------
kernel void k_gdn(device const half  *proj   [[buffer(0)]],
                  device float       *state  [[buffer(1)]],
                  device half        *tail   [[buffer(2)]],
                  device const half  *wconv  [[buffer(3)]],
                  device const float *A_log  [[buffer(4)]],
                  device const float *dtb    [[buffer(5)]],
                  device const half  *norm   [[buffer(6)]],
                  device half        *out    [[buffer(7)]],
                  device const half  *scr    [[buffer(8)]],
                  constant GdnArgs   &a      [[buffer(E8_ARGS_INDEX)]],
                  uint2 tgid [[threadgroup_position_in_grid]],
                  uint2 tid2 [[thread_position_in_threadgroup]],
                  uint2 ntg2 [[threads_per_threadgroup]]) {
    (void)proj; (void)tail; (void)wconv; (void)A_log; (void)dtb; (void)norm;
    const uint  tid = tid2.x;
    const uint  h   = tgid.x;
    const uint  dv0 = tgid.y * E8G_ROWS;
    const ulong sp  = a.Stile;
    const ulong osp = a.out_stride ? a.out_stride : a.Stile;
    const ulong os0 = a.out_s0;
    if (ntg2.x != E8G_TG || h >= (uint)E8_L_HEADS) return;   // threadgroup-uniform

    const uint lane = tid & 31u;
    const uint rg   = lane >> 2;                   // value row inside the warp
    const uint lr   = lane & 3u;                   // which 32-element block of dk
    const uint dv   = dv0 + (tid >> 5) * E8G_RPW + rg;
    const uint tq   = lr * E8G_M;

    device float *srow = state + ((ulong)h * E8G_D + dv) * E8G_D + tq;
    float S[E8G_M];
    #pragma unroll
    for (uint i = 0; i < E8G_M; i += 4) {
        float4 t = *(device const float4 *)(srow + i);
        S[i + 0] = t.x; S[i + 1] = t.y; S[i + 2] = t.z; S[i + 3] = t.w;
    }
    const int   St    = int(a.cols ? a.cols : a.Stile);
    const uint  so    = a.src_s0;                        // ticket 10: first source column
    const uint  kh    = E8G_KH(h);                   // q/k 槽按 k 头编号，v 槽按 v 头编号
    const ulong qbase = E8G_S_QN + kh * E8G_D + tq;  // this lane's 32 qn halves within a row
    const ulong kbase = E8G_S_KN + kh * E8G_D + tq;
    const ulong vbase = E8G_S_V  + h  * E8G_D + dv;

    for (int j = 0; j < St; j++) {
        const ulong row  = (ulong)(so + j) * E8G_S_ROW;
        const float g    = float(scr[row + E8G_S_G + h]);
        const float beta = float(scr[row + E8G_S_BETA + h]);
        const float vv   = float(scr[row + vbase]);

        // decay + kv_mem; kn is kept in registers (8 halves per vector) for the update pass.
        // 16-byte loads: a warp's four lane-in-row groups cover one 64-byte half of the row,
        // so each of these instructions is two 32-byte sectors instead of four.
        vec<half, 8> kh[E8G_M / 8];
        float ak0 = 0.0f, ak1 = 0.0f;
        #pragma unroll
        for (uint e = 0; e < E8G_M; e += 8) {
            kh[e / 8] = *(device const vec<half, 8> *)(scr + row + kbase + e);
            #pragma unroll
            for (uint q = 0; q < 8; q += 4) {
                float4 kk = float4(float(kh[e / 8][q + 0]), float(kh[e / 8][q + 1]),
                                   float(kh[e / 8][q + 2]), float(kh[e / 8][q + 3]));
                float s0 = S[e + q + 0] * g, s1 = S[e + q + 1] * g;
                float s2 = S[e + q + 2] * g, s3 = S[e + q + 3] * g;
                S[e + q + 0] = s0; S[e + q + 1] = s1; S[e + q + 2] = s2; S[e + q + 3] = s3;
                ak0 += s0 * kk.x; ak1 += s1 * kk.y;
                ak0 += s2 * kk.z; ak1 += s3 * kk.w;
            }
        }
        float kvm   = e8g_rowsum(ak0 + ak1);
        float delta = (vv - kvm) * beta;

        float ay0 = 0.0f, ay1 = 0.0f;
        #pragma unroll
        for (uint e = 0; e < E8G_M; e += 8) {
            vec<half, 8> qh = *(device const vec<half, 8> *)(scr + row + qbase + e);
            #pragma unroll
            for (uint q = 0; q < 8; q += 4) {
                float4 kk = float4(float(kh[e / 8][q + 0]), float(kh[e / 8][q + 1]),
                                   float(kh[e / 8][q + 2]), float(kh[e / 8][q + 3]));
                float4 qq = float4(float(qh[q + 0]), float(qh[q + 1]),
                                   float(qh[q + 2]), float(qh[q + 3]));
                float s0 = S[e + q + 0] + kk.x * delta, s1 = S[e + q + 1] + kk.y * delta;
                float s2 = S[e + q + 2] + kk.z * delta, s3 = S[e + q + 3] + kk.w * delta;
                S[e + q + 0] = s0; S[e + q + 1] = s1; S[e + q + 2] = s2; S[e + q + 3] = s3;
                ay0 += s0 * qq.x; ay1 += s1 * qq.y;
                ay0 += s2 * qq.z; ay1 += s3 * qq.w;
            }
        }
        float y = e8g_rowsum(ay0 + ay1);
        if (lr == 0u) out[((ulong)(h * E8G_D + dv)) * osp + os0 + (ulong)j] = half(y);
    }

    #pragma unroll
    for (uint i = 0; i < E8G_M; i += 4) {
        float4 t = float4(S[i + 0], S[i + 1], S[i + 2], S[i + 3]);
        *(device float4 *)(srow + i) = t;
    }
}

// ---------------------------------------------------------------------------------------
// k_gdn_norm: gated RMSNorm over the pre-norm y that k_gdn left in out[0:2048), in place.
//   out[h*128+dv][s] = yh * rsqrt(sum_dv(yh^2)/128 + eps) * sigmoid(z)*z * norm[dv]
// Warp per token: lane owns dv = 4*lane .. +4, so the mean over the head's 128 value channels
// is one simd_sum.  grid = (ceil(Stile/4), 16), tg = (128,).
// ---------------------------------------------------------------------------------------
kernel void k_gdn_norm(device const half  *proj   [[buffer(0)]],
                       device float       *state  [[buffer(1)]],
                       device half        *tail   [[buffer(2)]],
                       device const half  *wconv  [[buffer(3)]],
                       device const float *A_log  [[buffer(4)]],
                       device const float *dtb    [[buffer(5)]],
                       device const half  *norm   [[buffer(6)]],
                       device half        *out    [[buffer(7)]],
                       device const half  *scr    [[buffer(8)]],
                       constant GdnArgs   &a      [[buffer(E8_ARGS_INDEX)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]],
                       uint2 ntg2 [[threads_per_threadgroup]]) {
    (void)state; (void)tail; (void)wconv; (void)A_log; (void)dtb; (void)scr;
    const uint  tid  = tid2.x;
    const ulong sp   = a.Stile;
    const ulong osp  = a.out_stride ? a.out_stride : a.Stile;
    const ulong os0  = a.out_s0;
    const uint  h    = tgid.y;
    const uint  warp = tid >> 5;
    const uint  lane = tid & 31u;
    const int   s    = int(tgid.x) * int(ntg2.x >> 5) + int(warp);   // one token per warp
    if (s >= int(a.cols ? a.cols : a.Stile) || h >= (uint)E8_L_HEADS) return;

    const uint  dv  = lane * 4u;
    const ulong off = (ulong)(h * E8G_D + dv) * osp + os0 + (ulong)s;
    device half *o = out + off;
    device const half *z = proj + (ulong)E8G_CZ(h, dv) * sp + (ulong)a.src_s0
                                + (ulong)s;
    float y[4];
    float sq = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 4u; i++) { y[i] = float(o[(ulong)i * osp]); sq += y[i] * y[i]; }
    float nsc = rsqrt(simd_sum(sq) * (1.0f / float(E8G_D)) + a.eps);
    #pragma unroll
    for (uint i = 0; i < 4u; i++) {
        float zv = float(z[(ulong)i * sp]);
        o[(ulong)i * osp] = half(y[i] * nsc * (zv / (1.0f + exp(-zv))) * float(norm[dv + i]));
    }
}
