// E8c — fast Metal noncausal full-attention core: k_attn.
//
// Consumes the prepped qg / kv surfaces in place (see e8_common.h), so the driver only runs
// k_q_prep / k_kv_prep before it.  Everything is channel-major: x[c, s] at c * stride + s.
//
// --------------------------------------------------------------------------------------
// HOW TO LAUNCH  (the driver must use exactly this)
// --------------------------------------------------------------------------------------
//   buffers 0..3 = qg, kv, out, scratch        args = M.a_attn(T, Stile, s0, nk=, nq=)
//                                              (S6 batched tails: also nka=, segw=, nkw= --
//                                               every query row sees [0, nka) plus its own
//                                               `nkw`-row slot; segw=0 is the plain mask)
//   grid = (8 * ceil(Stile / AT_BM), 1, 1)     tg   = (AT_TG, 1, 1)
// with AT_BM / AT_TG as configured below.  One threadgroup per (query head, AT_BM-query tile
// of this call):  tgid -> head = tgid/tiles, tile = tgid % tiles.  Head-major on purpose: all
// concurrently resident threadgroups then read the same kv head's K/V, so the K+V working set
// of that head stays in the system cache instead of streaming out of DRAM once per tile.
//
//   Requires T % 8 == 0 (16-byte staging loads) and s0 + Stile <= T.
//   `scratch` is unused: the whole score block stays in registers, no (T,T) tensor.
//
// --------------------------------------------------------------------------------------
// E8c changes (this revision) — the GPU is an M2 Pro (16 cores, 5.65 TFLOPS fp16 MMA peak).
// Measured facts that drove them:
//   * one 256-thread threadgroup per core reaches only ~2.2 TFLOPS whatever the code does:
//     8 warps cannot keep the MMA pipe of a core busy (a pure register MMA loop at 256
//     threads / 1 tg per core measures 4.5 of 5.65 TFLOPS, at 384 threads 5.0, at 512 5.1).
//     => AT_BM was 64 (8 warps), is now 96 (12 warps): 1.39x on the same instruction mix.
//   * the P^T buffer is per-warp private: warp sg only ever stores/loads its own 8-query
//     column, so parking it as pt[AT_NW][AT_BN][8] instead of pt[AT_BN][AT_BM+8] removes the
//     padding, makes the 8x8 fragment loads bank-conflict-free, and frees ~5 KB of the 32 KB
//     threadgroup budget for more warps (measured +4%).
//   * staging chunk size (AT_KC) trades threadgroup memory against barrier count; 32 channels
//     leaves room for 2 threadgroups per core, 64 keeps the barrier count at 8 per key block.
//
// --------------------------------------------------------------------------------------
// Structure (flash attention over key blocks, online softmax in registers)
// --------------------------------------------------------------------------------------
// Threadgroup = AT_NW simdgroups; simdgroup sg owns query rows [sg*8, sg*8+8) of the tile.
// Per key block of BN = 64:
//
//   S[8 x 8]  = Q[8 x 256] @ K[256 x 8]     8 fp32 fragments, tg = key group 0..7
//   online softmax over the lane's own row  (row max/sum reduced across the 4 lanes that
//                                           share a row with simd_shuffle_xor 1 and 8)
//   P = exp2(...)  ->  Pp^T[k][m] in threadgroup memory (transposed: 16 half stores/lane)
//   C[eg]     = V[8e x 8t] @ Pt[8t x 8m]    32 fp32 fragments = this row block's whole
//                                           256-channel output, accumulated TRANSPOSED
//
// The transpose (C = O^T, rows are output channels, columns are queries) is what makes this
// shape work: the accumulator is 256 channels x 8 queries, so no operand fragment has to be
// re-loaded per channel group, and each lane's two accumulator elements belong to a fixed
// pair of queries.  Those two columns need the softmax rescale factor / 1/l of *other*
// lanes, which is two simd_shuffles per block (at_owner_of).
//
// K and V share the same two AT_KC x 64 threadgroup buffers, staged one chunk ahead of the
// MMAs: one threadgroup_barrier per chunk, and the global-load latency of the next chunk
// overlaps the current chunk's MMAs.  The last PV step prefetches the next block's first K
// chunk, so block boundaries cost nothing.
//
// Every 8x8 fragment is its own named variable rather than an array element.  That is not
// style: this compiler does not scalarise fragment arrays, so an indexed `simdgroup_float8x8
// C[32]` silently becomes thread-local memory and the kernel runs 4x slower (measured
// 1.14 vs 4.50 TFLOPS on the same loop).  The macros below exist only to spell that out.
//
// --------------------------------------------------------------------------------------
// Numerics
// --------------------------------------------------------------------------------------
// Scores, softmax and the output accumulation are fp32; only P and the final out[] are
// rounded to fp16.  The 1/16 attention scale is folded into the exponential's base-2
// argument (exp2(scale*log2e*s)) instead of into q, so q keeps exactly the reference's fp16
// values and only fp32 multiplies touch it.

#include "e8_common.h"

#include <metal_simdgroup_matrix>

#ifndef AT_BM
#define AT_BM     128u         // queries per threadgroup (8 per simdgroup)
#endif
#ifndef AT_TG
#define AT_TG     512u         // threads per threadgroup = AT_BM/8 * 32
#endif
#define AT_NW     (AT_TG / 32u) // simdgroups per threadgroup
#define AT_BN     64u          // keys per block
#ifndef AT_KC
#define AT_KC     32u          // channels staged per chunk (32 or 64)
#endif
#define AT_LDK    72u          // staged-chunk leading dim in halves: AT_BN + 8
#if (AT_KC == 64)
#define AT_NSUB   8u           // 8-channel sub-chunks per staging chunk
#else
#define AT_NSUB   4u
#endif
#define AT_NCHUNK (256u / AT_KC)   // staging chunks per 256-channel K or V block
#define AT_LOG2E  1.4426950408889634f

// 8x8 fragment geometry: lane l holds elements (row, col) and (row, col+1), with
//   row = ((l>>1)&3) | ((l>>4)&1)<<2        col = (l&1)<<1 | ((l>>3)&1)<<2
// verified on this GPU by simdgroup_store of a known matrix through a lane dump.
static inline uint at_row_of(uint l)   { return ((l >> 1) & 3u) | (((l >> 4) & 1u) << 2); }
static inline uint at_col_of(uint l)   { return ((l & 1u) << 1) | (((l >> 3) & 1u) << 2); }
// lane that owns query row m of the 8-row group (the 4 owners are owner, +1, +8, +9)
static inline uint at_owner_of(uint m) { return ((m & 3u) << 1) | ((m >> 2) << 4); }


// ---- QK: b_* = current K key-group fragments, n_* = one d-sub-chunk lookahead ----
#define AT_QK_LD(P, O) \
    simdgroup_load(b_0, (P) + (O) + 0u, AT_LDK); \
    simdgroup_load(b_1, (P) + (O) + 8u, AT_LDK); \
    simdgroup_load(b_2, (P) + (O) + 16u, AT_LDK); \
    simdgroup_load(b_3, (P) + (O) + 24u, AT_LDK); \
    simdgroup_load(b_4, (P) + (O) + 32u, AT_LDK); \
    simdgroup_load(b_5, (P) + (O) + 40u, AT_LDK); \
    simdgroup_load(b_6, (P) + (O) + 48u, AT_LDK); \
    simdgroup_load(b_7, (P) + (O) + 56u, AT_LDK);
#define AT_QK_LDN(P, O) \
    simdgroup_load(n_0, (P) + (O) + 0u, AT_LDK); \
    simdgroup_load(n_1, (P) + (O) + 8u, AT_LDK); \
    simdgroup_load(n_2, (P) + (O) + 16u, AT_LDK); \
    simdgroup_load(n_3, (P) + (O) + 24u, AT_LDK); \
    simdgroup_load(n_4, (P) + (O) + 32u, AT_LDK); \
    simdgroup_load(n_5, (P) + (O) + 40u, AT_LDK); \
    simdgroup_load(n_6, (P) + (O) + 48u, AT_LDK); \
    simdgroup_load(n_7, (P) + (O) + 56u, AT_LDK);
#define AT_QK_CP \
    b_0 = n_0; \
    b_1 = n_1; \
    b_2 = n_2; \
    b_3 = n_3; \
    b_4 = n_4; \
    b_5 = n_5; \
    b_6 = n_6; \
    b_7 = n_7;
#define AT_QK_MMA(A) \
    simdgroup_multiply_accumulate(s_0, (A), b_0, s_0); \
    simdgroup_multiply_accumulate(s_1, (A), b_1, s_1); \
    simdgroup_multiply_accumulate(s_2, (A), b_2, s_2); \
    simdgroup_multiply_accumulate(s_3, (A), b_3, s_3); \
    simdgroup_multiply_accumulate(s_4, (A), b_4, s_4); \
    simdgroup_multiply_accumulate(s_5, (A), b_5, s_5); \
    simdgroup_multiply_accumulate(s_6, (A), b_6, s_6); \
    simdgroup_multiply_accumulate(s_7, (A), b_7, s_7);

// ---- PV: v_* = V fragments for the current key sub-chunk, w_* = lookahead ----
#if (AT_KC == 64)
#define AT_PV_LD(P, O) \
    simdgroup_load(v_0, (P) + (O) + 0u * AT_LDK, AT_LDK); \
    simdgroup_load(v_1, (P) + (O) + 8u * AT_LDK, AT_LDK); \
    simdgroup_load(v_2, (P) + (O) + 16u * AT_LDK, AT_LDK); \
    simdgroup_load(v_3, (P) + (O) + 24u * AT_LDK, AT_LDK); \
    simdgroup_load(v_4, (P) + (O) + 32u * AT_LDK, AT_LDK); \
    simdgroup_load(v_5, (P) + (O) + 40u * AT_LDK, AT_LDK); \
    simdgroup_load(v_6, (P) + (O) + 48u * AT_LDK, AT_LDK); \
    simdgroup_load(v_7, (P) + (O) + 56u * AT_LDK, AT_LDK);
#define AT_PV_LDN(P, O) \
    simdgroup_load(w_0, (P) + (O) + 0u * AT_LDK, AT_LDK); \
    simdgroup_load(w_1, (P) + (O) + 8u * AT_LDK, AT_LDK); \
    simdgroup_load(w_2, (P) + (O) + 16u * AT_LDK, AT_LDK); \
    simdgroup_load(w_3, (P) + (O) + 24u * AT_LDK, AT_LDK); \
    simdgroup_load(w_4, (P) + (O) + 32u * AT_LDK, AT_LDK); \
    simdgroup_load(w_5, (P) + (O) + 40u * AT_LDK, AT_LDK); \
    simdgroup_load(w_6, (P) + (O) + 48u * AT_LDK, AT_LDK); \
    simdgroup_load(w_7, (P) + (O) + 56u * AT_LDK, AT_LDK);
#define AT_PV_CP \
    v_0 = w_0; \
    v_1 = w_1; \
    v_2 = w_2; \
    v_3 = w_3; \
    v_4 = w_4; \
    v_5 = w_5; \
    v_6 = w_6; \
    v_7 = w_7;
#else
#define AT_PV_LD(P, O) \
    simdgroup_load(v_0, (P) + (O) + 0u * AT_LDK, AT_LDK); \
    simdgroup_load(v_1, (P) + (O) + 8u * AT_LDK, AT_LDK); \
    simdgroup_load(v_2, (P) + (O) + 16u * AT_LDK, AT_LDK); \
    simdgroup_load(v_3, (P) + (O) + 24u * AT_LDK, AT_LDK);
#define AT_PV_LDN(P, O) \
    simdgroup_load(w_0, (P) + (O) + 0u * AT_LDK, AT_LDK); \
    simdgroup_load(w_1, (P) + (O) + 8u * AT_LDK, AT_LDK); \
    simdgroup_load(w_2, (P) + (O) + 16u * AT_LDK, AT_LDK); \
    simdgroup_load(w_3, (P) + (O) + 24u * AT_LDK, AT_LDK);
#define AT_PV_CP \
    v_0 = w_0; \
    v_1 = w_1; \
    v_2 = w_2; \
    v_3 = w_3;
#endif

#if (AT_KC == 64)      // one staging chunk = 64 channels = 8 accumulator frags
#define AT_PV_MMA0(PB) \
    simdgroup_multiply_accumulate(c_0, v_0, (PB), c_0); \
    simdgroup_multiply_accumulate(c_1, v_1, (PB), c_1); \
    simdgroup_multiply_accumulate(c_2, v_2, (PB), c_2); \
    simdgroup_multiply_accumulate(c_3, v_3, (PB), c_3); \
    simdgroup_multiply_accumulate(c_4, v_4, (PB), c_4); \
    simdgroup_multiply_accumulate(c_5, v_5, (PB), c_5); \
    simdgroup_multiply_accumulate(c_6, v_6, (PB), c_6); \
    simdgroup_multiply_accumulate(c_7, v_7, (PB), c_7);
#define AT_PV_MMA1(PB) \
    simdgroup_multiply_accumulate(c_8, v_0, (PB), c_8); \
    simdgroup_multiply_accumulate(c_9, v_1, (PB), c_9); \
    simdgroup_multiply_accumulate(c_10, v_2, (PB), c_10); \
    simdgroup_multiply_accumulate(c_11, v_3, (PB), c_11); \
    simdgroup_multiply_accumulate(c_12, v_4, (PB), c_12); \
    simdgroup_multiply_accumulate(c_13, v_5, (PB), c_13); \
    simdgroup_multiply_accumulate(c_14, v_6, (PB), c_14); \
    simdgroup_multiply_accumulate(c_15, v_7, (PB), c_15);
#define AT_PV_MMA2(PB) \
    simdgroup_multiply_accumulate(c_16, v_0, (PB), c_16); \
    simdgroup_multiply_accumulate(c_17, v_1, (PB), c_17); \
    simdgroup_multiply_accumulate(c_18, v_2, (PB), c_18); \
    simdgroup_multiply_accumulate(c_19, v_3, (PB), c_19); \
    simdgroup_multiply_accumulate(c_20, v_4, (PB), c_20); \
    simdgroup_multiply_accumulate(c_21, v_5, (PB), c_21); \
    simdgroup_multiply_accumulate(c_22, v_6, (PB), c_22); \
    simdgroup_multiply_accumulate(c_23, v_7, (PB), c_23);
#define AT_PV_MMA3(PB) \
    simdgroup_multiply_accumulate(c_24, v_0, (PB), c_24); \
    simdgroup_multiply_accumulate(c_25, v_1, (PB), c_25); \
    simdgroup_multiply_accumulate(c_26, v_2, (PB), c_26); \
    simdgroup_multiply_accumulate(c_27, v_3, (PB), c_27); \
    simdgroup_multiply_accumulate(c_28, v_4, (PB), c_28); \
    simdgroup_multiply_accumulate(c_29, v_5, (PB), c_29); \
    simdgroup_multiply_accumulate(c_30, v_6, (PB), c_30); \
    simdgroup_multiply_accumulate(c_31, v_7, (PB), c_31);
#else                  // one staging chunk = 32 channels = 4 accumulator frags
#define AT_PV_MMA0(PB) \
    simdgroup_multiply_accumulate(c_0, v_0, (PB), c_0); \
    simdgroup_multiply_accumulate(c_1, v_1, (PB), c_1); \
    simdgroup_multiply_accumulate(c_2, v_2, (PB), c_2); \
    simdgroup_multiply_accumulate(c_3, v_3, (PB), c_3);
#define AT_PV_MMA1(PB) \
    simdgroup_multiply_accumulate(c_4, v_0, (PB), c_4); \
    simdgroup_multiply_accumulate(c_5, v_1, (PB), c_5); \
    simdgroup_multiply_accumulate(c_6, v_2, (PB), c_6); \
    simdgroup_multiply_accumulate(c_7, v_3, (PB), c_7);
#define AT_PV_MMA2(PB) \
    simdgroup_multiply_accumulate(c_8, v_0, (PB), c_8); \
    simdgroup_multiply_accumulate(c_9, v_1, (PB), c_9); \
    simdgroup_multiply_accumulate(c_10, v_2, (PB), c_10); \
    simdgroup_multiply_accumulate(c_11, v_3, (PB), c_11);
#define AT_PV_MMA3(PB) \
    simdgroup_multiply_accumulate(c_12, v_0, (PB), c_12); \
    simdgroup_multiply_accumulate(c_13, v_1, (PB), c_13); \
    simdgroup_multiply_accumulate(c_14, v_2, (PB), c_14); \
    simdgroup_multiply_accumulate(c_15, v_3, (PB), c_15);
#define AT_PV_MMA4(PB) \
    simdgroup_multiply_accumulate(c_16, v_0, (PB), c_16); \
    simdgroup_multiply_accumulate(c_17, v_1, (PB), c_17); \
    simdgroup_multiply_accumulate(c_18, v_2, (PB), c_18); \
    simdgroup_multiply_accumulate(c_19, v_3, (PB), c_19);
#define AT_PV_MMA5(PB) \
    simdgroup_multiply_accumulate(c_20, v_0, (PB), c_20); \
    simdgroup_multiply_accumulate(c_21, v_1, (PB), c_21); \
    simdgroup_multiply_accumulate(c_22, v_2, (PB), c_22); \
    simdgroup_multiply_accumulate(c_23, v_3, (PB), c_23);
#define AT_PV_MMA6(PB) \
    simdgroup_multiply_accumulate(c_24, v_0, (PB), c_24); \
    simdgroup_multiply_accumulate(c_25, v_1, (PB), c_25); \
    simdgroup_multiply_accumulate(c_26, v_2, (PB), c_26); \
    simdgroup_multiply_accumulate(c_27, v_3, (PB), c_27);
#define AT_PV_MMA7(PB) \
    simdgroup_multiply_accumulate(c_28, v_0, (PB), c_28); \
    simdgroup_multiply_accumulate(c_29, v_1, (PB), c_29); \
    simdgroup_multiply_accumulate(c_30, v_2, (PB), c_30); \
    simdgroup_multiply_accumulate(c_31, v_3, (PB), c_31);
#endif

// ---- leaf helpers -------------------------------------------------------------------
#define AT_SCALE(C)  { C.thread_elements()[0] *= scl0; C.thread_elements()[1] *= scl1; }
#define AT_ZEROC(C)  { C.thread_elements()[0] = 0.0f;    C.thread_elements()[1] = 0.0f; }
#define AT_LDQ(Q, DG) { \
    device const half *qp = qg + (ulong)(qbase + (DG)) * QT + scol; \
    Q.thread_elements()[0] = qp[0]; Q.thread_elements()[1] = qp[QT]; }
#define AT_STORE(EG, E) { \
    device const half *gp = qg + (ulong)(head * 2u * uint(E8_HEAD_DIM) + uint(E8_HEAD_DIM) \
                                   + (E)) * QT + a.s0 + jc; \
    device half *op = out + (ulong)(head * uint(E8_HEAD_DIM) + (E)) * osp + os0 + jc; \
    const float y0 = c_##EG.thread_elements()[0] * iv0 * float(gp[0]); \
    if (jlast) { op[0] = half(y0); } \
    else { const float y1 = c_##EG.thread_elements()[1] * iv1 * float(gp[1]); \
           *(device half2 *)op = half2(half(y0), half(y1)); } }
#define AT_STAGE(P, CHAN, TB) at_stage(&kbuf[((P) == kb0) ? 0u : 1u][0][0], kv, tid, T, (CHAN), (TB));
// S6: visible keys are [0, nka) plus the query row's own slot [nkb, nkc).  With `segw` = 0 a
// caller passes nka = nkc = nk and nkb = 0, so this is exactly the pre-S6 `t < nk`.
static inline bool at_visible(uint t, uint nka, uint nkb, uint nkc) {
    return t < nka || (t >= nkb && t < nkc);
}
#define AT_SM1(K, S) { const uint tA = t0 + (K) * 8u + col_l;                        \
    float v0 = (S).thread_elements()[0] * K1, v1 = (S).thread_elements()[1] * K1;       \
    if (!at_visible(tA, nka, nkb, nkc))        v0 = -INFINITY;                          \
    if (!at_visible(tA + 1u, nka, nkb, nkc))   v1 = -INFINITY;                          \
    sv0[K] = v0; sv1[K] = v1; mx = max(mx, max(v0, v1)); }
#define AT_SM2(K) { const float p0 = exp2(sv0[K] - m_new), p1 = exp2(sv1[K] - m_new);     \
    sum += p0 + p1;                                                                     \
    const uint tl = (K) * 8u + col_l;      /* LOCAL key index: pt is a per-warp buffer */ \
    pt[sg][tl][row_l] = half(p0); pt[sg][tl + 1u][row_l] = half(p1); }

// threadgroup P^T is per warp: only its own 8-query column is ever stored or read, so the
// row stride is exactly 8 halves = 16 B, which puts the 8 rows of a fragment load on 8
// distinct 16-byte bank groups (0 conflicts) and saves the inter-warp padding.
#define AT_LDP    8u
#define AT_PV_PT \
    simdgroup_load(pbs_0, &pt[sg][0u][0u], AT_LDP); \
    simdgroup_load(pbs_1, &pt[sg][8u][0u], AT_LDP); \
    simdgroup_load(pbs_2, &pt[sg][16u][0u], AT_LDP); \
    simdgroup_load(pbs_3, &pt[sg][24u][0u], AT_LDP); \
    simdgroup_load(pbs_4, &pt[sg][32u][0u], AT_LDP); \
    simdgroup_load(pbs_5, &pt[sg][40u][0u], AT_LDP); \
    simdgroup_load(pbs_6, &pt[sg][48u][0u], AT_LDP); \
    simdgroup_load(pbs_7, &pt[sg][56u][0u], AT_LDP);

// ---- one pipeline step: stage chunk >>NXT<< first (its global loads then overlap the
// ---- current chunk's MMAs), run the chunk's MMAs, one barrier.
#define AT_QK_SUB(P, A, NEXTOFF) { AT_QK_LDN(P, NEXTOFF); AT_QK_MMA(A); AT_QK_CP }
#define AT_PV_SUB(P, TS, NEXTOFF, ACC, PB) { \
    AT_PV_LDN(P, NEXTOFF); ACC(PB); AT_PV_CP }

#if (AT_KC == 64)
#define AT_QK_STEP(CUR, NXT, CHAN, A0, A1, A2, A3, A4, A5, A6, A7) \
    { AT_STAGE(NXT, CHAN, t0);                         \
      AT_QK_LD(CUR, 0u);                               \
      AT_QK_SUB(CUR, A0, 8u * AT_LDK) \
      AT_QK_SUB(CUR, A1, 16u * AT_LDK) \
      AT_QK_SUB(CUR, A2, 24u * AT_LDK) \
      AT_QK_SUB(CUR, A3, 32u * AT_LDK) \
      AT_QK_SUB(CUR, A4, 40u * AT_LDK) \
      AT_QK_SUB(CUR, A5, 48u * AT_LDK) \
      AT_QK_SUB(CUR, A6, 56u * AT_LDK) \
      AT_QK_SUB(CUR, A7, 0u * AT_LDK) \
      threadgroup_barrier(mem_flags::mem_threadgroup); }
#else
#define AT_QK_STEP(CUR, NXT, CHAN, A0, A1, A2, A3) \
    { AT_STAGE(NXT, CHAN, t0);                         \
      AT_QK_LD(CUR, 0u);                               \
      AT_QK_SUB(CUR, A0, 8u * AT_LDK) \
      AT_QK_SUB(CUR, A1, 16u * AT_LDK) \
      AT_QK_SUB(CUR, A2, 24u * AT_LDK) \
      AT_QK_SUB(CUR, A3, 0u * AT_LDK) \
      threadgroup_barrier(mem_flags::mem_threadgroup); }
#endif
#define AT_PV_STEP(CUR, NXT, CHAN, TB, ACC) \
    { AT_STAGE(NXT, CHAN, TB);                         \
      AT_PV_LD(CUR, 0u);                               \
      AT_PV_PT;                                        \
      AT_PV_SUB(CUR, 0u,  8u, ACC, pbs_0) \
      AT_PV_SUB(CUR, 1u, 16u, ACC, pbs_1) \
      AT_PV_SUB(CUR, 2u, 24u, ACC, pbs_2) \
      AT_PV_SUB(CUR, 3u, 32u, ACC, pbs_3) \
      AT_PV_SUB(CUR, 4u, 40u, ACC, pbs_4) \
      AT_PV_SUB(CUR, 5u, 48u, ACC, pbs_5) \
      AT_PV_SUB(CUR, 6u, 56u, ACC, pbs_6) \
      AT_PV_SUB(CUR, 7u,  0u, ACC, pbs_7) \
      threadgroup_barrier(mem_flags::mem_threadgroup); }


// -------------------------------------------------------------------------------------
// k_attn
// -------------------------------------------------------------------------------------
// M2.2 第 3 片：本机的 MSL 编译器不接受 lambda（`lambda expressions are not supported in
// Metal`），而快核用一个捕获上下文的分片搬运闭包。这里把它展开成普通函数：参数与原闭包捕获的
// 变量一一对应（剩余使用到的 `AT_KC` / `AT_LDK` / `AT_TG` 是宏），循环体和分支逐字保留，
// 语义不变。改动仅限本仓库内的这份副本，`docs/` 里的旧实现不动。
static inline void at_stage(threadgroup half *buf, device const half *kv, uint tid, uint T,
                            uint chan_base, uint tbase) {
    #pragma unroll
    for (uint u = tid; u < AT_KC * 8u; u += AT_TG) {
        const uint c = u >> 3, j = u & 7u;
        const uint t = tbase + j * 8u;
        threadgroup uint4 *dst = (threadgroup uint4 *)(buf + c * AT_LDK + j * 8u);
        if (t + 8u <= T) {
            *dst = *(device const uint4 *)(kv + (ulong)(chan_base + c) * T + t);
        } else {
            *dst = uint4(0u, 0u, 0u, 0u);   // masked out by the score mask
        }
    }
}

[[max_total_threads_per_threadgroup(1024)]]
kernel void k_attn(device const half  *qg      [[buffer(0)]],
                   device const half  *kv      [[buffer(1)]],
                   device half        *out     [[buffer(2)]],
                   device float       *scratch [[buffer(3)]],
                   constant AttnArgs  &a       [[buffer(E8_ARGS_INDEX)]],
                   uint tgid [[threadgroup_position_in_grid]],
                   uint tid  [[thread_position_in_threadgroup]]) {
    // ---- which (query head, AT_BM-query tile) this threadgroup is
    const uint T     = a.T;
    // S7: qg may be a *window* buffer while kv is still the whole-sequence cache, so qg has its
    // own column count.  `qgT` = 0 means the two are one buffer (every call before S7), which is
    // also why the whole kernel can stay address-identical in that case.
    const uint QT    = a.qgT ? a.qgT : a.T;
    // nk = keys that actually exist.  T is the *padded* length, i.e. the column count of qg/kv,
    // and every column past nk is a zero row that used to enter the softmax as a real key.
    const uint nk    = min(a.nk, a.T);
    const uint Stile = a.Stile;
    const uint osp   = a.out_stride ? a.out_stride : Stile;
    const uint os0   = a.out_s0;
    // One threadgroup per (head, AT_BM queries), but only `nq` query rows exist, so the tile count
    // has to come from nq and not from Stile.  The two agree unless the caller shrank the grid,
    // and when they disagree every threadgroup decodes the wrong (head, j0): measured with the
    // Stile-based count and nq = 500 of 2048, h differed by 22 between tiles.
    const uint tiles = max(1u, (min(a.nq, Stile) + AT_BM - 1u) / AT_BM);
    const uint head  = tgid / tiles;
    if (head >= uint(E8_Q_HEADS)) return;               // grid overrun guard
    const uint j0  = (tgid % tiles) * AT_BM;            // query column within this call
    // ---- key visibility: [0, nka) always, plus this query block's own slot when `segw` != 0
    // A batched tail pass (S6) runs K tails as K slots of one tile: they share the prefix and
    // none may read another's rows, which `nk` alone cannot say.  `segw` is a multiple of AT_BM,
    // so a threadgroup is always inside one slot.  `segw` = 0 gives nka = nkc = nk, nkb = 0, i.e.
    // the single bound every pre-S6 caller passes.
    const uint nka  = a.segw ? a.nka : nk;
    const uint qpos = a.s0 + j0;                        // first query column of this threadgroup
    const uint slot = (a.segw && qpos > nka) ? (qpos - nka) / a.segw : 0u;
    const uint nkb  = a.segw ? nka + slot * a.segw : 0u;            // first key of the own slot
    const uint nkc  = a.segw ? min(nkb + a.nkw, T) : nk;            // one past its last key
    const uint kvh = head / uint(E8_Q_HEADS / E8_KV_HEADS);
    const float K1 = a.scale * AT_LOG2E;                // score -> base-2 logit

    // ---- lane / simdgroup geometry (which query row and which two columns a lane holds)
    const uint sg    = tid >> 5, l = tid & 31u;
    const uint row_l = at_row_of(l);                    // query row this lane owns
    const uint col_l = at_col_of(l);                    // one of 0, 2, 4, 6
    const uint own0  = at_owner_of(col_l);              // lane that owns row col_l
    const uint own1  = own0 + 2u;                       // ... and row col_l + 1
    const uint myq   = j0 + sg * 8u;                    // this simdgroup's first query column
    const uint qcol  = myq + col_l;

    // ---- K and V staging: two AT_KC x 64 buffers, reused by both phases (they never overlap)
    threadgroup half kbuf[2][AT_KC][AT_LDK];
    threadgroup half pt[AT_NW][AT_BN][AT_LDP];          // P^T, per-warp: [warp][key][query]
    (void)scratch;                                      // no (T,T) scratch needed

// -------------------------------------------------------------------------------------
// Fragments.  Named, never arrays (see the note at the top).
// -------------------------------------------------------------------------------------
    simdgroup_half8x8  q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, q_11, q_12, q_13, q_14, q_15, q_16, q_17, q_18, q_19, q_20, q_21, q_22, q_23, q_24, q_25, q_26, q_27, q_28, q_29, q_30, q_31;
    simdgroup_half8x8  b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7;
    simdgroup_half8x8  n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7;
    simdgroup_half8x8  v_0, v_1, v_2, v_3, v_4, v_5, v_6, v_7;
    simdgroup_half8x8  w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7;
    simdgroup_half8x8  pbs_0, pbs_1, pbs_2, pbs_3, pbs_4, pbs_5, pbs_6, pbs_7;   // P^T frags for the chunk
    simdgroup_float8x8 s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7;
    simdgroup_float8x8 c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15;
    simdgroup_float8x8 c_16, c_17, c_18, c_19, c_20, c_21, c_22, c_23, c_24, c_25, c_26, c_27, c_28, c_29, c_30, c_31;

    // ---- Q into registers, once per query tile: element (row_l, col_l) of fragment dg
    {
        const uint scol  = min(a.s0 + myq + row_l, min(a.s0 + Stile, QT) - 1u);
        const uint qbase = head * 2u * uint(E8_HEAD_DIM) + col_l;   // head*512 + col_l
        AT_LDQ(q_0, 0u)
        AT_LDQ(q_1, 8u)
        AT_LDQ(q_2, 16u)
        AT_LDQ(q_3, 24u)
        AT_LDQ(q_4, 32u)
        AT_LDQ(q_5, 40u)
        AT_LDQ(q_6, 48u)
        AT_LDQ(q_7, 56u)
        AT_LDQ(q_8, 64u)
        AT_LDQ(q_9, 72u)
        AT_LDQ(q_10, 80u)
        AT_LDQ(q_11, 88u)
        AT_LDQ(q_12, 96u)
        AT_LDQ(q_13, 104u)
        AT_LDQ(q_14, 112u)
        AT_LDQ(q_15, 120u)
        AT_LDQ(q_16, 128u)
        AT_LDQ(q_17, 136u)
        AT_LDQ(q_18, 144u)
        AT_LDQ(q_19, 152u)
        AT_LDQ(q_20, 160u)
        AT_LDQ(q_21, 168u)
        AT_LDQ(q_22, 176u)
        AT_LDQ(q_23, 184u)
        AT_LDQ(q_24, 192u)
        AT_LDQ(q_25, 200u)
        AT_LDQ(q_26, 208u)
        AT_LDQ(q_27, 216u)
        AT_LDQ(q_28, 224u)
        AT_LDQ(q_29, 232u)
        AT_LDQ(q_30, 240u)
        AT_LDQ(q_31, 248u)
    }
    AT_ZEROC(s_0)
    AT_ZEROC(s_1)
    AT_ZEROC(s_2)
    AT_ZEROC(s_3)
    AT_ZEROC(s_4)
    AT_ZEROC(s_5)
    AT_ZEROC(s_6)
    AT_ZEROC(s_7)
    AT_ZEROC(c_0)
    AT_ZEROC(c_1)
    AT_ZEROC(c_2)
    AT_ZEROC(c_3)
    AT_ZEROC(c_4)
    AT_ZEROC(c_5)
    AT_ZEROC(c_6)
    AT_ZEROC(c_7)
    AT_ZEROC(c_8)
    AT_ZEROC(c_9)
    AT_ZEROC(c_10)
    AT_ZEROC(c_11)
    AT_ZEROC(c_12)
    AT_ZEROC(c_13)
    AT_ZEROC(c_14)
    AT_ZEROC(c_15)
    AT_ZEROC(c_16)
    AT_ZEROC(c_17)
    AT_ZEROC(c_18)
    AT_ZEROC(c_19)
    AT_ZEROC(c_20)
    AT_ZEROC(c_21)
    AT_ZEROC(c_22)
    AT_ZEROC(c_23)
    AT_ZEROC(c_24)
    AT_ZEROC(c_25)
    AT_ZEROC(c_26)
    AT_ZEROC(c_27)
    AT_ZEROC(c_28)
    AT_ZEROC(c_29)
    AT_ZEROC(c_30)
    AT_ZEROC(c_31)
    float m_run = -INFINITY, l_run = 0.0f;

    // ---- cooperative staging of an AT_KC-channel x 64-key column block of kv
    // （原实现是这里的捕获闭包 `auto stage = [&](...)`，已提到文件顶部成为 `at_stage()`。）
    threadgroup half *const kb0 = &kbuf[0][0][0];
    threadgroup half *const kb1 = &kbuf[1][0][0];
    const uint kbase = kvh * uint(E8_HEAD_DIM);
    const uint vbase = uint(E8_KV_PROJ) + kvh * uint(E8_HEAD_DIM);
    const uint nb = (nkc + AT_BN - 1u) / AT_BN;   // S6: up to this block's own visible top

    AT_STAGE(kb0, kbase, 0u);                   // prologue: first K chunk
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint blk = 0; blk < nb; blk++) {
        const uint t0 = blk * AT_BN;
        AT_ZEROC(s_0)
        AT_ZEROC(s_1)
        AT_ZEROC(s_2)
        AT_ZEROC(s_3)
        AT_ZEROC(s_4)
        AT_ZEROC(s_5)
        AT_ZEROC(s_6)
        AT_ZEROC(s_7)
        // ---- QK, 256 K channels in chunks of AT_KC.  Between the frag echoes of this
        // ---- macro sits the whole trick: the next chunk's global loads are issued first.
#if (AT_KC == 64)
        AT_QK_STEP(kb0, kb1, kbase + 64u,
                    q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7)
        AT_QK_STEP(kb1, kb0, kbase + 128u,
                    q_8, q_9, q_10, q_11, q_12, q_13, q_14, q_15)
        AT_QK_STEP(kb0, kb1, kbase + 192u,
                    q_16, q_17, q_18, q_19, q_20, q_21, q_22, q_23)
        AT_QK_STEP(kb1, kb0, vbase,
                    q_24, q_25, q_26, q_27, q_28, q_29, q_30, q_31)
#else
        AT_QK_STEP(kb0, kb1, kbase + 32u, q_0, q_1, q_2, q_3)
        AT_QK_STEP(kb1, kb0, kbase + 64u, q_4, q_5, q_6, q_7)
        AT_QK_STEP(kb0, kb1, kbase + 96u, q_8, q_9, q_10, q_11)
        AT_QK_STEP(kb1, kb0, kbase + 128u, q_12, q_13, q_14, q_15)
        AT_QK_STEP(kb0, kb1, kbase + 160u, q_16, q_17, q_18, q_19)
        AT_QK_STEP(kb1, kb0, kbase + 192u, q_20, q_21, q_22, q_23)
        AT_QK_STEP(kb0, kb1, kbase + 224u, q_24, q_25, q_26, q_27)
        AT_QK_STEP(kb1, kb0, vbase, q_28, q_29, q_30, q_31)
#endif
        // ---- online softmax over this key block (fp32, per-lane row) + P^T staging
        {
            float sv0[8], sv1[8], mx = -INFINITY;
            AT_SM1(0, s_0)
            AT_SM1(1, s_1)
            AT_SM1(2, s_2)
            AT_SM1(3, s_3)
            AT_SM1(4, s_4)
            AT_SM1(5, s_5)
            AT_SM1(6, s_6)
            AT_SM1(7, s_7)
            mx    = max(mx, simd_shuffle_xor(mx, 1));
            mx    = max(mx, simd_shuffle_xor(mx, 8));
            const float m_new = max(m_run, mx);
            const float asc   = exp2(m_run - m_new);        // 0 on the first block
            m_run = m_new;
            float sum = 0.0f;
            AT_SM2(0)
            AT_SM2(1)
            AT_SM2(2)
            AT_SM2(3)
            AT_SM2(4)
            AT_SM2(5)
            AT_SM2(6)
            AT_SM2(7)
            sum   = sum + simd_shuffle_xor(sum, 1);
            sum   = sum + simd_shuffle_xor(sum, 8);
            l_run = l_run * asc + sum;
            // the two accumulator columns this lane owns belong to other lanes' rows
            const float scl0 = simd_shuffle(asc, ushort(own0));
            const float scl1 = simd_shuffle(asc, ushort(own1));
            {   // unconditional: on the first block asc = 0 and c is still zero, so the
                // `if (scl != 1.0f)` guard the old kernel carried was pure overhead
                // (measured 148.5 -> 132.9 ms at T=16384).
                AT_SCALE(c_0)
                AT_SCALE(c_1)
                AT_SCALE(c_2)
                AT_SCALE(c_3)
                AT_SCALE(c_4)
                AT_SCALE(c_5)
                AT_SCALE(c_6)
                AT_SCALE(c_7)
                AT_SCALE(c_8)
                AT_SCALE(c_9)
                AT_SCALE(c_10)
                AT_SCALE(c_11)
                AT_SCALE(c_12)
                AT_SCALE(c_13)
                AT_SCALE(c_14)
                AT_SCALE(c_15)
                AT_SCALE(c_16)
                AT_SCALE(c_17)
                AT_SCALE(c_18)
                AT_SCALE(c_19)
                AT_SCALE(c_20)
                AT_SCALE(c_21)
                AT_SCALE(c_22)
                AT_SCALE(c_23)
                AT_SCALE(c_24)
                AT_SCALE(c_25)
                AT_SCALE(c_26)
                AT_SCALE(c_27)
                AT_SCALE(c_28)
                AT_SCALE(c_29)
                AT_SCALE(c_30)
                AT_SCALE(c_31)
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // ---- PV, 256 V channels in chunks of AT_KC; the last step prefetches the next
        // ---- block's first K chunk so a block boundary costs no bubble.
        AT_PV_STEP(kb0, kb1, vbase + AT_KC, t0, AT_PV_MMA0)
        AT_PV_STEP(kb1, kb0, vbase + 2u * AT_KC, t0, AT_PV_MMA1)
#if (AT_KC == 32)
        AT_PV_STEP(kb0, kb1, vbase + 3u * AT_KC, t0, AT_PV_MMA2)
        AT_PV_STEP(kb1, kb0, vbase + 4u * AT_KC, t0, AT_PV_MMA3)
        AT_PV_STEP(kb0, kb1, vbase + 5u * AT_KC, t0, AT_PV_MMA4)
        AT_PV_STEP(kb1, kb0, vbase + 6u * AT_KC, t0, AT_PV_MMA5)
        AT_PV_STEP(kb0, kb1, vbase + 7u * AT_KC, t0, AT_PV_MMA6)
        AT_PV_STEP(kb1, kb0, kbase, t0 + AT_BN, AT_PV_MMA7)
#else
        AT_PV_STEP(kb0, kb1, vbase + 3u * AT_KC, t0, AT_PV_MMA2)
        AT_PV_STEP(kb1, kb0, kbase, t0 + AT_BN, AT_PV_MMA3)
#endif
    }

    // ---- epilogue: divide by l, multiply by the (already sigmoided) gate, fp16 out
    {
        const float iv  = 1.0f / l_run;
        const float iv0 = simd_shuffle(iv, ushort(own0));
        const float iv1 = simd_shuffle(iv, ushort(own1));
        const uint jc = qcol;
        if (jc < Stile) {
            const bool jlast = (jc + 1u >= Stile);
            AT_STORE(0, 0u + row_l)
            AT_STORE(1, 8u + row_l)
            AT_STORE(2, 16u + row_l)
            AT_STORE(3, 24u + row_l)
            AT_STORE(4, 32u + row_l)
            AT_STORE(5, 40u + row_l)
            AT_STORE(6, 48u + row_l)
            AT_STORE(7, 56u + row_l)
            AT_STORE(8, 64u + row_l)
            AT_STORE(9, 72u + row_l)
            AT_STORE(10, 80u + row_l)
            AT_STORE(11, 88u + row_l)
            AT_STORE(12, 96u + row_l)
            AT_STORE(13, 104u + row_l)
            AT_STORE(14, 112u + row_l)
            AT_STORE(15, 120u + row_l)
            AT_STORE(16, 128u + row_l)
            AT_STORE(17, 136u + row_l)
            AT_STORE(18, 144u + row_l)
            AT_STORE(19, 152u + row_l)
            AT_STORE(20, 160u + row_l)
            AT_STORE(21, 168u + row_l)
            AT_STORE(22, 176u + row_l)
            AT_STORE(23, 184u + row_l)
            AT_STORE(24, 192u + row_l)
            AT_STORE(25, 200u + row_l)
            AT_STORE(26, 208u + row_l)
            AT_STORE(27, 216u + row_l)
            AT_STORE(28, 224u + row_l)
            AT_STORE(29, 232u + row_l)
            AT_STORE(30, 240u + row_l)
            AT_STORE(31, 248u + row_l)
        }
    }
}

#undef AT_QK_LD
#undef AT_QK_LDN
#undef AT_QK_CP
#undef AT_QK_MMA
#undef AT_PV_LD
#undef AT_PV_LDN
#undef AT_PV_CP
#undef AT_PV_MMA0
#undef AT_PV_MMA1
#undef AT_PV_MMA2
#undef AT_PV_MMA3
#undef AT_PV_MMA4
#undef AT_PV_MMA5
#undef AT_PV_MMA6
#undef AT_PV_MMA7
#undef AT_QK_SUB
#undef AT_QK_STEP
#undef AT_PV_SUB
#undef AT_PV_STEP
#undef AT_SCALE
#undef AT_ZEROC
#undef AT_LDQ
#undef AT_STORE
#undef AT_SM1
#undef AT_SM2
