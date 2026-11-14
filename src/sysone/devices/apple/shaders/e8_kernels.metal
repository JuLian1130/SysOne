// E8b glue kernels + REFERENCE-quality cores.
//
// The glue kernels are what make the handover free: every one of them reads or writes an ANE
// IOSurface directly (the host wrapped it with newBufferWithBytesNoCopy, so the MTLBuffer
// *is* the IOSurface memory -- no staging, no host round trip).
//
// The two `*_ref` cores are deliberately naive.  They exist so the pipeline can be validated
// end to end and so the fast kernels in e8_attn.metal / e8_gdn.metal have an independent
// implementation to be checked against.  They are NOT the measured path: the reference
// attention needs an 8 x Stile x T fp32 score scratch, so it is only usable at small T.
//
// The prep kernels and the cores are a clean seam: `k_q_prep` / `k_kv_prep` do the
// per-head RMSNorm + partial rope + gate sigmoid in place (O(T) work), and both the fast and
// the reference core then consume exactly the same prepped buffers.
//
// Build: xcrun -sdk macosx metal -c e8_kernels.metal -I. -o /tmp/e8/k_glue.air

#include "e8_common.h"

// threadgroup reduction, one value per thread.  Two barriers (before write, after write) so
// it is safe to call once per loop iteration.
inline float e8_tg_sum(float v, threadgroup float *red, uint tid, uint nthreads) {
    v = simd_sum(v);
    uint nw = nthreads >> 5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint i = 0; i < nw; i++) t += red[i];
    return t;
}

inline float e8_tg_max(float v, threadgroup float *red, uint tid, uint nthreads) {
    v = simd_max(v);
    uint nw = nthreads >> 5;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 31u) == 0u) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = red[0];
    for (uint i = 1; i < nw; i++) t = max(t, red[i]);
    return t;
}

// ---------------------------------------------------------------------------------------
// k_norm_to: RMSNorm one column of the layer input into the proj input surface.
//   src io_b (HID, src_stride) column s0+j  ->  dst (HID, dst_stride) column dst_s0+j
// `dst_stride`/`dst_s0` are 0 for the old per-request layout, and the program's packed stride
// plus this request's slot offset for the short pack (one surface for all requests).
//
// One THREAD per column, `ntg` columns per threadgroup.  The reduction axis (channels) is the
// strided one, so the thread mapping must be transposed w.r.t. it: 32 consecutive lanes then
// read 32 consecutive columns of ONE channel = 64 contiguous bytes per instruction.  The old
// one-threadgroup-per-column mapping gave every lane its own 32-byte sector (2/32 bytes useful)
// and measured 4.3 GB/s = 390 ms per 16k call.  No reduction is needed now: a thread owns the
// whole column, so the sum over 2048 channels is a private accumulator.
// Two passes over the source (sum, then apply) because a column does not fit in registers.
// ---------------------------------------------------------------------------------------
kernel void k_norm_to(device const half *src   [[buffer(0)]],
                      device const float *w    [[buffer(1)]],
                      device half       *dst   [[buffer(2)]],
                      constant NormArgs &a     [[buffer(E8_ARGS_INDEX)]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint tid  [[thread_position_in_threadgroup]],
                      uint ntg  [[threads_per_threadgroup]]) {
    const uint j = tgid * ntg + tid;
    if (j >= a.Stile) return;
    const ulong s = a.s0 + j, stride = a.src_stride;
    const ulong dstride = a.dst_stride ? a.dst_stride : a.Stile;
    const ulong dbase = (ulong)a.dst_s0 + j;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    ulong p = s;
    for (uint c = 0; c < E8_HID; c += 8u) {          // 8 in flight: memory-level parallelism
        const float v0 = float(src[p]);
        const float v1 = float(src[p + stride]);
        const float v2 = float(src[p + 2u * stride]);
        const float v3 = float(src[p + 3u * stride]);
        const float v4 = float(src[p + 4u * stride]);
        const float v5 = float(src[p + 5u * stride]);
        const float v6 = float(src[p + 6u * stride]);
        const float v7 = float(src[p + 7u * stride]);
        acc0 += v0 * v0; acc1 += v1 * v1; acc2 += v2 * v2; acc3 += v3 * v3;
        acc4 += v4 * v4; acc5 += v5 * v5; acc6 += v6 * v6; acc7 += v7 * v7;
        p += 8u * stride;
    }
    const float scale = e8_rms_scale(((acc0 + acc1) + (acc2 + acc3)) +
                                     ((acc4 + acc5) + (acc6 + acc7)), E8_HID, a.eps);
    p = s;
    for (uint c = 0; c < E8_HID; c += 8u) {
        const half r0 = src[p];
        const half r1 = src[p + stride];
        const half r2 = src[p + 2u * stride];
        const half r3 = src[p + 3u * stride];
        const half r4 = src[p + 4u * stride];
        const half r5 = src[p + 5u * stride];
        const half r6 = src[p + 6u * stride];
        const half r7 = src[p + 7u * stride];
        const ulong d = (ulong)c * dstride + dbase;
        dst[d]              = half(float(r0) * scale * w[c]);
        dst[d + dstride]    = half(float(r1) * scale * w[c + 1u]);
        dst[d + 2u * dstride] = half(float(r2) * scale * w[c + 2u]);
        dst[d + 3u * dstride] = half(float(r3) * scale * w[c + 3u]);
        dst[d + 4u * dstride] = half(float(r4) * scale * w[c + 4u]);
        dst[d + 5u * dstride] = half(float(r5) * scale * w[c + 5u]);
        dst[d + 6u * dstride] = half(float(r6) * scale * w[c + 6u]);
        dst[d + 7u * dstride] = half(float(r7) * scale * w[c + 7u]);
        p += 8u * stride;
    }
}

// ---------------------------------------------------------------------------------------
// k_norm_resid: the ffn half of a layer.  Normalize y *and* hand the raw y to the `ffn`
// procedure as its residual, so y never needs a second copy.
//   src io_b (HID, src_stride) -> dst[0:HID] = rmsnorm(y), dst[HID:2HID] = y
// `dst_stride`/`dst_s0` are the packed-surface form of the same write (short pack): the source
// is the `out` procedure's packed io_b and the destination is the packed io_a for `ffn`.
// Same one-thread-per-column mapping as k_norm_to (see there for why), and the second pass
// reuses the value it already loaded for the residual instead of reading the source a 3rd time.
// ---------------------------------------------------------------------------------------
kernel void k_norm_resid(device const half *src   [[buffer(0)]],
                         device const float *w    [[buffer(1)]],
                         device half       *dst   [[buffer(2)]],
                         constant NormArgs &a     [[buffer(E8_ARGS_INDEX)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint tid  [[thread_position_in_threadgroup]],
                         uint ntg  [[threads_per_threadgroup]]) {
    const uint j = tgid * ntg + tid;
    if (j >= a.Stile) return;
    const ulong s = a.s0 + j, stride = a.src_stride;
    const ulong dstride = a.dst_stride ? a.dst_stride : a.Stile;
    const ulong dbase = (ulong)a.dst_s0 + j;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    ulong p = s;
    for (uint c = 0; c < E8_HID; c += 8u) {
        const float v0 = float(src[p]);
        const float v1 = float(src[p + stride]);
        const float v2 = float(src[p + 2u * stride]);
        const float v3 = float(src[p + 3u * stride]);
        const float v4 = float(src[p + 4u * stride]);
        const float v5 = float(src[p + 5u * stride]);
        const float v6 = float(src[p + 6u * stride]);
        const float v7 = float(src[p + 7u * stride]);
        acc0 += v0 * v0; acc1 += v1 * v1; acc2 += v2 * v2; acc3 += v3 * v3;
        acc4 += v4 * v4; acc5 += v5 * v5; acc6 += v6 * v6; acc7 += v7 * v7;
        p += 8u * stride;
    }
    const float scale = e8_rms_scale(((acc0 + acc1) + (acc2 + acc3)) +
                                     ((acc4 + acc5) + (acc6 + acc7)), E8_HID, a.eps);
    p = s;
    for (uint c = 0; c < E8_HID; c += 8u) {
        const half r0 = src[p];
        const half r1 = src[p + stride];
        const half r2 = src[p + 2u * stride];
        const half r3 = src[p + 3u * stride];
        const half r4 = src[p + 4u * stride];
        const half r5 = src[p + 5u * stride];
        const half r6 = src[p + 6u * stride];
        const half r7 = src[p + 7u * stride];
        const ulong d = (ulong)c * dstride + dbase;
        const ulong dr = d + (ulong)E8_HID * dstride;
        dst[d]                 = half(float(r0) * scale * w[c]);
        dst[d + dstride]       = half(float(r1) * scale * w[c + 1u]);
        dst[d + 2u * dstride]  = half(float(r2) * scale * w[c + 2u]);
        dst[d + 3u * dstride]  = half(float(r3) * scale * w[c + 3u]);
        dst[d + 4u * dstride]  = half(float(r4) * scale * w[c + 4u]);
        dst[d + 5u * dstride]  = half(float(r5) * scale * w[c + 5u]);
        dst[d + 6u * dstride]  = half(float(r6) * scale * w[c + 6u]);
        dst[d + 7u * dstride]  = half(float(r7) * scale * w[c + 7u]);
        dst[dr]                = r0;
        dst[dr + dstride]      = r1;
        dst[dr + 2u * dstride] = r2;
        dst[dr + 3u * dstride] = r3;
        dst[dr + 4u * dstride] = r4;
        dst[dr + 5u * dstride] = r5;
        dst[dr + 6u * dstride] = r6;
        dst[dr + 7u * dstride] = r7;
        p += 8u * stride;
    }
}

// ---------------------------------------------------------------------------------------
// k_copy: strided channel-major block copy.  Two uses:
//   (a) out_proj -> qg / kv  (the gather that keeps the whole-sequence K/V that the
//       noncausal full-attention core needs)
//   (b) io_b -> io_a[HID:]   (residual the `out` procedure adds)
// TG = 64 columns x 8 channels so x is coalesced along the contiguous axis.
// ---------------------------------------------------------------------------------------
kernel void k_copy(device const half *src  [[buffer(0)]],
                   device half       *dst  [[buffer(1)]],
                   constant CopyArgs &a    [[buffer(E8_ARGS_INDEX)]],
                   uint2 tgid [[threadgroup_position_in_grid]],
                   uint2 tid  [[thread_position_in_threadgroup]]) {
    uint j = tgid.x * 64u + tid.x;
    uint c = tgid.y * 8u  + tid.y;
    if (j >= a.Stile || c >= a.nchan) return;
    dst[(ulong)(a.dst_c0 + c) * a.dst_stride + a.s0 + j] =
        src[(ulong)(a.src_c0 + c) * a.src_stride + a.src_s0 + j];
}

// ---------------------------------------------------------------------------------------
// k_q_prep: in place on qg.  For query head h and query position s:
//   qg[h*512 + d]     = rmsnorm(q, q_norm) then partial rope   (d < 256)
//   qg[h*512 + 256+d] = sigmoid(gate)                          (d < 256)
// One threadgroup per (query, head), thread d handles channel d.
// ---------------------------------------------------------------------------------------
kernel void k_q_prep(device half        *qg     [[buffer(0)]],
                     device const float *q_norm [[buffer(1)]],
                     device const float *cosT   [[buffer(2)]],
                     device const float *sinT   [[buffer(3)]],
                     constant NormArgs  &a      [[buffer(E8_ARGS_INDEX)]],
                     uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint j = tgid.x, h = tgid.y, s = a.s0 + j, D = E8_HEAD_DIM, tid = tid2.x;
    // S6: the rope position of this column.  Outside a batched tail it *is* the column; inside
    // one, the slots' rows are the same requests' rows at their own positions (e8_fold).
    // A1: `qg` may be a K-slot ring, so the column a query lives at is not its position --
    // `pos0` is how far the ring slot base sits behind the tile's first token.
    const uint p = a.pos0 + e8_fold(a.nka, a.segw, s);
    threadgroup float qn[E8_HEAD_DIM];
    threadgroup float red[8];
    device half *base = qg + (ulong)(h * 2u * D) * a.src_stride;

    float q = float(base[(ulong)tid * a.src_stride + s]);
    float acc = e8_tg_sum(q * q, red, tid, D);
    qn[tid] = q * e8_rms_scale(acc, int(D), a.eps) * q_norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float lo = qn[tid], hi = qn[tid + 32u];
        e8_rot(lo, hi, cosT, sinT, p, tid);
        qn[tid] = lo; qn[tid + 32u] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    base[(ulong)tid * a.src_stride + s] = half(qn[tid]);
    base[(ulong)(D + tid) * a.src_stride + s] =
        half(1.0f / (1.0f + exp(-float(base[(ulong)(D + tid) * a.src_stride + s]))));
}

// ---------------------------------------------------------------------------------------
// k_kv_prep: in place on the k half of kv (channels 0:512).  v (channels 512:1024) is left
// untouched because it needs no normalization or rotation.
// Grid = (tokens, 2 kv heads), thread d handles channel d.
// ---------------------------------------------------------------------------------------
kernel void k_kv_prep(device half        *kv     [[buffer(0)]],
                      device const float *k_norm [[buffer(1)]],
                      device const float *cosT   [[buffer(2)]],
                      device const float *sinT   [[buffer(3)]],
                      constant NormArgs  &a      [[buffer(E8_ARGS_INDEX)]],
                      uint2 tgid [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint t = a.s0 + tgid.x, j = tgid.y, D = E8_HEAD_DIM, tid = tid2.x;
    // S6: rope position, not the slot column.  A1：kv 是整窗缓存，`pos0` 恒为 0（留在这里只是
    // 让两条 prep 核的写法一致）。
    const uint p = a.pos0 + e8_fold(a.nka, a.segw, t);
    threadgroup float kn[E8_HEAD_DIM];
    threadgroup float red[8];
    device half *base = kv + (ulong)(j * D) * a.src_stride;

    float k = float(base[(ulong)tid * a.src_stride + t]);
    float acc = e8_tg_sum(k * k, red, tid, D);
    kn[tid] = k * e8_rms_scale(acc, int(D), a.eps) * k_norm[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) {
        float lo = kn[tid], hi = kn[tid + 32u];
        e8_rot(lo, hi, cosT, sinT, p, tid);
        kn[tid] = lo; kn[tid + 32u] = hi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    base[(ulong)tid * a.src_stride + t] = half(kn[tid]);
}

// ---------------------------------------------------------------------------------------
// k_attn_ref: noncausal full-attention core, reference implementation.
//   b0 qg prepped (4096, T)   b1 kv prepped (1024, T)   b2 out (2*HID, Stile)
//   b3 scratch (8*Stile*T) float   -- only the reference needs this
// One threadgroup per (query, head); thread d owns channel d of that head.
// ---------------------------------------------------------------------------------------
kernel void k_attn_ref(device const half  *qg      [[buffer(0)]],
                       device const half  *kv      [[buffer(1)]],
                       device half        *out     [[buffer(2)]],
                       device float       *scr     [[buffer(3)]],
                       constant AttnArgs  &a       [[buffer(E8_ARGS_INDEX)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint j = tgid.x, h = tgid.y, s = a.s0 + j, D = E8_HEAD_DIM, T = a.T, tid = tid2.x;
    const uint nk = min(a.nk, a.T);   // keys that exist; the rest of T is zero padding
    const uint kvh = h / (E8_Q_HEADS / E8_KV_HEADS);
    threadgroup float qn[E8_HEAD_DIM];
    threadgroup float red[8];
    device float *sc = scr + (ulong)(h * a.Stile + j) * T;

    // q channels of head h live at [h*2*D, h*2*D+D); the gate is the next D channels
    qn[tid] = float(qg[(ulong)(h * 2u * D + tid) * T + s]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const half *kb = kv + (ulong)(kvh * D) * T;
    device const half *vb = kv + (ulong)(E8_KV_PROJ + kvh * D) * T;

    // scores: this thread owns keys t = tid, tid+256, ...
    for (uint t = tid; t < nk; t += 256u) {
        float acc = 0.0f;
        for (uint d = 0; d < D; d++) acc += qn[d] * float(kb[(ulong)d * T + t]);
        sc[t] = acc * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // softmax over the whole key axis
    float m = -INFINITY;
    for (uint t = tid; t < nk; t += 256u) m = max(m, sc[t]);
    m = e8_tg_max(m, red, tid, 256u);

    float ssum = 0.0f;
    for (uint t = tid; t < nk; t += 256u) { sc[t] = exp(sc[t] - m); ssum += sc[t]; }
    ssum = e8_tg_sum(ssum, red, tid, 256u);
    float inv = 1.0f / ssum;
    for (uint t = tid; t < nk; t += 256u) sc[t] *= inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float y = 0.0f;
    for (uint t = 0; t < nk; t++) y += sc[t] * float(vb[(ulong)tid * T + t]);
    float g = float(qg[(ulong)(h * 2u * D + D + tid) * T + s]);
    const uint osp = a.out_stride ? a.out_stride : a.Stile;
    out[(ulong)(h * D + tid) * osp + a.out_s0 + j] = half(y * g);
}

// ---------------------------------------------------------------------------------------
// k_gdn_ref: GDN core, reference implementation (all steps inline, one threadgroup per head,
// thread = dv, state in device memory).  Correct anywhere, fast nowhere.
//   b0 proj (8224, Stile)  b1 state (16*128*128 float)  b2 tail (6144*3 half)
//   b3 w_conv (6144*4 half) b4 A_log (16 float) b5 dt_bias (16 float) b6 norm (128 half)
//   b7 out (2*HID, Stile)  b8 scratch: qn/kn/vn (3*2048*Stile half)? -> not needed, inline
// ---------------------------------------------------------------------------------------
inline half e8_gdn_raw(device const half *proj, device const half *tail, constant GdnArgs &a,
                       uint c, int t) {
    if (t >= (int)a.s0) return proj[(ulong)c * a.Stile + (t - (int)a.s0)];
    if (a.first) return half(0.0f);
    int idx = t - (int)a.s0 + 3;       // -3..-1 -> 0..2
    return tail[(ulong)c * 3 + idx];
}

kernel void k_gdn_ref(device const half  *proj   [[buffer(0)]],
                      device float       *state  [[buffer(1)]],
                      device half        *tail   [[buffer(2)]],
                      device const half  *wconv  [[buffer(3)]],
                      device const float *A_log  [[buffer(4)]],
                      device const float *dtb    [[buffer(5)]],
                      device const half  *norm   [[buffer(6)]],
                      device half        *out    [[buffer(7)]],
                      constant GdnArgs   &a      [[buffer(E8_ARGS_INDEX)]],
                      uint  h   [[threadgroup_position_in_grid]],
                      uint  dv  [[thread_position_in_threadgroup]],
                      uint  ndv [[threads_per_threadgroup]]) {
    const uint D = E8_L_KD;            // == E8_L_VD
    if (ndv != D) return;
    const int s0 = int(a.s0);
    device float *st = state + (ulong)h * D * D + (ulong)dv * D;

    const float inv = rsqrt(float(D));
    const float rms_eps = a.eps * inv * inv;
    threadgroup float red[8];      // the 128 dv of one head ARE the threadgroup, so the
                                   // gated RMSNorm's mean over dv is a threadgroup reduction

    const uint ncol = a.cols ? a.cols : a.Stile;
    for (uint j = 0; j < ncol; j++) {
        int t = s0 + int(j);
        // ---- conv1d(K=4) + SiLU for this head's q, k, v rows
        float qv[E8_L_KD], kv[E8_L_KD], vv = 0.0f;
        float sumq = 0.0f, sumk = 0.0f;
        for (uint d = 0; d < D; d++) {
            uint cq = h * D + d;                       // q channel
            uint ck = E8_HID + cq;                     // k channel
            float pq = 0.0f, pk = 0.0f;
            for (uint k = 0; k < E8_CONV_K; k++) {
                pq += float(e8_gdn_raw(proj, tail, a, cq, t - 3 + (int)k)) * float(wconv[(ulong)cq * E8_CONV_K + k]);
                pk += float(e8_gdn_raw(proj, tail, a, ck, t - 3 + (int)k)) * float(wconv[(ulong)ck * E8_CONV_K + k]);
            }
            half hq = half(pq), hk = half(pk);
            float cqv = float(hq) / (1.0f + exp(-float(hq)));
            float ckv = float(hk) / (1.0f + exp(-float(hk)));
            qv[d] = cqv; kv[d] = ckv;
            sumq += cqv * cqv; sumk += ckv * ckv;
        }
        {
            uint cv = 2u * E8_HID + h * D + dv;        // v channel
            float pv = 0.0f;
            for (uint k = 0; k < E8_CONV_K; k++)
                pv += float(e8_gdn_raw(proj, tail, a, cv, t - 3 + (int)k)) * float(wconv[(ulong)cv * E8_CONV_K + k]);
            half hv = half(pv);
            vv = float(hv) / (1.0f + exp(-float(hv)));
        }
        // ---- L2 norm exactly as mlx-lm normalize_qk
        float sq = inv * inv * rsqrt(sumq / float(D) + rms_eps);
        float sk = inv * rsqrt(sumk / float(D) + rms_eps);
        // ---- gate / beta
        // a / b are ONE CHANNEL PER LINEAR HEAD (16 channels each, not 16 wide shared)
        float av = float(proj[(ulong)(E8_COUT_GDN - 16u + h) * a.Stile + j]);
        float bv = float(proj[(ulong)(E8_COUT_GDN - 32u + h) * a.Stile + j]);
        float g   = e8_gdn_gate(av, A_log[h], dtb[h]);
        float beta = 1.0f / (1.0f + exp(-bv));

        // ---- delta rule for this (head, dv) row
        //   state = state * g      (decay -- easy to drop and it costs a 10x error, not a
        //   subtle one: without it the state grows monotonically instead of tracking)
        for (uint d = 0; d < D; d++) st[d] *= g;
        float kv_mem = 0.0f;
        for (uint d = 0; d < D; d++) {
            float kn = kv[d] * sk;
            kv_mem += st[d] * kn;
        }
        float delta = (vv - kv_mem) * beta;
        float y = 0.0f;
        for (uint d = 0; d < D; d++) {
            float kn = kv[d] * sk;
            st[d] += kn * delta;
            y += st[d] * (qv[d] * sq);
        }
        // ---- gated RMSNorm (y is fp16-rounded first, exactly like the model path).
        // The mean is over the head's Dv values, which live one per thread here, so it has
        // to be reduced across the threadgroup -- using the thread's own yh is wrong (and
        // it is a ~10% error, not a rounding one).
        float yh = float(half(y));
        float n = rsqrt(e8_tg_sum(yh * yh, red, dv, ndv) / float(D) + a.eps);
        float z = float(proj[(ulong)(E8_FFN + h * D + dv) * a.Stile + j]);
        float o = yh * n * (z / (1.0f + exp(-z))) * float(norm[dv]);
        const uint osp = a.out_stride ? a.out_stride : a.Stile;
        out[(ulong)(h * D + dv) * osp + a.out_s0 + j] = half(o);
    }

    // ---- carry the last 3 raw qkv columns to the next tile
    if (ncol >= 3u) {
        for (uint k = 0; k < 3u; k++) {
            int t = s0 + int(ncol) - 3 + int(k);
            if (t < 0) continue;
            uint cq = h * D + dv;
            tail[(ulong)cq * 3 + k]              = e8_gdn_raw(proj, tail, a, cq, t);
            tail[(ulong)(E8_HID + cq) * 3 + k]   = e8_gdn_raw(proj, tail, a, E8_HID + cq, t);
            tail[(ulong)(2u * E8_HID + cq) * 3 + k] =
                e8_gdn_raw(proj, tail, a, 2u * E8_HID + cq, t);
        }
    }
}
