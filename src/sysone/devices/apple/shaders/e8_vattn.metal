// 视觉塔的非因果注意力：`k_vattn`。
//
// 与文本的 `k_attn` 不是同一个核：视觉是 head_dim 64（不是 `E8_HEAD_DIM` 256）、16 头对 16 头
// （没有 GQA 的 kv 头）、**非因果**、无 mask、无 gate、无环。q/k 的轴向 RoPE 已经在 host 上做完
// （`vision.py::rope_qk` 把 cos/sin 乘进 q、k 通道），所以这个核只做 softmax(QKᵀ)·V ——
// 这也正是可行性探针里对着 numpy 验到 `max|Δ| 6.1e-5` 的那个形态（票据 07 §7.6）。
//
// **P2.c 的形态是参考核**：一个线程组管 (一个 query, 一个 head)，通道主序，每个 query 都把整条
// K/V 重读一遍。探针在 L=1024 上量到 73 ms/block（旧引擎整段预算 3.20 ms 的 22.8×），所以它是
// **正确性锚点**，不是出货性能：P2.d 用 8×8 分片 + 128 query/线程组 + 键分块把它换掉，本核
// 的参数与输出布局保持不变。
//
// 键是**分块**的（`VA_CHUNK` 列一块，online softmax 重标定）：探针那份把全部 T 个分数放进
// 线程组内存（`sc[4096]` = 16 KB），1280² 的图就有 6400 个 token，那个写法直接越界。

#include "e8_common.h"

#define VA_CHUNK 1024u

// 与 Python 侧 `kernels.vattn_args` 的 `_VATTN` 格式串逐字节对应（24 B）。
struct VAttnArgs {
    uint  T;       // 0  每通道列数（qkv / out 的 stride）
    uint  nq;      // 1  这次产出的 query 列数
    uint  s0;      // 2  第一列 query 的偏移
    uint  hidden;  // 3  q、k、v 各自的通道数（= heads * dim）
    uint  dim;     // 4  head_dim（本核按 threadgroup 线程数 = dim 展开）
    float scale;   // 5  dim ** -0.5
};

inline float va_tg_max(float v, threadgroup float *red, uint tid, uint n) {
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = n >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] = max(red[tid], red[tid + off]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return red[0];
}

inline float va_tg_sum(float v, threadgroup float *red, uint tid, uint n) {
    red[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = n >> 1u; off > 0u; off >>= 1u) {
        if (tid < off) red[tid] = red[tid] + red[tid + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return red[0];
}

kernel void k_vattn(device const half  *qkv    [[buffer(0)]],
                    device half        *out    [[buffer(1)]],
                    constant VAttnArgs &a      [[buffer(E8_ARGS_INDEX)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint j = tgid.x, h = tgid.y, tid = tid2.x;
    const uint T = a.T, D = a.dim, HID = a.hidden;
    if (j >= a.nq || tid >= D) return;
    const uint s = a.s0 + j;

    // qkv 是通道主序：q 在 [0, HID)、k 在 [HID, 2·HID)、v 在 [2·HID, 3·HID)，头 h 占本段里
    // 的 [h·D, h·D + D) 行。out 只有 HID 行。
    const ulong qrow = (ulong)(h * D + tid) * T;
    const ulong kbase = (ulong)(HID + h * D) * T;
    const ulong vrow = (ulong)(2u * HID + h * D + tid) * T;

    threadgroup half  qs[64u];
    threadgroup float sc[VA_CHUNK];
    threadgroup float red[64u];

    qs[tid] = qkv[qrow + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY, l = 0.0f, y = 0.0f;
    for (uint t0 = 0; t0 < T; t0 += VA_CHUNK) {
        const uint cn = (T - t0 < VA_CHUNK) ? (T - t0) : VA_CHUNK;
        // 分数：本线程负责本块的键 u = tid, tid + D, ...
        for (uint u = tid; u < cn; u += D) {
            const ulong col = t0 + u;
            float acc = 0.0f;
            for (uint d = 0; d < D; d++) acc += float(qs[d]) * float(qkv[kbase + (ulong)d * T + col]);
            sc[u] = acc * a.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float cm = -INFINITY;
        for (uint u = tid; u < cn; u += D) cm = max(cm, sc[u]);
        cm = va_tg_max(cm, red, tid, D);
        const float mnew = max(m, cm);
        // 第一次进入时 m 是 -inf：`exp(-inf - mnew)` 是 0，正是"还没有历史"的意思。
        const float corr = (m == -INFINITY) ? 0.0f : exp(m - mnew);
        l *= corr;
        y *= corr;

        float part = 0.0f;
        for (uint u = tid; u < cn; u += D) {
            const float e = exp(sc[u] - mnew);
            sc[u] = e;
            part += e;
        }
        l += va_tg_sum(part, red, tid, D);
        m = mnew;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 加权和：每个线程只累加自己那一维，权重从线程组里的 `sc` 读（已归一化到 mnew）。
        for (uint u = 0; u < cn; u++) y += sc[u] * float(qkv[vrow + t0 + u]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    out[qrow + s] = (l > 0.0f) ? half(y / l) : half(0.0f);
}
