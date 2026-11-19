// 视觉塔的非因果注意力：`k_vattn`。
//
// 与文本的 `k_attn` 不是同一个核：视觉是 head_dim 64（不是 `E8_HEAD_DIM` 256）、16 头对 16 头
// （没有 GQA 的 kv 头）、**非因果**、无 mask、无 gate、无环。q/k 的轴向 RoPE 由同一文件里的
// `k_vrope` 先做完（原地作用在通道主序的 `[q | k | v]` 上），所以这个核只做 softmax(QKᵀ)·V。
//
// ---------------------------------------------------------------------------------------
// 形态（P2.d）：一个线程组管 **64 列 query × 1 个头**，键按 32 一档
// ---------------------------------------------------------------------------------------
//   * 8 个 simdgroup，每个 8 列 query：Q 与累加器 O 各 8 个分片，常驻寄存器。
//   * K/V 暂存进线程组内存，布局 `[通道][键]` —— 通道主序的全局缓冲里一行的键是连续的，所以
//     搬运就是直拷贝，Kᵀ/Vᵀ 都不需要转置（与 `k_attn` 的 `at_stage` 同一套约定）。
//   * 线程组内存 14 KB（K/V 各 5 KB、P^T 每 warp 0.5 KB）：一个核放得下两个线程组 = 16 个
//     warp，与 `k_attn`（512 线程 / 25 KB）同档的占用率。这不是美学：`e8_attn.metal` 顶部记着
//     "一个 256 线程的线程组只到 ~2.2 TFLOPS"，128 线程（4 warp）每核更站不满。
//     键档取 32 而不是 64，也是这个理由：64 键要把 P^T 抬到 8 KB、线程组内存 20 KB，就再也
//     回不到"一个核两个线程组"。
//   * 每档两次 `threadgroup_barrier`：一次在 K→QK 之前，一次在 P^T+V→PV 之前。
//
// 数值：在线 softmax 全程 fp32（m/l 是**按行**的），累加器每档按 `exp2(m - m_new)` 重标定；
// P 落一半字进线程组内存再当 A 操作数读回 —— 与 P2.c 的参考核同一条数值路径，只是参考核
// 每列 query 都把整条 K/V 重读一遍（L=1024 上 73 ms/block），这里每档只读一次、被 64 列摊开。
//
// 分片几何（量过，见 `.scratch/probe_layout.py` 与票据 07 §7.6c）：lane l 持有
// `(row, col)` 与 `(row, col+1)`，且一个 lane 恰好一行、一行的四个持有者只差 lane 的第 0/3 位。
// 所以按行归约是 `simd_shuffle_xor(1)` 加 `(8)`，按行取因子是 `simd_shuffle(..., owner_of(col))`。
//
// 记一条两个下标容易混的地方：Q/S 分片的**行**是 query（列是通道/键），而累加器 C 的**行**是
// 通道、**列**才是 query。把 Q 的取列也写成 `col_l` 不会报错，只会让每个 lane 拿到别人的 Q
// （实测 rel 0.5–0.8，且 max|Δ| 随 T 增大反而下降——错误的平均被摊薄了）。

#include "e8_common.h"
#include <metal_simdgroup_matrix>

#define VA_TG    256u                       // 线程数 = 8 个 simdgroup
#define VA_NW    (VA_TG / 32u)
#define VA_BQ    64u                        // 每线程组的 query 列数
#define VA_BK    32u                        // 每档的键数
#define VA_LDK   40u                        // 暂存的行跨距（32 键 + 8 半字填充）
#define VA_LDP    8u                        // P^T 的行跨距（8 列 query = 一个 16 字节组）
#define VA_LOG2E 1.4426950408889634f

// 与 Python 侧 `kernels.vattn_args` 的 `_VATTN` 格式串逐字节对应（24 B）。
struct VAttnArgs {
    uint  T;       // 0  qkv 每通道的列数（整图长度，也是键的总数）
    uint  nq;      // 1  这次产出的 query 列数
    uint  s0;      // 2  第一列 query 在 qkv 里的偏移
    uint  hidden;  // 3  q、k、v 各自的通道数（= heads * dim）
    uint  dim;     // 4  head_dim（本核写死 64，见 `kernels.vattn` 的守卫）
    float scale;   // 5  dim ** -0.5
    uint  oT;      // 6  out 每通道的列数（输出面自己的行跨距）
    uint  os0;     // 7  这一片 query 在 out 里的首列
};

// 8×8 分片的元素归属（行主序读入 / 写出时 lane 与格的对应）。
static inline uint va_row_of(uint l)   { return ((l >> 1) & 3u) | (((l >> 4) & 1u) << 2); }
static inline uint va_col_of(uint l)   { return ((l & 1u) << 1) | (((l >> 3) & 1u) << 2); }
// 行 m 的四个持有者里最小的那个 lane（另三个是 +1、+8、+9）
static inline uint va_owner_of(uint m) { return ((m & 3u) << 1) | ((m >> 2) << 4); }

// 键（或值）的一档搬进线程组内存：每线程一个 `[1 通道][8 键]` 的单位，VA_TG 个线程恰好
// 铺满 64 通道 × 4 组。块尾不足 8 个键的那组要逐半字补零 —— `T` 不保证是 8 的倍数
// （640×480 的图有 300 个 token），而 16 字节的向量搬运还额外要求行首对齐，也就是 `T % 8 == 0`。
static inline void va_stage(threadgroup half *buf, device const half *src, uint tid,
                            uint chan0, uint T, uint t0, uint vec) {
    const uint c = tid >> 2, j = tid & 3u;
    const uint col = t0 + j * 8u;
    threadgroup half *dst = buf + c * VA_LDK + j * 8u;
    const device half *p = src + (ulong)(chan0 + c) * T + col;
    if (vec && col + 8u <= T) {
        *(threadgroup uint4 *)dst = *(device const uint4 *)p;
    } else {
        for (uint i = 0u; i < 8u; i++) dst[i] = (col + i < T) ? p[i] : half(0);
    }
}

// Q：本 lane 的两个通道（col_l、col_l+1），落在本次调用的第 qcol 列 query 上。
#define VA_LDQ(Q, DG) { \
    device const half *qp = qkv + (ulong)(qbase + (DG) + col_l) * T + qrow; \
    Q.thread_elements()[0] = qp[0]; Q.thread_elements()[1] = qp[T]; }

// QKᵀ 的一档通道：b_0..b_3 是 4 个键瓦片（8 键一个），(Q) 是这一档的 Q 分片。
#define VA_QK(DG, Q) { \
    simdgroup_load(b_0, ktg + (DG) * VA_LDK +  0u, VA_LDK); \
    simdgroup_load(b_1, ktg + (DG) * VA_LDK +  8u, VA_LDK); \
    simdgroup_load(b_2, ktg + (DG) * VA_LDK + 16u, VA_LDK); \
    simdgroup_load(b_3, ktg + (DG) * VA_LDK + 24u, VA_LDK); \
    simdgroup_multiply_accumulate(s_0, (Q), b_0, s_0); \
    simdgroup_multiply_accumulate(s_1, (Q), b_1, s_1); \
    simdgroup_multiply_accumulate(s_2, (Q), b_2, s_2); \
    simdgroup_multiply_accumulate(s_3, (Q), b_3, s_3); }

// PV 的一档通道：累加器 (C) 对应 8 个通道 × 8 列 query，4 个键瓦片各自一遍。
#define VA_PV(DG, C) { \
    simdgroup_load(v_0, vtg + (DG) * VA_LDK +  0u, VA_LDK); \
    simdgroup_load(v_1, vtg + (DG) * VA_LDK +  8u, VA_LDK); \
    simdgroup_load(v_2, vtg + (DG) * VA_LDK + 16u, VA_LDK); \
    simdgroup_load(v_3, vtg + (DG) * VA_LDK + 24u, VA_LDK); \
    simdgroup_multiply_accumulate((C), v_0, pbs_0, (C)); \
    simdgroup_multiply_accumulate((C), v_1, pbs_1, (C)); \
    simdgroup_multiply_accumulate((C), v_2, pbs_2, (C)); \
    simdgroup_multiply_accumulate((C), v_3, pbs_3, (C)); }

#define VA_ZEROC(C) { C.thread_elements()[0] = 0.0f; C.thread_elements()[1] = 0.0f; }
#define VA_SCALE(C) { C.thread_elements()[0] *= scl0; C.thread_elements()[1] *= scl1; }

// 本档的分数：乘上 scale·log2e，键越界的置 -inf（尾档才有，`T % 32 != 0` 时）
#define VA_SM1(K, S) { const uint tA = t0 + (K) * 8u + col_l; \
    float v0 = (S).thread_elements()[0] * K1, v1 = (S).thread_elements()[1] * K1; \
    if (tA >= T)      v0 = -INFINITY; \
    if (tA + 1u >= T) v1 = -INFINITY; \
    sv0[K] = v0; sv1[K] = v1; mx = max(mx, max(v0, v1)); }

// 指数与 P^T 落一半字：P^T 是每 warp 私有的 `[键][query]`，只有本 warp 写、本 warp 读
#define VA_SM2(K) { const float p0 = exp2(sv0[K] - m_new), p1 = exp2(sv1[K] - m_new); \
    sum += p0 + p1; \
    const uint tl = (K) * 8u + col_l; \
    ptg[sg][tl][row_l] = half(p0); ptg[sg][tl + 1u][row_l] = half(p1); }

// 收尾：fp32 累加器 ÷ l，直接写回通道主序的 fp16 out。逐半字写（不用 half2）：`T` 与 `s0`
// 的奇偶决定了同一行相邻两列未必 4 字节对齐，而这一段每线程只跑 8 次。
// 第二列要再判一次 `jc + 1 < nq`：本 lane 的 `[1]` 落在下一列 query 上，而那一列可能已经
// 出了这次调用的范围（`nq` 是 8 的倍数时没有，否则就有）—— 漏了这一判，`T` 是奇数时最后一列
// 会写到 `T` 号列，也就是**下一个通道行的第 0 列**，把第 0 列 query 的输出踩掉（实测 T=25/33）。
#define VA_STORE(EG, E) { \
    device half *op = out + (ulong)(qbase + (E)) * a.oT + a.os0 + jc; \
    op[0] = half(c_##EG.thread_elements()[0] * iv0); \
    if (jc + 1u < a.nq) op[1] = half(c_##EG.thread_elements()[1] * iv1); }

kernel void k_vattn(device const half  *qkv    [[buffer(0)]],
                    device half        *out    [[buffer(1)]],
                    constant VAttnArgs &a      [[buffer(E8_ARGS_INDEX)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tid2 [[thread_position_in_threadgroup]]) {
    const uint T = a.T, D = a.dim, HID = a.hidden;
    const uint j0 = tgid.x * VA_BQ, h = tgid.y, tid = tid2.x;
    if (h >= HID / D || j0 >= a.nq) return;

    // ---- lane 几何：本 warp 管 8 列 query，本 lane 管其中第 row_l 行、第 col_l 与 col_l+1 列
    const uint sg = tid >> 5, lane = tid & 31u;
    const uint row_l = va_row_of(lane), col_l = va_col_of(lane);
    const uint own0 = va_owner_of(col_l), own1 = va_owner_of(col_l + 1u);
    const uint myq = j0 + sg * 8u;                  // 本 warp 的第一列 query（本次调用内）
    // 分片的两个下标分工不同，这里最容易写错：Q/S 的**行**才是 query（所以 Q 用 row_l 取列，
    // 累加器的行是通道）；累加器 C 的**列**才是 query（所以 C 用 col_l 取列）。
    const uint qcol = myq + col_l;                  // 本 lane 在累加器里负责的 query 列
    // 尾部瓦片里超出 nq 的列只用来占位：夹到最后一列，避免读到缓冲末尾之外
    const uint qrow = min(a.s0 + myq + row_l, T - 1u);
    const uint qbase = h * D;                       // q 段里本头的首通道
    const float K1 = a.scale * VA_LOG2E;            // 分数 → 底 2 的指数

    // ---- 线程组内存：K、V 各 5 KB（[通道][键]），P^T 每 warp 0.5 KB
    threadgroup half ktg[64u * VA_LDK];
    threadgroup half vtg[64u * VA_LDK];
    threadgroup half ptg[VA_NW][VA_BK][VA_LDP];

    // 分片逐个命名，绝不用数组：这个编译器不会把分片数组拆成寄存器（`e8_attn.metal` 顶部）。
    simdgroup_half8x8  q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7;
    simdgroup_half8x8  b_0, b_1, b_2, b_3;
    simdgroup_half8x8  v_0, v_1, v_2, v_3;
    simdgroup_half8x8  pbs_0, pbs_1, pbs_2, pbs_3;
    simdgroup_float8x8 s_0, s_1, s_2, s_3;
    simdgroup_float8x8 c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7;
    float m_run = -INFINITY, l_run = 0.0f;

    VA_LDQ(q_0, 0u)
    VA_LDQ(q_1, 8u)
    VA_LDQ(q_2, 16u)
    VA_LDQ(q_3, 24u)
    VA_LDQ(q_4, 32u)
    VA_LDQ(q_5, 40u)
    VA_LDQ(q_6, 48u)
    VA_LDQ(q_7, 56u)
    VA_ZEROC(c_0)
    VA_ZEROC(c_1)
    VA_ZEROC(c_2)
    VA_ZEROC(c_3)
    VA_ZEROC(c_4)
    VA_ZEROC(c_5)
    VA_ZEROC(c_6)
    VA_ZEROC(c_7)

    const uint blocks = (T + VA_BK - 1u) / VA_BK;
    const uint vec = ((T & 7u) == 0u) ? 1u : 0u;     // 16 字节搬运要求的对齐
    const uint kchan = HID + qbase;                 // k 段里本头的首通道
    const uint vchan = 2u * HID + qbase;            // v 段
    threadgroup half *const ktg0 = ktg;
    threadgroup half *const vtg0 = vtg;

    va_stage(ktg0, qkv, tid, kchan, T, 0u, vec);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint blk = 0u; blk < blocks; blk++) {
        const uint t0 = blk * VA_BK;
        VA_ZEROC(s_0)
        VA_ZEROC(s_1)
        VA_ZEROC(s_2)
        VA_ZEROC(s_3)
        // ---- QKᵀ：8 档通道 × 4 个键瓦片 = 32 次 MMA（本 warp 8 列 query × 32 键 × 64 通道）
        VA_QK(0u, q_0)
        VA_QK(8u, q_1)
        VA_QK(16u, q_2)
        VA_QK(24u, q_3)
        VA_QK(32u, q_4)
        VA_QK(40u, q_5)
        VA_QK(48u, q_6)
        VA_QK(56u, q_7)
        // 值的暂存紧跟在 QK 后面：全局加载的延迟被下面的 softmax 盖掉，且写的是另一块缓冲
        va_stage(vtg0, qkv, tid, vchan, T, t0, vec);

        // ---- 在线 softmax（fp32，按 lane 的行）：max → 重标定 → exp → 行和 → 落 P^T
        {
            float sv0[4], sv1[4], mx = -INFINITY;
            VA_SM1(0, s_0)
            VA_SM1(1, s_1)
            VA_SM1(2, s_2)
            VA_SM1(3, s_3)
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 8));
            const float m_new = max(m_run, mx);
            // 第一档 m_run 是 -inf：exp2(-inf - m_new) = 0，正是"还没有历史"的意思
            const float asc = exp2(m_run - m_new);
            m_run = m_new;
            float sum = 0.0f;
            VA_SM2(0)
            VA_SM2(1)
            VA_SM2(2)
            VA_SM2(3)
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 8);
            l_run = l_run * asc + sum;
            // 累加器的两列 query 各属别的 lane，所以重标定因子要按行取回来
            const float scl0 = simd_shuffle(asc, ushort(own0));
            const float scl1 = simd_shuffle(asc, ushort(own1));
            VA_SCALE(c_0)
            VA_SCALE(c_1)
            VA_SCALE(c_2)
            VA_SCALE(c_3)
            VA_SCALE(c_4)
            VA_SCALE(c_5)
            VA_SCALE(c_6)
            VA_SCALE(c_7)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K 已无人再读（上面的屏障排在所有 warp 的 QK 之后），下一档的 K 可以开始搬；
        // 它的加载延迟由下面的 PV 盖住。最后一档搬的是全零（块数保证没人读它）。
        va_stage(ktg0, qkv, tid, kchan, T, t0 + VA_BK, vec);

        simdgroup_load(pbs_0, &ptg[sg][0][0], VA_LDP);
        simdgroup_load(pbs_1, &ptg[sg][8][0], VA_LDP);
        simdgroup_load(pbs_2, &ptg[sg][16][0], VA_LDP);
        simdgroup_load(pbs_3, &ptg[sg][24][0], VA_LDP);
        // ---- PV：8 档通道 × 4 个键瓦片 = 32 次 MMA，累加器是（通道 × query）
        VA_PV(0u, c_0)
        VA_PV(8u, c_1)
        VA_PV(16u, c_2)
        VA_PV(24u, c_3)
        VA_PV(32u, c_4)
        VA_PV(40u, c_5)
        VA_PV(48u, c_6)
        VA_PV(56u, c_7)
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- 收尾：÷ l，写回通道主序的 fp16
    {
        const float iv = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
        const float iv0 = simd_shuffle(iv, ushort(own0));
        const float iv1 = simd_shuffle(iv, ushort(own1));
        const uint jc = qcol;
        if (jc < a.nq) {
            VA_STORE(0, 0u + row_l)
            VA_STORE(1, 8u + row_l)
            VA_STORE(2, 16u + row_l)
            VA_STORE(3, 24u + row_l)
            VA_STORE(4, 32u + row_l)
            VA_STORE(5, 40u + row_l)
            VA_STORE(6, 48u + row_l)
            VA_STORE(7, 56u + row_l)
        }
    }
}

#undef VA_LDQ
#undef VA_QK
#undef VA_PV
#undef VA_ZEROC
#undef VA_SCALE
#undef VA_SM1
#undef VA_SM2
#undef VA_STORE

// ---------------------------------------------------------------------------------------
// k_vrope: axial RoPE over q|k, in place, on the channel-major `[q | k | v]` buffer
// ---------------------------------------------------------------------------------------
// 这里不再是文本的 `k_q_prep` / `k_kv_prep` 那件事：文本的 rope 叠在环形槽位的列号上、只转
// `E8_ROTARY_DIM` 维；视觉的是整图一次、按 (行, 列) 两个轴算角度、配对是 `(d, d + head_dim/2)`
// 而不是相邻两维，而且没有 gate、没有环。旧引擎把这步放在 GPU 上（`mx_rot` 作用在已上传的
// `(heads, L, dim)`）；我们原来放在主机上（`vision.py::rope_qk`），量到占整图 33–39 %
// （票据 07 §7.8），所以搬到这里。
//
// 形态：一格 128 线程管**一个通道对**、扫 512 列
//   * 通道对序号 u ∈ [0, hidden)：`seg = u / (hidden/2)`（0 = q、1 = k）、`p = u % (hidden/2)`、
//     `h = p / (dim/2)`、`k = p % (dim/2)`；行号 `seg·hidden + h·dim + k`，配对行 `+dim/2`。
//   * 一格里按 grid-stride 扫 512 列：同一行的 32 个连续列落在 32 个连续 lane 上，所以
//     `row[t]`（每行沿 token 连续）与转置过的 `cos[k][t]` 都是 64 B 一条的合并读写。
//
// 数值：与 `rope_qk` 的 numpy 算式逐运算对应 —— 四个乘各舍入一次，再减/加各一次，最后落 f16。
// `#pragma clang fp contract(off)` 把"不折 fma"写死，不靠编译器的默认值：本机去掉它也是逐位
// 相同（A/B 过），但那是这份编译器的默认，不是 MSL 的保证；折了就会在最后一位上与主机不同。
// 判据是 `tests/integration/test_metal_kernels.py::test_vision_rope_matches_host_bitwise`。
#define VR_ITER  4u                       // 一格扫过的列数 = ntg * VR_ITER（与 `kernels.vrope` 对齐）

struct VRopeArgs {
    uint T;       // 0  每通道列数（qkv 的 stride，也是列总数）
    uint hidden;  // 1  q 的通道数（k 同）
    uint dim;     // 2  head_dim（偶数：配对是 (k, k + dim/2)）
};

kernel void k_vrope(device half       *qkv  [[buffer(0)]],
                    device const half *cosb [[buffer(1)]],
                    device const half *sinb [[buffer(2)]],
                    constant VRopeArgs &a   [[buffer(E8_ARGS_INDEX)]],
                    uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 tid2 [[thread_position_in_threadgroup]],
                    uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint T = a.T, HID = a.hidden, D = a.dim, H = D >> 1;
    const uint u = tgid.y, tid = tid2.x, ntg = ntg2.x;
    if (u >= HID) return;
    const uint seg = u / (HID >> 1), p = u % (HID >> 1);
    const uint k = p % H, hd = p / H;
    device half *lo = qkv + ((ulong)seg * HID + (ulong)hd * D + k) * T;
    device half *hi = lo + (ulong)H * T;
    const device half *tc = cosb + (ulong)k * T;
    const device half *ts = sinb + (ulong)k * T;
    const uint base = tgid.x * (ntg * VR_ITER);
    for (uint t = base + tid; t < base + ntg * VR_ITER && t < T; t += ntg) {
#pragma clang fp contract(off)
        const float x0 = float(lo[t]), x1 = float(hi[t]);
        const float c = float(tc[t]), s = float(ts[t]);
        const float a0 = x0 * c, a1 = x1 * s;          // 各舍入一次
        const float b0 = x1 * c, b1 = x0 * s;
        lo[t] = half(a0 - a1);
        hi[t] = half(b0 + b1);
    }
}
