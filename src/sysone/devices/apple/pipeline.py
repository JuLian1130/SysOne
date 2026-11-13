"""ANE + Metal 的零拷贝流水线：一层三段 ANE 计算 + GPU 快核，中间不落 host 内存。

两种宽度，故意分开：

- `bucket`：ANE program 的列数。Apple 的 program 对 S 有实测上限（4096，0.56 GB wired），
  所以编译产物按它铺，一层一层固定下来；
- `width`：GPU 侧缓冲（`seed` / 隐藏态 / `qg` / `kv`）的列数，等于这一轮要算的 token 总数。

`width == bucket` 时就是"一桶跑完"（1k/4k 的路径）；`width > bucket` 时 ANE 按桶分段算，
而 GPU 侧的注意力仍然一次看**整条** kv —— 16k 就是这么跑的（4 × 4096 桶，注意力看 16k 键）。
这不是优化，是能力：桶宽 4096 的 qg/kv 根本装不下第 2 段的尾巴要看的键。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from ...domain.attention import AttentionRole
from ...errors import ConfigError
from . import kernels
from .compile import AneLayer
from .geometry import Geometry, mude_geometry
from .metal import LANE_LONG, LANE_SMALL, MetalBuffer, MetalRuntime

EPS = 1e-6


@dataclass
class LayerBuffers:
    """一层的 ANE 面与 GPU 侧的权重表。

    `surfaces` 里是 **wrap 过的** `MetalBuffer`（不是 ANE `Surface`），所以 GPU 核与 ANE
    程序看的是同一块内存；释放必须走 `free()`，它会对每张面 `unwind`。
    """

    ane: AneLayer
    role: AttentionRole
    surfaces: dict[str, MetalBuffer]
    weights: dict[str, MetalBuffer]
    # 这一层用哪一套 qg/kv（相邻两个全注意力层的投影与注意力会同时在飞）。
    canvas: int = 0

    def surface(self, key: str) -> MetalBuffer:
        return self.surfaces[key]

    def free(self) -> None:
        # 先解绑（`e8m_wrap` 在 dylib 里按 IOSurface 指针缓存句柄，不 unwrap 会让下一轮
        # 拿到指向已释放内存的句柄），再释放权重表。
        for buffer in self.surfaces.values():
            buffer.free()
        self.surfaces.clear()
        for buffer in self.weights.values():
            buffer.free()
        self.weights.clear()


def _float_buffer(runtime: MetalRuntime, array: np.ndarray, dtype) -> MetalBuffer:
    data = np.ascontiguousarray(np.asarray(array, dtype))
    buffer = runtime.alloc(max(data.nbytes, 4))
    view = buffer.array(dtype, data.shape)
    view[:] = data
    return buffer


class MudePipeline:
    """20 层的零拷贝流水线（mude 几何：hidden 2048 / ffn 6144 / 8 全注意力层）。"""

    def __init__(
        self,
        *,
        runtime: MetalRuntime,
        layers: list[AneLayer],
        roles: list[AttentionRole],
        params: list[dict[str, np.ndarray]],
        cos: np.ndarray,
        sin: np.ndarray,
        geometry: Geometry | None = None,
        bucket: int,
        width: int,
    ) -> None:
        self._runtime = runtime
        self._geometry = mude_geometry() if geometry is None else geometry
        self._geometry.validate()
        self._hidden = self._geometry.hid
        self._bucket = int(bucket)
        self._width = int(width)
        if self._width < self._bucket:
            raise ConfigError("傍宽不能小于桶宽", width=width, bucket=bucket)
        self.buffers: list[LayerBuffers] = []
        try:
            for index, layer in enumerate(layers):
                entry = LayerBuffers(ane=layer, role=roles[index], surfaces={}, weights={})
                for key, surface in layer.surfaces.items():
                    entry.surfaces[key] = runtime.wrap(surface)
                self.buffers.append(entry)
                self._load_weights(entry, params[index])
            self._cos = _float_buffer(runtime, cos, np.float32)
            self._sin = _float_buffer(runtime, sin, np.float32)
            self._alloc_shared()
            # 每一列往下游推进的层数预算（对应旧引擎的 `RING = 8 // NCH`）：它是"注意力窗口里
            # 填多少 ANE 活"的旋钮 —— 取 4 时一列能推 3 个线性层 + 下一个全注意力层的投影，
            # 恰好 ≈ 一个 16k 注意力瓦片的 135 ms。
            self._lookahead = 4
            self._chunks_list: list[tuple[int, int]] = []
            self._length = 0
            self._prefix = 0
            self._dead = 0
            # 只有模型的最后一层能把前缀瓦片的第二相省掉：别层的 `h` 还有读者。
            self._last_is_full = bool(self.buffers) and (
                self.buffers[-1].role is AttentionRole.FULL
            )
            # S3 的回退开关。关掉它只是不省那些瓦片，答案不该变 —— 对拍拿它做 A/B。
            self._skip_dead = True
            self._proj_done: list[int] = []
            self._linear_first: list[bool] = []
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ 装载

    def _load_weights(self, entry: LayerBuffers, params: dict[str, np.ndarray]) -> None:
        # buf(1) 在这些核里声明成 `device const float *`（fp32 表），与 fp16 的 qg/kv/out
        # 不同；按 fp16 上传会让核把半个 half 当成一个 float 读，最上游 xn 就错。
        for key in ("input_norm", "post_norm"):
            entry.weights[key] = _float_buffer(self._runtime, params[key], np.float32)
        if entry.role is AttentionRole.FULL:
            entry.weights["q_norm"] = _float_buffer(self._runtime, params["q_norm"], np.float32)
            entry.weights["k_norm"] = _float_buffer(self._runtime, params["k_norm"], np.float32)
        else:
            entry.weights["conv"] = _float_buffer(self._runtime, params["conv"], np.float16)
            entry.weights["a_log"] = _float_buffer(self._runtime, params["a_log"], np.float32)
            entry.weights["dt_bias"] = _float_buffer(self._runtime, params["dt_bias"], np.float32)
            entry.weights["norm"] = _float_buffer(self._runtime, params["norm"], np.float16)

    def _alloc_shared(self) -> None:
        rt, hidden, width, bucket = self._runtime, self._hidden, self._width, self._bucket
        geom = self._geometry
        self.seed = rt.alloc(2 * hidden * width)
        # 隐藏态只有一张：逐段"读出本段、写回本段"，所以同一张 `[hidden, width]` 当输入又当输出。
        # 段内所有读（`norm_to`、残差搬运）都在写（`ffn_out` 回填）之前完成，段与段的列不重叠，
        # 所以就地是安全的；这也是旧引擎 `e8_a1m.py` 的 `w_h` 的做法（它连乒乓缓冲都没有）。
        self.x = rt.alloc(2 * hidden * width)
        # 注意力的输出，全宽、单通道带（残差不经过这里，见 `_body`）。
        self.attn = rt.alloc(2 * hidden * width)
        # [rows, width] 的 fp16 缓冲。**两套**：波前里相邻两个全注意力层的投影与注意力会同时在飞
        # （层的列段工作在本层注意力窗口里推进），一套会被写坏。全注意力层共 5 层，按序数奇偶交替。
        self.qg_sets = [rt.alloc(2 * geom.qg_rows * width) for _ in range(2)]
        self.kv_sets = [rt.alloc(2 * geom.kv_rows * width) for _ in range(2)]
        self.gscratch = rt.alloc(2 * 8192 * bucket)
        # 快核不读 scratch，但 kernel 声明了 buffer(3)，必须绑定。全宽的参考核才要 (T,T)。
        self.attn_scratch = rt.alloc(32)
        # GDN 的递归状态与卷积尾巴**逐层独立**：波前里两层线性层的列段会交错执行，
        # 共用一张状态会让后一层的清零把前一层的递推冲掉（每层 1 MB + 尾巴几十 KB）。
        self.states = [rt.alloc(4 * geom.state_floats) for _ in self.buffers]
        self.tails = [rt.alloc(2 * geom.conv_dim * 3) for _ in self.buffers]
        self._last: MetalBuffer = self.seed

    # ------------------------------------------------------------------ 执行

    def _chunks(self, length: int) -> Iterator[tuple[int, int]]:
        """把 `length` 切成不超过桶宽的段，返回 `(起始列, 列数)`。"""
        step = self._bucket
        for start in range(0, length, step):
            yield start, min(step, length - start)

    def run(self, embed: np.ndarray, *, length: int, prefix: int = 0) -> None:
        """跑完 20 层。`embed` 是 `[length, hidden]` 的 embedding 输出（fp32）。

        **列段波前**（第 3 步 B 段）：不再"一层做完再做下一层"，而是按列段推进 —— 每写完
        一层某一列的 `x`，下一层**同一列**就能开工（就地 `x` 的列互不重叠）。于是全注意力层的
        注意力窗口（540 ms/层）里填的不是它自己的 ANE（只有 ~40 ms），而是紧随其后的三个线性层
        的列段工作（3 × 136 ms）与再下一个全注意力层的投影（66 ms）——把 ANE 的 3.6 s 摊进
        GPU 的 2.8 s 里。这里没有并发求值（ANE 只有一条固件队列、一次 eval 在飞），重叠完全
        来自"CPU 在等 ANE/GPU 之前先把另一侧的活排满"。

        `prefix` 非 0 时还会省掉最后一层的死瓦片（S3，见下面的注释）：出货口径
        （T=16884 / P=16384）那个形状上，最后一层有 4/5 的瓦片整段落在前缀里。
        """
        if not 0 <= prefix <= length <= self._width:
            raise ConfigError("长度或 prefix 越界", length=length, prefix=prefix, width=self._width)
        view = self.seed.array(np.float16, (self._hidden, self._width))
        view[:] = 0
        view[:, :length] = np.asarray(embed, np.float32).T.astype(np.float16)
        self._chunks_list = list(self._chunks(length))
        self._length = int(length)
        self._prefix = int(prefix)
        # S3（旧引擎的 `LASTDEAD`，`docs/ANE/anebench/e8_a3.py:291-302`）：模型最后一层里整段落在
        # 公共前缀中的瓦片只欠 K/V —— 前缀行的注意力输出与 MLP 输出都没有读者（读出在 query
        # 行上，K/V 出自投影那一步，不依赖注意力）。所以这些瓦片丢掉第二相（`attn` 与它之后的
        # `out`/`norm2`/`ffn`/`tail`）。
        #
        # 最后一个瓦片永远不丢：`prefix == length` 时读出那一行自己也在前缀里，它就变成读者了。
        dead = 0
        if self._skip_dead and prefix > 0 and self._last_is_full:
            dead = sum(1 for start, n in self._chunks_list if start + n <= prefix)
            dead = min(dead, len(self._chunks_list) - 1)
        self._dead = dead
        # 每个全注意力层分两阶段（投影 → 注意力 body），阶段边界就是旧引擎 `prep` 那道屏障。
        self._proj_done = [0] * len(self.buffers)
        self._linear_first = [True] * len(self.buffers)
        ordinal = 0
        for entry in self.buffers:
            if entry.role is AttentionRole.FULL:
                entry.canvas = ordinal % 2
                ordinal += 1
            else:
                entry.canvas = 0
        # 先按列推进：第 0 层没有前驱，一列一列往下游递。递到某个全注意力层时它只做投影，
        # 等它的最后一列投影完成，就由 `_advance` 内部一次性把它的 body 跑完（并继续往下游递）。
        for column in range(len(self._chunks_list)):
            self._advance(0, column, self._lookahead)
        self._wait_all()
        self._last = self.x

    # ---- 两条命令队列 -------------------------------------------------------

    def _wait(self, lane: int = LANE_SMALL) -> None:
        """提交该 lane 的命令缓冲并等到队尾。

        只有真的要用它刚写出的数据时才调：`kernels.py` 里 `k_norm_to`/`k_norm_resid`/
        `k_copy`/`k_gdn` 都排在 lane 1，`k_q_prep`/`k_kv_prep`/`k_attn` 排在 lane 0。多等
        另一条 lane 会把那边还在跑的活拖成串行，等于把两条队列并回一条。
        """
        self._runtime.commit(lane)
        self._runtime.wait(lane)

    def _wait_all(self) -> None:
        """两条 lane 都等到队尾（整层/整链收尾、读回最终隐藏态时才用）。"""
        self._runtime.sync_all()

    def _copy(
        self,
        src: MetalBuffer,
        dst: MetalBuffer,
        *,
        nchan: int,
        src_c0: int = 0,
        dst_c0: int = 0,
        n: int = 0,
        s0: int = 0,
        src_s0: int = 0,
        src_stride: int = 0,
        dst_stride: int = 0,
    ) -> None:
        """跨宽缓冲的通道带搬运：`n=0` 表示整宽。"""
        kernels.copy_columns(
            self._runtime, src, dst,
            nchan=nchan, src_c0=src_c0, dst_c0=dst_c0,
            stile=n or self._width, s0=s0,
            src_stride=src_stride or self._width,
            dst_stride=dst_stride or self._width,
            src_s0=src_s0,
        )

    # ---- 列段波前 -----------------------------------------------------------

    def _advance(self, index: int, column: int, budget: int) -> None:
        """把第 `index` 层的第 `column` 列段往下游推进，最多再走 `budget` 层。

        线性层：这一列做完（投影 → GDN → out → FFN，写回 `x` 的同一列），再把下一层同一列递一步。

        全注意力层分两段。投影逐列就能做（在上一层 body 的窗口里推进）；但注意力要看本层
        **全部列**的 k/v，所以最后一列投影完成后才进入 body —— 那一刻就是旧引擎 `attn_steps`
        的两相边界（`e8_a3.py:352-399` 的 `prep` 屏障）。body 的每一列收完，就继续往下游递
        （那一列的 `x` 刚写好，下一层同一列立刻可开工）。

        `budget` = 这一列还能往下走几层，默认 `_lookahead`。它决定每个注意力窗口里填多少
        ANE 活：太小 GPU 空转，太大就把下游抢到前面做完、窗口里反而没活。
        """
        if budget <= 0 or index >= len(self.buffers):
            return
        entry = self.buffers[index]
        start, n = self._chunks_list[column]
        cur = self.seed if index == 0 else self.x
        if entry.role is AttentionRole.FULL:
            if self._proj_done[index] >= len(self._chunks_list):
                return              # 已进入（或已完成）body 阶段，这一列由下面的循环驱动
            self._proj_chunk(entry, start, n, cur)
            self._finish_proj(entry, start, n)
            self._proj_done[index] += 1
            if self._proj_done[index] < len(self._chunks_list):
                return              # k/v 还没攒够，body 不能开工
            self._wait(LANE_SMALL)  # 分段投影的 qg/kv 搬运在 lane 1，prep 在 lane 0 上要读它
            self._prep(entry)
            dead = self._dead_tiles(index)
            self._dispatch_attention(entry, dead)   # 一次提交全部瓦片，GPU 从头跑到尾
            for jj, (body_start, body_n) in enumerate(self._chunks_list):
                if jj >= dead:                      # S3：前缀瓦片没有第二相
                    self._body_column(entry, body_start, body_n)
                self._advance(index + 1, jj, self._lookahead)
            return
        self._linear_column(entry, index, start, n, cur)
        self._advance(index + 1, column, budget - 1)

    def _proj_chunk(self, entry: LayerBuffers, start: int, n: int, cur: MetalBuffer) -> None:
        """一段的输入归一化 + 投影（发射，不回收）。"""
        kernels.norm_to(
            self._runtime, cur, entry.weights["input_norm"], entry.surface("proj_in"),
            stile=n, src_stride=self._width, eps=EPS, s0=start,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._wait(LANE_SMALL)          # GPU 只是排队，ANE 紧接着要读 proj_in
        entry.ane.issue_proc("proj")

    def _finish_proj(self, entry: LayerBuffers, start: int, n: int) -> None:
        """回收投影，并把 `proj_out` 的两段分别搬进全宽 qg / kv。"""
        entry.ane.reap_proc()
        qg, kv = self._qgkv(entry)
        self._copy(entry.surface("proj_out"), qg, nchan=self._geometry.qg_rows, n=n, s0=start,
                   src_stride=self._bucket, dst_stride=self._width)
        self._copy(entry.surface("proj_out"), kv, nchan=self._geometry.kv_rows,
                   src_c0=self._geometry.qg_rows, n=n, s0=start,
                   src_stride=self._bucket, dst_stride=self._width)

    def _prep(self, entry: LayerBuffers) -> None:
        """全宽的 q/k 归一化 + 部分 RoPE（旧引擎的 `prep`，序列级屏障之后才做）。"""
        qg, kv = self._qgkv(entry)
        kernels.q_prep(self._runtime, qg, entry.weights["q_norm"],
                       self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                       geometry=self._geometry)
        kernels.kv_prep(self._runtime, kv, entry.weights["k_norm"],
                        self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                        geometry=self._geometry)

    def _dead_tiles(self, index: int) -> int:
        """这一层有几个瓦片整段落在公共前缀里（S3）；只有模型的最后一层能省。"""
        return self._dead if index == len(self.buffers) - 1 else 0

    def _dispatch_attention(self, entry: LayerBuffers, dead: int = 0) -> None:
        """把注意力瓦片**一个瓦片一次提交**，GPU 从头到尾连续跑。

        `commit` 提交的是当前打开的命令缓冲 —— 若 4 个瓦片只提交一次，它们就落在同一个 CB 里，
        之后 `wait_upto(1)` 会把四个一起等完，交错又退化。一个瓦片一个 CB，才能"只等第 j 个"。
        第一个**实发**的 CB 里还带着 `_prep` 的两个 dispatch（lane 0 的队列顺序即执行顺序）。

        S3 的死瓦片（`dead`，前 `dead` 个）不在这里提交：它们的注意力没有读者，`k_attn` 也就
        不该跑。提交数与 `_body_column` 的 `wait_upto(1)` 次数要一一对应，所以两边跳过同一批。
        """
        qg, kv = self._qgkv(entry)
        for jj, chunk in enumerate(self._chunks_list):
            if jj < dead:
                continue
            self._attention_tile(entry, chunk, qg, kv)
            self._runtime.commit(LANE_LONG)

    def _qgkv(self, entry: LayerBuffers) -> tuple[MetalBuffer, MetalBuffer]:
        """这一层用哪一套 qg/kv：相邻两个全注意力层的投影与注意力会同时在飞。"""
        return self.qg_sets[entry.canvas], self.kv_sets[entry.canvas]

    def _body_column(self, entry: LayerBuffers, start: int, n: int) -> None:
        """收一列注意力：等第 `jj` 个瓦片 → out（含残差）→ 门控归一 → FFN → 写回 `x`。"""
        self._runtime.wait_upto(LANE_LONG, 1)   # 只等队首这一个瓦片，后面的还在跑
        self._copy(self.attn, entry.surface("out_in"), nchan=self._hidden,
                   n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        self._copy(self.x, entry.surface("out_in"), nchan=self._hidden,
                   dst_c0=self._hidden, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        self._wait(LANE_SMALL)              # out_in 两半就绪
        entry.ane.issue_proc("out")
        entry.ane.reap_proc()               # norm_residual 要读 out_out
        kernels.norm_residual(
            self._runtime, entry.surface("out_out"), entry.weights["post_norm"],
            entry.surface("ffn_in"),
            stile=n, src_stride=self._bucket, eps=EPS,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._wait(LANE_SMALL)              # ffn_in 就绪
        entry.ane.issue_proc("ffn")
        entry.ane.reap_proc()
        self._copy(entry.surface("ffn_out"), self.x, nchan=self._hidden,
                   n=n, s0=start, src_stride=self._bucket, dst_stride=self._width)
        self._wait(LANE_SMALL)              # 这一列的 x 落定，下游可以接着读

    def _attention_tile(
        self, entry: LayerBuffers, chunk: tuple[int, int],
        qg: MetalBuffer, kv: MetalBuffer,
    ) -> None:
        """一段 query 的注意力 dispatch；`prefix_noncausal` 的边界若落在这段内就拆两次。

        段内的 query 只受 `nk` 一个上界约束，所以"按段发射"与"整条一次发射"在数值上等价 ——
        `test_metal_chain.py` 的 `prefix ∈ {0,4}` 用例守住这一点。
        """
        start, n = chunk
        end = start + n
        if self._prefix <= 0 or self._prefix >= self._length:
            kernels.attention(
                self._runtime, qg, kv, self.attn, self.attn_scratch,
                T=self._width, stile=n, nk=self._length, nq=n, s0=start,
                out_stride=self._width, out_s0=start, fast=True, geometry=self._geometry,
            )
            return
        boundary = min(end, self._prefix)
        if start < boundary:
            kernels.attention(
                self._runtime, qg, kv, self.attn, self.attn_scratch,
                T=self._width, stile=boundary - start, nk=self._prefix, nq=boundary - start,
                s0=start, out_stride=self._width, out_s0=start, fast=True,
                geometry=self._geometry,
            )
        if boundary < end:
            kernels.attention(
                self._runtime, qg, kv, self.attn, self.attn_scratch,
                T=self._width, stile=end - boundary, nk=self._length, nq=end - boundary,
                s0=boundary, out_stride=self._width, out_s0=boundary, fast=True,
                geometry=self._geometry,
            )

    # ---- 线性（GDN）层 ------------------------------------------------------

    def _linear_column(
        self, entry: LayerBuffers, index: int, start: int, n: int, cur: MetalBuffer,
    ) -> None:
        """一列的 GDN 层：投影 → GDN（递归状态与卷积尾巴跨段接力）→ out → FFN → 写回 `x`。

        状态与尾巴**逐层独立**（`self.states[index]` / `self.tails[index]`）：波前里两层线性层
        的列段会交错执行，共用一张 1 MB 状态会被后一层的清零冲掉。
        """
        state, tail = self.states[index], self.tails[index]
        if self._linear_first[index]:
            # 每层的递归状态都从零开始（prefill、无 cache）：`k_gdn` 自己不清零，
            # 旧引擎也是在每层前把 w_state 填零（docs/ANE/anebench/e8_a1m.py:969-971）。
            state.array(np.float32,
                        (self._geometry.l_v_heads, self._geometry.l_k_dim,
                         self._geometry.l_v_dim))[:] = 0
        kernels.norm_to(
            self._runtime, cur, entry.weights["input_norm"], entry.surface("proj_in"),
            stile=n, src_stride=self._width, eps=EPS, s0=start,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._wait(LANE_SMALL)
        entry.ane.issue_proc("proj")
        # 残差搬运只读 `cur`、只写 `out_in` 的后半区，与在飞的投影（读写 `proj_in` /
        # `proj_out`）不冲突，所以排在回收之前 —— 一次 eval 在飞时 CPU 继续编码。
        self._copy(cur, entry.surface("out_in"), nchan=self._hidden,
                   dst_c0=self._hidden, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        entry.ane.reap_proc()
        kernels.gdn(
            self._runtime, entry.surface("proj_out"), state, tail,
            entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
            entry.weights["norm"], entry.surface("out_in"), self.gscratch,
            # GDN 核里的 `Stile` 既是列界也是**通道步长**（`proj[c*Stile + j]`），所以必须
            # 传桶宽而不是本段的有效列数；`T` 是它不看的多余字段。
            stile=self._bucket, eps=EPS, first=self._linear_first[index], T=self._bucket,
            out_stride=self._bucket, fast=True, geometry=self._geometry,
        )
        self._linear_first[index] = False
        self._wait(LANE_SMALL)   # ANE 的 out 要读 GDN 写出的 out_in 前半区
        entry.ane.issue_proc("out")
        entry.ane.reap_proc()    # norm_residual 要读 out_out
        kernels.norm_residual(
            self._runtime, entry.surface("out_out"), entry.weights["post_norm"],
            entry.surface("ffn_in"),
            stile=n, src_stride=self._bucket, eps=EPS,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._wait(LANE_SMALL)
        entry.ane.issue_proc("ffn")
        entry.ane.reap_proc()
        self._copy(entry.surface("ffn_out"), self.x, nchan=self._hidden,
                   n=n, s0=start, src_stride=self._bucket, dst_stride=self._width)
        self._wait(LANE_SMALL)   # 这一列的 x 落定，下游可以接着读

    # ------------------------------------------------------------------ 收尾

    def last_hidden(self, length: int) -> np.ndarray:
        """最后一层的最后一条 token（`[hidden]` fp32），供 norm + 读出。"""
        view = self._last.array(np.float16, (self._hidden, self._width))
        return np.asarray(view[:, length - 1], np.float32)

    def close(self) -> None:
        for entry in self.buffers:
            entry.free()
        self.buffers.clear()
        for name in ("seed", "x", "attn", "gscratch", "attn_scratch", "_cos", "_sin"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.free()
        for name in ("qg_sets", "kv_sets", "states", "tails"):
            group = getattr(self, name, None) or ()
            for buffer in group:
                buffer.free()
