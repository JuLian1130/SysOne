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
        # 注意力的输出，全宽、单通道带（残差不经过这里，见 `_out_and_ffn`）。
        self.attn = rt.alloc(2 * hidden * width)
        # [rows, width] 的 fp16 缓冲：行数来自几何（qg 是 2*q_heads*head_dim，kv 是 2*kv_proj）。
        self.qg = rt.alloc(2 * geom.qg_rows * width)
        self.kv = rt.alloc(2 * geom.kv_rows * width)
        self.state = rt.alloc(4 * geom.state_floats)
        self.tail = rt.alloc(2 * geom.conv_dim * 3)
        self.gscratch = rt.alloc(2 * 8192 * bucket)
        # 快核不读 scratch，但 kernel 声明了 buffer(3)，必须绑定。全宽的参考核才要 (T,T)。
        self.attn_scratch = rt.alloc(32)
        self._last: MetalBuffer = self.seed

    # ------------------------------------------------------------------ 执行

    def _chunks(self, length: int) -> Iterator[tuple[int, int]]:
        """把 `length` 切成不超过桶宽的段，返回 `(起始列, 列数)`。"""
        step = self._bucket
        for start in range(0, length, step):
            yield start, min(step, length - start)

    def run(self, embed: np.ndarray, *, length: int, prefix: int = 0) -> None:
        """跑完 20 层。`embed` 是 `[length, hidden]` 的 embedding 输出（fp32）。"""
        if not 0 <= prefix <= length <= self._width:
            raise ConfigError("长度或 prefix 越界", length=length, prefix=prefix, width=self._width)
        view = self.seed.array(np.float16, (self._hidden, self._width))
        view[:] = 0
        view[:, :length] = np.asarray(embed, np.float32).T.astype(np.float16)
        cur: MetalBuffer = self.seed
        nxt: MetalBuffer = self.x
        for entry in self.buffers:
            if entry.role is AttentionRole.FULL:
                self._full_layer(entry, length, prefix, cur, nxt)
            else:
                self._linear_layer(entry, length, cur, nxt)
            # 从第二层起，输入和输出都是 `x` 本身（就地）；第一层的输入仍是 `seed`。
            cur = nxt
            nxt = self.x
        self._wait_all()
        self._last = cur

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

    # ---- 全注意力层 ---------------------------------------------------------

    def _full_layer(
        self, entry: LayerBuffers, length: int, prefix: int,
        cur: MetalBuffer, nxt: MetalBuffer,
    ) -> None:
        """先按桶分段算投影（ANE 列数受限、边算边把 `proj_out` 搬进全宽的 qg/kv），
        再按桶分段做注意力与 out/FFN，两相交错。"""
        proj = entry.surface("proj_out")
        for start, n in self._chunks(length):
            kernels.norm_to(
                self._runtime, cur, entry.weights["input_norm"], entry.surface("proj_in"),
                stile=n, src_stride=self._width, eps=EPS, s0=start,
                dst_stride=self._bucket, dst_s0=0,
            )
            self._wait(LANE_SMALL)          # GPU 只是排队，ANE 紧接着要读 proj_in
            # 发射即返回：投影在飞的时候 CPU 不阻塞，这段时间可以用来编码与本段无关的
            # 搬运/计算（旧引擎 `ISSUE_EVAL` 的用法）。投影要把 `proj_out` 交给下一段，
            # 所以紧接着回收 —— 真正的叠法要等两套 proj 槽位（见 docs 的 M2.2 第 5 片）。
            entry.ane.issue_proc("proj")
            entry.ane.reap_proc()
            self._copy(proj, self.qg, nchan=self._geometry.qg_rows, n=n, s0=start,
                       src_stride=self._bucket, dst_stride=self._width)
            self._copy(proj, self.kv, nchan=self._geometry.kv_rows, src_c0=self._geometry.qg_rows,
                       n=n, s0=start, src_stride=self._bucket, dst_stride=self._width)
        # 分段投影的 `proj_out → qg/kv` 搬运在 lane 1 上，而下面两个 prep 在 lane 0 上；
        # 两者碰的是同一批缓冲，所以必须等 lane 1 排空（否则 prep 之后的 qg 又被裸拷贝覆盖）。
        self._wait(LANE_SMALL)
        kernels.q_prep(self._runtime, self.qg, entry.weights["q_norm"],
                       self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                       geometry=self._geometry)
        kernels.kv_prep(self._runtime, self.kv, entry.weights["k_norm"],
                        self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                        geometry=self._geometry)
        # 相位 B（步骤图）：逐段"注意力 → out → norm2 → ffn"，把 ANE 的 out/ffn 埋进
        # 下一段的 attention 里。这是旧引擎 `attn_steps` 的两相结构（`e8_a3.py:352-399`）：
        # 每段的 `k_attn` 独立发射（`nk` 是全局上界、`s0`/`out_s0` 是本段列），段与段之间
        # 没有数据依赖，所以可以先把下一段的 attention 排进 lane 0，再用本轮 ANE 求值回收成本段。
        chunks = list(self._chunks(length))
        self._start_attention_tile(chunks[0], length, prefix)
        for index, (start, n) in enumerate(chunks):
            self._wait(LANE_LONG)               # 本段 attention 完成（顺带排空上一段的 lane 0）
            self._copy(self.attn, entry.surface("out_in"), nchan=self._hidden,
                       n=n, s0=0, src_s0=start,
                       src_stride=self._width, dst_stride=self._bucket)
            self._copy(cur, entry.surface("out_in"), nchan=self._hidden,
                       dst_c0=self._hidden, n=n, s0=0, src_s0=start,
                       src_stride=self._width, dst_stride=self._bucket)
            self._wait(LANE_SMALL)              # out_in 两半就绪
            entry.ane.issue_proc("out")
            if index + 1 < len(chunks):
                # ANE 做 out 的这段时间里，下一段的 attention 已经在 lane 0 上跑起来了。
                self._start_attention_tile(chunks[index + 1], length, prefix)
            entry.ane.reap_proc()               # norm_residual 要读 out_out
            kernels.norm_residual(
                self._runtime, entry.surface("out_out"), entry.weights["post_norm"],
                entry.surface("ffn_in"),
                stile=n, src_stride=self._bucket, eps=EPS,
                dst_stride=self._bucket, dst_s0=0,
            )
            self._wait(LANE_SMALL)              # ffn_in 就绪
            entry.ane.issue_proc("ffn")         # 在飞：下一段的 attention 同时也在跑
            entry.ane.reap_proc()
            self._copy(entry.surface("ffn_out"), nxt, nchan=self._hidden,
                       n=n, s0=start, src_stride=self._bucket, dst_stride=self._width)
            self._wait(LANE_SMALL)              # ffn_out 已落进 nxt

    def _attention_tile(self, chunk: tuple[int, int], length: int, prefix: int) -> None:
        """一段 query 的注意力 dispatch；`prefix_noncausal` 的边界若落在这段内就拆两次。

        段内的 query 只受 `nk` 一个上界约束，所以"按段发射"与"整条一次发射"在数值上等价 ——
        `test_metal_chain.py` 的 `prefix ∈ {0,4}` 用例守住这一点。
        """
        start, n = chunk
        end = start + n
        if prefix <= 0 or prefix >= length:
            kernels.attention(
                self._runtime, self.qg, self.kv, self.attn, self.attn_scratch,
                T=self._width, stile=n, nk=length, nq=n, s0=start,
                out_stride=self._width, out_s0=start, fast=True, geometry=self._geometry,
            )
            return
        boundary = min(end, prefix)
        if start < boundary:
            kernels.attention(
                self._runtime, self.qg, self.kv, self.attn, self.attn_scratch,
                T=self._width, stile=boundary - start, nk=prefix, nq=boundary - start,
                s0=start, out_stride=self._width, out_s0=start, fast=True,
                geometry=self._geometry,
            )
        if boundary < end:
            kernels.attention(
                self._runtime, self.qg, self.kv, self.attn, self.attn_scratch,
                T=self._width, stile=end - boundary, nk=length, nq=end - boundary,
                s0=boundary, out_stride=self._width, out_s0=boundary, fast=True,
                geometry=self._geometry,
            )

    def _start_attention_tile(
        self, chunk: tuple[int, int], length: int, prefix: int,
    ) -> None:
        """发射一段注意力并**立刻提交** lane 0。

        `commit` 才是把命令缓冲交给 GPU，`dispatch` 只是编码；只有提交了，下一段的
        attention 才会在本轮的 ANE 求值期间真的跑起来 —— 不提交就等到 `_wait(LANE_LONG)`
        才一起提交，"交错"退化成串行（实测省不出任何时间，正是这一处漏掉的 commit）。
        """
        self._attention_tile(chunk, length, prefix)
        self._runtime.commit(LANE_LONG)

    # ---- 线性（GDN）层 ------------------------------------------------------

    def _linear_layer(
        self, entry: LayerBuffers, length: int, cur: MetalBuffer, nxt: MetalBuffer,
    ) -> None:
        """按桶分段：投影 → GDN（递归状态与卷积尾巴跨段接力）→ out → FFN。"""
        # 每层的递归状态都从零开始（prefill、无 cache）：`k_gdn` 自己不清零，
        # 旧引擎也是在每层前把 w_state 填零（docs/ANE/anebench/e8_a1m.py:969-971）。
        self.state.array(np.float32,
                         (self._geometry.l_v_heads, self._geometry.l_k_dim,
                          self._geometry.l_v_dim))[:] = 0
        first = True
        for start, n in self._chunks(length):
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
                self._runtime, entry.surface("proj_out"), self.state, self.tail,
                entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
                entry.weights["norm"], entry.surface("out_in"), self.gscratch,
                # GDN 核里的 `Stile` 既是列界也是**通道步长**（`proj[c*Stile + j]`），所以必须
                # 传桶宽而不是本段的有效列数；`T` 是它不看的多余字段。
                stile=self._bucket, eps=EPS, first=first, T=self._bucket,
                out_stride=self._bucket, fast=True, geometry=self._geometry,
            )
            first = False
            self._wait(LANE_SMALL)   # ANE 的 out 要读 GDN 写出的 out_in 前半区
            self._out_and_ffn(entry, length, nxt, only=(start, n))

    def _out_and_ffn(
        self, entry: LayerBuffers, length: int, nxt: MetalBuffer,
        *, only: tuple[int, int] | None = None,
    ) -> None:
        """`out` 投影（含残差加）→ 门控前归一 → FFN，按桶分段写回 `nxt` 的对应列。

        只有线性（GDN）层走这里：它在同一段的循环里已经把 GDN 输出与残差写进 `out_in` 的两半，
        所以这里只需要补 out → norm2 → ffn 这一段。全注意力层走 `_full_layer` 里那段交错流水，
        它的 `out_in` 由注意力结果与残差就地拼成。

        **段与段之间的软件流水**：某一段的 `ffn` 在 ANE 上跑着的时候，下一段的 `out_in`
        已经在 lane 1 的队列里等着了（`ffn` 只读写 `ffn_in`/`ffn_out`，与 `out_in` 无关），
        所以 `ffn` 结束时下一段的输入也已就绪；上一段的 `ffn_out → nxt` 回填则排在下一段
        `out` 发射之前。没有这一步流水，每段都要"等 ANE 完全空闲再开始搬输入"。
        """
        chunks = [only] if only is not None else list(self._chunks(length))
        pending: tuple[int, int] | None = None   # 上一次已发射、尚未回填的 ffn 属于哪一段
        for start, n in chunks:
            self._wait(LANE_SMALL)              # out_in 两半就绪
            if pending is not None:
                entry.ane.reap_proc()           # 上一段的 ffn 算完了，ffn_out 可读
                self._copy(entry.surface("ffn_out"), nxt, nchan=self._hidden,
                           n=pending[1], s0=pending[0],
                           src_stride=self._bucket, dst_stride=self._width)
            entry.ane.issue_proc("out")
            entry.ane.reap_proc()               # norm_residual 要读 out_out
            kernels.norm_residual(
                self._runtime, entry.surface("out_out"), entry.weights["post_norm"],
                entry.surface("ffn_in"),
                stile=n, src_stride=self._bucket, eps=EPS,
                dst_stride=self._bucket, dst_s0=0,
            )
            self._wait(LANE_SMALL)              # ffn_in 就绪
            entry.ane.issue_proc("ffn")         # 在飞：下一段的搬运可以叠上来
            pending = (start, n)
        if pending is not None:
            entry.ane.reap_proc()
            self._copy(entry.surface("ffn_out"), nxt, nchan=self._hidden,
                       n=pending[1], s0=pending[0],
                       src_stride=self._bucket, dst_stride=self._width)
            self._wait(LANE_SMALL)              # ffn_out 已落进 nxt，下一层可以读

    # ------------------------------------------------------------------ 收尾

    def last_hidden(self, length: int) -> np.ndarray:
        """最后一层的最后一条 token（`[hidden]` fp32），供 norm + 读出。"""
        view = self._last.array(np.float16, (self._hidden, self._width))
        return np.asarray(view[:, length - 1], np.float32)

    def close(self) -> None:
        for entry in self.buffers:
            entry.free()
        self.buffers.clear()
        for name in ("seed", "x", "attn", "qg", "kv", "state", "tail",
                     "gscratch", "attn_scratch", "_cos", "_sin"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.free()
