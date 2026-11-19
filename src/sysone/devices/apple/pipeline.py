"""ANE + Metal 的零拷贝流水线：一层三段 ANE 计算 + GPU 快核，中间不落 host 内存。

两种宽度，故意分开：

- `bucket`：ANE program 的列数。Apple 的 program 对 S 有实测上限（4096，0.56 GB wired），
  所以编译产物按它铺，一层一层固定下来；
- `width`：GPU 侧缓冲（隐藏态 / `qg` / `kv`）的列数，等于这一轮要算的 token 总数。

`width == bucket` 时就是"一桶跑完"（1k/4k 的路径）；`width > bucket` 时 ANE 按桶分段算，
而 GPU 侧的注意力仍然一次看**整条** kv —— 16k 就是这么跑的（4 × 4096 桶，注意力看 16k 键）。
这不是优化，是能力：桶宽 4096 的 qg/kv 根本装不下第 2 段的尾巴要看的键。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ...domain.attention import AttentionRole
from ...errors import ConfigError, NotReadyError
from . import kernels
from .compile import AneLayer, SurfaceSet
from .geometry import Geometry, mude_geometry
from .longpass import LongPass
from .metal import LANE_LONG, LANE_SMALL, MetalBuffer, MetalRuntime
from .residency import Residency
from .schedule import (
    NCH,
    Chain,
    Step,
    attention_steps,
    blocks,
    gdn_steps,
    ring_columns,
    ring_slot,
)

EPS = 1e-6

# 一层的瓦片链数（旧引擎 `NCH = 2`，`e8_a3.py:109`）：再多就把每条链排得太短，等 `core` 时
# 没有别的链可喂。
CHAINS = NCH


def pool_slots(roles: Sequence[AttentionRole]) -> int:
    """表面池要几套：每条 `(层, 瓦片奇偶)` 链独占一套，一个 block 里同时最多这么多条。

    一条链独占一套是**必须**的，不是省事的：相邻两层的链现在真的同时在飞 —— 第 i 层还在读
    自己那套的 `proj_in` 时，第 i+1 层的 `norm` 已经在写它的 `proj_in`。旧的手写波前在面上
    是严格串行的（一层一列做完才递下一层），所以一套面够用；步骤图驱动按层交错，就必须分套。
    """
    return CHAINS * max(len(group) for group in blocks(roles))


@dataclass
class LayerBuffers:
    """一层的 ANE 面（按槽位）与 GPU 侧的权重表。

    `surfaces` 是**整个表面池**的 wrap 视图：入口是 ANE 槽位（= 池里的第几套），值是那一套的
    四张面。GPU 核与 ANE 程序看的是同一块内存。这些面属于池、由流水线统一 wrap 一次：同一张面
    wrap 两次在 dylib 里拿到同一个句柄（`e8_mtl.m:241` 按 IOSurface 指针缓存），`unwind` 一次
    就够，所以 `free()` 只放权重，面留给 `MudePipeline.close()`。
    """

    ane: AneLayer
    role: AttentionRole
    surfaces: list[dict[str, MetalBuffer]]
    weights: dict[str, MetalBuffer]

    def surface(self, key: str, slot: int = 0) -> MetalBuffer:
        return self.surfaces[slot][key]

    def free(self) -> None:
        for buffer in self.weights.values():
            buffer.free()
        self.weights.clear()


def _float_buffer(runtime: MetalRuntime, array: np.ndarray, dtype) -> MetalBuffer:
    data = np.ascontiguousarray(np.asarray(array, dtype))
    buffer = runtime.alloc(max(data.nbytes, 4))
    view = buffer.array(dtype, data.shape)
    view[:] = data
    return buffer


AT_BM = 128   # 注意力的 query 块（`e8_attn.metal`），槽宽必须是它的整数倍


@dataclass(frozen=True)
class SlotLayout:
    """批尾巴的槽位布局（票据 10）：一题一格，格宽 `slotw`，打包区一共 `nt` 列。

    `per` 是一瓦片几格（`bucket // slotw`），`starts[j]` 是第 j 题尾巴在画布上的绝对首列。
    """

    slotw: int
    per: int
    ntiles: int
    nt: int
    starts: tuple[int, ...]


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
        pool: Sequence[SurfaceSet],
        residency: Residency | None = None,
    ) -> None:
        self._runtime = runtime
        # 长 pass 的闸：跨进程、只对 `length > 4096` 生效（见 `longpass.py` 的推导）。
        self._longpass = LongPass()
        self._geometry = mude_geometry() if geometry is None else geometry
        self._geometry.validate()
        self._hidden = self._geometry.hid
        # 核心输出宽度（`o_proj` 的输入行数）：2B 恰好等于 hidden，4B 是 4096 ≠ 2560。
        self._core_out = self._geometry.core_out
        self._bucket = int(bucket)
        self._width = int(width)
        if self._width < self._bucket:
            raise ConfigError("傍宽不能小于桶宽", width=width, bucket=bucket)
        # `qg` 的环：每条瓦片链一个槽位（票据 06）。整窗只有一片瓦片时一个槽位就够，
        # 所以 1k/4k 那种 `width == bucket` 的路不会因为成环多花内存。
        self._ring = ring_columns(self._width, self._bucket)
        if not pool:
            raise ConfigError("表面池至少要有一套面")
        if self._geometry.kv_rows > pool[0].out_in_rows:
            # 拆投影之后 kv 段借 `out_in` 的前 kv_rows 行（`compile.SET_KEYS`），借不出落脚点
            # 就别装作能拆 —— 否则 ANE 会写过界，那是坏答案不是慢答案。
            raise ConfigError("kv 行数超过 out_in 面，拆投影没有落脚点",
                              kv_rows=self._geometry.kv_rows,
                              out_in_rows=pool[0].out_in_rows)
        need = pool_slots(roles)
        if len(pool) < need:
            raise ConfigError("表面池不够：一条 (层, 瓦片奇偶) 链要独占一套",
                              sets=len(pool), needed=need)
        # 四张面在池里只 wrap 一次（见 `LayerBuffers`）。包出来的句柄归流水线。
        self._wrapped: list[dict[str, MetalBuffer]] = []
        self.buffers: list[LayerBuffers] = []
        # 常驻窗口：`None` 表示 20 层全装（今天的行为）。块边界由 `run` 驱动它换窗。
        self._residency = residency
        try:
            self._wrapped = [
                {key: runtime.wrap(surface) for key, surface in st.surfaces.items()}
                for st in pool
            ]
            for index, layer in enumerate(layers):
                entry = LayerBuffers(ane=layer, role=roles[index],
                                     surfaces=self._wrapped, weights={})
                self.buffers.append(entry)
                self._load_weights(entry, params[index])
            self._cos = _float_buffer(runtime, cos, np.float32)
            self._sin = _float_buffer(runtime, sin, np.float32)
            # 表是 `[width, rotary_dim/2]`，核按位置列号取行。带图的请求会把前 `length` 行换成
            # M-RoPE 的三轴表（`set_rope`），下一趟纯文本请求再盖回来 —— 表只按请求变、不按层变。
            self._rope_rows = int(cos.shape[0])
            self._rope_half = int(cos.shape[1])
            self._alloc_shared()
            # 每一列往下游推进的层数预算（对应旧引擎的 `RING = 8 // NCH`）：它是"注意力窗口里
            # 填多少 ANE 活"的旋钮 —— 取 4 时一列能推 3 个线性层 + 下一个全注意力层的投影，
            # 恰好 ≈ 一个 16k 注意力瓦片的 135 ms。
            # 每层拆成几条瓦片链。第 2 步一条，第 3 步切成两条（`NCH`）。
            self._nch = CHAINS
            self._chunks_list: list[tuple[int, int]] = []
            self._length = 0
            self._prefix = 0
            self._causal = False
            self._dead = 0
            # 只有模型的最后一层能把前缀瓦片的第二相省掉：别层的 `h` 还有读者。
            self._last_is_full = bool(self.buffers) and (
                self.buffers[-1].role is AttentionRole.FULL
            )
            # S3 的回退开关。关掉它只是不省那些瓦片，答案不该变 —— 对拍拿它做 A/B。
            self._skip_dead = True
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
        # 隐藏态只有一张：逐段"读出本段、写回本段"，所以同一张 `[hidden, width]` 当输入又当输出。
        # 段内所有读（`norm_to`、残差搬运）都在写（`out_out` 回填）之前完成，段与段的列不重叠，
        # 所以就地是安全的；这也是旧引擎 `e8_a1m.py` 的 `w_h` 的做法（它连乒乓缓冲都没有）。
        # 它同时是第 0 层的输入：embedding 直接写进来（旧引擎的 `w_h` 也只有一张，embedding
        # 就写在它上面）。同一瓦片里 `norm` 读它在先、`tail` 写它在后，同一条 lane 上有序。
        self.x = rt.alloc(2 * hidden * width)
        # 注意力输出**没有**单独的缓冲：那一片直接写进本槽位 `out_in` 的前 `core_out` 行
        # （旧引擎的 `ia = w_io_a[k]`），从 `core_out` 起的 x 半由残差搬运填上
        # （`_step_out_in` / `_step_core`）。
        # **一份** qg / 一份 kv 就够：`blocks()` 在每个全注意力层之前切开、`run` 在块之间整段
        # 排空，所以同一时刻只有一个全注意力层碰这两张画布（16k 的活性账：同时"开着"的层恒为
        # 1）。两套是 27c6250 的"列波前"留下的 —— 那一版块间不排空，相邻两个全注意力层真的
        # 会同时在飞。
        #
        # `qg` 是**环**（票据 06）：全注意力层的投影拆成 kv 遍 + q 遍之后，q 从 `proj_q` 到
        # `attn` 只活一个瓦片，所以它只需要每条瓦片链一个槽位（`self._ring` 列），不再整窗。
        # `kv` 仍是整窗缓存 —— 非因果注意力每个 query 都要读整条 kv，这一条没变。
        self.qg = rt.alloc(2 * geom.qg_rows * self._ring)
        self.kv = rt.alloc(2 * geom.kv_rows * width)
        # 一个 token 一行，行距由几何算（`Geometry.gdn_scratch_halves`），与 `e8_gdn.metal` 的
        # `E8G_S_ROW` 是同一个数：4B 的 v 段是 2B 的两倍，写死 8192 会少一半。
        self.gscratch = rt.alloc(2 * geom.gdn_scratch_halves * bucket)
        # 快核不读 scratch，但 kernel 声明了 buffer(3)，必须绑定。全宽的参考核才要 (T,T)。
        self.attn_scratch = rt.alloc(32)
        # GDN 的递归状态与卷积尾巴**逐层独立**：波前里两层线性层的列段会交错执行，
        # 共用一张状态会让后一层的清零把前一层的递推冲掉（每层 1 MB + 尾巴几十 KB）。
        self.states = [rt.alloc(4 * geom.state_floats) for _ in self.buffers]
        self.tails = [rt.alloc(2 * geom.conv_dim * 3) for _ in self.buffers]
        self._last: MetalBuffer = self.x
        # 共享前缀（`run_shared`）里尾巴的隐藏态：题与题共用同一段列，块边界上要按题各存一份。
        # 它只在一次共享请求期间存在（`run_shared` 结束就 `free`），不常驻。
        self._tail_hidden: MetalBuffer | None = None
        self._tail_cols = 0
        self._fresh = True
        # 批尾巴（票据 10）的相位状态：`segw` = 槽宽（0 = 逐题），`_slot_counts` = 各槽位行数，
        # `_slot_per` = 一瓦片几格，`_slot_snapshot` = 本块前缀末状态（逐槽开跑前写回）。
        self._segw = 0
        self._slot_counts: tuple[int, ...] = ()
        self._slot_per = 0
        self._slot_snapshot: dict[int, tuple[np.ndarray, np.ndarray]] | None = None

    # ------------------------------------------------------------------ 执行

    def _chunks(self, length: int) -> Iterator[tuple[int, int]]:
        """把 `length` 切成不超过桶宽的段，返回 `(起始列, 列数)`。"""
        step = self._bucket
        for start in range(0, length, step):
            yield start, min(step, length - start)

    def _write_rows(self, view: np.ndarray, rows: np.ndarray, s0: int = 0) -> None:
        """把 `[n, hidden]` 的行写进 `view` 的 `[s0, s0 + n)` 列（fp16，按桶转置）。

        画布的入口只剩这一条：`run` 的整条 prompt、`run_shared` 的前缀列与尾巴列都走它。
        整宽一次转置要先物化一份 `[hidden, length]` 的 fp32（16k 是 134 MB），按桶切，临时量
        是一桶（`bucket × hidden × 4`，1024 桶下 8 MB）。
        """
        data = np.asarray(rows)
        step = self._bucket
        for start in range(0, data.shape[0], step):
            block = np.asarray(data[start:start + step], np.float16)
            view[:, s0 + start:s0 + start + block.shape[0]] = block.T

    def run(self, rows: np.ndarray, *, length: int, prefix: int = 0,
            causal: bool = False) -> None:
        """跑完 20 层。`rows` 是 `[length, hidden]` 的 embedding，按桶写进 `x` 画布。

        **步骤图驱动**：一层被拆成 `schedule.py` 的订单式 `(chain, step)` 序列 ——
        `norm → proj → core/gather → out → norm2 → ffn → tail`，一步一个命令缓冲，
        提交/回收的账记在每条 lane 上，靠 dylib 的 `wait_upto` 精确等到"写这一步输入的那个
        缓冲"，而不是整条 lane。重叠因此来自"CPU 在等 ANE 之前已经把另一侧的活排满"。

        每层拆成 `NCH` 条瓦片链（`schedule.py` 的 `gdn_steps` / `attention_steps`）；`block`
        之间整段排空，因为表面池与 qg/kv 画布是跨层复用的。

        `prefix` 非 0 时还会省掉最后一层的死瓦片（S3，见下面 `dead` 的注释）。

        `[length, width)` 那几列**不清零**：调度只覆盖 `[0, length)` 切出来的瓦片，核读的
        每一列都在自己那一相写过（探针 `.scratch/probe_stale_canvas.py` 拿"短 → 长 → 短"
        逐位对拍过，两种读法 × 两档 prefix 都是 0）。
        """
        if not 0 <= prefix <= length <= self._width:
            raise ConfigError("长度或 prefix 越界", length=length, prefix=prefix, width=self._width)
        if int(np.asarray(rows).shape[0]) != length:
            raise ConfigError("行数与长度不一致", rows=int(np.asarray(rows).shape[0]), length=length)
        # 长 pass 串行化：画布与工作区按这一趟的长度分配，两个进程同时跑长的会把整机压垮
        # （`longpass.py`）。短 pass 不过闸，所以常态请求不受影响。
        with self._longpass.hold(length):
            view = self.x.array(np.float16, (self._hidden, self._width))
            self._write_rows(view, rows, 0)
            self._chunks_list = list(self._chunks(length))
            self._prepare(length, prefix, fresh=True, causal=causal)
            for index, block_of_layers in enumerate(self._blocks()):
                if self._residency is not None:
                    self._residency.begin(index)
                self._run_block(block_of_layers)
                self._wait_all()    # 块之间排空：表面池与 qg/kv 画布是跨层复用的
                if self._residency is not None:
                    # 排空之后才卸：卸掉的层在这一块里刚刚用完，下一次需要它到下一轮的块
                    # （`blocks()` 的块按层号单调，窗口只往后滚）。
                    self._residency.end(index)
            if self._residency is not None:
                self._residency.finish()   # 一轮结束落在待命窗口上（服务契约"回到前 W 层"）

    def run_shared(self, prefix_rows: np.ndarray, tails: Sequence[np.ndarray], *,
                   prefix: int, causal: bool = False) -> list[np.ndarray]:
        """一个请求内多题共享前缀：前缀一遍，每题只跑自己那条尾巴。

        `prefix_rows` 是 `[prefix, hidden]` 的 embedding，`tails` 每题一条 `[q_j, hidden]`
        （`q_j` 是尾巴的 token 数）。返回每题最后一层的 `[hidden]` fp32。

        **批尾巴（票据 10）**：能批就一个尾巴相把所有尾巴跑完（`_slot_layout` 给布局），
        批不了就原样逐题（`_run_shared_serial`）。题面长度不齐、且尾巴的掩码是双向的（补零
        行对读出可见），所以槽位规则除了 `segw` 还要一张按槽位的可行数表。

        复用单位是**一个块**：`blocks()` 在每个全注意力层之前切开，一块里至多一个全注意力层，
        所以一块 `kv` 画布就够 —— 前缀的 K/V 在本块里写完就被本块的尾巴读掉，下一块再覆盖。
        GDN 的递归状态与卷积尾巴**逐层独立**：前缀相跑完就抄到 host，尾巴相开始前再写回去
        （`first=False`，卷积尾巴继续用前缀的最后 3 行）。跨相传递就这一处。

        `prefix` 不限长度：批尾巴对任何前缀都开（槽 0 逐位、槽 1..N 差 ≤5e-3 logits，见
        `_slot_layout` 的数值契约），只有打包区装不下时才退回逐题。旧引擎的
        `--merge-align floor/pad` 在这里没有用：floor 把末尾 `prefix % 64` 行挪进每条尾巴后，
        那些行在尾巴相里是 query 行（旧引擎的尾巴相同样如此：它传 `prefix=Pc`、`nk=T_in`，
        `docs/ANE/anebench/e8_service.py:1251,1284`），实测答案移 0.08–0.11 logits —— 比"不齐
        直接批"的 ≤5e-3 大一个量级以上（票据 10 第 3 轮）；pad 在 mude 的双向前缀下也全可见。
        """
        if not 0 < prefix <= self._width:
            raise ConfigError("共享前缀的长度越界", prefix=prefix, width=self._width)
        if not tails:
            raise ConfigError("共享前缀至少需要一条尾巴")
        lengths = [int(np.asarray(tail).shape[0]) for tail in tails]
        if min(lengths) <= 0:
            raise ConfigError("尾巴至少要有一个 token", lengths=lengths)
        # 这个请求最长的那一趟是 `prefix + 最长尾巴`，闸按它算（批量路径同宽）。
        with self._longpass.hold(prefix + max(lengths)):
            layout = self._slot_layout(prefix, lengths)
            if layout is not None:
                return self._run_shared_batched(prefix_rows, tails, prefix=prefix, layout=layout,
                                                causal=causal)
            if prefix + max(lengths) > self._width:
                raise ConfigError("共享前缀 + 最长尾巴超出缓冲宽度",
                                  prefix=prefix, tail=max(lengths), width=self._width)
            return self._run_shared_serial(prefix_rows, tails, prefix=prefix, causal=causal)

    def _run_shared_serial(self, prefix_rows: np.ndarray, tails: Sequence[np.ndarray], *,
                           prefix: int, causal: bool = False) -> list[np.ndarray]:
        """逐题的尾巴相（批不了时的路径，行为与票据 07 一字不差）。"""
        lengths = [int(np.asarray(tail).shape[0]) for tail in tails]
        longest = max(lengths)
        self._ensure_tail_hidden(len(tails), longest)
        view = self.x.array(np.float16, (self._hidden, self._width))
        hidden = self._tail_hidden.array(np.float16, (self._hidden, self._tail_cols))
        self._write_rows(view, prefix_rows, 0)
        for index, tail in enumerate(tails):
            self._write_rows(hidden, tail, index * longest)
        prefix_chunks = list(self._chunks(prefix))
        tail_chunks = [list(self._chunks_from(prefix, prefix + q)) for q in lengths]
        result: list[np.ndarray | None] = [None] * len(tails)
        groups = self._blocks()
        try:
            for index, block_of_layers in enumerate(groups):
                if self._residency is not None:
                    self._residency.begin(index)
                self._chunks_list = list(prefix_chunks)
                self._prepare(prefix, prefix, fresh=True, causal=causal)
                self._run_block(block_of_layers)
                self._wait_all()
                snapshot = self._snapshot(block_of_layers)
                last_block = index == len(groups) - 1
                for question, chunks in enumerate(tail_chunks):
                    base, q = question * longest, lengths[question]
                    self._restore(block_of_layers, snapshot)
                    view[:, prefix:prefix + q] = hidden[:, base:base + q]
                    self._chunks_list = list(chunks)
                    self._prepare(prefix + q, prefix, fresh=False, causal=causal)
                    self._run_block(block_of_layers)
                    self._wait_all()
                    if last_block:
                        result[question] = self.last_hidden(prefix + q)
                    hidden[:, base:base + q] = view[:, prefix:prefix + q]
                if self._residency is not None:
                    self._residency.end(index)
            if self._residency is not None:
                self._residency.finish()
        finally:
            # 尾巴的隐藏态只服务这一次请求；题数/尾巴长度一变就换一块（不按见过的形状常驻）
            self._free_tail_hidden()
        if any(row is None for row in result):
            raise NotReadyError("共享前缀漏读了某题的读出", questions=len(tails))
        return [row for row in result if row is not None]

    def _slot_layout(self, prefix: int, lengths: Sequence[int]) -> SlotLayout | None:
        """批尾巴的槽位布局；批不了就 None（调用方原样逐题）。

        条件：每条尾巴装得进一个桶、槽宽整除桶宽（这样槽位边界落在瓦片边界上、一个槽不跨
        瓦片）、`per <= 8`（可见行数表只有 8 槽），以及**宽缓冲装得下整个打包区** —— 打包区
        是整桶的整数倍，调用方要在前缀之后留出这一桶的余量。**前缀长度不设条件**：打包区从
        `prefix` 自己起切，`prefix` 不必是 64 或桶的整数倍。

        数值契约（票据 10 第 3 轮）：槽 0 的 key 与逐题整趟落在同样的绝对列上，**逐位相同**；
        槽 1..N 的 key 从 `prefix + j*slotw` 起，而 online softmax 按**绝对列号的 64 列块**分组
        （`e8_attn.metal` 的 `AT_BN`）—— 只有槽 0 能复现参考里"末尾 `prefix % 64` 个前缀 key
        与本题开头几个 key 同块"的分组（`prefix` 对齐时没有这个半块，所以四个槽都逐位）。
        实测（真实 payload、4 题）：槽 1..N 差 4.8e-3 logits / 5.4e-4 选项概率，argmax 不变，
        见 `.scratch/probe_batch_align.py`。
        """
        questions = len(lengths)
        longest = max(lengths)
        slotw = -(-int(longest) // AT_BM) * AT_BM
        if (questions < 2 or longest > self._bucket
                or slotw % AT_BM or self._bucket % slotw):
            return None
        per = self._bucket // slotw
        # `AttnArgs` 只有 8 槽 × 16 bit 的可见行数表（`kernels.attn_args` 对第 9 槽起静默丢弃），
        # 而 `per` 是 `bucket // slotw`：bucket 1024/槽宽 128 恰好 8，bucket 2048 就 16。
        if per > 8:
            return None
        ntiles = -(-questions // per)
        nt = ntiles * self._bucket
        if prefix + nt > self._width:
            return None
        return SlotLayout(slotw=slotw, per=per, ntiles=ntiles, nt=nt,
                          starts=tuple(prefix + j * slotw for j in range(questions)))

    def _run_shared_batched(self, prefix_rows: np.ndarray, tails: Sequence[np.ndarray], *,
                            prefix: int, layout: SlotLayout,
                            causal: bool = False) -> list[np.ndarray]:
        """批尾巴：N 条尾巴一人一格排进打包区，整条请求只跑一个尾巴相。

        尾巴的隐藏态**不用 scratch**：每格占的是自己的列，块与块之间原样留在画布上（逐题
        那条路要 scratch 是因为所有题共用同一段列、互相覆盖）。
        """
        counts = [int(np.asarray(tail).shape[0]) for tail in tails]
        view = self.x.array(np.float16, (self._hidden, self._width))
        self._write_rows(view, prefix_rows, 0)
        # 打包区先清一遍：padding 列不是任何一题的尾巴，但它们的 K/V 照样会被算（掩码看不见
        # 它们），留上一趟的残值只会让这趟的中间量没有定义。
        view[:, prefix:prefix + layout.nt] = 0
        for index, tail in enumerate(tails):
            self._write_rows(view, tail, layout.starts[index])
        prefix_chunks = list(self._chunks(prefix))
        # 打包区的瓦片**从 prefix 自己起切**（不要求前缀桶对齐）：槽宽整除桶宽，所以槽位
        # 边界天然落在瓦片边界上，一个槽不会跨瓦片。
        tail_chunks = list(self._chunks_from(prefix, prefix + layout.nt))
        result: list[np.ndarray | None] = [None] * len(tails)
        groups = self._blocks()
        try:
            for index, block_of_layers in enumerate(groups):
                if self._residency is not None:
                    self._residency.begin(index)
                self._chunks_list = list(prefix_chunks)
                self._prepare(prefix, prefix, fresh=True, causal=causal)
                self._run_block(block_of_layers)
                self._wait_all()
                snapshot = self._snapshot(block_of_layers)
                self._slot_snapshot = snapshot
                self._chunks_list = list(tail_chunks)
                self._prepare(prefix + layout.nt, prefix, fresh=False,
                              segw=layout.slotw, counts=counts, causal=causal)
                self._run_block(block_of_layers)
                self._wait_all()
                self._slot_snapshot = None
                if index == len(groups) - 1:
                    for question, count in enumerate(counts):
                        result[question] = self.last_hidden(layout.starts[question] + count)
                if self._residency is not None:
                    self._residency.end(index)
            if self._residency is not None:
                self._residency.finish()
        finally:
            self._slot_snapshot = None
        if any(row is None for row in result):
            raise NotReadyError("共享前缀漏读了某题的读出", questions=len(tails))
        return [row for row in result if row is not None]

    def _slot_table(self, tile: int) -> tuple[int, ...]:
        """这一瓦片里各槽位的可行行数（按片内相对槽位排，表长 = `per`）。"""
        per = self._slot_per
        table = []
        for index in range(per):
            question = tile * per + index
            table.append(self._slot_counts[question] if question < len(self._slot_counts) else 0)
        return tuple(table)

    def _tile_slots(self, tile: int) -> list[tuple[int, int]]:
        """这一瓦片里真正有尾巴的 (片内列偏移, 行数)；空槽位没有尾巴，跳过。"""
        return [(index * self._segw, count)
                for index, count in enumerate(self._slot_table(tile)) if count > 0]

    def _restore_layer(self, layer: int) -> None:
        """把本块前缀相留下的末状态写回这一层（批尾巴逐槽开跑前用）。"""
        snapshot = self._slot_snapshot or {}
        state, tail = snapshot.get(layer, (None, None))
        if state is None or tail is None:
            raise NotReadyError("批尾巴缺少前缀末状态", layer=layer)
        self.states[layer].array(np.uint8)[:] = state
        self.tails[layer].array(np.uint8)[:] = tail


    # ---- 相 / 块 -----------------------------------------------------------

    def _blocks(self) -> tuple[tuple[int, ...], ...]:
        return blocks([entry.role for entry in self.buffers])

    def _chunks_from(self, start: int, end: int) -> Iterator[tuple[int, int]]:
        """从 `start` 起把 `[start, end)` 切成不超过桶宽的段：尾巴的列不要求桶对齐。"""
        position = int(start)
        while position < int(end):
            size = min(self._bucket, int(end) - position)
            yield position, size
            position += size

    def _prepare(self, length: int, prefix: int, *, fresh: bool,
                 segw: int = 0, counts: Sequence[int] = (), causal: bool = False) -> None:
        """开一个执行相：掩码模型、S3 的死瓦片，以及各条链的游标账。

        `fresh=False` 表示这一相接着上一相的状态跑（共享前缀的尾巴相）：GDN 的递归状态与
        卷积尾巴由调用方先写回，块开头的清零要跳过（见 `_run_block`）。
        """
        self._length = int(length)
        self._prefix = int(prefix)
        self._fresh = bool(fresh)
        # M3：这一相的掩码是不是三角。它与 `segw` 正交：批尾巴的槽位规则之上再按行收到自己
        # 那一列（见 `kernels.attn_args` 的 `caus`）。
        self._causal = bool(causal)
        # 批尾巴（票据 10）：这一相里每条尾巴占一个槽位，槽宽 `segw`、各槽位行数 `counts`。
        # `segw = 0` 是逐题整趟/前缀相，所有环节按老规则走。
        self._segw = int(segw)
        self._slot_counts = tuple(int(count) for count in counts)
        self._slot_per = (self._bucket // int(segw)) if segw else 0
        # S3（旧引擎的 `LASTDEAD`，`docs/ANE/anebench/e8_a3.py:291-302`）：模型最后一层里整段落在
        # 公共前缀中的瓦片只欠 K/V —— 前缀行的注意力输出与 MLP 输出都没有读者（读出在 query
        # 行上，K/V 出自投影那一步，不依赖注意力）。所以这些瓦片丢掉第二相（`attn` 与它之后的
        # `out`/`norm2`/`ffn`/`tail`）。
        #
        # 最后一个瓦片永远不丢：`prefix == length` 时读出那一行自己也在前缀里，它就变成读者了。
        dead = 0
        if self._skip_dead and self._prefix > 0 and self._last_is_full:
            dead = sum(1 for start, n in self._chunks_list if start + n <= self._prefix)
            dead = min(dead, len(self._chunks_list) - 1)
        self._dead = dead
        # 每条 lane 的命令缓冲账：本相提交了几个、其中前几个已经结算。
        self._committed = {LANE_LONG: 0, LANE_SMALL: 0}
        self._completed = {LANE_LONG: 0, LANE_SMALL: 0}
        # 哪一层哪些瓦片的 `h` 已经写好（= 上一层的 `tail` 编码过）。相的边界上清空：
        # 尾巴相的 `norm` 只认本相写过的 `h`，前缀相的瓦片号不能替它作证。
        self._h: dict[int, set[int]] = {index: set() for index in range(len(self.buffers))}
        self._preps: dict[int, bool] = {}
        self._prev: dict[int, int | None] = {}
        # 每层本相已经编码了几个 gather（两相屏障要的是**两条奇偶链**的合计）。
        self._gathers: dict[int, int] = {}
        # 每层最后编码的那个 `core` 瓦片（两条奇偶链共用递归状态与卷积尾巴，只能按序）。
        self._core_enc: dict[int, int] = {}

    def _snapshot(self, block: Sequence[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """把本块各 GDN 层的前缀末状态抄到 host（尾巴相要按题回放）。"""
        snapshot: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for index in block:
            if self.buffers[index].role is AttentionRole.FULL:
                continue
            snapshot[index] = (np.array(self.states[index].array(np.uint8)),
                               np.array(self.tails[index].array(np.uint8)))
        return snapshot

    def _restore(self, block: Sequence[int], snapshot: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """尾巴相开始前把前缀末状态写回；只在块排空之后调用（不会与在飞的工作抢内存）。"""
        for index in block:
            state, tail = snapshot.get(index, (None, None))
            if state is None:
                continue
            self.states[index].array(np.uint8)[:] = state
            self.tails[index].array(np.uint8)[:] = tail

    def _ensure_tail_hidden(self, questions: int, longest: int) -> None:
        """按 (题数, 最长尾巴) 备一块尾巴隐藏态的 scratch（不复用别的形状）。"""
        columns = int(questions) * int(longest)
        self._free_tail_hidden()
        self._tail_hidden = self._runtime.alloc(2 * self._hidden * columns)
        self._tail_cols = columns

    def _free_tail_hidden(self) -> None:
        if self._tail_hidden is not None:
            self._tail_hidden.free()
            self._tail_hidden = None
        self._tail_cols = 0

    # ---- 命令缓冲账 ---------------------------------------------------------

    def _commit(self, chain: Chain, lane: int, work: str, tile: int) -> None:
        """把刚编码完的那一步提交成一个命令缓冲，并记下它的序号。"""
        self._runtime.commit(lane)
        self._committed[lane] += 1
        chain.cbs[(work, tile)] = (lane, self._committed[lane])

    def _need(self, lane: int, index: int) -> None:
        """等到这条 lane 上第 `index` 个命令缓冲结算。

        等整条 lane（`wait`）会把那边还在跑的活也拖成串行 —— 那些活可能正好是下几层的、
        与本步无关的；`wait_upto` 只结算队首这么多个。执行期错误只在 `wait_upto` 碰到的
        缓冲上才会被记进错误串（`e8_mtl.m:403-405`），所以每次等到都要查一次。
        """
        pending = index - self._completed[lane]
        if pending <= 0:
            return
        ms = self._runtime.wait_upto(lane, pending)
        self._completed[lane] = index
        if ms < 0:
            raise NotReadyError("GPU 命令缓冲执行失败", lane=lane, index=index,
                                reason=self._runtime.error())

    def _wait_for(self, chain: Chain, work: str, tile: int) -> None:
        """等到这条链这一步写出的那个缓冲结算完（不是整条 lane）。"""
        dep = chain.cbs.get((work, tile))
        if dep is not None:
            self._need(*dep)

    def _wait_all(self) -> None:
        """两条 lane 都等到队尾（块尾与整链收尾用）。"""
        for lane in (LANE_LONG, LANE_SMALL):
            self._need(lane, self._committed[lane])

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
        lane: int = LANE_SMALL,
    ) -> None:
        """跨宽缓冲的通道带搬运：`n=0` 表示整宽。"""
        kernels.copy_columns(
            self._runtime, src, dst,
            nchan=nchan, src_c0=src_c0, dst_c0=dst_c0,
            stile=n or self._width, s0=s0,
            src_stride=src_stride or self._width,
            dst_stride=dst_stride or self._width,
            src_s0=src_s0,
            lane=lane,
        )

    # ---- block / 链 / 步骤 --------------------------------------------------

    def _run_block(self, block: Sequence[int]) -> None:
        """跑一个 block：块内各层的链按步骤图互相推进，块尾由 `run` 整段排空。

        `blocks()` 在每个全注意力层之前切开，所以一个块里至多一个全注意力层、且它在第 0 位；
        那个注意力窗口里填的就是本块后面那几个线性层的活（旧引擎的 block 形状，
        `e8_a3.py:2235-2253`）。

        GDN 的递归状态与卷积尾巴是**逐层独立**的，在块开头清零。块之间已经排空，所以这个
        CPU 写不会与任何在飞的 GPU/ANE 工作抢同一张缓冲。
        """
        ntiles = len(self._chunks_list)
        last = len(self.buffers) - 1
        for position, index in enumerate(block):
            self._prev[index] = block[position - 1] if position else None
            entry = self.buffers[index]
            if entry.role is AttentionRole.FULL:
                continue
            if not self._fresh:
                # 共享前缀的尾巴相：状态是 `_restore` 写回的前缀末状态，不能清
                continue
            self.states[index].array(
                np.float32,
                (self._geometry.l_v_heads, self._geometry.l_k_dim, self._geometry.l_v_dim),
            )[:] = 0
        chains: list[Chain] = []
        for position, index in enumerate(block):
            entry = self.buffers[index]
            dead = self._dead if index == last else 0
            for par in range(self._nch):
                steps = (
                    attention_steps(ntiles, par, dead=dead, nch=self._nch)
                    if entry.role is AttentionRole.FULL
                    else gdn_steps(ntiles, par, nch=self._nch)
                )
                slot = self._nch * position + par
                chains.append(Chain(layer=index, steps=steps, sets=(slot,), slot=slot, par=par))
        while True:
            for chain in chains:
                step = chain.cur()
                if step is None or not self._ready(chain, step):
                    continue
                self._issue(chain, step)
                chain.advance()
            if all(chain.done for chain in chains):
                return

    def _ready(self, chain: Chain, step: Step) -> bool:
        """这一步能不能现在编码。判据只有"会不会读到别人还没写好的东西"：

        - `ane`：整台 ANE 只有一条固件队列、一次一个 eval 在飞，所以前一次没回收就不能发射；
        - `norm`：要读上一层同一瓦片的 `h`，而 `h` 是上一层 `tail` 那一步写的；
        - `core`：本层的递归状态与卷积尾巴是两条奇偶链共用的，所以瓦片 `s` 的 core 必须排在
          瓦片 `s-1` 的 core **编码之后**（lane 1 一条有序队列，编码序即执行序）；
        - `proj`（只出现在线性层）：用**和 `core` 同一个瓦片序 gate**。`proj` 一发射就占住整台
          ANE，而只有本链的 `core(t)` 会回收它 —— 若让 `proj(t)` 跑到 `core(t-1)` 前面，就会
          出现"一条链握着 ANE 等另一条链的 `core(t-1)`"、而另一条握着自己的 `proj` 等 ANE 的环
          （票据 06 实测：4 瓦片、桶 64 时死锁；A1 之前没触发只是步数少、跑得没那么前）。共用
          一个 gate 没有额外代价：ANE 本来就一次只有一个 eval 在飞。
        - `prep_kv`：全序列屏障 —— 本层所有瓦片的 `gather_kv`（两条链的合计）都得先编码完；
        - `attn`：得先有 `prep_kv` 编码。`prep_q` 不需要屏障：它只碰自己那条链的环槽位，
          而 `gather_q → prep_q → attn` 三步在同一条 lane 的编码序上连着排（票据 06）。

        其余依赖（`proj` 要读 `norm` 写的 `proj_in`、`out` 要读注意力 / `core` 写的那半……）
        不是"能不能编码"而是"数据在哪"，在 `_issue` 里用 `_wait_for` 精确等到那一个命令缓冲。
        """
        work = step.work
        if step.kind == "ane" and self.buffers[chain.layer].ane.pending:
            return False
        if work == "norm":
            prev = self._prev.get(chain.layer)
            return prev is None or step.tile in self._h[prev]
        if work in ("core", "proj"):
            # `proj` 只在 GDN 层的步表里；`core` 只在 GDN 层。全注意力层的 `proj_kv`/`proj_q`
            # 不在这里：它们的回收步（`gather_kv`/`gather_q`）紧跟在本链下一步、不被任何门挡住。
            return self._core_enc.get(chain.layer, -1) >= step.tile - 1
        if work == "prep_kv":
            return self._gathers.get(chain.layer, 0) >= len(self._chunks_list)
        if work == "attn":
            return self._preps.get(chain.layer, False)
        return True

    def _issue(self, chain: Chain, step: Step) -> None:
        """编码/发射一步。**不阻塞在 ANE 上**：eval 发射后立刻返回，谁消费输出谁回收。"""
        entry = self.buffers[chain.layer]
        work, tile = step.work, step.tile
        if work == "norm":
            self._step_norm(chain, entry, tile)
        elif work in ("proj", "proj_kv", "proj_q"):
            self._wait_for(chain, "norm", tile)
            # 同一槽位的上一片 `gather` 还在读这张面（lane 0），而 ANE 的写不在任何一条 lane 上：
            # 不显式等它，投影的写就会与那一片的读重叠 —— 首趟拷贝冷页慢，实测块 1（第一个
            # 全注意力层）的输出与后续趟差 2.6e-1（票据 02）。GDN 层没有这个洞：读 `proj_out`
            # 的 gdn 核在 `core` 那个 CB 里，与后来的 `norm2` 同一条 lane，等 `norm2` 等于等它。
            # 拆投影之后三条投影各自盯自己那张面的上一片读（票据 06）：
            #   `proj`    → `proj_out` ← `gather`
            #   `proj_q`  → `proj_out` ← `gather_q`
            #   `proj_kv` → `out_in`   ← `gather_kv`
            self._wait_for(chain, work.replace("proj", "gather"), tile - self._nch)
            entry.ane.issue_proc(work, slot=chain.slot)
        elif work == "gather_kv":
            self._step_gather_kv(chain, entry, tile)
        elif work == "gather_q":
            self._step_gather_q(chain, entry, tile)
        elif work == "prep_kv":
            self._prep_kv(entry)
            self._preps[chain.layer] = True
            self._commit(chain, LANE_LONG, "prep_kv", -1)
        elif work == "prep_q":
            self._step_prep_q(chain, entry, tile)
        elif work == "attn":
            self._step_attn(chain, entry, tile)
        elif work == "core":
            self._step_core(chain, entry, tile)
        elif work == "out":
            self._wait_for(chain, "attn" if entry.role is AttentionRole.FULL else "core", tile)
            if entry.role is AttentionRole.FULL:
                self._step_out_in(chain, entry, tile)
            entry.ane.issue_proc("out", slot=chain.slot)
        elif work == "norm2":
            self._step_norm2(chain, entry, tile)
        elif work == "ffn":
            self._wait_for(chain, "norm2", tile)
            entry.ane.issue_proc("ffn", slot=chain.slot)
        elif work == "tail":
            self._step_tail(chain, entry, tile)
        else:
            raise ConfigError("未知的 step", work=work, layer=chain.layer)

    # ---- 每一步 -------------------------------------------------------------

    def _step_norm(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`norm`：输入归一化写进 `proj_in`（ANE 投影那一段的入口）。"""
        start, n = self._chunks_list[tile]
        kernels.norm_to(
            self._runtime, self.x, entry.weights["input_norm"],
            entry.surface("proj_in", chain.slot),
            stile=n, src_stride=self._width, eps=EPS, s0=start,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._commit(chain, LANE_SMALL, "norm", tile)

    def _step_gather_kv(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`gather_kv`：把 `proj_kv` 写在 `out_in` 前 `kv_rows` 行的那一段搬进整窗的 `kv`。

        lane 0 是因为 `prep_kv` 也在 lane 0 —— 两者之间不需要跨 lane 的同步，编码序即执行序
        （旧引擎的 `gather` 同样在长 lane 上）。
        """
        entry.ane.reap_proc()                    # proj_kv 的输出就在 out_in 的前 kv_rows 行
        start, n = self._chunks_list[tile]
        self._copy(entry.surface("out_in", chain.slot), self.kv,
                   nchan=self._geometry.kv_rows, n=n, s0=start,
                   src_stride=self._bucket, dst_stride=self._width, lane=LANE_LONG)
        self._commit(chain, LANE_LONG, "gather_kv", tile)
        self._gathers[chain.layer] = self._gathers.get(chain.layer, 0) + 1

    def _step_gather_q(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`gather_q`：把 `proj_q` 的 q|gate 半搬进 `qg` 的环槽位（`ring_slot`）。

        目标是环槽位而不是整窗列，所以 `dst_stride` 是环的列数（`self._ring`）；源仍是桶宽的
        面，从列 0 开始读。
        """
        entry.ane.reap_proc()                    # proj_q 的输出就在 proj_out 上
        start, n = self._chunks_list[tile]
        slot = ring_slot(tile, self._bucket)
        self._copy(entry.surface("proj_out", chain.slot), self.qg,
                   nchan=self._geometry.qg_rows, n=n, s0=slot,
                   src_stride=self._bucket, dst_stride=self._ring, lane=LANE_LONG)
        self._commit(chain, LANE_LONG, "gather_q", tile)

    def _step_attn(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`attn`：这一段 query 的注意力，**直写**进本槽位 `out_in` 的前 `core_out` 行
        （旧引擎的 `ia`）。

        直写省掉"先落一条全宽的注意力结果、再搬进面里"那一次搬运 —— 那半区本来就是 `out`
        那一段 ANE 要读的 `[v|x]` 的 v 半，搬到别处再搬回来是白搬（16k 每瓦片 16 MB）。
        """
        start, n = self._chunks_list[tile]
        self._attention_tile(entry, chain.slot, (start, n), self.qg, self.kv,
                             ring_slot(tile, self._bucket), tile=tile)
        self._commit(chain, LANE_LONG, "attn", tile)

    def _step_out_in(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """注意力层的 `out` 入口：只搬 **x 半**（这一层的输入残差，从 `core_out` 行起），
        v 半是注意力直写进去的。

        必须在注意力的命令缓冲结算之后才能**编码**（同一张面里读它刚写的东西不安全，
        见 `schedule.py` 的头注释）；两半在互不相干的通道带上，所以这片搬运与注意力之间
        没有真的数据依赖，等的只是缓冲边界。
        """
        start, n = self._chunks_list[tile]
        self._copy(self.x, entry.surface("out_in", chain.slot), nchan=self._hidden,
                   dst_c0=self._core_out, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        self._commit(chain, LANE_SMALL, "out_in", tile)
        self._wait_for(chain, "out_in", tile)      # ANE 从面上读，必须已经结算

    def _step_core(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`core`：GDN 的四段快核，加上这一瓦片的残差搬运（写进 `out_in` 的 x 半）。

        `first` 只对瓦片 0 为真：递归状态与卷积尾巴在这个块开头已经清零，瓦片 0 没有前驱列。
        """
        start, n = self._chunks_list[tile]
        state, tail = self.states[chain.layer], self.tails[chain.layer]
        entry.ane.reap_proc()                    # 投影的输出就在 proj_out 上
        self._copy(self.x, entry.surface("out_in", chain.slot), nchan=self._hidden,
                   dst_c0=self._core_out, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        if self._segw:
            # 批尾巴（票据 10）：一个瓦片里有好几个槽位，每槽一段列、一份递归状态、一份
            # 卷积尾巴。核的读写都从 `s0` 起，所以先把这一槽的前缀末状态写回，再按该槽的行数
            # 扫自己那段。写状态前必须等上一槽的 gdn 结算（Metal 的编码序管不到主机写）。
            for lo, cnt in self._tile_slots(tile):
                self._runtime.wait(LANE_SMALL)
                self._restore_layer(chain.layer)
                kernels.gdn(
                    self._runtime, entry.surface("proj_out", chain.slot), state, tail,
                    entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
                    entry.weights["norm"], entry.surface("out_in", chain.slot), self.gscratch,
                    stile=self._bucket, cols=cnt, eps=EPS, first=False,
                    T=self._bucket, out_stride=self._bucket, src_s0=lo, out_s0=lo,
                    fast=True, geometry=self._geometry,
                )
        else:
            kernels.gdn(
                self._runtime, entry.surface("proj_out", chain.slot), state, tail,
                entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
                entry.weights["norm"], entry.surface("out_in", chain.slot), self.gscratch,
                # GDN 核里的 `Stile` 既是列界也是**通道步长**（`proj[c*Stile + j]`），所以必须
                # 传桶宽而不是本段的有效列数；`T` 是它不看的多余字段。
                # `first` 只在"新序列的第一个瓦片"上为真：共享前缀的尾巴相沿用前缀的卷积尾巴。
                # `cols=n`：部分瓦片的卷积尾巴要取自最后 3 个**真实**列，而不是桶尾的 padding。
                stile=self._bucket, cols=n, eps=EPS, first=(tile == 0 and self._fresh),
                T=self._bucket, out_stride=self._bucket, fast=True, geometry=self._geometry,
            )
        self._commit(chain, LANE_SMALL, "core", tile)
        self._core_enc[chain.layer] = tile

    def _step_norm2(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`norm2`：`[rmsnorm(out) | out]` 写进 `out_in`（ANE FFN 那一段的入口）。

        `out_in` 就是 `out` 那一段刚读过的那张面（`SET_KEYS` 的复用）：`reap_proc()` 等它
        回收，既是"读 `out_out` 前要看见 ANE 的写"，也是"覆写 `out_in` 前要等 ANE 读完"。
        """
        start, n = self._chunks_list[tile]
        entry.ane.reap_proc()
        kernels.norm_residual(
            self._runtime, entry.surface("out_out", chain.slot), entry.weights["post_norm"],
            entry.surface("out_in", chain.slot),
            stile=n, src_stride=self._bucket, eps=EPS,
            dst_stride=self._bucket, dst_s0=0,
        )
        self._commit(chain, LANE_SMALL, "norm2", tile)

    def _step_tail(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`tail`：把 `out_out`（FFN 的输出）写回 `x` 的同一列，并把这一瓦片记成"h 已写好"。

        下游层的 `norm` 也在 lane 1、且编码在这之后：编码序即执行序，所以"编码过"就够下游
        读了，这里不必再等这个缓冲结算。
        """
        start, n = self._chunks_list[tile]
        entry.ane.reap_proc()                    # ffn 的输出在 out_out 上
        self._copy(entry.surface("out_out", chain.slot), self.x, nchan=self._hidden,
                   n=n, s0=start, src_stride=self._bucket, dst_stride=self._width)
        self._commit(chain, LANE_SMALL, "tail", tile)
        self._h[chain.layer].add(tile)

    def _prep_kv(self, entry: LayerBuffers) -> None:
        """本相窗口的 k 归一 + 部分 RoPE（旧引擎 `prep` 的 k 半，序列级屏障之后才做）。

        q 不在这里：投影拆开之后 q 只活一个瓦片，归一与 RoPE 挪进了每个瓦片的 `prep_q`
        （见 `_step_prep_q`）—— 这也是 `qg` 能成环的原因（票据 06）。

        窗口是本相的 `_chunks_list` 从第一段首列到末段末列。共享前缀的尾巴相只归一自己那几
        列：前缀列在前缀相已经 prep 过，再来一遍会把 RoPE 叠两次。`kv` 是整窗缓存，列号就是
        位置，所以 `pos0` 恒为 0（`s0` 已经是绝对列号）。
        """
        chunks = self._chunks_list
        start = chunks[0][0]
        end = chunks[-1][0] + chunks[-1][1]
        kernels.kv_prep(self._runtime, self.kv, entry.weights["k_norm"],
                        self._cos, self._sin, stile=end - start, s0=start,
                        src_stride=self._width, eps=EPS, geometry=self._geometry,
                        nka=self._prefix if self._segw else 0, segw=self._segw)

    def _step_prep_q(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`prep_q`：环槽位里的 q 做 RMSNorm+半维 RoPE、gate 做 sigmoid（`k_q_prep`，原地）。

        `qg` 成环之后列号不是 rope 位置：槽位从 `ring_slot(tile)` 起，而这一瓦片第一个 token
        在整条序列里的位置是 `start`，所以 `pos0 = start - 槽位`。同一槽位每 `nch` 个瓦片被
        重写一次，而写之前上一次的 `attn` 已经在这条 lane 上执行完（编码序即执行序）。
        """
        start, n = self._chunks_list[tile]
        slot = ring_slot(tile, self._bucket)
        kernels.q_prep(self._runtime, self.qg, entry.weights["q_norm"],
                       self._cos, self._sin, stile=n, s0=slot, src_stride=self._ring,
                       eps=EPS, pos0=start - slot, geometry=self._geometry,
                       nka=self._prefix if self._segw else 0, segw=self._segw)
        self._commit(chain, LANE_LONG, "prep_q", tile)

    def _attention_tile(
        self, entry: LayerBuffers, slot: int, chunk: tuple[int, int],
        qg: MetalBuffer, kv: MetalBuffer, qslot: int, *, tile: int = 0,
    ) -> None:
        """一段 query 的注意力 dispatch；`prefix_noncausal` 的边界若落在这段内就拆两次。

        段内的 query 只受 `nk` 一个上界约束，所以"按段发射"与"整条一次发射"在数值上等价 ——
        `test_metal_chain.py` 的 `prefix ∈ {0,4}` 用例守住这一点。

        两套列号是分开的：`qg` 按环槽位读（`s0 = qslot`、`qgT = self._ring`），`kv` 仍按整窗
        缓存读（`T = self._width`、`nk` 是可见上界）。`k_attn` 早就把这两件事拆成了两个字段。

        输出写进本槽位 `out_in` 的前 `core_out` 行：面里只有桶宽列，所以 `out_s0` 是**段内**
        偏移（拆成
        两次时第二次从 `boundary - start` 列接着写），`out_stride` 必须显式传桶宽
        （传 0 的话核会退回 `Stile`，而那时 `Stile` 只是这一段的有效列数）。
        """
        start, n = chunk
        end = start + n
        out = entry.surface("out_in", slot)
        # 批尾巴（票据 10）：query 行按槽位只看 `[0, prefix)` + 自己那 `cnt_j` 行。`qseq` 是这
        # 一调用第一行 query 的**序列列号**（qg 成环后 `s0` 只是环槽位，不是位置）。
        rule: dict = {}
        if self._segw:
            rule = dict(nka=self._prefix, segw=self._segw, nkw_slots=self._slot_table(tile))
        # M3：causal 时这一段的可见上界就是它自己的末列（末行的位置 + 1）。双向时是全长。
        # `caus` 交给核里按行再收 —— 两件事分开：这个上界省掉整块看不见的 key 块，按行的项
        # 才给出三角。共享前缀的两刀同理：前一半的上界就是 `prefix`（= `boundary`）。
        caus = 1 if self._causal else 0
        top = end if self._causal else self._length
        if self._prefix <= 0 or self._prefix >= self._length:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=n, nk=top, nq=n, s0=qslot, qgT=self._ring,
                out_stride=self._bucket, out_s0=0, fast=True, geometry=self._geometry,
                qseq=start, caus=caus, **rule,
            )
            return
        # 边界钳在本段内：前缀整段落在本段之前时（`prefix <= start`）这整段都是 query 侧，
        # `nk` 取全长；原先把 `s0` 退回 `prefix` 会在邻段上重复算一遍同样的列。
        boundary = min(end, max(start, self._prefix))
        if start < boundary:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=boundary - start, nk=self._prefix, nq=boundary - start,
                s0=qslot, qgT=self._ring, out_stride=self._bucket, out_s0=0, fast=True,
                geometry=self._geometry, qseq=start, caus=caus, **rule,
            )
        if boundary < end:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=end - boundary, nk=top, nq=end - boundary,
                s0=qslot + (boundary - start), qgT=self._ring, out_stride=self._bucket,
                out_s0=boundary - start, fast=True, geometry=self._geometry,
                qseq=boundary, caus=caus, **rule,
            )

    # ------------------------------------------------------------------ 收尾

    def set_rope(self, cos: np.ndarray, sin: np.ndarray) -> None:
        """把本请求的 RoPE 表盖到前 `cos.shape[0]` 行（`[width, rotary_dim/2]` 的前缀）。

        带图的题面三轴位置不同（`models/vision/merge.mrope_positions`），而核是从缓冲里按列号
        读表的，所以表必须逐请求更新。只盖用得到的那几行是够的（核按 `pos < length` 取址），
        但**盖过之后要负责盖回去**：这张表是缓冲，下一次纯文本请求若沿用就会拿图像位置算文本
        token（实测会让文本 logits 变）。调用方（`SplitChain._prepare_rope`）用 `_rope_dirty`
        记着写过多少行，恢复时盖上同样的行数。
        """
        rows = int(cos.shape[0])
        if rows > self._rope_rows or cos.shape[1] != self._rope_half:
            raise ValueError(
                f"RoPE 表形状 {tuple(cos.shape)} 与流水线的 "
                f"({self._rope_rows}, {self._rope_half}) 不符")
        self._cos.array(np.float32, (self._rope_rows, self._rope_half))[:rows] = np.asarray(
            cos, np.float32)
        self._sin.array(np.float32, (self._rope_rows, self._rope_half))[:rows] = np.asarray(
            sin, np.float32)

    def last_hidden(self, length: int) -> np.ndarray:
        """最后一层的最后一条 token（`[hidden]` fp32），供 norm + 读出。"""
        view = self._last.array(np.float16, (self._hidden, self._width))
        return np.asarray(view[:, length - 1], np.float32)

    def hidden_rows(self, length: int) -> np.ndarray:
        """最后一层的**全部** `length` 个位置（`[length, hidden]` fp32）。

        `_last` 是整张 `[hidden, width]` 画布，所以整段都在，只是平时只取末位。逐位置对拍
        （numpy 链 vs Metal 链）与逐层 trace 需要这一份；numpy 那条路本来给的就是整段。
        """
        view = self._last.array(np.float16, (self._hidden, self._width))
        return np.asarray(view[:, :length].T, np.float32)

    def close(self) -> None:
        residency, self._residency = getattr(self, "_residency", None), None
        if residency is not None:
            residency.finish()          # 先收尾装卸线程：它们可能正在重绑下面的面
        for entry in self.buffers:
            entry.free()
        self.buffers.clear()
        # 解绑池里的四张面：必须在它们的所有者 `SurfaceSet.free()` 之前（`e8m_wrap` 的缓存
        # 按 IOSurface 指针键，地址会被下一次分配复用）。
        for wrapped in self._wrapped:
            for buffer in wrapped.values():
                buffer.free()
            wrapped.clear()
        self._wrapped.clear()
        self._free_tail_hidden()
        for name in ("x", "qg", "kv", "gscratch", "attn_scratch", "_cos", "_sin"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.free()
        for name in ("states", "tails"):
            for buffer in getattr(self, name, None) or ():
                buffer.free()
