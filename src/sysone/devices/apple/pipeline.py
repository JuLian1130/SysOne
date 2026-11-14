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
from .metal import LANE_LONG, LANE_SMALL, MetalBuffer, MetalRuntime
from .residency import Residency
from .schedule import NCH, Chain, Step, attention_steps, blocks, gdn_steps

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
        self._geometry = mude_geometry() if geometry is None else geometry
        self._geometry.validate()
        self._hidden = self._geometry.hid
        self._bucket = int(bucket)
        self._width = int(width)
        if self._width < self._bucket:
            raise ConfigError("傍宽不能小于桶宽", width=width, bucket=bucket)
        if not pool:
            raise ConfigError("表面池至少要有一套面")
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
            self._alloc_shared()
            # 每一列往下游推进的层数预算（对应旧引擎的 `RING = 8 // NCH`）：它是"注意力窗口里
            # 填多少 ANE 活"的旋钮 —— 取 4 时一列能推 3 个线性层 + 下一个全注意力层的投影，
            # 恰好 ≈ 一个 16k 注意力瓦片的 135 ms。
            # 每层拆成几条瓦片链。第 2 步一条，第 3 步切成两条（`NCH`）。
            self._nch = CHAINS
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
        # 注意力输出**没有**单独的缓冲：那一片直接写进本槽位的 `out_in` 前半（旧引擎的
        # `ia = w_io_a[k]`），后半由残差搬运填上（`_step_out_in` / `_step_core`）。
        # **一份** qg / 一份 kv 就够：`blocks()` 在每个全注意力层之前切开、`run` 在块之间整段
        # 排空，所以同一时刻只有一个全注意力层碰这两张画布（16k 的活性账：同时"开着"的层恒为
        # 1）。两套是 27c6250 的"列波前"留下的 —— 那一版块间不排空，相邻两个全注意力层真的
        # 会同时在飞。
        self.qg = rt.alloc(2 * geom.qg_rows * width)
        self.kv = rt.alloc(2 * geom.kv_rows * width)
        self.gscratch = rt.alloc(2 * 8192 * bucket)
        # 快核不读 scratch，但 kernel 声明了 buffer(3)，必须绑定。全宽的参考核才要 (T,T)。
        self.attn_scratch = rt.alloc(32)
        # GDN 的递归状态与卷积尾巴**逐层独立**：波前里两层线性层的列段会交错执行，
        # 共用一张状态会让后一层的清零把前一层的递推冲掉（每层 1 MB + 尾巴几十 KB）。
        self.states = [rt.alloc(4 * geom.state_floats) for _ in self.buffers]
        self.tails = [rt.alloc(2 * geom.conv_dim * 3) for _ in self.buffers]
        self._last: MetalBuffer = self.x

    # ------------------------------------------------------------------ 执行

    def _chunks(self, length: int) -> Iterator[tuple[int, int]]:
        """把 `length` 切成不超过桶宽的段，返回 `(起始列, 列数)`。"""
        step = self._bucket
        for start in range(0, length, step):
            yield start, min(step, length - start)

    def run(self, embed: np.ndarray, *, length: int, prefix: int = 0) -> None:
        """跑完 20 层。`embed` 是 `[length, hidden]` 的 embedding 输出（fp32）。

        **步骤图驱动**：一层被拆成 `schedule.py` 的订单式 `(chain, step)` 序列 ——
        `norm → proj → core/gather → out → norm2 → ffn → tail`，一步一个命令缓冲，
        提交/回收的账记在每条 lane 上，靠 dylib 的 `wait_upto` 精确等到"写这一步输入的那个
        缓冲"，而不是整条 lane。重叠因此来自"CPU 在等 ANE 之前已经把另一侧的活排满"。

        每层拆成 `NCH` 条瓦片链（`schedule.py` 的 `gdn_steps` / `attention_steps`）；`block`
        之间整段排空，因为表面池与 qg/kv 画布是跨层复用的。

        `prefix` 非 0 时还会省掉最后一层的死瓦片（S3，见下面 `dead` 的注释）。
        """
        if not 0 <= prefix <= length <= self._width:
            raise ConfigError("长度或 prefix 越界", length=length, prefix=prefix, width=self._width)
        view = self.x.array(np.float16, (self._hidden, self._width))
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
        # 每条 lane 的命令缓冲账：本轮提交了几个、其中前几个已经结算。
        self._committed = {LANE_LONG: 0, LANE_SMALL: 0}
        self._completed = {LANE_LONG: 0, LANE_SMALL: 0}
        # 哪一层哪些瓦片的 `h` 已经写好（= 上一层的 `tail` 编码过）。
        self._h: dict[int, set[int]] = {index: set() for index in range(len(self.buffers))}
        self._preps: dict[int, bool] = {}
        self._prev: dict[int, int | None] = {}
        # 每层已经编码了几个 gather（两相屏障要的是**两条奇偶链**的合计）。
        self._gathers: dict[int, int] = {}
        # 每层最后编码的那个 `core` 瓦片（两条奇偶链共用递归状态与卷积尾巴，只能按序）。
        self._core_enc: dict[int, int] = {}
        roles = [entry.role for entry in self.buffers]
        for index, block_of_layers in enumerate(blocks(roles)):
            if self._residency is not None:
                self._residency.begin(index)
            self._run_block(block_of_layers)
            self._wait_all()        # 块之间排空：表面池与 qg/kv 画布是跨层复用的
            if self._residency is not None:
                # 排空之后才卸：卸掉的层在这一块里刚刚用完，下一次需要它到下一轮的块
                # （`blocks()` 的块按层号单调，窗口只往后滚）。
                self._residency.end(index)
        if self._residency is not None:
            self._residency.finish()   # 一轮结束落在待命窗口上（服务契约"回到前 W 层"）

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
        - `prep`：全序列屏障 —— 本层所有瓦片（两条链的合计）的第一相都得先编码完；
        - `attn`：得先有 `prep` 编码。

        其余依赖（`proj` 要读 `norm` 写的 `proj_in`、`out` 要读注意力 / `core` 写的那半……）
        不是"能不能编码"而是"数据在哪"，在 `_issue` 里用 `_wait_for` 精确等到那一个命令缓冲。
        """
        work = step.work
        if step.kind == "ane":
            return not self.buffers[chain.layer].ane.pending
        if work == "norm":
            prev = self._prev.get(chain.layer)
            return prev is None or step.tile in self._h[prev]
        if work == "core":
            return self._core_enc.get(chain.layer, -1) >= step.tile - 1
        if work == "prep":
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
        elif work in ("proj1", "proj"):
            self._wait_for(chain, "norm", tile)
            # 同一槽位的上一片 `gather` 还在读这张 `proj_out`（lane 0），而 ANE 的写不在任何一条
            # lane 上：不显式等它，`proj` 的写就会与那一片的读重叠 —— 首趟拷贝冷页慢，实测
            # 块 1（第一个全注意力层）的输出与后续趟差 2.6e-1。GDN 层没有这个洞：读 `proj_out`
            # 的 gdn 核在 `core` 那个 CB 里，与后来的 `norm2` 同一条 lane，等 `norm2` 等于等它。
            self._wait_for(chain, "gather", tile - self._nch)
            entry.ane.issue_proc("proj", slot=chain.slot)
        elif work == "gather":
            self._step_gather(chain, entry, tile)
        elif work == "prep":
            self._prep(entry)
            self._preps[chain.layer] = True
            self._commit(chain, LANE_LONG, "prep", -1)
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

    def _step_gather(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`gather`：把 `proj_out` 的两段搬进全宽的 `qg` / `kv`，走 lane 0。

        lane 0 是因为 `prep` 也在 lane 0 —— 两者之间不需要跨 lane 的同步，编码序即执行序
        （旧引擎的 `gather` 同样在长 lane 上）。
        """
        entry.ane.reap_proc()                    # 投影的输出就在 proj_out 上
        start, n = self._chunks_list[tile]
        qg, kv = self.qg, self.kv
        self._copy(entry.surface("proj_out", chain.slot), qg, nchan=self._geometry.qg_rows, n=n, s0=start,
                   src_stride=self._bucket, dst_stride=self._width, lane=LANE_LONG)
        self._copy(entry.surface("proj_out", chain.slot), kv, nchan=self._geometry.kv_rows,
                   src_c0=self._geometry.qg_rows, n=n, s0=start,
                   src_stride=self._bucket, dst_stride=self._width, lane=LANE_LONG)
        self._commit(chain, LANE_LONG, "gather", tile)
        self._gathers[chain.layer] = self._gathers.get(chain.layer, 0) + 1

    def _step_attn(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`attn`：这一段 query 的注意力，**直写**进本槽位 `out_in` 的前半（旧引擎的 `ia`）。

        直写省掉"先落一条全宽的注意力结果、再搬进面里"那一次搬运 —— 那半区本来就是 `out`
        那一段 ANE 要读的 `[v|x]` 的 v 半，搬到别处再搬回来是白搬（16k 每瓦片 16 MB）。
        """
        start, n = self._chunks_list[tile]
        qg, kv = self.qg, self.kv
        self._attention_tile(entry, chain.slot, (start, n), qg, kv)
        self._commit(chain, LANE_LONG, "attn", tile)

    def _step_out_in(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """注意力层的 `out` 入口：只搬**后半**（这一层的输入残差），前半是注意力直写进去的。

        必须在注意力的命令缓冲结算之后才能**编码**（同一张面里读它刚写的东西不安全，
        见 `schedule.py` 的头注释）；两半在互不相干的通道带上，所以这片搬运与注意力之间
        没有真的数据依赖，等的只是缓冲边界。
        """
        start, n = self._chunks_list[tile]
        self._copy(self.x, entry.surface("out_in", chain.slot), nchan=self._hidden,
                   dst_c0=self._hidden, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        self._commit(chain, LANE_SMALL, "out_in", tile)
        self._wait_for(chain, "out_in", tile)      # ANE 从面上读，必须已经结算

    def _step_core(self, chain: Chain, entry: LayerBuffers, tile: int) -> None:
        """`core`：GDN 的四段快核，加上这一瓦片的残差搬运（写进 `out_in` 的后半）。

        `first` 只对瓦片 0 为真：递归状态与卷积尾巴在这个块开头已经清零，瓦片 0 没有前驱列。
        """
        start, n = self._chunks_list[tile]
        state, tail = self.states[chain.layer], self.tails[chain.layer]
        entry.ane.reap_proc()                    # 投影的输出就在 proj_out 上
        self._copy(self.x, entry.surface("out_in", chain.slot), nchan=self._hidden,
                   dst_c0=self._hidden, n=n, s0=0, src_s0=start,
                   src_stride=self._width, dst_stride=self._bucket)
        kernels.gdn(
            self._runtime, entry.surface("proj_out", chain.slot), state, tail,
            entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
            entry.weights["norm"], entry.surface("out_in", chain.slot), self.gscratch,
            # GDN 核里的 `Stile` 既是列界也是**通道步长**（`proj[c*Stile + j]`），所以必须
            # 传桶宽而不是本段的有效列数；`T` 是它不看的多余字段。
            stile=self._bucket, eps=EPS, first=tile == 0, T=self._bucket,
            out_stride=self._bucket, fast=True, geometry=self._geometry,
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

    def _prep(self, entry: LayerBuffers) -> None:
        """全宽的 q/k 归一化 + 部分 RoPE（旧引擎的 `prep`，序列级屏障之后才做）。"""
        qg, kv = self.qg, self.kv
        kernels.q_prep(self._runtime, qg, entry.weights["q_norm"],
                       self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                       geometry=self._geometry)
        kernels.kv_prep(self._runtime, kv, entry.weights["k_norm"],
                        self._cos, self._sin, stile=self._width, src_stride=self._width, eps=EPS,
                        geometry=self._geometry)

    def _attention_tile(
        self, entry: LayerBuffers, slot: int, chunk: tuple[int, int],
        qg: MetalBuffer, kv: MetalBuffer,
    ) -> None:
        """一段 query 的注意力 dispatch；`prefix_noncausal` 的边界若落在这段内就拆两次。

        段内的 query 只受 `nk` 一个上界约束，所以"按段发射"与"整条一次发射"在数值上等价 ——
        `test_metal_chain.py` 的 `prefix ∈ {0,4}` 用例守住这一点。

        输出写进本槽位 `out_in` 的前半：面里只有桶宽列，所以 `out_s0` 是**段内**偏移（拆成
        两次时第二次从 `boundary - start` 列接着写），`out_stride` 必须显式传桶宽
        （传 0 的话核会退回 `Stile`，而那时 `Stile` 只是这一段的有效列数）。
        """
        start, n = chunk
        end = start + n
        out = entry.surface("out_in", slot)
        if self._prefix <= 0 or self._prefix >= self._length:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=n, nk=self._length, nq=n, s0=start,
                out_stride=self._bucket, out_s0=0, fast=True, geometry=self._geometry,
            )
            return
        # 边界钳在本段内：前缀整段落在本段之前时（`prefix <= start`）这整段都是 query 侧，
        # `nk` 取全长；原先把 `s0` 退回 `prefix` 会在邻段上重复算一遍同样的列。
        boundary = min(end, max(start, self._prefix))
        if start < boundary:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=boundary - start, nk=self._prefix, nq=boundary - start,
                s0=start, out_stride=self._bucket, out_s0=0, fast=True,
                geometry=self._geometry,
            )
        if boundary < end:
            kernels.attention(
                self._runtime, qg, kv, out, self.attn_scratch,
                T=self._width, stile=end - boundary, nk=self._length, nq=end - boundary,
                s0=boundary, out_stride=self._bucket, out_s0=boundary - start, fast=True,
                geometry=self._geometry,
            )

    # ------------------------------------------------------------------ 收尾

    def last_hidden(self, length: int) -> np.ndarray:
        """最后一层的最后一条 token（`[hidden]` fp32），供 norm + 读出。"""
        view = self._last.array(np.float16, (self._hidden, self._width))
        return np.asarray(view[:, length - 1], np.float32)

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
        for name in ("x", "qg", "kv", "gscratch", "attn_scratch", "_cos", "_sin"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer.free()
        for name in ("states", "tails"):
            for buffer in getattr(self, name, None) or ():
                buffer.free()
