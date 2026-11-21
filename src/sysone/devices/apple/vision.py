"""视觉塔：patch / 24 block / merger 共 26 个 ANE program，外加 GPU 上的非因果注意力。

塔的形状来自 `VisionSpec`（两个出货 checkpoint 逐项相同，只有 merger 的输出宽度不同），所以
一份实现两个模型。**不按 `model_type` 分支**：h2o 的 `vision_config.model_type` 写的是
`qwen3_5` 而不是 `qwen3_5_vision`（见 `weights/vision.py` 的说明）。

四段约定是这张图里唯一"跑得通但语义错"的地方，全在 host 上、都有单元测试
（`tests/unit/test_vision_image.py`）：

* **patchify 是块主序**（块行, 块列, 块内行, 块内列），不是栅格序 —— 厂商 processor 这么排，
  merger 的 `x.view(-1, 4·hidden)` 与 pos_embed 的插值索引按的是同一个顺序；
* **时间轴的复制在通道轴内部**（`(c, t, py, px)`），不是把两份 patch 拼在末尾；
* **pos_embed 按 `align_corners=True` 双线性插值**（每轴 2 个抽头 = 每 token 4 个），从
  `sqrt(num_position_embeddings)` 见方的表插到每张图的 `(gh, gw)` 网格；
* **轴向 RoPE 的 cos/sin 两半相同**（每轴 `head_dim/2/2` 个频率，两轴拼成 `head_dim/2` 个
  角度，再 `cat` 一次），与 `rotate_half` 的半交换配对。

`encode` 的输入是 P1 产出的归一图 `(H, W, 3)` fp32（`p·2/255 - 1`），输出是
`(L/4, out_hidden)` fp16 —— 与厂商 `Qwen3_5VisionModel` 的 `pooler_output` 同一个形状。
**patchify 在这里**（不在 preprocess）：块的 2×2 边界与 merger 的合并是同一件事。

一个 ANE program 一次只有 `s` 列宽，所以一图超过 `s` 个 token 就要分趟：残差流留在 host 上
（`(L, hidden)` fp16），每趟把切片写进面、算完再读回来。**分趟与不分趟逐位相同**（每一列的
线性层结果与其它列无关，注意力仍按整图算）—— 这条是门之一，也是"`s` 只是桶宽、不是能力"的
证明。
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...errors import NotReadyError, UnsupportedMediaError
from ...models.descriptor import ModelDescriptor, VisionSpec
from ...models.vision import contract as vision_contract
from ...models.vision import merge
from ...models.vision import preprocess
from ...weights import vision as vw
from ...weights.checkpoint import WeightStore
from ...weights.packing.ane_blob import build as build_blob
from . import image
from . import vision_plan as vp
from .chain import _artifacts_root, _program_root
from .compile import ProgramFiles, cached_program, write_program
from .kernels import copy_columns, vrope, vattn
from .metal import LANE_SMALL, MetalBuffer, MetalRuntime
from .runtime import AneRuntime, Surface

# 一个进程里可以有多个塔（对拍脚本会同时开两个模型），ANE 的程序名必须唯一。
_TOWERS = itertools.count()

# 按网格缓存的位置表最多留几个尺寸（1024² 的一份是 8 MB；塔空闲时整体卸载）。
_GRID_CACHE = 4


# ------------------------------------------------------------ host 侧的四段约定


def patchify(img: np.ndarray, *, patch: int, merge: int, temporal: int) -> np.ndarray:
    """`(H, W, 3)` → `(L, 3·t·p·p)`：块主序 · 通道向量按 `(c, t, py, px)`。

    厂商 `Qwen2VLImageProcessor.patchify` 的实际动作是：`(c, gh/m, m, p, gw/m, m, p)` 重排成
    `(gh/m, gw/m, mr, mc, c, p, p)`，再在 `c` 与 `p` 之间 `unsqueeze` 一根时间轴、`expand`、
    摊平。所以**时间轴的复制在通道轴内部**。这里写成等价的闭式；单元测试对逐字转写验逐字节
    相同（那份转写才是"厂商的代码"）。
    """
    c = int(img.shape[2])
    grid_h, grid_w = img.shape[0] // patch, img.shape[1] // patch
    x = img.reshape(grid_h // merge, merge, patch, grid_w // merge, merge, patch, c)
    x = x.transpose(0, 3, 1, 4, 6, 2, 5)                            # (br, bc, mr, mc, c, py, px)
    x = np.repeat(x[:, :, :, :, :, None, :, :], temporal, axis=5)   # (br, bc, mr, mc, c, t, py, px)
    return np.ascontiguousarray(x.reshape(grid_h * grid_w, c * temporal * patch * patch))


def block_positions(grid_h: int, grid_w: int, *, merge: int) -> tuple[np.ndarray, np.ndarray]:
    """每个 token 的 `(行, 列)`，块主序 —— 厂商 `get_vision_position_ids` 的同一串 reshape。

    栅格序（`arange(gh*gw)`）会让每四列的 2×2 合并把一个块里不同位置的 token 拼到一起，
    形状合法而语义错，所以这里不是"怎么排都行"。
    """
    rows, cols = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    shape = (grid_h // merge, merge, grid_w // merge, merge)
    return (rows.reshape(shape).transpose(0, 2, 1, 3).reshape(-1),
            cols.reshape(shape).transpose(0, 2, 1, 3).reshape(-1))


def pos_embed_rows(table: np.ndarray, *, grid_h: int, grid_w: int, merge: int,
                   side: int) -> np.ndarray:
    """把 `(side², hidden)` 的 pos_embed 表插值到 `(L, hidden)`，4 抽头（每轴 2 个）。

    `align_corners=True` 的闭式：`src = 位置 · (side-1) / (size-1)`，抽头 `floor + {0,1}`
    夹到 `[0, side-1]`，权重 `max(0, 1 - |src - 抽头|)`。两轴外积成 4 个 2D 抽头。
    """
    rows, cols = block_positions(grid_h, grid_w, merge=merge)
    h_taps, h_weights = _taps(rows, grid_h, side)
    w_taps, w_weights = _taps(cols, grid_w, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    # 逐抽头累加，而不是先建 `(L, 4, hidden)`：L=6400 时那张中间表是 105 MB。
    out = table[indices[:, 0]] * weights[:, 0:1]
    for j in range(1, 4):
        out += table[indices[:, j]] * weights[:, j:j + 1]
    return out


def rope_tables(rows: np.ndarray, cols: np.ndarray, *, head_dim: int,
                theta: float) -> tuple[np.ndarray, np.ndarray]:
    """轴向 RoPE 的 `(cos, sin)`，各 `(L, head_dim)`。

    每轴 `head_dim/2/2` 个频率（`1/θ^(2i/(head_dim/2))`），两个轴的角度拼成 `head_dim/2` 个，
    再 `cat` 一次 —— 于是 cos 的两半相同，配对的是 `rotate_half` 的 `(d, d + head_dim/2)`。
    """
    half = head_dim // 2
    inv_freq = theta ** (-np.arange(0, half, 2, dtype=np.float64) / half)
    angles = np.concatenate([rows[:, None] * inv_freq, cols[:, None] * inv_freq], axis=1)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return np.concatenate([cos, cos], axis=1), np.concatenate([sin, sin], axis=1)


def rope_qk(qkv: np.ndarray, cos: np.ndarray, sin: np.ndarray, *,
            hidden: int, head_dim: int) -> np.ndarray:
    """把 `[q | k | v]` 里的 q、k 两段就地转成 RoPE 之后的值（`(3·hidden, L)` 通道主序）。

    `(d, d + head_dim/2)` 配对：`out[d] = x[d]·cos[d] - x[d+½]·sin[d]`、
    `out[d+½] = x[d+½]·cos[d] + x[d]·sin[d]`。cos/sin 的两半相同，所以一对抽头共用一个角 ——
    这正是厂商 `recomposition_frequencies` 里那次 `cat` 的含义。

    **这不是执行路径**：`encode` 用 GPU 上的 `kernels.vrope`（同一算式，逐位对齐）。这份主机实现
    留着当参考，是 `tests/unit/test_vision_image.py`（对厂商 `rotate_half`）与
    `tests/integration/test_metal_kernels.py`（对核）两条判据的 oracle。
    """
    half = head_dim // 2
    heads = hidden // head_dim
    c = np.asarray(cos[:, :half], np.float32).T[None]     # (1, ½, L)
    s = np.asarray(sin[:, :half], np.float32).T[None]
    for base in (0, hidden):                           # q 段、k 段
        view = qkv[base:base + hidden].reshape(heads, head_dim, -1)
        lo = np.asarray(view[:, :half], np.float32)
        hi = np.asarray(view[:, half:], np.float32)
        view[:, :half] = lo * c - hi * s
        view[:, half:] = hi * c + lo * s
    return qkv


def _taps(pos: np.ndarray, size: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    """一维的双线性抽头：`(索引 (n, 2) int64, 权重 (n, 2) fp32)`。"""
    src = pos.astype(np.float32) * (side - 1) / max(size - 1, 1)
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor[:, None].astype(np.int64) + offsets[None, :], 0, side - 1)
    weights = np.maximum(0.0, 1.0 - np.abs(src[:, None] - floor[:, None] - offsets[None, :]))
    return taps, weights.astype(np.float32)


# ------------------------------------------------------------------ 程序与面


@dataclass(frozen=True)
class _Program:
    """一个 ANE program、它的产物、以及每个 procedure 的面规格 `(cin, s, cout, s)`。"""

    name: str
    files: ProgramFiles
    io: dict[str, tuple[tuple[int, ...], int, int]]
    # 产物是**缓存命中**（没读权重、没装 blob），而不是"重算了一遍、结果恰好一样"。
    # `files.reused` 是后者：`write_program` 拿新算出的字节与磁盘比，一样就置 True —— 那 6 s
    # 的 blob 构建已经花掉了。`doctor` 报的命中率必须与装载同一判据，所以分开记。
    hit: bool = False

    def run(self, runtime: AneRuntime, proc: str, slot: int = 0) -> float:
        ms = runtime.eval_slot(self.name, vp.PROCS[proc], int(slot))
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, proc=proc, slot=slot,
                                reason=runtime.error())
        return ms

    def close(self, runtime: AneRuntime) -> None:
        runtime.unload(self.name)


@dataclass(frozen=True)
class _Placed:
    """一张图在打包列空间里的位置，以及它自己的三张表。

    `padded` 是它占的列数（今天等于 `length`：各图之间不补空，注意力按 (瓦片, 图) 逐段发，
    不要求段的长度是 64 的倍数）。留着这个字段是因为 merger 的 4 列一组要求图边界落在组边界上，
    而"是否需要补"由那一处的守卫决定，不由打包层猜。
    """

    lo: int
    length: int
    padded: int
    tokens: np.ndarray
    pos: np.ndarray
    cos: np.ndarray
    sin: np.ndarray


@dataclass(frozen=True)
class _Tile:
    """一个 token 瓦片独占的两张面，以及它们在 GPU 侧的零拷贝包装。

    瓦片**不共享面**：qkv 与注意力上下文都是"这一片刚算出来、下一片才用"的，共享就要拷，
    而拷贝正是要删掉的东西（票据 07 §7.19）。`qkv_dev` / `ctx_dev` 是同一块内存的
    `MTLBuffer` 包装（`e8m_wrap`），所以 GPU 直接从 ANE 的面上读 q、往 ANE 的面上写上下文，
    两个方向都不经过主机。

    `ctx` 是 `hidden` 高：GPU 把注意力上下文直接写进它，`p1` 把它当作 0 号输入面（1 号是残差
    面）。两者都不经过主机 —— 双输入 `p1` 是 §7.23 之后的形状。
    """

    qkv: Surface
    ctx: Surface
    qkv_dev: MetalBuffer
    ctx_dev: MetalBuffer
    qkv_view: np.ndarray
    ctx_view: np.ndarray


class VisionTower:
    """一座塔：26 个 program + 按瓦片划分的面 + 每次 `encode` 的整图设备缓冲。

    面的所有权在塔上（不归 program）：ANE 只有一条固件队列、一次只有一个 eval 在飞。**但面
    按瓦片分**，每个瓦片一对 (qkv, ctx)，各自按槽位绑到同一个 program 上（`attach_slot` 的
    `slot`）。这不是为了并发，而是为了让相邻两段接力：`p1` 的输出就是下一层 `main` 的入口，
    `main` 的 qkv 出口就是注意力的入口 —— 面在 ANE 与 GPU 之间传来传去，主机一次都不碰。

    残差流是例外：它按**奇偶**两套面轮转（第 i 层读一套、写另一套）。同一套面既当入口又当出口
    不行，因为 `p1` 一边读 `res` 一边往同一段写；两套就够，因为第 i+1 层的入口正是第 i 层的
    出口。

    瓦片数由图像尺寸决定（`s` 是列数上限），所以面与槽位都是**第一次见这个瓦片数时**才建、
    才绑 —— 一张 512² 的图只建 1 个瓦片，不会为 1024² 预留。
    """

    def __init__(
        self,
        store: WeightStore,
        *,
        descriptor: ModelDescriptor,
        s: int,
        runtime: AneRuntime,
        metal: MetalRuntime,
        artifacts_dir: str | Path | None = None,
    ) -> None:
        spec: VisionSpec | None = descriptor.vision
        if spec is None:
            raise NotReadyError("这个模型没有视觉塔", model=descriptor.model_id)
        if spec.rope_type != "axial":
            raise NotReadyError("视觉塔只实现了轴向 RoPE", model=descriptor.model_id,
                                rope_type=spec.rope_type)
        self.store, self.descriptor, self.spec = store, descriptor, spec
        self.runtime, self.metal = runtime, metal
        self.s = int(s)
        self.merge = int(spec.spatial_merge)
        if self.merge != 2:
            # merger 的合并是**写死的 2×2**（`vision_plan._merge2x2`，与厂商
            # `x.view(-1, 4·hidden)` 一致）：换一个 merge 值需要另一个 MIL 构造器，不是改个常数。
            raise NotReadyError("视觉塔只实现了 2×2 的空间合并", merge=self.merge)
        self.unit = self.merge ** 2          # 一个合并块吃掉几列
        if self.s % self.unit:
            # `self.q` 是整除，桶宽不整除 4 会让最后一趟少算几列 —— 静默丢列，所以在这里拦。
            raise NotReadyError("视觉塔的桶宽必须能被 merger 的 2×2 合并整除",
                                s=self.s, merge=self.merge)
        self.q = self.s // self.unit          # 合并后的列数
        self.hidden = int(spec.hidden)
        self.ffn = int(spec.intermediate_size)
        self.out_hidden = int(spec.out_hidden_size)
        self.heads = int(spec.num_heads)
        self.head_dim = self.hidden // self.heads
        self.scale = self.head_dim ** -0.5
        self.side = int(round(spec.num_position_embeddings ** 0.5))
        self.root = vw.tower_root(store)
        self.directory = _program_root(_artifacts_root(artifacts_dir), descriptor,
                                       s=self.s) / "vision"
        self.stats = {"evals": 0, "ane_ms": 0.0, "gpu_ms": 0.0, "gpu_busy_ms": 0.0,
                      "host_ms": 0.0}
        self._grid_cache: OrderedDict[tuple[int, int], tuple[np.ndarray, ...]] = OrderedDict()
        self._header = f"vision{next(_TOWERS)}"

        # patch 的输入宽度从**权重形状**取：写死 1536 就是第二份塔的形状。
        self.patch_entries = vw.patch_entries(store, self.root)
        self.cin = int(self.patch_entries["pe"].shape[1])
        if self.cin != spec.patch_size ** 2 * spec.temporal_patch_size * 3:
            raise NotReadyError("patch 权重的列数与 config 的 patch 几何对不上",
                                columns=self.cin, patch=spec.patch_size,
                                temporal=spec.temporal_patch_size)
        self.pos_table = vw.pos_embed(store, self.root)
        # 不随瓦片变的四张面：patch 的入口（每片覆写一次）、merger 的两张中间面。
        self._surfaces = self._make_surfaces()
        self._tiles: list[_Tile] = []
        # 残差流：两套（奇偶轮转）× 瓦片数。`_hidden_views` 与之一一对应，省掉每次的 ctypes。
        self._hidden: tuple[list[Surface], list[Surface]] = ([], [])
        self._hidden_views: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])
        self._depth = int(spec.depth)
        self._programs: dict[str, _Program] = {}
        # `(program, procedure) -> 输出面`。面归塔所有，程序本身不认识它，所以这块账记在塔上：
        # 读输出（merger 的最终出口）时按它开视图，瓦片切换时它也只有一个值（同一种输出）。
        self._outputs: dict[tuple[str, str], Surface] = {}
        self._programs["patch"] = self._build_patch()
        for index in range(self._depth):
            self._programs[f"block{index:02d}"] = self._build_block(index)
        self._programs["merger"] = self._build_merger()

    # -- 构造 ----------------------------------------------------------------
    def _make_surfaces(self) -> dict[str, Surface]:
        """不随瓦片变的四张面。

        `patch_in` 每片覆写一次（它的读者是同一次 eval，顺序很紧）；merger 的两张中间面本来就
        只有一个瓦片在用，且 `merged` 要回主机，所以它们不按瓦片分。
        """
        plan = (
            ("patch_in", self.cin, self.s),
            ("m1o", self.ffn, self.q),
            ("m2o", self.out_hidden, self.q),
        )
        surfaces: dict[str, Surface] = {}
        try:
            for key, rows, cols in plan:
                surfaces[key] = self.runtime.surface(rows * cols * 2)
        except NotReadyError:
            for surface in surfaces.values():
                surface.free()
            raise
        return surfaces

    def _ensure_tiles(self, count: int) -> None:
        """把瓦片 0..count-1 的面建出来、绑好。第一次见这个尺寸时付一次，之后白拿。"""
        while len(self._tiles) < count:
            index = len(self._tiles)
            qkv = self.runtime.surface(3 * self.hidden * self.s * 2)
            ctx = self.runtime.surface(self.hidden * self.s * 2)
            try:
                tile = _Tile(
                    qkv=qkv, ctx=ctx,
                    qkv_dev=self.metal.wrap(qkv), ctx_dev=self.metal.wrap(ctx),
                    qkv_view=qkv.array(np.float16, (1, 3 * self.hidden, 1, self.s))[0, :, 0],
                    ctx_view=ctx.array(np.float16, (1, self.hidden, 1, self.s))[0, :, 0],
                )
                for slot in (0, 1):
                    surface = self.runtime.surface(self.hidden * self.s * 2)
                    self._hidden[slot].append(surface)
                    self._hidden_views[slot].append(
                        surface.array(np.float16, (1, self.hidden, 1, self.s))[0, :, 0])
            except NotReadyError:
                qkv.free()
                ctx.free()
                raise
            self._tiles.append(tile)
            self._attach_tile(index)

    def _attach_tile(self, index: int) -> None:
        """把一个瓦片的槽位绑到所有相关的 (program, procedure) 上。

        `p1` 有两张输入面：`ctx`（注意力上下文）与 `res`（残差）。两张面按字典序命名，所以
        `ctx` 是 0 号、`res` 是 1 号，面的顺序就是参数顺序（`vision_plan` 的模块说明）。
        """
        tile = self._tiles[index]
        attach = self.runtime.attach_slot
        self._outputs[("patch", "main")] = self._hidden[0][index]
        attach(self._programs["patch"].name, vp.PROCS["main"], index,
               self._surfaces["patch_in"], self._hidden[0][index])
        for layer in range(self._depth):
            block = self._programs[f"block{layer:02d}"]
            parity = layer % 2
            # `p1` 是双输入（上下文 + 残差）→ 用 attach_multi；`main` 单输入走 attach_slot。
            attach(block.name, vp.PROCS[vp.MAIN], index,
                   self._hidden[parity][index], tile.qkv)
            self.runtime.attach_multi(block.name, vp.PROCS[vp.P1],
                                      [tile.ctx, self._hidden[parity][index]],
                                      self._hidden[1 - parity][index], slot=index)
            self._outputs[(block.name, vp.MAIN)] = tile.qkv
            self._outputs[(block.name, vp.P1)] = self._hidden[1 - parity][index]
        merger = self._programs["merger"]
        attach(merger.name, vp.PROCS["main"], index,
               self._hidden[self._tail_set][index], self._surfaces["m1o"])
        self._outputs[(merger.name, "main")] = self._surfaces["m1o"]
        self._outputs[(merger.name, "p1")] = self._surfaces["m2o"]

    @property
    def _tail_set(self) -> int:
        """最后一层的残差落在哪一套面上（`depth` 为偶数时是 0 号）。"""
        return 1 - ((self._depth - 1) % 2)

    def _source(self, kind: str, index: int | None = None) -> dict:
        """产物的身份 —— 与 `artifact_state` 是同一个函数（`doctor` 报的命中率不能与装载分家）。"""
        return _identity(self.store, self.spec, self.s, kind, index)

    def _assemble(self, name: str, directory: Path, source: dict,
                  io: dict[str, tuple[tuple[int, ...], int, int]], build: "callable") -> _Program:
        """产物命中就只装载；不命中才读权重、装 blob、生成 MIL。

        `build()` 只在未命中时调用 —— 一个 block 的 12 张张量读成 fp32、再摊成 fp16 blob 是
        ~50 MB 的活，而缓存命中时那 24 × 50 MB 一分都不该花。`io` 与几何无关地先算出来，
        所以命中路径不需要 blob 的位置表。
        """
        files = cached_program(directory, source=source, s=self.s)
        hit = files is not None
        if files is None:
            files = build(directory, source)
        files.load(self.runtime, name)
        return _Program(name=name, files=files, io=io, hit=hit)

    def _build_patch(self) -> _Program:
        io = vp.patch_io(cin=self.cin, hidden=self.hidden, s=self.s)

        def build(directory: Path, source: dict) -> ProgramFiles:
            blob, table = build_blob(self.patch_entries, quantize=False)
            plan = vp.patch_plan(hidden=self.hidden, s=self.s,
                                 entries=self.patch_entries, blob=table)
            return write_program(directory, mil=plan.mil, blob=blob, s=self.s,
                                 meta={"kind": "patch", "cin": self.cin}, source=source)

        return self._assemble(f"{self._header}_patch", self.directory / "patch",
                              self._source("patch"), io, build)

    def _build_block(self, index: int) -> _Program:
        io = vp.block_io(hidden=self.hidden, ffn=self.ffn, s=self.s)

        def build(directory: Path, source: dict) -> ProgramFiles:
            entries = vw.block_entries(self.store, self.root, index)
            blob, table = build_blob(entries, quantize=False)
            plan = vp.block_plan(hidden=self.hidden, ffn=self.ffn, s=self.s,
                                 entries=entries, blob=table)
            return write_program(directory, mil=plan.mil, blob=blob, s=self.s,
                                 meta={"kind": "block", "index": index}, source=source)

        return self._assemble(f"{self._header}_block_{index:02d}",
                              self.directory / f"block{index:02d}",
                              self._source("block", index), io, build)

    def _build_merger(self) -> _Program:
        io = vp.merger_io(hidden=self.hidden, ffn=self.ffn,
                          out_hidden=self.out_hidden, s=self.s)

        def build(directory: Path, source: dict) -> ProgramFiles:
            entries = vw.merger_entries(self.store, self.root)
            blob, table = build_blob(entries, quantize=False)
            plan = vp.merger_plan(hidden=self.hidden, ffn=self.ffn,
                                  out_hidden=self.out_hidden, s=self.s,
                                  entries=entries, blob=table)
            return write_program(directory, mil=plan.mil, blob=blob, s=self.s,
                                 meta={"kind": "merger"}, source=source)

        program = self._assemble(f"{self._header}_merger", self.directory / "merger",
                                 self._source("merger"), io, build)
        # merger 的 `p1` 只有一张面（中间面 m1o → m2o），与瓦片无关，所以绑槽 0 就够。
        self.runtime.attach(program.name, vp.PROCS["p1"],
                            self._surfaces["m1o"], self._surfaces["m2o"])
        return program

    # -- 跑 ------------------------------------------------------------------
    def encode(self, img: np.ndarray) -> np.ndarray:
        """`(H, W, 3)` fp32 归一图 → `(L/spatial_merge², out_hidden)` fp16。"""
        return self.encode_many([img])[0]

    def encode_many(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """一次请求的几张图**拼在一条列空间里**过塔：返回每张图各自的 `(L/4, out_hidden)`。

        为什么要拼：ANE 一次只算一个桶宽的列，一图 2304 个 token 要 3 趟，三张图就是 9 趟；拼成
        一条 6912 列的空间只要 7 趟（1024² 这种已经对齐的形状则一点不省）。拼的前提是**注意力
        只能看自己那张图的键** —— `k_vattn` 的 `k0`/`k1` 就是为此加的（票据 07 §7.24）。

        各图之间**不需要**按桶补空：注意力是按 (瓦片, 图) 逐段发的，一段的 query 数不必是 64 的
        倍数；只有 merger 的 2×2 合并要求每张图的长度是 4 的倍数（smart_resize 保证，这里补
        一个守卫）。逐位与"一张一张过"相同 —— 列与列之间本来就没有耦合。
        """
        spec = self.spec
        started = time.perf_counter()
        before = (self.stats["ane_ms"], self.stats["gpu_ms"])
        placed = []
        lo = 0
        for img in images:
            item = self._place(img)
            placed.append(_Placed(lo=lo, length=item.length, padded=item.padded,
                                  tokens=item.tokens, pos=item.pos, cos=item.cos, sin=item.sin))
            lo += item.padded
        width = lo
        if width == 0:
            raise NotReadyError("没有可编码的图像", images=len(images))
        tiles = list(self._tiles_of(width))
        self._ensure_tiles(len(tiles))
        half_dim = self.head_dim // 2

        # 打包后的输入与表：每张图写进自己的列区间，补齐的那几列留零。
        tokens = np.zeros((self.cin, width), np.float16)
        pos_cm = np.zeros((self.hidden, width), np.float16)
        cos_p = np.zeros((half_dim, width), np.float16)
        sin_p = np.zeros((half_dim, width), np.float16)
        for item in placed:
            tokens[:, item.lo:item.lo + item.length] = item.tokens
            pos_cm[:, item.lo:item.lo + item.length] = item.pos
            cos_p[:, item.lo:item.lo + item.length] = item.cos[:, :half_dim].T
            sin_p[:, item.lo:item.lo + item.length] = item.sin[:, :half_dim].T
        try:
            # 1) patch 投影 + pos_embed：出口直接落在这个瓦片的残差面上，位置编码就地加上。
            for index, (lo, n) in enumerate(tiles):
                self._put("patch_in", self.cin, tokens[:, lo:lo + n])
                self._call(self._programs["patch"], "main", slot=index)
                self._hidden_views[0][index][:, :n] += pos_cm[:, lo:lo + n]

            # 2) 24 个 block（见 §7.21 的跨层调度）
            qkv_bufs = [self.metal.alloc(3 * self.hidden * width * 2) for _ in range(2)]
            cos_dev = self.metal.alloc(half_dim * width * 2)
            sin_dev = self.metal.alloc(half_dim * width * 2)
            try:
                cos_dev.array(np.float16, (half_dim, width))[...] = cos_p
                sin_dev.array(np.float16, (half_dim, width))[...] = sin_p
                self._layer_main(0, tiles, qkv_bufs[0], width)
                for layer in range(self._depth):
                    block = self._programs[f"block{layer:02d}"]
                    # 上一层抄进 lane 1 的 qkv 必须在这一趟 vrope 之前落地。
                    self.stats["gpu_ms"] += self.metal.wait(LANE_SMALL)
                    vrope(self.metal, qkv_bufs[layer % 2], cos_dev, sin_dev, T=width,
                          hidden=self.hidden, dim=self.head_dim)
                    for j, (lo, n) in enumerate(tiles):
                        self._attention(qkv_bufs[layer % 2], j, lo, n, width, placed)
                        self.metal.commit()
                    following = layer + 1
                    for j, (lo, n) in enumerate(tiles):
                        self._await_gpu()
                        self._call(block, vp.P1, slot=j)
                        if following < self._depth:
                            self._call(self._programs[f"block{following:02d}"], vp.MAIN, slot=j)
                            copy_columns(
                                self.metal, self._tiles[j].qkv_dev, qkv_bufs[following % 2],
                                nchan=3 * self.hidden, src_c0=0, dst_c0=0, stile=n, s0=lo,
                                src_stride=self.s, dst_stride=width)
                            self.metal.commit(LANE_SMALL)
            finally:
                for dev in (*qkv_bufs, cos_dev, sin_dev):
                    dev.free()

            # 3) merger：每趟把 2×2 合成 4·hidden，再 fc1 → gelu → fc2
            merger = self._programs["merger"]
            merged = np.empty((width // self.unit, self.out_hidden), np.float16)
            for index, (lo, n) in enumerate(tiles):
                self._call(merger, "main", slot=index)
                self._call(merger, "p1")
                merged[lo // self.unit:(lo + n) // self.unit] = \
                    self._take(merger, "p1", n // self.unit).T
        finally:
            after = (self.stats["ane_ms"], self.stats["gpu_ms"])
            busy = (after[0] - before[0]) + (after[1] - before[1])
            self.stats["host_ms"] += (time.perf_counter() - started) * 1e3 - busy
        return [merged[item.lo // self.unit:(item.lo + item.length) // self.unit]
                for item in placed]

    def _place(self, img: np.ndarray) -> "_Placed":
        """一张图 → 打包布局里的位置与它自己的三张表（位置编码、cos、sin）。"""
        spec = self.spec
        img = np.asarray(img, np.float32)
        if img.ndim != 3 or img.shape[2] != 3:
            raise NotReadyError("视觉塔的输入必须是 (H, W, 3)", shape=list(img.shape))
        height, width = int(img.shape[0]), int(img.shape[1])
        patch, merge = int(spec.patch_size), self.merge
        if height % patch or width % patch:
            raise NotReadyError("图像尺寸必须是 patch 的整数倍", shape=[height, width], patch=patch,
                                hint="P1 的 smart_resize 负责这一点")
        grid_h, grid_w = height // patch, width // patch
        if grid_h % merge or grid_w % merge:
            raise NotReadyError("patch 网格必须是空间合并的整数倍", grid=[grid_h, grid_w],
                                merge=merge)
        length = grid_h * grid_w
        if length % self.unit:
            # merger 的 2×2 合并是按列 4 个一组切的，图与图的边界要落在组边界上 —— 否则一张图
            # 的最后几列会跟下一张图的前几列合成一个块。补到 4 的倍数最省（不补到 64：那会把
            # 打包省下来的趟数又还回去）。
            raise NotReadyError("一张图的 token 数必须是空间合并的整数倍",
                                length=length, unit=self.unit, grid=[grid_h, grid_w])
        pos_cm, cos16, sin16 = self._grid_tables(grid_h, grid_w)
        tokens = np.ascontiguousarray(
            patchify(img, patch=patch, merge=merge,
                     temporal=int(spec.temporal_patch_size)).T).astype(np.float16)
        return _Placed(lo=0, length=length, padded=length, tokens=tokens,
                       pos=pos_cm, cos=cos16, sin=sin16)

    def _layer_main(self, layer: int, tiles: list, buffer: MetalBuffer,
                    length: int) -> None:
        """一整层的 `main`，随后把每个瓦片的 qkv 面抄进这一层的设备缓冲。

        抄一次是因为注意力要读**整图**的键，而 ANE 一次只写一个瓦片的宽度。抄的动作放在 GPU 上
        （`k_copy`）：主机上那 67 ms（1024²）变成设备上的几毫秒，而且**排在 lane 1**——注意力在
        lane 0 上跑，两条队互不排队。`wait(LANE_SMALL)` 在下一层的 `vrope` 之前把它们排空。
        """
        block = self._programs[f"block{layer:02d}"]
        for index, (lo, n) in enumerate(tiles):
            self._call(block, vp.MAIN, slot=index)
            copy_columns(
                self.metal, self._tiles[index].qkv_dev, buffer,
                nchan=3 * self.hidden, src_c0=0, dst_c0=0, stile=n, s0=lo,
                src_stride=self.s, dst_stride=length)
            self.metal.commit(LANE_SMALL)

    def _attention(self, qkv_dev, tile: int, lo: int, nq: int, width: int,
                   placed: Sequence["_Placed"]) -> None:
        """发一个瓦片的注意力：读整条的 qkv，写这个瓦片的上下文面（GPU 直接写面）。

        一个瓦片可能横跨两张图，所以按**图**逐段发：每段的 query 数只取这张图在这一瓦片里的
        那部分，键区间也只取这张图（`k0`/`k1`）。段与段之间没有依赖，同一个命令缓冲一次提交。
        """
        end = lo + nq
        for item in placed:
            start = max(lo, item.lo)
            stop = min(end, item.lo + item.length)
            if stop <= start:
                continue
            vattn(self.metal, qkv_dev, self._tiles[tile].ctx_dev, T=width, nq=stop - start,
                  s0=start, hidden=self.hidden, dim=self.head_dim, scale=self.scale,
                  out_cols=self.s, out_offset=start - lo,
                  k0=item.lo, k1=item.lo + item.length)

    def _grid_tables(self, grid_h: int, grid_w: int) -> tuple[np.ndarray, ...]:
        """`(gh, gw)` 决定的三张表：pos_embed（通道主序 fp16）、cos、sin（各 `(L, ½dim)` fp16）。

        只取决于网格尺寸，与像素无关，所以同尺寸的第二张图起是白拿的（旧引擎 §4.10.1 记
        `pos_embed` 1024² 要 46.5 ms、同尺寸第二张起 0.002 ms；我们现在是 15 ms）。
        容量按 4 个尺寸封顶：一张 1024² 的表是 8 MB，塔空闲时整个卸载，不需要更复杂的策略。
        """
        key = (int(grid_h), int(grid_w))
        hit = self._grid_cache.get(key)
        if hit is not None:
            self._grid_cache.move_to_end(key)
            return hit
        spec = self.spec
        rows, cols = block_positions(grid_h, grid_w, merge=self.merge)
        pos = pos_embed_rows(self.pos_table, grid_h=grid_h, grid_w=grid_w, merge=self.merge,
                             side=self.side).astype(np.float16)
        cos, sin = rope_tables(rows, cols, head_dim=self.head_dim,
                               theta=float(spec.rope_theta))
        half = self.head_dim // 2
        entry = (np.ascontiguousarray(pos.T),
                 cos[:, :half].astype(np.float16), sin[:, :half].astype(np.float16))
        self._grid_cache[key] = entry
        if len(self._grid_cache) > _GRID_CACHE:
            self._grid_cache.popitem(last=False)
        return entry

    # -- 面与小工具 -----------------------------------------------------------
    def _tiles_of(self, length: int):
        for lo in range(0, length, self.s):
            yield lo, min(self.s, length - lo)

    def _view(self, key: str, rows: int) -> np.ndarray:
        surface = self._surfaces[key]
        cols = self.q if key in ("m1o", "m2o") else self.s
        return surface.array(np.float16, (1, rows, 1, cols))[0, :, 0]

    def _put(self, key: str, rows: int, data: np.ndarray) -> None:
        view = self._view(key, rows)
        view[:, :data.shape[1]] = data

    def _take(self, program: _Program, proc: str, cols: int) -> np.ndarray:
        """读输出面的前 `cols` 列（**视图**，调用方要么当场用完，要么自己拷走）。

        输出面由 `_attach_tile` 记在 `_outputs` 上（面归塔所有，程序不认识它），行列规格来自
        `io`。瓦片只有一块出口面会被外部读（merger 的最终出口），所以这里不区分瓦片。
        """
        _, cout, cols_out = program.io[proc]
        surface = self._outputs[(program.name, proc)]
        return surface.array(np.float16, (1, cout, 1, cols_out))[0, :, 0][:, :cols]

    def _await_gpu(self) -> None:
        """等**队首**那一片注意力完成，两个口径分开记。

        `gpu_ms` 是"等了多少"（墙钟）：串行时它等于 GPU 忙时，交错之后它只剩没藏住的那一截，
        所以它同时也是"这次重叠亏了多少"的度量。`gpu_busy_ms` 是 GPU 真在跑的时间
        （`e8m_wait_upto` 里的 `g_last_gpu_ms`），两条排布下应当相同——不同就是争用。

        **必须是 `wait_upto(0, 1)` 而不是 `wait()`**：`wait()` 等于 `wait_upto(0, -1)`，把整条
        队列一起结算 —— 队里压着几片就等几片，还会把它们的 GPU 时间**各自累加一遍**。实测把
        四片一次发出去之后仍用 `wait()`：GPU 忙时从 448 涨到 1 768（正好 4 倍），墙钟 749 → 950。
        """
        self.stats["gpu_ms"] += self.metal.wait_upto(0, 1)
        self.stats["gpu_busy_ms"] += self.metal.gpu_ms()

    def _call(self, program: _Program, proc: str, slot: int = 0) -> None:
        self.stats["ane_ms"] += program.run(self.runtime, proc, slot)
        self.stats["evals"] += 1

    # -- 生命周期 ------------------------------------------------------------
    @property
    def reuse(self) -> tuple[int, int]:
        """产物命中数 / 总数：与 `cached_program` 同一判据（`doctor` 与门共用这一个）。"""
        return (sum(1 for p in self._programs.values() if p.hit), len(self._programs))

    @property
    def bytes(self) -> int:
        """常驻字节：26 份 blob（ANE 权重）+ 面 + 按网格缓存的位置表。

        缓存那一项是**跑起来之后才涨**的（一张 1024² 的表 8 MB，最多 4 个尺寸），所以刚装好的
        塔报的还是小块；`/metrics` 要的是"现在占多少"，这里就得跟着涨。
        """
        blobs = sum(p.files.blob_path.stat().st_size for p in self._programs.values())
        cached = sum(a.nbytes for entry in self._grid_cache.values() for a in entry)
        fixed = sum(surface.nbytes for surface in self._surfaces.values())
        hidden = sum(surface.nbytes for slot in self._hidden for surface in slot)
        tiles = sum(tile.qkv.nbytes + tile.ctx.nbytes for tile in self._tiles)
        return blobs + fixed + hidden + tiles + cached

    def close(self) -> None:
        for program in self._programs.values():
            program.close(self.runtime)
        self._programs.clear()
        for tile in self._tiles:
            # 包装句柄只能解绑、不能 free（dylib 按 IOSurface 指针缓存它）——
            # 顺序也重要：先解绑再放面，否则句柄会指向已释放的内存。
            tile.qkv_dev.free()
            tile.ctx_dev.free()
            tile.qkv.free()
            tile.ctx.free()
        self._tiles.clear()
        for slot in self._hidden:
            for surface in slot:
                surface.free()
            slot.clear()
        self._hidden_views = ([], [])
        for surface in self._surfaces.values():
            surface.free()
        self._surfaces.clear()


# ------------------------------------------------------------------ 生命周期


@dataclass(frozen=True)
class VisionArtifacts:
    """塔的产物在磁盘上的状态（不装载、不碰 ANE，`doctor` 用）。"""

    directory: Path
    hits: int
    total: int

    @property
    def missing(self) -> int:
        return self.total - self.hits


def artifact_state(
    store: WeightStore, *, descriptor: ModelDescriptor, s: int,
    artifacts_dir: str | Path | None = None,
) -> VisionArtifacts:
    """这份权重在 `artifacts_dir` 里已经有哪些塔的产物：判据与装载**完全相同**。

    与 `chain.artifact_state` 同一个形态、同一个理由：`doctor` 说"全命中"而装载照样重编，
    这个报告就没有意义了。所以身份字典由塔自己在构造时用的那一份（`_identity`）算，
    这里只读磁盘。
    """
    spec = descriptor.vision
    if spec is None:
        raise NotReadyError("这个模型没有视觉塔", model=descriptor.model_id)
    kinds: list[tuple[str, int | None]] = [("patch", None), ("merger", None)]
    kinds += [("block", index) for index in range(int(spec.depth))]
    directory = _program_root(_artifacts_root(artifacts_dir), descriptor, s=int(s)) / "vision"
    hits = 0
    for kind, index in kinds:
        name = f"block{index:02d}" if kind == "block" else kind
        source = _identity(store, spec, int(s), kind, index)
        if cached_program(directory / name, source=source, s=int(s)) is not None:
            hits += 1
    return VisionArtifacts(directory=directory, hits=hits, total=len(kinds))


def _identity(store: WeightStore, spec: VisionSpec, s: int, kind: str,
              index: int | None) -> dict:
    """一份塔产物的身份：权重容器指纹 + 塔的形状 + 桶宽 + 段数 + 是哪一份。

    `cached_program` 只比这个字典，不比哈希（那是缓存要省掉的那一段），所以凡是会改变产物
    的输入都得进来 —— 尤其是**塔的形状**：同一份 `family-model` 目录下换了 config 的检查点
    不能认领彼此的产物。`procs` 同理：段名决定面槽位，三段与两段的产物互相认领会绑错面。
    """
    return {
        "weights": store.fingerprint(),
        "kind": kind,
        "procs": "+".join(sorted(vp.PROCS)),
        "index": index,
        "s": s,
        "hidden": spec.hidden,
        "ffn": spec.intermediate_size,
        "depth": spec.depth,
        "heads": spec.num_heads,
        "patch": spec.patch_size,
        "merge": spec.spatial_merge,
        "temporal": spec.temporal_patch_size,
        "out_hidden": spec.out_hidden_size,
        "pos": spec.num_position_embeddings,
    }


class VisionSlot:
    """懒加载 + 空闲卸载：**注册了但没常驻 = 0 内存**（票据 07 §2.5 #5.1–#5.5）。

    两条都不是风格，是旧引擎量出来的：

    * **"挂起"省不了内存**（627.6 MB 里只还 20–56 MB，§4.12）⇒ 这里只有"装"和"真卸载"；
      卸载就是 `VisionTower.close()`：26 个 program 卸掉、面释放。
    * **不常驻**（`e8e-service-plan.md` T6）⇒ 第一次带图请求才把塔装上。首启的预处理只编
      26 份产物、编完就归还（`release()`），0 常驻的契约不变；这样第二次启动既不重编，也
      不用为一张可能永远不会来的图付 630–690 MB。

    `idle` 秒之后没有新请求就卸（`threading.Timer`，到点先查"最近一次使用"再决定，所以并发
    请求不会把一个正在跑的塔卸掉）。`idle = 0` 表示"每次请求之后立刻卸"，那是 `finish()`
    的事 —— 一次请求里有几张图时不该装一次卸一次。

    **塔不是可重入的**（一套复用的面、一份 GPU 缓冲），所以 `encode` 全程持锁：这条路径本来
    就受 ANE 那条固件队列的串行约束，锁只是把"必须串行"这件事写明。
    """

    def __init__(self, build: "callable", *, spec: VisionSpec, idle: float = 60.0) -> None:
        if idle < 0:
            raise ValueError(f"空闲卸载的秒数不能为负：{idle}")
        self._build = build
        # 规格在这里而不是在塔上取：`media()` 要用它跑像素管线，而那时塔可能还没装
        # （`--vision-idle 0` 的请求之间塔是不在的）。
        self._spec = spec
        self._idle = float(idle)
        self._tower: VisionTower | None = None
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._last_use = 0.0
        self._loads = 0
        self._unloads = 0
        self._load_s = 0.0
        self._evals = 0
        self._ane_ms = 0.0
        self._gpu_ms = 0.0
        self._gpu_busy_ms = 0.0

    # -- 装载与卸载 ----------------------------------------------------------
    def warm(self) -> VisionTower:
        """只装不跑：首启预处理用它把 26 份产物提前建出来，随后由 `release()` 归还。"""
        with self._lock:
            return self._ensure()

    def _ensure(self) -> VisionTower:
        if self._tower is None:
            started = time.perf_counter()
            tower = self._build()
            self._load_s = time.perf_counter() - started
            self._tower = tower
            self._loads += 1
        self._last_use = time.monotonic()
        return self._tower

    def _drop(self) -> None:
        tower, self._tower = self._tower, None
        if tower is not None:
            tower.close()
            self._unloads += 1

    def _arm(self) -> None:
        """按"现在"重排一次卸载计时；`idle = 0` 不排（卸载在 `finish()` 里）。"""
        if self._idle <= 0.0:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            timer = threading.Timer(self._idle, self._fire)
            timer.daemon = True
            self._timer = timer
            timer.start()

    def _fire(self) -> None:
        with self._lock:
            if self._tower is None:
                return
            if time.monotonic() - self._last_use < self._idle:
                # 期间又被用过：那一次的 `_arm()` 已经排了新的计时器。
                return
            self._drop()

    def finish(self) -> None:
        """一次请求的收尾：`idle = 0` 时立刻卸，其它情况等计时器。"""
        if self._idle <= 0.0:
            with self._lock:
                self._drop()

    def release(self) -> None:
        """建完产物就把设备上的塔还回去：槽位保留，下一次 `media()` 会重新装。

        与 `close()` 同一个动作，区别在调用方意图：这里只归还"为编译产物暂装的那一份"内存，
        不是进程收尾；槽位本身仍可再次 `_ensure()`。
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._drop()

    def close(self) -> None:
        self.release()

    # -- 跑 ------------------------------------------------------------------
    def encode(self, img: np.ndarray) -> np.ndarray:
        return self.encode_many([img])[0]

    def encode_many(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """一次请求的几张图一起过塔（拼成一条列空间，见 `VisionTower.encode_many`）。"""
        with self._lock:
            tower = self._ensure()
            keys = ("evals", "ane_ms", "gpu_ms", "gpu_busy_ms")
            before = [tower.stats.get(k, 0.0) for k in keys]
            rows = tower.encode_many(images)
            after = [tower.stats.get(k, 0.0) for k in keys]
            self._evals += after[0] - before[0]
            self._ane_ms += after[1] - before[1]
            self._gpu_ms += after[2] - before[2]
            self._gpu_busy_ms += after[3] - before[3]
            self._last_use = time.monotonic()
        self._arm()
        return rows

    def media(self, refs: Sequence[object]) -> merge.Media:
        """一次请求的图片 → `Media`：解码、16 MP 的拒收、像素管线、塔、行数与合并网格。

        错误一律带 `images[i]:` 前缀（与上游 `load_images` 同一句法）：客户端要知道是第几张
        坏，而不是"你这批图有问题"。
        """
        spec = self._spec
        counts: list[int] = []
        grids: list[tuple[int, int]] = []
        pixels: list[np.ndarray] = []
        prepared: list[np.ndarray] = []
        unit = int(spec.spatial_merge)
        for index, ref in enumerate(refs):
            data = getattr(ref, "data", b"")
            if not isinstance(data, (bytes, bytearray, memoryview)) or not data:
                raise UnsupportedMediaError(
                    f"images[{index}]: 这一项没有解码后的字节", length=len(data or b""))
            try:
                width, height = image.size(bytes(data))
            except UnsupportedMediaError as exc:
                raise UnsupportedMediaError(
                    f"images[{index}]: {exc.message}", **exc.context) from exc
            if width * height > vision_contract.MAX_SOURCE_PIXELS:
                raise UnsupportedMediaError(
                    f"images[{index}]: the image has {width * height:,} pixels; "
                    f"at most {vision_contract.MAX_SOURCE_PIXELS:,} are accepted",
                    width=width, height=height)
            try:
                rgb = image.decode(bytes(data))
            except UnsupportedMediaError as exc:
                raise UnsupportedMediaError(
                    f"images[{index}]: {exc.message}", **exc.context) from exc
            pixels, grid = preprocess.prepare(rgb, spec=spec)
            # `prepare` 给的是 patch 网格；`Media` 要的是**合并之后**的网格（厂商同口径）。
            gh, gw = int(grid[0, 1]) // unit, int(grid[0, 2]) // unit
            if gh * gw < 1:
                raise UnsupportedMediaError(
                    f"images[{index}]: 缩到 {pixels.shape[1]}×{pixels.shape[0]} 之后连一个合并块都没有",
                    shape=list(pixels.shape), merge=unit)
            counts.append(gh * gw)
            grids.append((gh, gw))
            prepared.append(pixels)
        # 整批一起过塔：拼成一条列空间，比逐张少几个 ANE 趟（票据 07 §7.24）。
        blocks = [np.asarray(rows) for rows in self.encode_many(prepared)]
        return merge.Media(counts=tuple(counts), grids=tuple(grids),
                           rows=np.concatenate(blocks).astype(np.float16))

    # -- 观测 ----------------------------------------------------------------
    def stats(self) -> dict:
        with self._lock:
            tower = self._tower
            return {
                "loaded": tower is not None,
                "bytes": 0 if tower is None else int(tower.bytes),
                "reuse": [0, 0] if tower is None else list(tower.reuse),
                "loads": self._loads,
                "unloads": self._unloads,
                "load_s": round(self._load_s, 3),
                "idle_s": self._idle,
                # 这几个是**累计**的（塔卸载之后也还在），否则 `/metrics` 会把历史抹掉。
                "evals": self._evals,
                "ane_ms": round(self._ane_ms, 3),
                "gpu_ms": round(self._gpu_ms, 3),
                "gpu_busy_ms": round(self._gpu_busy_ms, 3),
            }
