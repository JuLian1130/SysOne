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
from .kernels import vattn
from .metal import MetalRuntime
from .runtime import AneRuntime, Surface

# 一个进程里可以有多个塔（对拍脚本会同时开两个模型），ANE 的程序名必须唯一。
_TOWERS = itertools.count()


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
    """一个 ANE program、它的产物、以及每个 procedure 的面尺寸 `(cin, s, cout, s)`。"""

    name: str
    files: ProgramFiles
    io: dict[str, tuple[int, int, int, int]]
    procs: dict[str, tuple[Surface, Surface]]

    def run(self, runtime: AneRuntime, proc: str) -> float:
        ms = runtime.eval(self.name, vp.PROCS[proc])
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, proc=proc,
                                reason=runtime.error())
        return ms

    def close(self, runtime: AneRuntime) -> None:
        runtime.unload(self.name)


class VisionTower:
    """一座塔：26 个 program + 一套复用的面 + 每次 `encode` 的两块 Metal buffer。

    面的所有权在塔上（不归 program）：ANE 只有一条固件队列、一次只有一个 eval 在飞，所以 24 个
    block 轮流用同一套面（与文本链 20 层共享一套面是同一条理由）。`close()` 先卸 program 再放面
    —— 卸载会把每次 `attach` 加在面上的引用 CFRelease 掉，面才真的回收。
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
        self.stats = {"evals": 0, "ane_ms": 0.0, "gpu_ms": 0.0, "host_ms": 0.0}
        self._header = f"vision{next(_TOWERS)}"

        # patch 的输入宽度从**权重形状**取：写死 1536 就是第二份塔的形状。
        self.patch_entries = vw.patch_entries(store, self.root)
        self.cin = int(self.patch_entries["pe"].shape[1])
        if self.cin != spec.patch_size ** 2 * spec.temporal_patch_size * 3:
            raise NotReadyError("patch 权重的列数与 config 的 patch 几何对不上",
                                columns=self.cin, patch=spec.patch_size,
                                temporal=spec.temporal_patch_size)
        self.pos_table = vw.pos_embed(store, self.root)
        self._programs: dict[str, _Program] = {}
        self._surfaces = self._make_surfaces()
        self._programs["patch"] = self._build_patch()
        for index in range(int(spec.depth)):
            self._programs[f"block{index:02d}"] = self._build_block(index)
        self._programs["merger"] = self._build_merger()

    # -- 构造 ----------------------------------------------------------------
    def _make_surfaces(self) -> dict[str, Surface]:
        """一套复用的面（`(行, s)` fp16，通道在前）。所有权在塔上，`close()` 统一释放。"""
        plan = (
            ("patch_in", self.cin),
            ("hidden_in", self.hidden),
            ("qkv", 3 * self.hidden),
            ("attn", 2 * self.hidden),
            ("p1o", self.hidden),
            ("hidden_out", self.hidden),
            ("m1o", self.ffn),
            ("m2o", self.out_hidden),
        )
        surfaces: dict[str, Surface] = {}
        try:
            for key, rows in plan:
                cols = self.q if key in ("m1o", "m2o") else self.s
                surfaces[key] = self.runtime.surface(rows * cols * 2)
        except NotReadyError:
            for surface in surfaces.values():
                surface.free()
            raise
        return surfaces

    def _source(self, kind: str, index: int | None = None) -> dict:
        """产物的身份 —— 与 `artifact_state` 是同一个函数（`doctor` 报的命中率不能与装载分家）。"""
        return _identity(self.store, self.spec, self.s, kind, index)

    def _assemble(self, name: str, directory: Path, source: dict,
                  io: dict[str, tuple[int, int, int, int]],
                  build: "callable", procs: dict[str, tuple[str, str]]) -> _Program:
        """产物命中就只装载；不命中才读权重、装 blob、生成 MIL。

        `build()` 只在未命中时调用 —— 一个 block 的 12 张张量读成 fp32、再摊成 fp16 blob 是
        ~50 MB 的活，而缓存命中时那 24 × 50 MB 一分都不该花。`io` 与几何无关地先算出来，
        所以命中路径不需要 blob 的位置表。
        """
        files = cached_program(directory, source=source, s=self.s)
        if files is None:
            files = build(directory, source)
        files.load(self.runtime, name)
        bound = {proc: (self._surfaces[src], self._surfaces[dst]) for proc, (src, dst) in procs.items()}
        for proc, (src, dst) in bound.items():
            self.runtime.attach(name, vp.PROCS[proc], src, dst)
        return _Program(name=name, files=files, io=io, procs=bound)

    def _build_patch(self) -> _Program:
        io = vp.patch_io(cin=self.cin, hidden=self.hidden, s=self.s)

        def build(directory: Path, source: dict) -> ProgramFiles:
            blob, table = build_blob(self.patch_entries, quantize=False)
            plan = vp.patch_plan(hidden=self.hidden, s=self.s,
                                 entries=self.patch_entries, blob=table)
            return write_program(directory, mil=plan.mil, blob=blob, s=self.s,
                                 meta={"kind": "patch", "cin": self.cin}, source=source)

        return self._assemble(f"{self._header}_patch", self.directory / "patch",
                              self._source("patch"), io, build,
                              {"main": ("patch_in", "hidden_out")})

    def _build_block(self, index: int) -> _Program:
        io = vp.block_io(hidden=self.hidden, ffn=self.ffn, s=self.s)

        def build(directory: Path, source: dict) -> ProgramFiles:
            entries = vw.block_entries(self.store, self.root, index)
            blob, table = build_blob(entries, quantize=False)
            plan = vp.block_plan(hidden=self.hidden, ffn=self.ffn, s=self.s,
                                 entries=entries, blob=table)
            return write_program(directory, mil=plan.mil, blob=blob, s=self.s,
                                 meta={"kind": "block", "index": index}, source=source)

        return self._assemble(
            f"{self._header}_block_{index:02d}", self.directory / f"block{index:02d}",
            self._source("block", index), io, build,
            {"main": ("hidden_in", "qkv"), "p1": ("attn", "p1o"),
             "p2": ("p1o", "hidden_out")},
        )

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

        return self._assemble(f"{self._header}_merger", self.directory / "merger",
                              self._source("merger"), io, build,
                              {"main": ("hidden_in", "m1o"), "p1": ("m1o", "m2o")})

    # -- 跑 ------------------------------------------------------------------
    def encode(self, img: np.ndarray) -> np.ndarray:
        """`(H, W, 3)` fp32 归一图 → `(L/spatial_merge², out_hidden)` fp16。"""
        spec = self.spec
        started = time.perf_counter()
        before = (self.stats["ane_ms"], self.stats["gpu_ms"])
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
        tokens = patchify(img, patch=patch, merge=merge,
                          temporal=int(spec.temporal_patch_size)).astype(np.float16)
        pos = pos_embed_rows(self.pos_table, grid_h=grid_h, grid_w=grid_w, merge=merge,
                             side=self.side).astype(np.float16)
        rows, cols = block_positions(grid_h, grid_w, merge=merge)
        cos, sin = rope_tables(rows, cols, head_dim=self.head_dim, theta=float(spec.rope_theta))
        cos16, sin16 = cos.astype(np.float16), sin.astype(np.float16)

        residual = np.empty((length, self.hidden), np.float16)
        try:
            # 1) patch 投影 + pos_embed（两块 fp16 相加，与厂商的 `h + pos` 同一步）
            for lo, n in self._tiles(length):
                self._put("patch_in", self.cin, tokens[lo:lo + n].T)
                self._call(self._programs["patch"], "main")
                residual[lo:lo + n] = self._take(self._programs["patch"], "main", n).T + pos[lo:lo + n]

            # 2) 24 个 block：每层先把整图的 qkv 收齐，注意力按**整图**算，再分趟喂回去
            qkv = np.empty((3 * self.hidden, length), np.float16)
            qkv_dev = self.metal.alloc(3 * self.hidden * length * 2)
            ctx_dev = self.metal.alloc(self.hidden * length * 2)
            try:
                qkv_view = qkv_dev.array(np.float16, (3 * self.hidden, length))
                ctx_view = ctx_dev.array(np.float16, (self.hidden, length))
                for index in range(int(spec.depth)):
                    block = self._programs[f"block{index:02d}"]
                    for lo, n in self._tiles(length):
                        self._put("hidden_in", self.hidden, residual[lo:lo + n].T)
                        self._call(block, "main")
                        qkv[:, lo:lo + n] = self._take(block, "main", n)
                    rope_qk(qkv, cos16, sin16, hidden=self.hidden, head_dim=self.head_dim)
                    qkv_view[...] = qkv
                    vattn(self.metal, qkv_dev, ctx_dev, T=length, nq=length,
                          hidden=self.hidden, dim=self.head_dim, scale=self.scale)
                    self.stats["gpu_ms"] += self.metal.sync()
                    for lo, n in self._tiles(length):
                        attn = self._view("attn", 2 * self.hidden)
                        attn[:self.hidden, :n] = ctx_view[:, lo:lo + n]
                        attn[self.hidden:, :n] = residual[lo:lo + n].T
                        self._call(block, "p1")
                        self._call(block, "p2")          # 读 `p1o`、写 `hidden_out`
                        residual[lo:lo + n] = self._take(block, "p2", n).T
            finally:
                qkv_dev.free()
                ctx_dev.free()

            # 3) merger：每趟把 2×2 合成 4·hidden，再 fc1 → gelu → fc2
            merged = np.empty((length // self.unit, self.out_hidden), np.float16)
            for lo, n in self._tiles(length):
                self._put("hidden_in", self.hidden, residual[lo:lo + n].T)
                self._call(self._programs["merger"], "main")
                self._call(self._programs["merger"], "p1")
                merged[lo // self.unit:(lo + n) // self.unit] = \
                    self._take(self._programs["merger"], "p1", n // self.unit).T
        finally:
            after = (self.stats["ane_ms"], self.stats["gpu_ms"])
            busy = (after[0] - before[0]) + (after[1] - before[1])
            self.stats["host_ms"] += (time.perf_counter() - started) * 1e3 - busy
        return merged

    # -- 面与小工具 -----------------------------------------------------------
    def _tiles(self, length: int):
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
        """读输出面的前 `cols` 列（**视图**，调用方要么当场用完，要么自己拷走）。"""
        out = program.procs[proc][1]
        cout = program.io[proc][2]
        s_out = program.io[proc][3]
        view = out.array(np.float16, (1, cout, 1, s_out))[0, :, 0]
        return view[:, :cols]

    def _call(self, program: _Program, proc: str) -> None:
        self.stats["ane_ms"] += program.run(self.runtime, proc)
        self.stats["evals"] += 1

    # -- 生命周期 ------------------------------------------------------------
    @property
    def reuse(self) -> tuple[int, int]:
        """产物命中数 / 总数：与 `cached_program` 同一判据（`doctor` 与门共用这一个）。"""
        return (sum(1 for p in self._programs.values() if p.files.reused), len(self._programs))

    @property
    def bytes(self) -> int:
        """常驻字节：26 份 blob（ANE 权重）+ 一套面 —— 与 `programs` 的装载期账一致。"""
        blobs = sum(p.files.blob_path.stat().st_size for p in self._programs.values())
        return blobs + sum(surface.nbytes for surface in self._surfaces.values())

    def close(self) -> None:
        for program in self._programs.values():
            program.close(self.runtime)
        self._programs.clear()
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
    """一份塔产物的身份：权重容器指纹 + 塔的形状 + 桶宽 + 是哪一份。

    `cached_program` 只比这个字典，不比哈希（那是缓存要省掉的那一段），所以凡是会改变产物
    的输入都得进来 —— 尤其是**塔的形状**：同一份 `family-model` 目录下换了 config 的检查点
    不能认领彼此的产物。
    """
    return {
        "weights": store.fingerprint(),
        "kind": kind,
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
    * **不预热**（`e8e-service-plan.md` T6）⇒ 第一次带图请求才建塔。启动时就把塔装上等于选了
      常驻，0 常驻的契约当场作废。

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

    # -- 装载与卸载 ----------------------------------------------------------
    def warm(self) -> VisionTower:
        """只装不跑：`sysone compile --vision` 用它把 26 份产物提前建出来。"""
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

    def close(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._drop()

    # -- 跑 ------------------------------------------------------------------
    def encode(self, img: np.ndarray) -> np.ndarray:
        with self._lock:
            tower = self._ensure()
            before = (tower.stats["evals"], tower.stats["ane_ms"], tower.stats["gpu_ms"])
            rows = tower.encode(img)
            after = (tower.stats["evals"], tower.stats["ane_ms"], tower.stats["gpu_ms"])
            self._evals += after[0] - before[0]
            self._ane_ms += after[1] - before[1]
            self._gpu_ms += after[2] - before[2]
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
        blocks: list[np.ndarray] = []
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
            blocks.append(np.asarray(self.encode(pixels)))
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
                # 这三个是**累计**的（塔卸载之后也还在），否则 `/metrics` 会把历史抹掉。
                "evals": self._evals,
                "ane_ms": round(self._ane_ms, 3),
                "gpu_ms": round(self._gpu_ms, 3),
            }
