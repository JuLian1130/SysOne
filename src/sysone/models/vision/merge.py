"""合流：占位符展开、M-RoPE 位置、行覆盖 —— 视觉塔与文本栈之间的那道接缝。

三件事，都在这儿且都在这儿只做一次（旧引擎 `e8_mm.py` 的三个纯函数原样搬过来）：

    ids, spans = expand_image_tokens(ids, counts)      # 1 ↔ 多：一张图展开成 L/4 个占位
    positions  = mrope_positions(ids, counts, grids)   # (3, T)，时间/行/列三轴
    splice_rows(x, spans, rows)                        # 用塔的输出盖掉 span 内的行

**契约只有一条**：题面里一张图占**一个** `<|image_pad|>` 占位符，塔为它产 `n_i` 行，占位符在
**token id 层**展开成 `n_i` 个同 id 的 token，塔的行**按顺序**盖掉恰好那几行的输入嵌入。占位符
自己那份 embedding 因此是垃圾 —— 设计如此。

为什么展开必须发生在 tokenize **之后**、并且要返回 span 而不是一个偏移：每张图展开的长度不同，
第二张图的起点取决于第一张展开了多少。返回 span 让"图 i 的行落在哪儿"这件事只有一个来源。

`positions` 的三轴不是可选的：`Qwen3_5TextRotaryEmbedding` 一个 token 读三个坐标，频率对 `i`
取轴 `i % 3`（`:func:`mrope_cos_sin` 在设备层做这件事）。文本 token 的三轴相等；图像 token
不同（`(时间, 行, 列)`），少了它图能"跑通"、注意力却是错的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Media:
    """一次请求的图像侧：塔已经算完，这里只记"产了几行 + 网格多大"。

    `counts[i]` 是第 i 张图交给文本栈的行数（`gh*gw/merge²`），`grids[i]` 是它在**合并之后**的
    `(gh, gw)`。两者都要：行数说"展开成几个占位符"，网格说"这些占位符在图上怎么摆" —— 后者
    推不出来（`n = gh*gw` 有无穷多组解，而且 6 个 token 可能是一行六列也可能是一列六行）。
    """

    counts: tuple[int, ...]
    grids: tuple[tuple[int, int], ...]
    rows: np.ndarray
    """`(Σ counts, out_hidden)`：每张图的塔输出按顺序拼起来（塔自己出 fp16，链会转 float32）。"""

    def __post_init__(self) -> None:
        if len(self.counts) != len(self.grids):
            raise ValueError(
                f"counts 与 grids 条数不一致：{len(self.counts)} vs {len(self.grids)}")
        if not self.counts:
            raise ValueError("Media 至少要有 1 张图")
        given = sum(self.counts)
        if self.rows.shape[0] != given:
            raise ValueError(
                f"塔给了 {self.rows.shape[0]} 行，counts 却要求 {given} 行")

    @property
    def images(self) -> int:
        return len(self.counts)


@dataclass(frozen=True)
class Placement:
    """一道题展开之后的落点：`spans` 与塔的行一一对应，`positions` 是 `(3, T)`。

    链只吃这个 —— 它不需要知道图是从哪个文件来的、网格多大。
    """

    spans: tuple[tuple[int, int], ...]
    rows: np.ndarray
    positions: np.ndarray


def expand_image_tokens(
    ids: Sequence[int], counts: Sequence[int], *, image_token_id: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """一张图一个占位符 → `counts[i]` 个；返回 `(展开后的 id, spans)`。

    `spans[i] = (start, n)` 是第 i 张图的行在序列里的落点，按序列顺序。

    **数不对就报错，不猜**：占位符数与图数不一致，说明渲染和图片列表各说各话；照旧覆盖只会把
    一张图的行盖到另一张图的位置上 —— 形状全对、答案是错的。
    """
    ids = list(ids)
    got = sum(1 for token in ids if token == image_token_id)
    if got != len(counts):
        raise ValueError(f"题面有 {got} 个占位符，却有 {len(counts)} 张图")
    out: list[int] = []
    spans: list[tuple[int, int]] = []
    index = 0
    for token in ids:
        if token != image_token_id:
            out.append(token)
            continue
        n = int(counts[index])
        if n < 1:
            raise ValueError(f"第 {index} 张图产出 {n} 行")
        spans.append((len(out), n))
        out.extend([image_token_id] * n)
        index += 1
    return out, spans


def mrope_positions(
    ids: Sequence[int], counts: Sequence[int], grids: Sequence[tuple[int, int]],
    *, image_token_id: int,
) -> np.ndarray:
    """`(3, T)` int32 位置表，见 `qwen3_vl._get_mrope_input_positions`：

    文本段 `arange(n)` 三轴相同、按 `st_idx` 平移；图像段 `np.indices((1, gh, gw))`（即
    `(base, base+row, base+col)`）+ `text_len + st_idx`；之后 `st_idx = 已产出的最大值 + 1`。

    `ids` 是**未展开**的题面（一张图一个占位符），`counts` 是每张图产几行。
    """
    ids = list(ids)
    placeholders = sum(1 for token in ids if token == image_token_id)
    if placeholders != len(counts):
        raise ValueError(f"题面有 {placeholders} 个占位符，却有 {len(counts)} 张图")
    if len(grids) != len(counts):
        raise ValueError(f"grids 与 counts 条数不一致：{len(grids)} vs {len(counts)}")
    total = len(ids) - placeholders + int(sum(counts))
    positions = np.zeros((3, total), dtype=np.int32)
    at = 0                       # ids 里的下标
    out = 0                      # positions 里的下标（展开后）
    start = 0                    # st_idx
    index = 0
    while at < len(ids):
        if ids[at] != image_token_id:
            run = at
            while run < len(ids) and ids[run] != image_token_id:
                run += 1
            n = run - at
            positions[:, out:out + n] = np.arange(start, start + n, dtype=np.int32)
            out += n
            start += n
            at = run
            continue
        n = int(counts[index])
        gh, gw = (int(value) for value in grids[index])
        if gh * gw < n:
            raise ValueError(f"第 {index} 张图：网格 {gh}×{gw} = {gh * gw} 行 < {n}")
        grid = np.indices((1, gh, gw)).reshape(3, -1)[:, :n] + start
        positions[:, out:out + n] = grid
        out += n
        start = int(grid.max()) + 1
        at += 1
        index += 1
    return positions[:, :out]


def splice_rows(
    x: np.ndarray, spans: Sequence[tuple[int, int]], rows: np.ndarray
) -> np.ndarray:
    """用 `rows` 盖掉 `x` 在 `spans` 那几段的行（原地），返回 `x`。

    `spans` 来自 `expand_image_tokens`，`rows` 是所有图的塔输出按同一顺序拼起来的，所以这就是
    `masked_scatter` 的偏移已经解好的版本（`_merge_multimodal_embeddings` / `_scatter_mm_embedding`）。
    区间之外一个字节都不碰 —— 这正是"合流只覆盖 span 内"那条门的对象。
    """
    at = 0
    for position, n in spans:
        block = rows[at:at + n]
        if block.shape[0] != n:
            raise ValueError(f"span 要 {n} 行，只拿到 {block.shape[0]} 行")
        if position < 0 or position + n > x.shape[0]:
            raise ValueError(
                f"span [{position}, {position + n}) 超出被嵌入的 {x.shape[0]} 行；"
                "跨瓦片边界的图不支持")
        x[position:position + n] = block
        at += n
    if at != rows.shape[0]:
        raise ValueError(f"{rows.shape[0] - at} 行没用上")
    return x


@dataclass(frozen=True)
class Splice:
    """一段连续 token 里的行覆盖：`spans` 相对**这一段的起点**，`rows` 只含这一段用到的行。

    共享前缀那条路要把同一条序列拆成"前缀 + 每条尾巴"分别嵌入，而 `Placement` 的 span 是整条
    序列上的绝对列号。`split_placement` 负责换算，算不了的（图块跨在前缀边界上）就报错 ——
    静默丢一段等于图没进去。
    """

    spans: tuple[tuple[int, int], ...]
    rows: np.ndarray


def split_placement(placement: Placement, at: int) -> tuple[Splice | None, Splice | None]:
    """按列号 `at` 把落点切成 `(前缀段, 尾巴段)`；某一段没有图就是 `None`。

    图块**整个**落在一边才合法：跨在边界上的话，两段各自的行都对不上号，只能报错。
    """
    head: list[tuple[int, int]] = []
    head_rows: list[np.ndarray] = []
    tail: list[tuple[int, int]] = []
    tail_rows: list[np.ndarray] = []
    offset = 0
    for position, n in placement.spans:
        block = placement.rows[offset:offset + n]
        offset += n
        if position + n <= at:
            head.append((position, n))
            head_rows.append(block)
        elif position >= at:
            tail.append((position - at, n))
            tail_rows.append(block)
        else:
            raise ValueError(
                f"span [{position}, {position + n}) 跨在 {at} 这个边界上；"
                "图块必须整个落在前缀或整个落在尾巴里")
    return (
        Splice(tuple(head), np.concatenate(head_rows) if head_rows else placement.rows[:0]),
        Splice(tuple(tail), np.concatenate(tail_rows) if tail_rows else placement.rows[:0]),
    )


def attach(
    ids: Sequence[int], media: Media, *, image_token_id: int
) -> tuple[list[int], Placement]:
    """展开 + 定位一次做完：`(未展开的题面, 塔的输出)` → `(展开后的题面, 落点)`。

    这是策略层唯一要调的合流入口（`split` / `pooled` 共用），三件纯函数各自仍然单独可测。
    """
    expanded, spans = expand_image_tokens(ids, media.counts, image_token_id=image_token_id)
    positions = mrope_positions(ids, media.counts, media.grids, image_token_id=image_token_id)
    if positions.shape[1] != len(expanded):
        raise ValueError(
            f"位置表 {positions.shape[1]} 列与展开后的 {len(expanded)} 个 token 不一致")
    return expanded, Placement(spans=tuple(spans), rows=media.rows, positions=positions)


__all__ = [
    "Media",
    "Placement",
    "Splice",
    "attach",
    "expand_image_tokens",
    "mrope_positions",
    "splice_rows",
    "split_placement",
]
