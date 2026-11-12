"""层计划：把 32/20 层切成"块"，块边界落在全注意力层之前。

为什么在 FULL 层之前切：全注意力层要先把整个序列的 K/V 算完才能出第一行查询，
GDN 层则逐 token 增量、不必等；把 FULL 当作块的起点，层环（ring）的填充/排空自然
按"跨层流水"发生。`ring` 是层环容量，块不能超过它，否则常驻放不下。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.attention import AttentionRole
from ..errors import ConfigError


@dataclass(frozen=True)
class Block:
    """一个块：连续若干层，`starts_with_full` 表示首层是全注意力。"""

    index: int
    layers: tuple[int, ...]
    starts_with_full: bool

    def __len__(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class LayerPlan:
    """整模型的块划分 + 每层角色。"""

    blocks: tuple[Block, ...]
    roles: tuple[AttentionRole, ...]

    def block_of(self, layer: int) -> int:
        """层号 → 块号；越界抛 `IndexError`。"""
        for block in self.blocks:
            if layer in block.layers:
                return block.index
        raise IndexError(f"层 {layer} 不在任何块中")


def plan_blocks(layer_types: tuple[AttentionRole, ...], ring: int) -> tuple[Block, ...]:
    """按 FULL 层切段，再贪心合并到不超过 `ring` 层。"""
    if ring < 1:
        raise ConfigError("ring 必须 >= 1", ring=ring)
    if not layer_types:
        raise ConfigError("layer_types 不能为空")

    segments: list[tuple[int, int]] = []
    start = 0
    for index, role in enumerate(layer_types):
        if role is AttentionRole.FULL and index > 0:
            segments.append((start, index))
            start = index
    segments.append((start, len(layer_types)))

    merged: list[tuple[int, int]] = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            if segment[1] - segment[0] + previous[1] - previous[0] <= ring:
                merged[-1] = (previous[0], segment[1])
                continue
        merged.append(segment)

    blocks = tuple(
        Block(
            index=index,
            layers=tuple(range(start_index, end_index)),
            starts_with_full=layer_types[start_index] is AttentionRole.FULL,
        )
        for index, (start_index, end_index) in enumerate(merged)
    )
    return blocks


def plan_layers(
    layer_types: tuple[AttentionRole, ...], ring: int
) -> LayerPlan:
    """`plan_blocks` 的完整版（含角色列表）。"""
    return LayerPlan(blocks=plan_blocks(layer_types, ring), roles=layer_types)
