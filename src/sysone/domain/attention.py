"""注意力语义：角色 × 掩码模式，以及"每窗口 key 上界"的掩码计划。

关键事实（见 docs/sysone/attention.md）：
- 三个模型都是混合结构：`LINEAR` 层（Gated DeltaNet）始终因果，`FULL` 层才可能双向；
- 因果/双向不是引擎的全局开关，而是**每种掩码模式在每窗口上的 key 上界**；
- `prefix_noncausal` 下，prefix 行只看 prefix，tail 行看 prefix + 自己的整个 tail，
  边界处是单向桥。所以"部分因果"只能体现为窗口边界，不能在同一窗口内混用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class AttentionRole(StrEnum):
    """层在结构里的角色。"""

    FULL = "full"
    LINEAR = "linear"


class MaskMode(StrEnum):
    """全注意力层看到的上下文范围。"""

    CAUSAL = "causal"
    NONCAUSAL_FULL = "noncausal_full"
    PREFIX_NONCAUSAL = "prefix_noncausal"


@dataclass(frozen=True)
class AttnWindow:
    """一个查询窗口（`q_start..q_start+q_len`）及其可读的 key 上界。

    `nk` 是硬上界（不含）：窗口内所有查询最多读到 key `nk-1`。
    双向块与单向桥同窗时，用 `nk_boundary` / `k_boundary` 把该窗口的查询分两段：
    - `q_start..q_start+k_boundary-1` 读到 key `< nk_boundary`（只读前缀）；
    - 其余查询读到 key `< nk`（读全部）。
    """

    q_start: int
    q_len: int
    nk: int
    nk_boundary: int | None = None
    k_boundary: int = 0

    def q_end(self) -> int:
        return self.q_start + self.q_len


@dataclass(frozen=True)
class MaskPlan:
    """一整条序列的掩码分解（窗口列表 + 语义参考掩码）。"""

    mode: MaskMode
    seq_len: int
    prefix_len: int
    tile: int
    windows: tuple[AttnWindow, ...]

    def as_reference(self) -> np.ndarray:
        """按语义直接生成 `[L, L]` 布尔掩码（True = 可见）。

        dtype 用 bool，形状与内核消费的参考实现一致；M0 没有 padding，故不含 padding 行。
        """
        length = self.seq_len
        queries = np.arange(length, dtype=np.int64)[:, None]
        keys = np.arange(length, dtype=np.int64)[None, :]
        if self.mode is MaskMode.CAUSAL:
            return keys <= queries
        if self.mode is MaskMode.NONCAUSAL_FULL:
            return np.ones((length, length), dtype=bool)
        prefix = self.prefix_len
        return (queries >= prefix) | (keys < prefix)

    def check(self) -> list[str]:
        """返回不变量违例（空列表 = 通过）。

        不变量：
        1. 窗口覆盖 `[0, seq_len)` 且互不重叠；
        2. 每窗 `nk` 在 `[q_start, seq_len]` 内；
        3. 模式语义：causal 每窗 `nk == min(seq_len, q_end)`；noncausal_full 每窗 `nk == seq_len`；
           prefix_noncausal 的前缀窗 `nk == prefix`、尾窗 `nk == seq_len`；
        4. 跨边界窗口必须同时给出 `nk_boundary == prefix` 与 `k_boundary == prefix - q_start`；
        5. `tile > 0` 时窗口边界必须落在 tile 倍数（或 0 / prefix / seq_len）。
        """
        violations: list[str] = []
        length, prefix, tile = self.seq_len, self.prefix_len, self.tile
        if length <= 0:
            violations.append(f"seq_len 必须为正: {length}")
        if not 0 <= prefix <= length:
            violations.append(f"prefix_len 越界: {prefix} (seq_len={length})")
        # `prefix_len` 只是"前 P 个 token 被多个问题共享"这一结构事实；causal 与
        # noncausal_full 的语义不依赖它（causal 的三角本身就保证了 prefix 行只看 prefix）。
        if not self.windows:
            violations.append("windows 为空")
            return violations

        cursor = 0
        allowed_edges = {0, length}
        if tile > 0:
            allowed_edges |= set(range(0, length + 1, tile))
        if 0 < prefix < length:
            allowed_edges.add(prefix)

        for index, window in enumerate(self.windows):
            if window.q_len <= 0:
                violations.append(f"窗口 {index}: q_len 必须为正 ({window.q_len})")
            if window.q_start != cursor:
                violations.append(f"窗口 {index}: q_start={window.q_start} 期望 {cursor}（不连续）")
            cursor = window.q_end()
            if window.q_end() > length:
                violations.append(f"窗口 {index}: q_end={window.q_end()} 超出 seq_len={length}")
                continue
            if not window.q_start <= window.nk <= length:
                violations.append(
                    f"窗口 {index}: nk={window.nk} 越界 [q_start={window.q_start}, {length}]"
                )
            violations.extend(self._check_mode(index, window))
            if tile > 0 and length > tile:
                if window.q_start not in allowed_edges or window.q_end() not in allowed_edges:
                    violations.append(
                        f"窗口 {index}: 边界 [{window.q_start},{window.q_end()}) 未对齐 tile={tile}"
                    )

        if cursor != length:
            violations.append(f"窗口覆盖不完整：结束于 {cursor}，seq_len={length}")
        return violations

    def _check_mode(self, index: int, window: AttnWindow) -> list[str]:
        out: list[str] = []
        if self.mode is MaskMode.CAUSAL:
            expected = min(self.seq_len, window.q_end())
            if window.nk != expected:
                out.append(f"窗口 {index}: causal 的 nk 应为 q_end={expected}，实际 {window.nk}")
            if window.nk_boundary is not None:
                out.append(f"窗口 {index}: causal 不应有 nk_boundary")
            return out
        if self.mode is MaskMode.NONCAUSAL_FULL:
            if window.nk != self.seq_len:
                out.append(f"窗口 {index}: noncausal_full 的 nk 应为 seq_len={self.seq_len}")
            if window.nk_boundary is not None:
                out.append(f"窗口 {index}: noncausal_full 不应有 nk_boundary")
            return out

        prefix = self.prefix_len
        if window.q_end() <= prefix:
            if window.nk != prefix:
                out.append(f"窗口 {index}: prefix 窗的 nk 应为 prefix={prefix}，实际 {window.nk}")
            if window.nk_boundary is not None:
                out.append(f"窗口 {index}: 完全在前缀内的窗口不应有 nk_boundary")
        elif window.q_start >= prefix:
            if window.nk != self.seq_len:
                out.append(f"窗口 {index}: tail 窗的 nk 应为 seq_len={self.seq_len}")
            if window.nk_boundary is not None:
                out.append(f"窗口 {index}: 完全在尾部的窗口不应有 nk_boundary")
        else:
            if window.nk != self.seq_len:
                out.append(f"窗口 {index}: 跨界窗口的 nk 应为 seq_len={self.seq_len}")
            if window.nk_boundary != prefix:
                out.append(f"窗口 {index}: 跨界窗口的 nk_boundary 应为 prefix={prefix}")
            expected_k = prefix - window.q_start
            if window.k_boundary != expected_k:
                out.append(f"窗口 {index}: 跨界窗口的 k_boundary 应为 {expected_k}，实际 {window.k_boundary}")
        if window.nk_boundary is not None and not 0 < window.nk_boundary < window.nk:
            out.append(f"窗口 {index}: nk_boundary 必须在 (0, nk) 内")
        return out

    def summary(self) -> str:
        """一行人类可读摘要，用于日志与 `doctor`。"""
        edges = ", ".join(f"{w.q_start}->{w.q_end()}:nk{w.nk}" for w in self.windows)
        return f"{self.mode.value} L={self.seq_len} P={self.prefix_len} tile={self.tile} [{edges}]"
