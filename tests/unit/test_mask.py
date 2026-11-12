"""掩码计划：三种模式的窗口规则与参考掩码一致。"""

from __future__ import annotations

import numpy as np
import pytest

from sysone.domain.attention import AttnWindow, MaskMode, MaskPlan
from sysone.engine.planner import build_mask_plan


def expected_mask(mode: MaskMode, length: int, prefix: int) -> np.ndarray:
    queries = np.arange(length)[:, None]
    keys = np.arange(length)[None, :]
    if mode is MaskMode.CAUSAL:
        return keys <= queries
    if mode is MaskMode.NONCAUSAL_FULL:
        return np.ones((length, length), dtype=bool)
    return (queries >= prefix) | (keys < prefix)


@pytest.mark.parametrize("mode", list(MaskMode))
@pytest.mark.parametrize("tile", [0, 1, 2, 4, 8])
def test_plan_is_clean(mode: MaskMode, tile: int) -> None:
    for length in (1, 2, 3, 9, 17):
        prefix = 0 if mode is not MaskMode.PREFIX_NONCAUSAL else max(1, length // 2)
        plan = build_mask_plan(mode, length, prefix, tile)
        assert plan.check() == []
        assert plan.as_reference().shape == (length, length)
        assert np.array_equal(plan.as_reference(), expected_mask(mode, length, prefix))


def test_prefix_windows_are_split() -> None:
    plan = build_mask_plan(MaskMode.PREFIX_NONCAUSAL, seq_len=10, prefix_len=6, tile=4)
    starts = [window.q_start for window in plan.windows]
    assert starts == [0, 4, 6, 8]
    assert plan.windows[0].nk == 6
    assert plan.windows[2].nk == 10
    assert all(window.nk_boundary is None for window in plan.windows)


def test_straddling_window_carries_boundary() -> None:
    plan = build_mask_plan(
        MaskMode.PREFIX_NONCAUSAL, seq_len=10, prefix_len=6, tile=4, split_prefix=False
    )
    straddling = [window for window in plan.windows if window.nk_boundary is not None]
    assert len(straddling) == 1
    window = straddling[0]
    assert (window.q_start, window.nk, window.nk_boundary, window.k_boundary) == (4, 10, 6, 2)
    assert plan.check() == []


def test_causal_prefix_is_structural_only() -> None:
    """causal 的三角本身就让 prefix 行只看 prefix，`prefix_len` 不参与语义。"""
    plan = build_mask_plan(MaskMode.CAUSAL, seq_len=6, prefix_len=3, tile=2)
    assert plan.check() == []
    assert np.array_equal(plan.as_reference(), expected_mask(MaskMode.CAUSAL, 6, 3))


def test_bad_windows_are_reported() -> None:
    plan = MaskPlan(
        mode=MaskMode.NONCAUSAL_FULL,
        seq_len=4,
        prefix_len=0,
        tile=2,
        windows=(AttnWindow(q_start=0, q_len=2, nk=4),),
    )
    problems = plan.check()
    assert problems
    assert any("覆盖" in problem for problem in problems)
