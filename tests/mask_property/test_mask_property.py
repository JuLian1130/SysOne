"""掩码属性测试：任意 (模式, 长度, 前缀, tile) 的计划都要自洽且等于参考掩码。"""

from __future__ import annotations

import random

import numpy as np

from sysone.domain.attention import MaskMode
from sysone.engine.planner import build_mask_plan

_TILES = (0, 1, 2, 3, 4, 5, 7, 8, 16, 37)


def expected_mask(mode: MaskMode, length: int, prefix: int) -> np.ndarray:
    queries = np.arange(length)[:, None]
    keys = np.arange(length)[None, :]
    if mode is MaskMode.CAUSAL:
        return keys <= queries
    if mode is MaskMode.NONCAUSAL_FULL:
        return np.ones((length, length), dtype=bool)
    return (queries >= prefix) | (keys < prefix)


def test_two_hundred_random_plans() -> None:
    rng = random.Random(20261112)
    modes = list(MaskMode)
    for _ in range(200):
        mode = rng.choice(modes)
        length = rng.randint(1, 40)
        tile = rng.choice(_TILES)
        prefix = 0 if mode is not MaskMode.PREFIX_NONCAUSAL else rng.randint(0, length)
        plan = build_mask_plan(mode, length, prefix, tile)
        problems = plan.check()
        assert problems == [], (mode, length, prefix, tile, problems)
        assert np.array_equal(plan.as_reference(), expected_mask(mode, length, prefix)), (
            mode, length, prefix, tile
        )


def test_straddling_variant() -> None:
    rng = random.Random(7)
    for _ in range(50):
        length = rng.randint(2, 40)
        tile = rng.choice(_TILES[1:])
        prefix = rng.randint(1, length - 1)
        plan = build_mask_plan(
            MaskMode.PREFIX_NONCAUSAL, length, prefix, tile, split_prefix=False
        )
        assert plan.check() == []
        assert np.array_equal(plan.as_reference(), expected_mask(MaskMode.PREFIX_NONCAUSAL, length, prefix))
