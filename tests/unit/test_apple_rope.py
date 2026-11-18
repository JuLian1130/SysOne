"""RoPE 位置表的接缝：任意位置都要能查表，三轴位置不能悄悄被拍平。

`_rope_table` 是"0..s-1"的便利包装，真正做表的是 `_rope_positions`。合流之后一条 prompt 里
每个 token 的位置不再是它自己的下标（图上那些行按 `(t, h, w)` 三段算），所以"表从位置来"
这件事必须在 P0 就立住 —— 断言它今天就成立，比等 P3 撞上再回头改便宜。
"""

from __future__ import annotations

import numpy as np
import pytest

from sysone.devices.apple.chain import _rope_positions, _rope_table
from sysone.errors import NotReadyError


def test_the_table_is_the_prefix_of_the_same_function() -> None:
    """`_rope_table(theta, s, …)` 就是位置 `0..s-1` 的那几行 —— 不为它另写一套算法。"""
    theta, rotary_dim = 10_000.0, 64
    full_cos, full_sin = _rope_table(theta, 512, rotary_dim)
    cos, sin = _rope_positions(theta, np.arange(512), rotary_dim)
    assert np.array_equal(full_cos, cos) and np.array_equal(full_sin, sin)


def test_any_positions_can_be_looked_up() -> None:
    """位置可以是任意顺序、可以重复：合流时图上那些行根本不是连续下标。"""
    theta, rotary_dim = 10_000.0, 64
    full_cos, full_sin = _rope_table(theta, 200, rotary_dim)
    positions = np.array([7, 3, 199, 0, 7])
    cos, sin = _rope_positions(theta, positions, rotary_dim)
    assert cos.dtype == np.float32 and cos.shape == (positions.size, rotary_dim // 2)
    assert sin.shape == cos.shape
    assert np.array_equal(cos, full_cos[positions])
    assert np.array_equal(sin, full_sin[positions])


def test_three_axis_positions_are_not_flattened_silently() -> None:
    """M-RoPE 的 `(3, T)` 还没接：明确报错，不要拿时间轴顶替三段。

    静默退化成 `positions[0]` 会让所有图块的行/列位置都等于时间位置 —— 塔能跑、注意力是错的，
    而且只有在与厂商对拍时才看得出来。
    """
    with pytest.raises(NotReadyError) as excinfo:
        _rope_positions(10_000.0, np.zeros((3, 8), dtype=np.float64), 64)
    assert excinfo.value.context["positions"] == (3, 8)
    assert "7.4" in str(excinfo.value.context["hint"])
