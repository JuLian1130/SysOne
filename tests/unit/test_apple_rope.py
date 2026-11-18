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


def test_three_axis_positions_compose_the_way_the_vendor_does() -> None:
    """`(3, T)` 按厂商 `recomposition_frequencies` 拼频率，不是拿时间轴顶替三段。

    抄的是 `Qwen3VLTextRotaryEmbedding`：

        freqs_thw = freq[0]                            # 时间轴打底
        for dim, offset in ((1, 1), (2, 2)):           # 行轴、列轴
            length = section[dim] * 3
            freqs_thw[..., offset:length:3] = freq[dim][..., offset:length:3]

    行轴占 `i % 3 == 1` 的频率、列轴占 `i % 3 == 2`，其余留在时间轴。这条用例的重点是后两句
    断言：三行如果被拍平成同一行，它们会相等 —— 而那正是只在端到端对拍时才显形的那种错。
    """
    theta, rotary_dim, section = 10_000.0, 64, (11, 11, 10)
    positions = np.array([[0, 5, 5, 5], [0, 3, 3, 3], [0, 7, 7, 7]], dtype=np.float64)
    cos, sin = _rope_positions(theta, positions, rotary_dim, section=section, interleaved=True)

    frequencies = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angles = positions[:, :, None] * frequencies[None, None, :]
    want = angles[0].copy()
    for axis, offset in ((1, 1), (2, 2)):
        window = slice(offset, section[axis] * 3, 3)
        want[:, window] = angles[axis][:, window]
    np.testing.assert_allclose(cos, np.cos(want), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(sin, np.sin(want), rtol=1e-6, atol=1e-6)

    assert not np.array_equal(cos[1], cos[0]), "行轴没有生效（退化成了时间轴）"
    assert not np.array_equal(cos[2], cos[0]), "列轴没有生效（退化成了时间轴）"
    # 一维那条路逐位不变
    flat_cos, _ = _rope_positions(theta, np.arange(4), rotary_dim)
    assert np.array_equal(
        flat_cos, np.cos(np.arange(4)[:, None] * frequencies[None, :]).astype(np.float32))


def test_positions_of_the_wrong_rank_are_refused() -> None:
    """`(2, T)` 这种既不是文本也不是三轴的形状要当场报错，不能猜一个。"""
    with pytest.raises(NotReadyError) as excinfo:
        _rope_positions(10_000.0, np.zeros((2, 8), dtype=np.float64), 64)
    assert excinfo.value.context["positions"] == (2, 8)
