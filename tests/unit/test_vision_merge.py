"""合流的三个纯函数：占位符展开、M-RoPE 位置、行覆盖（都不碰设备）。

这四条判据各自钉一种"形状全对、答案是错的"错法：

* 展开成 span 而不是一个偏移：第二张图的起点取决于第一张展开了多少；
* 位置表的三轴：图像 token 是 `(base, base+row, base+col)`，图之后的文本从 `max+1` 接着数
  （不是接着列号）—— 退化成一维会让所有行/列位置都等于时间位置；
* 行覆盖只动 span 那几段：盖错行等于图没进去、或者把邻题的文本盖了；
* 数不对就报错：占位符数与图数不符时照旧覆盖，会把一张图的行放到另一张的位置上。
"""

from __future__ import annotations

import numpy as np
import pytest

from sysone.models.vision.merge import (
    Media,
    Placement,
    attach,
    expand_image_tokens,
    mrope_positions,
    splice_rows,
    split_placement,
)

IMAGE = 7


def test_expansion_returns_spans_that_account_for_the_earlier_images() -> None:
    """第 i 张图的起点依赖前面各自展开了多少 —— 所以只能返回 span，不能返回一个偏移。"""
    ids = [1, 2, IMAGE, 3, IMAGE, 4]
    expanded, spans = expand_image_tokens(ids, (4, 3), image_token_id=IMAGE)
    assert expanded == [1, 2, IMAGE, IMAGE, IMAGE, IMAGE, 3, IMAGE, IMAGE, IMAGE, 4]
    assert spans == [(2, 4), (7, 3)]
    # 第二个 span 的起点不是"第一个 + 1"，而是"第一个展开完 + 中间那段文本"
    assert spans[1][0] == spans[0][0] + spans[0][1] + 1


def test_expansion_refuses_a_count_mismatch() -> None:
    """占位符与图数不符：报错。照旧覆盖会把一张图的行放到另一张的位置上。"""
    with pytest.raises(ValueError, match="占位符"):
        expand_image_tokens([1, IMAGE, IMAGE], (4,), image_token_id=IMAGE)
    with pytest.raises(ValueError, match="占位符"):
        expand_image_tokens([1, IMAGE], (4, 3), image_token_id=IMAGE)


def test_expansion_refuses_an_empty_image() -> None:
    with pytest.raises(ValueError, match="0 行"):
        expand_image_tokens([IMAGE], (0,), image_token_id=IMAGE)


def test_positions_are_three_axis_and_continue_after_the_image() -> None:
    """图像段是 `(base, base+row, base+col)`；图之后的文本三轴相同、从 `max+1` 起。"""
    ids = [10, 11, IMAGE, 12]
    counts, grids = (4,), ((2, 2),)
    positions = mrope_positions(ids, counts, grids, image_token_id=IMAGE)

    assert positions.shape == (3, 7)
    assert positions.dtype == np.int32
    # 前两个文本 token：三轴相同，从 0 数
    assert positions[:, :2].tolist() == [[0, 1], [0, 1], [0, 1]]
    # 图像：2×2 的网格，base = 2
    assert positions[:, 2:6].tolist() == [[2, 2, 2, 2], [2, 2, 3, 3], [2, 3, 2, 3]]
    # 图之后的文本：接着 `max + 1 = 4`，不是接着列号
    assert positions[:, 6].tolist() == [4, 4, 4]


def test_positions_of_the_second_image_start_after_the_text_that_follows() -> None:
    ids = [IMAGE, 12, IMAGE]
    positions = mrope_positions(ids, (4, 1), ((2, 2), (1, 1)), image_token_id=IMAGE)
    # 第一张 2×2：base 0，行/列 0..1，max = 1 ⇒ 下一个 st = 2
    assert positions[:, 0:4].tolist() == [[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1]]
    # 中间的文本 token 在 2，于是第二张图 base = 3
    assert positions[:, 4].tolist() == [2, 2, 2]
    assert positions[:, 5].tolist() == [3, 3, 3]


def test_positions_refuse_a_grid_too_small_for_the_rows() -> None:
    with pytest.raises(ValueError, match="网格"):
        mrope_positions([IMAGE], (5,), ((2, 2),), image_token_id=IMAGE)


def test_splice_covers_exactly_the_spans_and_nothing_else() -> None:
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    before = x.copy()
    rows = np.full((3, 2), -1.0, np.float32)
    splice_rows(x, [(2, 2), (7, 1)], rows)
    keep = np.ones(10, dtype=bool)
    keep[2:4] = keep[7] = False
    assert np.array_equal(x[keep], before[keep]), "span 之外被改动了"
    assert np.array_equal(x[2:4], rows[:2]) and np.array_equal(x[7], rows[2])


def test_splice_refuses_unused_rows_and_out_of_range_spans() -> None:
    x = np.zeros((4, 2), np.float32)
    with pytest.raises(ValueError, match="没用上"):
        splice_rows(x, [(0, 1)], np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="超出"):
        splice_rows(x, [(3, 3)], np.zeros((3, 2), np.float32))


def test_media_checks_rows_against_counts() -> None:
    with pytest.raises(ValueError, match="行"):
        Media(counts=(4,), grids=((2, 2),), rows=np.zeros((3, 8), np.float32))
    with pytest.raises(ValueError, match="条数"):
        Media(counts=(4,), grids=((2, 2), (1, 1)), rows=np.zeros((4, 8), np.float32))
    media = Media(counts=(4, 1), grids=((2, 2), (1, 1)), rows=np.zeros((5, 8), np.float32))
    assert media.images == 2


def test_attach_returns_a_placement_whose_positions_match_the_ids() -> None:
    media = Media(counts=(4,), grids=((2, 2),), rows=np.arange(8, dtype=np.float32).reshape(4, 2))
    expanded, placement = attach([1, IMAGE, 2], media, image_token_id=IMAGE)
    assert len(expanded) == placement.positions.shape[1] == 6
    assert placement.spans == ((1, 4),)
    assert np.array_equal(placement.rows, media.rows)


def test_split_placement_keeps_the_images_on_one_side() -> None:
    """共享前缀那条路把序列拆两段嵌入，图块必须整个落在一边。"""
    media = Media(counts=(4,), grids=((2, 2),), rows=np.full((4, 2), 9.0, np.float32))
    placement = Placement(spans=((1, 4),), rows=media.rows,
                          positions=np.zeros((3, 6), np.int32))
    head, tail = split_placement(placement, 5)
    assert head.spans == ((1, 4),) and np.array_equal(head.rows, media.rows)
    assert tail.spans == () and tail.rows.shape == (0, 2)

    head, tail = split_placement(placement, 1)      # 图整段落在尾巴里
    assert head.spans == ()
    assert tail.spans == ((0, 4),) and np.array_equal(tail.rows, media.rows)

    with pytest.raises(ValueError, match="跨在"):
        split_placement(placement, 3)
