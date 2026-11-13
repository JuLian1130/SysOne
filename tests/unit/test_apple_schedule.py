"""双链步骤图的不变式：纯结构断言，不碰 GPU / ANE。"""

import pytest

from sysone.devices.apple.metal import LANE_LONG, LANE_SMALL
from sysone.devices.apple.schedule import (
    Chain,
    attention_steps,
    blocks,
    gdn_steps,
)
from sysone.domain.attention import AttentionRole

FULL_AT = (3, 7, 11, 15, 19)


def _mude_roles() -> tuple[AttentionRole, ...]:
    return tuple(
        AttentionRole.FULL if index in FULL_AT else AttentionRole.LINEAR
        for index in range(20)
    )


def test_blocks_start_each_attention_layer():
    assert blocks(_mude_roles()) == (
        (0, 1, 2),
        (3, 4, 5, 6),
        (7, 8, 9, 10),
        (11, 12, 13, 14),
        (15, 16, 17, 18),
        (19,),
    )


def test_every_block_has_at_most_one_attention_layer_and_it_is_first():
    roles = _mude_roles()
    for group in blocks(roles):
        full = [index for index in group if roles[index] is AttentionRole.FULL]
        assert len(full) <= 1
        if full:
            assert full[0] == group[0]


def test_blocks_respect_the_ring():
    """`RING` 比注意力块还小时再切一刀，且切口仍落在注意力层上。"""
    roles = _mude_roles()
    groups = blocks(roles, ring=2)
    assert all(len(group) <= 2 for group in groups)
    assert (3, 4) in groups


def test_gdn_chain_builds_seven_steps_per_owned_tile():
    steps = gdn_steps(6)
    assert len(steps) == 3 * 7
    assert [step.work for step in steps[:7]] == [
        "norm",
        "proj",
        "core",
        "out",
        "norm2",
        "ffn",
        "tail",
    ]
    assert [step.tile for step in steps] == [
        tile for tile in (0, 2, 4) for _ in range(7)
    ]


def test_gdn_chain_owns_only_its_parity():
    assert [step.tile for step in gdn_steps(6, par=1)] == [
        tile for tile in (1, 3, 5) for _ in range(7)
    ]


def test_the_two_chains_of_a_layer_partition_the_tiles():
    first = {step.tile for step in gdn_steps(7, par=0)}
    second = {step.tile for step in gdn_steps(7, par=1)}
    assert first.isdisjoint(second)
    assert first | second == set(range(7))


def test_one_chain_covers_every_tile_in_order():
    """`nch=1`（第 2 步的单链驱动）下一条链走全部瓦片，且每个相里都是瓦片序。

    卷积尾巴与递归状态是按瓦片序接力的，所以"一条链"不只是少一条，它还把顺序交回给一条链。
    全注意力层的两相各自从瓦片 0 从头走一遍（第一相按瓦片攒 K/V，第二相按瓦片算 body）。
    """
    steps = gdn_steps(5, nch=1)
    assert [step.tile for step in steps] == [tile for tile in range(5) for _ in range(7)]

    steps = attention_steps(5, nch=1)
    prep = next(i for i, step in enumerate(steps) if step.work == "prep")
    assert [step.tile for step in steps[:prep]] == [tile for tile in range(5) for _ in range(3)]
    assert [step.tile for step in steps[prep + 1:]] == [
        tile for tile in range(5) for _ in range(5)
    ]


def test_attention_sequence_is_gathers_then_prep_then_attention():
    steps = attention_steps(4, par=0)
    gathers = [i for i, step in enumerate(steps) if step.work == "gather"]
    preps = [i for i, step in enumerate(steps) if step.work == "prep"]
    attns = [i for i, step in enumerate(steps) if step.work == "attn"]
    assert len(preps) == 1
    assert preps[0] > max(gathers)
    assert preps[0] < min(attns)


def test_only_the_first_chain_carries_the_prep():
    assert not any(step.work == "prep" for step in attention_steps(4, par=1))


def test_attention_chain_builds_eight_steps_per_owned_tile_plus_one_prep():
    steps = attention_steps(4, par=0)
    assert len(steps) == 2 * 8 + 1
    assert [step.work for step in steps[:6]] == ["norm", "proj1", "gather"] * 2
    assert steps[6].work == "prep"
    assert [step.work for step in steps[7:12]] == [
        "attn",
        "out",
        "norm2",
        "ffn",
        "tail",
    ]


@pytest.mark.parametrize("dead", [0, 1, 3])
def test_dead_tiles_keep_the_first_pass_only(dead):
    """S3：整段落在前缀里的瓦片只欠 K/V，第二相（注意力与 MLP）整段丢掉。"""
    steps = attention_steps(5, par=0, dead=dead)
    second_pass = {"attn", "out", "norm2", "ffn", "tail"}
    for tile in (0, 2, 4):
        kept = {
            step.work
            for step in steps
            if step.tile == tile and step.work in second_pass
        }
        assert kept == (set() if tile < dead else second_pass)
        # 第一相不受影响：这些瓦片的 K/V 仍要 gather 出来
        assert {"norm", "proj1", "gather"} <= {
            step.work for step in steps if step.tile == tile
        }


def test_the_readout_tiles_always_keep_their_tail():
    """`tail` 是唯一写 `h` 的 step：`dead` 只允许丢掉前缀瓦片，query 行永远有 tail。"""
    for dead in (0, 1, 2):
        tailed = {
            step.tile
            for step in attention_steps(4, par=0, dead=dead)
            if step.work == "tail"
        }
        assert tailed == {tile for tile in (0, 2) if tile >= dead}


def test_gpu_steps_use_one_of_the_two_lanes():
    for steps in (gdn_steps(4), attention_steps(4, par=0)):
        for step in steps:
            if step.kind == "gpu":
                assert step.lane in (LANE_LONG, LANE_SMALL)


def test_the_long_lane_carries_the_attention_and_the_gather():
    """lane 0 只放长任务：gather / prep / attn；lane 1 放小核。"""
    lane = {step.work: step.lane for step in attention_steps(4, par=0)}
    assert lane["gather"] == lane["prep"] == lane["attn"] == LANE_LONG
    assert lane["norm"] == lane["tail"] == LANE_SMALL


def test_chain_cursor_walks_the_steps_and_cycles_the_surface_set():
    steps = attention_steps(4, par=0)
    chain = Chain(layer=3, steps=steps, sets=(0, 1), slot=0)
    walked = []
    while chain.cur() is not None:
        walked.append(chain.cur().work)
        chain.advance()
    assert walked == [step.work for step in steps]
    assert chain.done
    assert chain.cur() is None
    assert [chain.set_of(tile) for tile in range(4)] == [0, 1, 0, 1]
