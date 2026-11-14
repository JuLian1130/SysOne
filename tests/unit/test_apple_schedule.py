"""双链步骤图的不变式：纯结构断言，不碰 GPU / ANE。"""

import pytest

from sysone.devices.apple.metal import LANE_LONG, LANE_SMALL
from sysone.devices.apple.schedule import (
    NCH,
    Chain,
    attention_steps,
    blocks,
    gdn_steps,
    ring_columns,
    ring_slot,
)
from sysone.domain.attention import AttentionRole

FULL_AT = (3, 7, 11, 15, 19)

# 全注意力层的两相步表（`attention_steps`）：第一相只产 K/V，第二相产 q 并立刻用它。
KV_PHASE = ("norm", "proj_kv", "gather_kv")
Q_PHASE = ("norm", "proj_q", "gather_q", "prep_q", "attn", "out", "norm2", "ffn", "tail")
BODY = ("attn", "out", "norm2", "ffn", "tail")


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
    """`nch=1`（第 2 步的单链驱动）下一条链走全部瓦片，且每个相里都是瓦片序。"""
    steps = gdn_steps(5, nch=1)
    assert [step.tile for step in steps] == [tile for tile in range(5) for _ in range(7)]

    steps = attention_steps(5, nch=1)
    prep = next(i for i, step in enumerate(steps) if step.work == "prep_kv")
    assert [step.tile for step in steps[:prep]] == [tile for tile in range(5) for _ in range(3)]
    assert [step.tile for step in steps[prep + 1:]] == [
        tile for tile in range(5) for _ in range(9)
    ]


def test_attention_sequence_is_kv_gathers_then_prep_then_attention():
    """`prep_kv` 是本层唯一的序列级屏障，且整条链的注意力都排在它后面。

    第一相只产 K/V（非因果注意力每个 query 都要读整条 kv），所以 `k` 的归一与 RoPE 必须等
    所有瓦片的 kv 都搬完；第二相的 q 只活一个瓦片，不需要这道屏障。
    """
    steps = attention_steps(4, par=0)
    gathers = [i for i, step in enumerate(steps) if step.work == "gather_kv"]
    preps = [i for i, step in enumerate(steps) if step.work == "prep_kv"]
    attns = [i for i, step in enumerate(steps) if step.work == "attn"]
    assert len(preps) == 1
    assert preps[0] > max(gathers)
    assert preps[0] < min(attns)


def test_the_q_gather_sits_next_to_its_attention():
    """A1 的全部安全性：`gather_q → prep_q → attn` 同一瓦片、同一条 lane 上连着排。

    这三步挨着，q 的寿命就只有这一个瓦片；环槽位每 `NCH` 个瓦片才被同一条链重写一次，
    而那时上一次的 `attn` 早已执行完（lane 0 的编码序即执行序）。
    """
    for par in (0, 1):
        steps = attention_steps(6, par=par)
        works = [step.work for step in steps]
        for index, work in enumerate(works):
            if work != "gather_q":
                continue
            tile = steps[index].tile
            assert works[index + 1] == "prep_q"
            assert works[index + 2] == "attn"
            assert [steps[index + k].tile for k in (1, 2)] == [tile, tile]
            assert {steps[index + k].lane for k in (0, 1, 2)} == {LANE_LONG}


def test_attention_and_body_stay_interleaved():
    """第二相必须按瓦片交错：把整层注意力提前实测两头都更慢（见 `attention_steps` 的注释）。

    这里的判据是"每个 `attn` 后面紧跟着它自己的 `out`"，也就是 lane 0 上永远只有一片注意力在飞。
    """
    for par in (0, 1):
        steps = attention_steps(6, par=par)
        attns = [i for i, step in enumerate(steps) if step.work == "attn"]
        assert len(attns) == 3
        for index in attns:
            assert steps[index + 1].work == "out"
            assert steps[index + 1].tile == steps[index].tile


def test_only_the_first_chain_carries_the_prep_kv():
    assert not any(step.work == "prep_kv" for step in attention_steps(4, par=1))


def test_attention_chain_builds_twelve_steps_per_owned_tile_plus_one_prep():
    steps = attention_steps(4, par=0)
    assert len(steps) == 2 * 12 + 1
    assert [step.work for step in steps[:6]] == list(KV_PHASE * 2)
    assert steps[6].work == "prep_kv"
    assert [step.work for step in steps[7:16]] == list(Q_PHASE)


def test_the_ring_slot_is_stable_per_chain():
    """槽位按奇偶循环：同一条链的每个瓦片落在同一个槽位，两条链各占一个。

    这条就是"环深 ≥ 链条数"在步表上的化身 —— 只要 `NCH` 个槽位，不必整窗。
    """
    for tile in range(8):
        assert ring_slot(tile, 64) == (tile % NCH) * 64
    for par in (0, 1):
        slots = {ring_slot(tile, 64) for tile in range(par, 8, NCH)}
        assert len(slots) == 1, f"par={par} 的瓦片落在多个槽位上"
    assert ring_slot(0, 64) != ring_slot(1, 64)


@pytest.mark.parametrize(
    "width,bucket,columns",
    [
        (1024, 1024, 1024),      # 1k：一片瓦片，一个槽位就够（不成环，一字节也不多花）
        (256, 64, 128),          # 四片瓦片：两个槽位
        (16384, 1024, 2048),     # 16k 桶 1024：两条链各 1024 列
        (16384, 4096, 8192),     # 16k 桶 4096：两条链各 4096 列
    ],
)
def test_ring_columns(width: int, bucket: int, columns: int) -> None:
    assert ring_columns(width, bucket) == columns


@pytest.mark.parametrize("dead", [0, 1, 3])
def test_dead_tiles_keep_the_first_pass_only(dead):
    """S3：整段落在前缀里的瓦片只欠 K/V，第二相（q 投影与注意力、MLP）整段丢掉。"""
    steps = attention_steps(5, par=0, dead=dead)
    second_pass = {"proj_q", "gather_q", "prep_q", "attn", "out", "norm2", "ffn", "tail"}
    for tile in (0, 2, 4):
        kept = {
            step.work
            for step in steps
            if step.tile == tile and step.work in second_pass
        }
        assert kept == (set() if tile < dead else second_pass)
        # 第一相不受影响：这些瓦片的 K/V 仍要 gather 出来
        assert set(KV_PHASE) <= {
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


def test_the_long_lane_carries_the_attention_and_the_preps():
    """lane 0 只放长任务：两条 gather / 两条 prep / attn；lane 1 放小核。"""
    lane = {step.work: step.lane for step in attention_steps(4, par=0)}
    for work in ("gather_kv", "gather_q", "prep_kv", "prep_q", "attn"):
        assert lane[work] == LANE_LONG, work
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
