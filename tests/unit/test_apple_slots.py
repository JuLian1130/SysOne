"""批尾巴的槽位布局（票据 10）：纯函数，不需要 ANE。"""

from types import SimpleNamespace

import pytest

from sysone.devices.apple.pipeline import MudePipeline

STUB = SimpleNamespace(_bucket=1024, _width=20480)   # 16k 前缀 + 两块打包区


def _layout(prefix: int, lengths: tuple[int, ...], *, round_up: bool = False):
    return MudePipeline._slot_layout(STUB, prefix, lengths, round_up=round_up)


def test_layout_packs_equal_and_unequal_tails() -> None:
    """四条约 60 token 的尾巴：槽宽 128、每桶 8 格、一个桶的打包区。"""
    layout = _layout(16384, (60, 60, 60, 60))
    assert layout is not None
    assert (layout.slotw, layout.per, layout.ntiles, layout.nt) == (128, 8, 1, 1024)
    assert layout.starts == (16384, 16512, 16640, 16768)

    layout = _layout(16384, (60, 63, 66, 69))
    assert layout is not None and layout.nt == 1024 and layout.starts[1] == 16384 + 128


def test_layout_widens_the_slot_for_longer_tails() -> None:
    """500 token 的尾巴：槽宽 512、每桶 2 格、两个桶（八格装不下四格以外的余量）。"""
    layout = _layout(16384, (500, 500, 500, 500))
    assert layout is not None
    assert (layout.slotw, layout.per, layout.ntiles, layout.nt) == (512, 2, 2, 2048)


@pytest.mark.parametrize("prefix,lengths", [
    (16384, (60,)),                       # 一题没有可批的东西
    (16384, (1025, 60)),                  # 尾巴比一个桶还长
])
def test_layout_declines(prefix: int, lengths: tuple[int, ...]) -> None:
    assert _layout(prefix, lengths) is None


def test_layout_takes_any_prefix() -> None:
    """前缀不必落在 64 或桶边界：打包区从 `prefix` 自己起切。

    对不齐的前缀，槽 0 与逐题整趟同列（逐位），槽 1..N 的 key 从 `prefix + j*slotw` 起，
    差在 64-key 块的分组（见 `_slot_layout` 的数值契约与 `.scratch/probe_batch_align.py`）。
    """
    for prefix in (16384 + 1, 16384 + 32, 16384 + 64):
        layout = _layout(prefix, (60, 60, 60, 60))
        assert layout is not None, f"prefix={prefix} 被挡掉了"
        assert (layout.slotw, layout.per, layout.ntiles, layout.nt) == (128, 8, 1, 1024)
        assert layout.starts == tuple(prefix + j * 128 for j in range(4))


def test_layout_refuses_more_slots_than_the_table_holds() -> None:
    """`AttnArgs` 的可见行数表是 8 槽 × 16 bit（`kernels.attn_args` 对第 9 槽起丢弃）。"""
    stub = SimpleNamespace(_bucket=2048, _width=8192)
    assert MudePipeline._slot_layout(stub, 2048, (128, 128, 128, 128)) is None


def test_layout_needs_a_bucket_of_slack() -> None:
    """宽缓冲只到前缀（真实 `serve` 的形状）：打包区没地方放，退回逐题。"""
    stub = SimpleNamespace(_bucket=1024, _width=16384)
    assert MudePipeline._slot_layout(stub, 16384, (60, 60)) is None


# 桶 1024 下"能打包"的形状是**离散的四段**（槽宽必须整除桶宽），不是"短就能批"。
# 这是"只对短提示词开合批"最容易被误判的地方：615 token 的 prompt 槽宽 640，一样进不去。
@pytest.mark.parametrize("length,packable", [
    (64, True),      # (0,128] → 槽宽 128、每桶 8 格
    (128, True),
    (129, True),     # (128,256] → 槽宽 256、每桶 4 格
    (256, True),
    (257, False),    # 槽宽 384 不整除 1024
    (384, False),
    (385, True),     # (384,512] → 槽宽 512、每桶 2 格
    (512, True),
    (513, False),    # 槽宽 640/768/896 都不整除 1024
    (615, False),
    (896, False),
    (897, True),     # (896,1024] → 槽宽 1024、每桶 1 格
    (1024, True),
    (1025, False),   # 超过桶宽：一条序列会跨瓦片，而 GDN 的递归状态只有一份
])
def test_tail_layout_packs_only_the_bands_whose_slot_divides_the_bucket(length: int,
                                                                      packable: bool) -> None:
    layout = _layout(0, (length, length))
    assert (layout is not None) is packable, f"长度 {length} 的判定变了"


def test_layout_uses_the_longest_sequence_for_the_slot() -> None:
    """槽宽由**最长**那条决定：混进一条落在洞里的长度，整批一起退回逐题。"""
    assert _layout(0, (100, 100)) is not None
    assert _layout(0, (100, 615)) is None


def test_block_diagonal_rounds_the_slot_up_to_fit_every_short_shape() -> None:
    """块对角把槽宽取到最近的“整除桶宽的 128 倍数”：落在洞里的长度也能合（最差 per=1）。

    h2o 真实题面 p50 是 615 token —— 贴紧取整给出 640，`640 ∤ 1024`，旧规则下它永远合不上。
    向上取到 1024 只多算 padding 列，而 ANE 按瓦片计费（一瓦片一条序列时成本与列宽无关）。
    """
    hole = _layout(0, (615, 615), round_up=True)
    assert hole is not None and (hole.slotw, hole.per) == (1024, 1)

    band = _layout(0, (257, 257), round_up=True)
    assert band is not None and (band.slotw, band.per) == (512, 2)

    small = _layout(0, (100, 100), round_up=True)
    assert small is not None and (small.slotw, small.per) == (128, 8), "能贴紧时不该被抬高"

    assert _layout(0, (1025, 1025), round_up=True) is None, "超过桶宽仍然不行"
    assert _layout(0, (615, 615)) is None, "批尾巴仍按贴紧槽宽，不吃这个放宽"
