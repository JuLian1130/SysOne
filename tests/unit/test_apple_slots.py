"""批尾巴的槽位布局（票据 10）：纯函数，不需要 ANE。"""

from types import SimpleNamespace

import pytest

from sysone.devices.apple.pipeline import MudePipeline

STUB = SimpleNamespace(_bucket=1024, _width=20480)   # 16k 前缀 + 两块打包区


def _layout(prefix: int, lengths: tuple[int, ...]):
    return MudePipeline._slot_layout(STUB, prefix, lengths)


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
    (16384 + 1, (60, 60)),                # 前缀不在 key 块（64）边界
    (16384 + 32, (60, 60)),               # 同上：离块边界只差 32 也不行
    (16384, (1025, 60)),                  # 尾巴比一个桶还长
])
def test_layout_declines(prefix: int, lengths: tuple[int, ...]) -> None:
    assert _layout(prefix, lengths) is None


def test_layout_only_needs_the_key_block_alignment() -> None:
    """64 对齐就够（1088 实测逐位）：桶（1024）对齐是更严的旧条件，不是正确性边界。"""
    layout = _layout(16384 + 64, (60, 60, 60, 60))
    assert layout is not None
    assert (layout.slotw, layout.per, layout.ntiles, layout.nt) == (128, 8, 1, 1024)
    assert layout.starts[0] == 16384 + 64


def test_layout_refuses_more_slots_than_the_table_holds() -> None:
    """`AttnArgs` 的可见行数表是 8 槽 × 16 bit（`kernels.attn_args` 对第 9 槽起丢弃）。"""
    stub = SimpleNamespace(_bucket=2048, _width=8192)
    assert MudePipeline._slot_layout(stub, 2048, (128, 128, 128, 128)) is None


def test_layout_needs_a_bucket_of_slack() -> None:
    """宽缓冲只到前缀（真实 `serve` 的形状）：打包区没地方放，退回逐题。"""
    stub = SimpleNamespace(_bucket=1024, _width=16384)
    assert MudePipeline._slot_layout(stub, 16384, (60, 60)) is None
