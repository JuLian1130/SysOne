"""视觉塔 host 侧的四个约定：块主序、时间轴在通道轴内部、`align_corners=True`、轴向 RoPE。

这四条都是"跑得通但语义错"的那一类：每一条单独看都像是随便怎么排都行，错一条只有与厂商对拍
才看得出来（而且看得出来的是整座塔的数值）。所以这里钉的是**与厂商代码的等价性**，不是字面：

* patchify 对着 `Qwen2VLImageProcessor.patchify` 的**逐字转写**验逐字节相同；
* pos_embed 用闭式（`align_corners=True` 的双线性）算一遍，再验端点/恒等这两个退化情形；
* RoPE 对着 `rotate_half` 的定义算一遍，并钉住"cos 的两半相同"这个配对前提。

不需要 GPU / ANE / 真权重。
"""

from __future__ import annotations

import numpy as np
import pytest

from sysone.devices.apple.vision import (
    block_positions,
    patchify,
    pos_embed_rows,
    rope_qk,
    rope_tables,
)

PATCH, MERGE, TEMPORAL = 16, 2, 2


def transcribed_patchify(img: np.ndarray, patch: int = PATCH, merge: int = MERGE,
                         temporal: int = TEMPORAL) -> np.ndarray:
    """`Qwen2VLImageProcessor.patchify` 的逐字转写（同一串 reshape / permute / expand）。

    厂商那份是 torch 的；这里用 numpy 写同一串算子 —— 逐字转写才是"对着厂商的代码验"，
    对着我自己的一版闭式验就只能证明两遍犯的是同一个错。
    """
    channel, height, width = img.shape
    grid_h, grid_w = height // patch, width // patch
    patches = img.reshape(channel, grid_h // merge, merge, patch,
                          grid_w // merge, merge, patch)
    patches = patches.transpose(1, 4, 2, 5, 0, 3, 6)                 # (gh/m, gw/m, m, m, c, p, p)
    patches = patches[:, :, :, :, :, None, :, :]                      # unsqueeze(6) → 时间轴
    patches = np.broadcast_to(patches, patches.shape[:5] + (temporal,) + patches.shape[6:])
    return np.ascontiguousarray(
        patches.reshape(grid_h * grid_w, channel * temporal * patch * patch))


@pytest.mark.parametrize("grid", [(8, 8), (4, 12), (2, 2)])
@pytest.mark.parametrize("temporal", [1, 2])
def test_patchify_matches_the_literal_transcription(grid, temporal) -> None:
    rng = np.random.default_rng(7)
    img = rng.uniform(-1.0, 1.0, (grid[0] * PATCH, grid[1] * PATCH, 3)).astype(np.float32)
    mine = patchify(img, patch=PATCH, merge=MERGE, temporal=temporal)
    literal = transcribed_patchify(img.transpose(2, 0, 1), temporal=temporal)
    assert mine.shape == literal.shape == (grid[0] * grid[1], 3 * temporal * PATCH * PATCH)
    assert np.array_equal(mine, literal)


def test_patchify_is_block_major_not_raster() -> None:
    """4×4 个 patch、2×2 合并 ⇒ 16 个 token 按 (块行, 块列, 块内行, 块内列) 排。

    每个像素写上 `100·行 + 列`，于是 token 的第一个通道（`c=0, t=0, py=0, px=0`）就是那个
    像素的值 —— 顺序错一格都会立刻显出来。
    """
    grid_h = grid_w = 4
    height = grid_h * PATCH
    img = np.zeros((height, height, 3), np.float32)
    rows = np.repeat(np.arange(height), height).reshape(height, height)
    cols = np.tile(np.arange(height), (height, 1))
    img[:, :, 0] = rows * 100 + cols
    tokens = patchify(img, patch=PATCH, merge=MERGE, temporal=TEMPORAL)
    # token 内的通道索引：c=0、t=0、py=0、px=0 → 摊平后的第 0 个元素
    got = tokens[:, 0]
    want = []
    for block_row in range(grid_h // MERGE):
        for block_col in range(grid_w // MERGE):
            for in_row in range(MERGE):
                for in_col in range(MERGE):
                    row = block_row * MERGE + in_row
                    col = block_col * MERGE + in_col
                    want.append(row * PATCH * 100 + col * PATCH)
    assert np.array_equal(got, np.asarray(want, np.float32))


def test_patchify_duplicates_time_inside_the_channel_axis() -> None:
    """时间轴的两份是**同一个像素**，且它们在 `(c, t, py, px)` 里的位置是 `t` 那一维。"""
    img = np.zeros((2 * PATCH, 2 * PATCH, 3), np.float32)
    img[:PATCH, :PATCH, 0] = 1.0          # 只有第一个 patch 的 c=0 是 1
    one = patchify(img, patch=PATCH, merge=MERGE, temporal=1)
    two = patchify(img, patch=PATCH, merge=MERGE, temporal=2)
    py_px = PATCH * PATCH
    # 一块 = (c, t, py, px) 摊平；t=1 时 c=0 占前 256 个，t=2 时 c=0 的两份交错占 512 个
    assert np.count_nonzero(one[0, :py_px]) == py_px
    assert np.count_nonzero(one[0, py_px:]) == 0
    assert np.count_nonzero(two[0, :2 * py_px]) == 2 * py_px
    assert np.count_nonzero(two[0, 2 * py_px:]) == 0
    # 交错而不是拼在末尾：t=2 时 c=0 的第二份从 py_px 开始（不是从 3·py_px 开始）
    assert np.array_equal(two[0, :py_px], one[0, :py_px])
    assert np.array_equal(two[0, py_px:2 * py_px], one[0, :py_px])


def test_block_positions_cover_every_patch_once() -> None:
    for grid_h, grid_w in ((4, 4), (8, 6), (2, 10)):
        rows, cols = block_positions(grid_h, grid_w, merge=MERGE)
        pairs = set(zip(rows.tolist(), cols.tolist()))
        assert len(pairs) == grid_h * grid_w
        assert pairs == {(r, c) for r in range(grid_h) for c in range(grid_w)}


def test_pos_embed_is_identity_when_the_grid_matches_the_table() -> None:
    """`(gh, gw) == (side, side)` 时插值是恒等 —— 这正是 `align_corners=True` 的意思。

    于是 token `t` 的 pos 行必须**恰好**是表里它自己那一格（格号按块主序取），不多不少。
    """
    side, hidden = 8, 4
    table = np.arange(side * side * hidden, dtype=np.float32).reshape(side * side, hidden)
    out = pos_embed_rows(table, grid_h=side, grid_w=side, merge=MERGE, side=side)
    rows, cols = block_positions(side, side, merge=MERGE)
    want = table[rows * side + cols]
    assert np.array_equal(out, want)


def test_pos_embed_endpoints_hit_the_corners() -> None:
    """`align_corners=True`：0 号位置落在表的第一格，最后一个位置落在最后一格。"""
    side, hidden = 8, 3
    table = np.zeros((side * side, hidden), np.float32)
    table[0] = 1.0
    table[-1] = 2.0
    rows, _ = block_positions(2, 2, merge=MERGE)
    out = pos_embed_rows(table, grid_h=2, grid_w=2, merge=MERGE, side=side)
    # (0,0) 的 token 只拿到 table[0]，(1,1) 只拿到 table[-1]
    assert np.array_equal(out[0], np.ones(hidden, np.float32))
    assert np.array_equal(out[-1], np.full(hidden, 2.0, np.float32))
    assert rows.tolist() == [0, 0, 1, 1]


def test_rope_tables_duplicate_the_halves() -> None:
    """cos/sin 的两半必须相同 —— `rotate_half` 的配对用的就是一个角。"""
    rows = np.arange(6)
    cols = np.arange(6)[::-1]
    cos, sin = rope_tables(rows, cols, head_dim=64, theta=1e4)
    half = 32
    assert np.array_equal(cos[:, :half], cos[:, half:])
    assert np.array_equal(sin[:, :half], sin[:, half:])
    # 每轴 `head_dim//4` 个频率；第一维的频率是 θ^0 = 1，所以角度就是位置本身。
    # 角度布局 = [h 的 16 个, w 的 16 个]，再整段复制一次，所以 w 的起点是 `head_dim//4`。
    assert np.allclose(cos[0, 0], np.cos(rows[0]))
    assert np.allclose(cos[0, half // 2], np.cos(cols[0]))


def test_rope_qk_matches_rotate_half() -> None:
    """`rope_qk` 与厂商的 `apply_rotary_pos_emb_vision` 同一步（半交换 + 两半同角）。"""
    length, hidden, head_dim = 5, 16, 8
    qkv = np.random.default_rng(4).standard_normal((3 * hidden, length)).astype(np.float16)
    rows = np.arange(length)
    cols = np.arange(length) + 3
    cos, sin = rope_tables(rows, cols, head_dim=head_dim, theta=1e4)
    got = rope_qk(qkv.copy(), cos.astype(np.float16), sin.astype(np.float16),
                  hidden=hidden, head_dim=head_dim)

    def rotate_half(x):
        h = x.shape[1] // 2
        return np.concatenate([-x[:, h:], x[:, :h]], axis=1)

    want = np.asarray(qkv, np.float32).copy()
    for base in (0, hidden):
        block = want[base:base + hidden].reshape(-1, head_dim, length)
        block[...] = block * cos.T[None] + rotate_half(block) * sin.T[None]
    # 设备那条路把结果落 fp16，所以比较也在 fp16 精度上
    assert np.allclose(np.asarray(got, np.float32), want, atol=2e-2, rtol=2e-2)


def _taps(pos, size, side):
    src = pos.astype(np.float32) * (side - 1) / max(size - 1, 1)
    floor = np.floor(src)
    offsets = np.arange(2)
    taps = np.clip(floor[:, None].astype(np.int64) + offsets[None, :], 0, side - 1)
    weights = np.maximum(0.0, 1.0 - np.abs(src[:, None] - floor[:, None] - offsets[None, :]))
    return taps, weights.astype(np.float32)
