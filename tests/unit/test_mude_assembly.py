"""mude 的 ANE 装配：行拼接的宽度、FFN 的 K 切分、norm 的 +1。"""

import zlib

import numpy as np
import pytest

from sysone.domain.attention import AttentionRole
from sysone.domain.work import ffn_segments
from sysone.devices.apple import blobplan
from sysone.models.families import mude

HIDDEN, FFN = 2048, 6144
PREFIX = "language_model"


def _fake_read(shapes: dict[str, tuple[int, ...]]):
    """按名字定种：同一个张量名无论第几次读都给出同样的数，便于断言拼接顺序。"""
    def read(name: str) -> np.ndarray:
        if name not in shapes:
            raise AssertionError(f"没料到会读 {name}")
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        return np.ascontiguousarray(rng.standard_normal(shapes[name]).astype(np.float16))

    return read


def _linear_shapes(layer: int) -> dict[str, tuple[int, ...]]:
    base = f"language_model.layers.{layer}"
    attn = f"{base}.linear_attn"
    return {
        f"{attn}.in_proj_qkv.weight": (6144, HIDDEN),
        f"{attn}.in_proj_z.weight": (HIDDEN, HIDDEN),
        f"{attn}.in_proj_a.weight": (16, HIDDEN),
        f"{attn}.in_proj_b.weight": (16, HIDDEN),
        f"{attn}.out_proj.weight": (HIDDEN, HIDDEN),
        f"{base}.mlp.gate_proj.weight": (FFN, HIDDEN),
        f"{base}.mlp.up_proj.weight": (FFN, HIDDEN),
        f"{base}.mlp.down_proj.weight": (HIDDEN, FFN),
    }


def _full_shapes(layer: int) -> dict[str, tuple[int, ...]]:
    base = f"language_model.layers.{layer}"
    attn = f"{base}.self_attn"
    return {
        f"{attn}.q_proj.weight": (4096, HIDDEN),   # q 2048 + 门控 2048
        f"{attn}.k_proj.weight": (512, HIDDEN),
        f"{attn}.v_proj.weight": (512, HIDDEN),
        f"{attn}.o_proj.weight": (HIDDEN, HIDDEN),
        f"{base}.mlp.gate_proj.weight": (FFN, HIDDEN),
        f"{base}.mlp.up_proj.weight": (FFN, HIDDEN),
        f"{base}.mlp.down_proj.weight": (HIDDEN, FFN),
    }


def _linear_shapes_4b(layer: int) -> dict[str, tuple[int, ...]]:
    """h2o 的线性层：hidden 2560，value 头 32×128 = 4096，ffn 9216。"""
    hidden = 2560
    base = f"language_model.layers.{layer}"
    attn = f"{base}.linear_attn"
    return {
        f"{attn}.in_proj_qkv.weight": (8192, hidden),     # qk 2048 + v 4096 + 2×1024 卷积窗
        f"{attn}.in_proj_z.weight": (4096, hidden),
        f"{attn}.in_proj_a.weight": (32, hidden),
        f"{attn}.in_proj_b.weight": (32, hidden),
        f"{attn}.out_proj.weight": (hidden, 4096),
        f"{base}.mlp.gate_proj.weight": (9216, hidden),
        f"{base}.mlp.up_proj.weight": (9216, hidden),
        f"{base}.mlp.down_proj.weight": (hidden, 9216),
    }


def test_linear_layer_projection_is_8224_rows():
    entries = blobplan.ane_layer_weights(_fake_read(_linear_shapes(0)), layer=0,
                                     layer_type=AttentionRole.LINEAR, hidden=HIDDEN, ffn=FFN, prefix=PREFIX)
    assert entries["proj"].shape == (8224, HIDDEN), "qkv 6144 + z 2048 + a 16 + b 16"
    assert entries["o"].shape == (HIDDEN, HIDDEN)
    assert entries["gate"].shape == entries["up"].shape == (FFN, HIDDEN)
    segments = ffn_segments(HIDDEN, FFN)
    assert [columns.stop - columns.start for columns in segments] == [HIDDEN] * 3
    assert entries["down0"].shape == entries["down1"].shape == entries["down2"].shape == (HIDDEN, HIDDEN)


def test_full_layer_projection_is_5120_rows():
    entries = blobplan.ane_layer_weights(_fake_read(_full_shapes(3)), layer=3,
                                     layer_type=AttentionRole.FULL, hidden=HIDDEN, ffn=FFN, prefix=PREFIX)
    assert entries["proj"].shape == (5120, HIDDEN), "q 4096（含门控）+ k 512 + v 512"
    assert entries["o"].shape == (HIDDEN, HIDDEN)
    assert set(entries) == {"proj", "o", "gate", "up", "down0", "down1", "down2"}


def test_down_is_split_along_k_axis_in_order():
    """三段 down 按列拼回去必须等于原矩阵，顺序不能乱。"""
    read = _fake_read(_linear_shapes(0))
    down = read("language_model.layers.0.mlp.down_proj.weight")
    entries = blobplan.ane_layer_weights(read, layer=0, layer_type=AttentionRole.LINEAR,
                                     hidden=HIDDEN, ffn=FFN, prefix=PREFIX)
    rebuilt = np.concatenate([entries["down0"], entries["down1"], entries["down2"]], axis=1)
    np.testing.assert_array_equal(rebuilt, down)


def _no_parts(name: str) -> None:
    return None


def test_quantized_layer_takes_the_containers_codes():
    """容器自带码时，条目就是它的码与标度 —— 行拼与列切都不重算。

    这里给一份"故意与 amax 不符"的码（码全是 ±1、标度是整行的），如果实现又按 amax
    量化了一遍，码就不会再是 ±1，测试立刻失败。
    """
    shapes = _linear_shapes(0)
    read = _fake_read(shapes)

    def parts(name: str):
        rows = shapes[name][0]
        cols = shapes[name][1]
        codes = np.where((np.arange(rows)[:, None] + np.arange(cols)[None, :]) % 2 == 0, 1, -1)
        return codes.astype(np.int8), np.full(rows, 0.0625, dtype=np.float32)

    entries = blobplan.ane_layer_quantized(
        read, parts, layer=0, layer_type=AttentionRole.LINEAR, hidden=HIDDEN, ffn=FFN,
        prefix=PREFIX,
    )
    codes, scales = entries["down0"]
    assert codes.shape == (HIDDEN, HIDDEN) and codes.dtype == np.int8
    # 列切的是第 0 段，所以每行的列号从 0 开始重新计 —— 与切片前的棋盘格一致。
    assert np.array_equal(codes[0], np.where(np.arange(HIDDEN) % 2 == 0, 1, -1))
    np.testing.assert_array_equal(scales, np.full(HIDDEN, 0.0625, dtype=np.float16))
    # 三段按列拼回去还是那张棋盘格。
    rebuilt = np.concatenate(
        [entries[f"down{i}"][0] for i in range(len(ffn_segments(HIDDEN, FFN)))], axis=1
    )
    np.testing.assert_array_equal(
        rebuilt,
        np.where((np.arange(HIDDEN)[:, None] + np.arange(FFN)[None, :]) % 2 == 0, 1, -1),
    )
    # 投影是行拼接：前 6144 行来自 qkv，所以第 6144 行开始的棋盘格相位仍按各张量自己的行号。
    assert entries["proj"][0].shape == (8224, HIDDEN)
    assert entries["proj"][1].shape == (8224,)


def test_quantized_layer_re_quantizes_what_the_container_keeps_as_is():
    """容器里按 16 bit 存的张量（`parts` 给 None）在这里现量化，码不再是棋盘格。"""
    shapes = _linear_shapes(0)
    read = _fake_read(shapes)
    entries = blobplan.ane_layer_quantized(
        read, _no_parts, layer=0, layer_type=AttentionRole.LINEAR, hidden=HIDDEN, ffn=FFN,
        prefix=PREFIX,
    )
    codes, scales = entries["gate"]
    assert codes.shape == (FFN, HIDDEN)
    # 现量化的码会用到 -127..127 的整个量程，不可能只有 ±1。
    assert int(np.abs(codes).max()) > 1
    assert scales.shape == (FFN,)
    expected = read("language_model.layers.0.mlp.gate_proj.weight").astype(np.float16)
    np.testing.assert_allclose(
        codes.astype(np.float32) * scales.astype(np.float32)[:, None].astype(np.float16).astype(np.float32),
        expected.astype(np.float32),
        atol=float(np.abs(expected).max()) / 127.0 * 1.01,
    )


def test_the_checkpoint_needs_the_plus_one_norm_offset():
    """这份 checkpoint 的 norm 是零中心存的，真值是 `w + 1`（README §10.22）。

    这个数由家族递给设备层（`WeightPlan.norm_offset`，`SplitChain` 从它取），设备层不再认识它；
    改成 0 时第 0 层的 cos 会掉到 0.53，所以它是与权重文件绑定的契约，不是调参。
    """
    assert mude.NORM_OFFSET == 1.0


def test_4b_geometry_splits_the_ffn_into_three_full_parts_and_a_tail():
    """9216/2560 = 3.6：三段 2560 加一段 1536，覆盖整个 K 维。

    段长不齐是 4B 的形状事实，不是可以四舍五入掉的误差 —— 少 0.6 段就是少算 1536 列。
    """
    segments = ffn_segments(2560, 9216)
    assert [columns.stop - columns.start for columns in segments] == [2560, 2560, 2560, 1536]
    assert [columns.start for columns in segments] == [0, 2560, 5120, 7680]
    assert segments[-1].stop == 9216
    entries = blobplan.ane_layer_weights(_fake_read(_linear_shapes_4b(0)), layer=0,
                                     layer_type=AttentionRole.LINEAR, hidden=2560, ffn=9216, prefix=PREFIX)
    assert entries["down3"].shape == (2560, 1536)
    rebuilt = np.concatenate([entries[f"down{i}"] for i in range(4)], axis=1)
    np.testing.assert_array_equal(rebuilt, _fake_read(_linear_shapes_4b(0))(
        "language_model.layers.0.mlp.down_proj.weight"))


def test_the_shipping_factories_warn_before_they_touch_the_chain(caplog, monkeypatch):
    """两个执行器工厂在装链之前就出声：形状不在实测白名单里 → 一行告警，然后照常继续。

    这条钉的是"接上了"而不是"拦住了"：夹具缺 tokenizer / serve_config，工厂走到那一步会抛，
    所以这里只要求告警在**失败之前**已经发出（顺序正是这条检查存在的理由 —— 它只读 config，
    却要在任何"看起来像结果"的运行之前留痕）。
    """
    from helpers import FIXTURES

    from sysone.devices.apple import platform as apple_platform
    from sysone.engine.strategies.pooled import pooled_runner
    from sysone.engine.strategies.split import split_runner
    from sysone.errors import SysOneError
    from sysone.models.families import h2o as h2o_family
    from sysone.models.families import mude as mude_family

    monkeypatch.setattr(apple_platform, "MEASURED_GEOMETRIES", frozenset())
    for build, strategy in (
        (lambda: split_runner(FIXTURES / "mude", recipe=mude_family.RECIPE,
                              weights=mude_family.weight_plan(FIXTURES / "mude")), "split"),
        (lambda: pooled_runner(FIXTURES / "h2o", recipe=h2o_family.RECIPE,
                               weights=h2o_family.weight_plan(FIXTURES / "h2o")), "pooled"),
    ):
        caplog.clear()
        with caplog.at_level("WARNING", logger="sysone.devices.apple.platform"):
            try:
                build()
            except SysOneError:
                pass  # 夹具没有分词器/答面配置：这里只关心告警已经发出
        messages = [record.getMessage() for record in caplog.records]
        assert any("实测白名单" in message and strategy in message for message in messages), (
            f"{strategy} 工厂装链前没有告警：{messages}"
        )
