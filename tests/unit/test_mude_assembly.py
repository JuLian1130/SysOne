"""mude 的 ANE 装配：行拼接的宽度、FFN 的 K 切分、norm 的 +1。"""

import zlib

import numpy as np
import pytest

from sysone.domain.attention import AttentionRole
from sysone.domain.work import ffn_parts
from sysone.models.families import mude

HIDDEN, FFN = 2048, 6144


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


def test_linear_layer_projection_is_8224_rows():
    entries = mude.ane_layer_weights(_fake_read(_linear_shapes(0)), layer=0,
                                     layer_type=AttentionRole.LINEAR)
    assert entries["proj"].shape == (8224, HIDDEN), "qkv 6144 + z 2048 + a 16 + b 16"
    assert entries["o"].shape == (HIDDEN, HIDDEN)
    assert entries["gate"].shape == entries["up"].shape == (FFN, HIDDEN)
    parts = ffn_parts(HIDDEN, FFN)
    assert parts == 3
    assert entries["down0"].shape == entries["down1"].shape == entries["down2"].shape == (HIDDEN, HIDDEN)


def test_full_layer_projection_is_5120_rows():
    entries = mude.ane_layer_weights(_fake_read(_full_shapes(3)), layer=3,
                                     layer_type=AttentionRole.FULL)
    assert entries["proj"].shape == (5120, HIDDEN), "q 4096（含门控）+ k 512 + v 512"
    assert entries["o"].shape == (HIDDEN, HIDDEN)
    assert set(entries) == {"proj", "o", "gate", "up", "down0", "down1", "down2"}


def test_down_is_split_along_k_axis_in_order():
    """三段 down 按列拼回去必须等于原矩阵，顺序不能乱。"""
    read = _fake_read(_linear_shapes(0))
    down = read("language_model.layers.0.mlp.down_proj.weight")
    entries = mude.ane_layer_weights(read, layer=0, layer_type=AttentionRole.LINEAR)
    rebuilt = np.concatenate([entries["down0"], entries["down1"], entries["down2"]], axis=1)
    np.testing.assert_array_equal(rebuilt, down)


def test_norm_gets_plus_one():
    raw = np.zeros(8, dtype=np.float16)
    assert np.allclose(mude.normalize_norm(raw), 1.0)
    assert mude.normalize_norm(raw).dtype == np.float32


def test_4b_geometry_is_rejected_by_the_shared_rule():
    with pytest.raises(Exception) as excinfo:
        ffn_parts(2560, 9216)
    assert "整除" in str(excinfo.value)
