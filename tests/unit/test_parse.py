"""读三份真实 config，检查几何与能力声明。"""

from __future__ import annotations

import dataclasses

import pytest

from helpers import FIXTURES
from sysone.devices.apple.geometry import Geometry, mude_geometry
from sysone.domain.attention import AttentionRole, MaskMode
from sysone.errors import ConfigError
from sysone.models.families import h2o, mude, torchcast


def test_mude_geometry() -> None:
    descriptor = mude.parse_config(FIXTURES / "mude")
    assert descriptor.family == "mude"
    assert descriptor.num_layers == 20
    assert descriptor.hidden == 2048
    assert descriptor.ffn == 6144
    assert (descriptor.heads, descriptor.kv_heads, descriptor.head_dim) == (8, 2, 256)
    full = [index for index, role in enumerate(descriptor.layer_types) if role is AttentionRole.FULL]
    assert full == [3, 7, 11, 15, 19]
    assert descriptor.tied_embeddings is True
    assert descriptor.vision is not None and descriptor.vision.model_type == "qwen3_5_vision"


def test_mude_readout_and_modes() -> None:
    descriptor = mude.parse_config(FIXTURES / "mude")
    assert descriptor.readout.kind == "code_rows"
    assert descriptor.readout.rows == 255
    assert descriptor.readout.file == "readout.safetensors"
    assert descriptor.readout.temperature == pytest.approx(1.2645396590233329)
    assert descriptor.default_mode is MaskMode.PREFIX_NONCAUSAL
    assert descriptor.supported_modes == frozenset(MaskMode)
    assert descriptor.trained_modes == frozenset({MaskMode.PREFIX_NONCAUSAL})


def test_h2o_geometry() -> None:
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.num_layers == 32
    assert descriptor.hidden == 2560
    assert descriptor.kv_heads == 4
    assert descriptor.mtp_layers == 1
    assert descriptor.tied_embeddings is True
    assert descriptor.readout.kind == "pooled"
    assert descriptor.default_mode is MaskMode.NONCAUSAL_FULL
    assert descriptor.trained_modes == frozenset({MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL})


def test_torchcast_untied() -> None:
    descriptor = torchcast.parse_config(FIXTURES / "torchcast")
    assert descriptor.tied_embeddings is False
    assert descriptor.readout.kind == "letter_rows"
    assert descriptor.readout.weight == "lm_head.weight"
    assert descriptor.supported_modes == frozenset({MaskMode.NONCAUSAL_FULL})


def test_heads_need_not_cover_hidden() -> None:
    """4B 的 `heads * head_dim = 4096 != hidden 2560`，校验不该要求相等。"""
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.heads * descriptor.head_dim != descriptor.hidden
    descriptor.validate()  # 不抛异常


def test_h2o_linear_geometry_keeps_v_apart_from_k() -> None:
    """4B 的 GDN v 头数是 k 的两倍（32 对 16），descriptor 必须分开记。

    这条不变量是 4B 上引擎的门槛：`value_dim = 4096 ≠ hidden 2560`，任何"v 就是 k 的
    副本"的写法都会把 `in_proj_z`、递归状态与 `cout_gdn` 的宽度算错。
    """
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.linear_key_heads == 16
    assert descriptor.linear_value_heads == 32  # v 头数是 k 的两倍，2B 上两者相等
    assert (descriptor.linear_key_dim, descriptor.linear_value_dim) == (128, 128)
    # qk_dim 2048、value_dim 4096，**都不等于 hidden 2560**（2B 恰好三者重合，4B 不是）
    assert descriptor.linear_key_heads * descriptor.linear_key_dim == 2048
    assert descriptor.linear_value_heads * descriptor.linear_value_dim == 4096
    assert descriptor.conv_kernel == 4
    assert descriptor.partial_rotary_factor == pytest.approx(0.25)


def test_geometry_projection_matches_mude_hardcoding() -> None:
    """`Geometry.from_descriptor` 投影出的几何要与 `mude_geometry()` 逐项相同。

    这是设备层不再 `import models.families.mude`、改由 descriptor 供货的前提：投影算出的
    每个 `#define` 都得是 mude 今天看到的那个值，否则"纯重构"就不成立。
    """
    assert Geometry.from_descriptor(mude.parse_config(FIXTURES / "mude")) == mude_geometry()


def test_gdn_scratch_row_keeps_the_2b_pitch_and_widens_for_4b() -> None:
    """`k_gdn` 的 scratch 行距：2B 保持 8192 的既有刻度，4B 因为 v 段翻倍而加宽。

    `e8_gdn.metal` 的 `E8G_S_ROW` 用同一套公式算这个数，宿主按它分配 buffer；两边算不
    一致就是越界写。运行时守卫是 `test_metal_gdn*.py` 的对拍，这里钉住数字本身。
    """
    assert mude_geometry().gdn_scratch_halves == 8192
    four_b = Geometry.from_descriptor(h2o.parse_config(FIXTURES / "h2o"))
    assert four_b.gdn_scratch_halves == 9216
    need = 2 * four_b.qk_dim + four_b.value_dim + 2 * four_b.l_v_heads
    assert four_b.gdn_scratch_halves >= need


def test_missing_linear_geometry_is_a_config_error() -> None:
    """有线性层却缺几何时解析期就报错，不留到设备层分配缓冲那一步。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    broken = dataclasses.replace(descriptor, linear_value_heads=None)
    with pytest.raises(ConfigError):
        broken.validate()


def test_linear_geometry_is_not_required_without_linear_layers() -> None:
    """只有全注意力层的模型不需要这四项，`validate()` 不该因为它们缺席而报错。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    all_full = dataclasses.replace(
        descriptor,
        layer_types=tuple(AttentionRole.FULL for _ in descriptor.layer_types),
        linear_key_heads=None,
        linear_key_dim=None,
        linear_value_heads=None,
        linear_value_dim=None,
    )
    all_full.validate()  # 不抛异常


def test_missing_config_is_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        mude.parse_config(tmp_path)
