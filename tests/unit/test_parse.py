"""读三份真实 config，检查几何与能力声明。"""

from __future__ import annotations

import pytest

from helpers import FIXTURES
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


def test_missing_config_is_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        mude.parse_config(tmp_path)
