"""Torchcast Decision 4B 家族：解绑 lm_head，只支持整段双向。"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan
from ..registry import register_family
from . import qwen35


def parse_config(model_dir: Path) -> ModelDescriptor:
    """读 `lm_head` 的候选字母行；没有因果读法（引擎里也没有开关）。"""
    config = qwen35.load_json(model_dir / "config.json")
    return qwen35.descriptor_from(
        config,
        family="torchcast",
        model_id=model_dir.name,
        readout=ReadoutPlan(kind="letter_rows", weight="lm_head.weight"),
        supported_modes=frozenset({MaskMode.NONCAUSAL_FULL}),
        trained_modes=frozenset({MaskMode.NONCAUSAL_FULL}),
        default_mode=MaskMode.NONCAUSAL_FULL,
        tied_embeddings=False,
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


register_family("torchcast", parse_config, build_layer_plan,
                strategy="vendor", source=__name__)
