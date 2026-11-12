"""H2O-Lightning-4B 家族：pooled readout + selector，因果与双向都训练过。"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan
from ..registry import register_family
from . import qwen35


def parse_config(model_dir: Path) -> ModelDescriptor:
    """读出方案是厂商的 pooled readout + selector，M1 由 vendor 策略实现。"""
    config = qwen35.load_json(model_dir / "config.json")
    modes = frozenset({MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL})
    return qwen35.descriptor_from(
        config,
        family="h2o",
        model_id=model_dir.name,
        readout=ReadoutPlan(kind="pooled"),
        supported_modes=modes,
        trained_modes=modes,
        default_mode=MaskMode.NONCAUSAL_FULL,
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


register_family("h2o", parse_config, build_layer_plan, source=__name__)
