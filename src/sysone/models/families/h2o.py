"""H2O-Lightning-4B 家族：pooled readout + selector，因果与双向都训练过。"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan, WeightPlan
from ..registry import register_family
from . import qwen35

# 这份 checkpoint 的 norm 也是零中心存的（真值 `w + 1`）：厂商实现
# `Qwen3_5RMSNorm.forward` 就是 `output * (1.0 + self.weight)`
# （`transformers/models/qwen3_5/modeling_qwen3_5.py:850-856`，票据 15 §18.1）。
# 例外只有 GDN 的门控 norm `Qwen3_5RMSNormGated`（:217-231），它是标准 RMSNorm、不补偏移，
# 装载侧（`chain._layer_params`）一直都把它排除在外。
NORM_OFFSET = 1.0

# h2o 的张量比 mude 多一层 `model.`：词表是 `model.language_model.embed_tokens.weight`，
# 层是 `model.language_model.layers.N…`（与 `int8_manifest.json` 里的名字一致）。
PREFIX = "model.language_model"


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


def weight_plan(model_dir: Path) -> WeightPlan:
    """h2o 交给设备层的那四项。读出是 `pooled`（没有码表文件），设备链还不消费它。"""
    config = qwen35.load_json(Path(model_dir) / "config.json")
    return WeightPlan(
        descriptor=parse_config(Path(model_dir)),
        prefix=PREFIX,
        norm_offset=NORM_OFFSET,
        rope_theta=qwen35.rope_theta(config),
    )


register_family("h2o", parse_config, build_layer_plan, source=__name__)
