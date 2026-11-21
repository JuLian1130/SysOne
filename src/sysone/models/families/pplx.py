"""Perplexity pplx-decider-flash 家族：d3 码表读出的 4B 决策检查点，只训练过整段双向。

与 d3-mini 是同一份检查点形态（也即 mude 的 `d3-code-readout-v1` 线格式），所以两家共用
`mude.RECIPE`：题面、255 个回答码与熵置信度不另写。v1.2 与 v1.3 是两份同格式权重，家族只看
目录形态，版本由目录名进 `model_id`（`Provenance.model` 会原样报出来）—— 换版本不改代码。

与 d3 逐项相同的三件事：`language_model` 前缀、4B 几何（32 层、hidden 2560、ffn 9216）、
原生掩码只有 `noncausal_full`。最后一条意味着没有前缀复用，只有跨请求合批；§8.8 量到的
因果臂（137 题 743 s → 191 s、29/137 条翻转）是未训练读法的代价，不声明成能力。

准备约定与 d3 相同：合分片时丢掉 `visual.*`，config 里可能还留着 `vision_config`，所以用
`qwen35.text_only` 读 —— 否则描述符会声明一座没有权重的塔。
"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ...weights.checkpoint import stored_quant
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan, WeightPlan
from ..registry import register_family
from . import mude, qwen35

# 与 d3 / mude 同形：`language_model.layers.N…`，没有 `model.` 那层。
PREFIX = "language_model"
# Qwen3.5 的 norm 是零中心存的，真值 `w + 1`。
NORM_OFFSET = 1.0

MODES = frozenset({MaskMode.NONCAUSAL_FULL})


def parse_config(model_dir: Path) -> ModelDescriptor:
    """按 d3 线格式读描述符；v1.2 / v1.3 的差别只该在权重与温度里。"""
    config = qwen35.text_only(qwen35.load_json(model_dir / "config.json"))
    decision = qwen35.load_json(model_dir / "decision_config.json")
    temperature = decision.get("temperature")
    return qwen35.descriptor_from(
        config,
        family="pplx",
        model_id=model_dir.name,
        readout=ReadoutPlan(
            kind="code_rows",
            weight="weight",
            file="readout.safetensors",
            rows=mude.CODE_ROWS,
            temperature=float(temperature) if temperature is not None else None,  # type: ignore[arg-type]
        ),
        supported_modes=MODES,
        trained_modes=MODES,
        default_mode=MaskMode.NONCAUSAL_FULL,
        quant=stored_quant(model_dir),
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


def weight_plan(model_dir: Path) -> WeightPlan:
    config = qwen35.load_json(Path(model_dir) / "config.json")
    return WeightPlan(
        descriptor=parse_config(Path(model_dir)),
        prefix=PREFIX,
        norm_offset=NORM_OFFSET,
        rope_theta=qwen35.rope_theta(config),
    )


# 与 mude / d3 共用同一份配方：同一线格式，题面与码表没有第二套定义。
RECIPE = mude.RECIPE

register_family("pplx", parse_config, build_layer_plan,
                strategy="split", weight_plan=weight_plan, recipe=RECIPE, source=__name__)
