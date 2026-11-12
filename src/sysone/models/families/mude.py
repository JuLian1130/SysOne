"""mude 1.0 家族：Qwen3.5-2B 剪到 20 层 + 255 路码表读出。"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan
from ..registry import register_family
from . import qwen35

# `d3_runtime.py` 的 MAX_OPTIONS；readout 表是 [255, hidden]。
CODE_ROWS = 255


def parse_config(model_dir: Path) -> ModelDescriptor:
    """读 `config.json` 与 `decision_config.json`，产出 `ModelDescriptor`。"""
    config = qwen35.load_json(model_dir / "config.json")
    decision_path = model_dir / "decision_config.json"
    decision = qwen35.load_json(decision_path) if decision_path.exists() else {}

    mode = MaskMode(str(decision.get("attention_mode", MaskMode.PREFIX_NONCAUSAL.value)))
    temperature = decision.get("temperature")
    readout = ReadoutPlan(
        kind="code_rows",
        weight="weight",
        file="readout.safetensors",
        rows=CODE_ROWS,
        temperature=float(temperature) if temperature is not None else None,  # type: ignore[arg-type]
    )
    return qwen35.descriptor_from(
        config,
        family="mude",
        model_id=model_dir.name,
        readout=readout,
        supported_modes=frozenset(MaskMode),
        # 发布 checkpoint 只在 prefix_noncausal 掩码下做过决策 SFT（NOTICE:8）。
        trained_modes=frozenset({MaskMode.PREFIX_NONCAUSAL}),
        default_mode=mode,
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


register_family("mude", parse_config, build_layer_plan, source=__name__)
