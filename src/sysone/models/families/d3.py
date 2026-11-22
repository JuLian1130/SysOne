"""vLLM-SR d3-mini 家族：d3 码表读出的 4B 决策检查点，只训练过整段双向。

与 mude 是**同一份线格式**（`d3-code-readout-v1`：255 个回答码、`readout.safetensors`、
`decision_config.json` 的 `temperature`），所以题面、码表与置信度口径直接共用 `mude.RECIPE`，
不另写一份 —— 两份检查点不同的地方只有这里声明的三件事：

- 张量前缀与 2B 同形（`language_model.layers.N…`，没有 `model.` 那层）；
- 4B 几何（32 层、hidden 2560、ffn 9216），与 h2o / torchcast 同一条 `SplitChain`；
- 原生掩码只有 `noncausal_full`：全双向没有前缀可复用（`use_prefix=False`），所以只有
  跨请求合批这一条收益，题与题之间的共享前缀拿不到（掩码定义，不是没做）。

**因果臂不在这里声明**：§8.8 量过它（137 题 739 s → 186 s，但 37/137 条判词翻掉），那是
"未训练读法的代价"，与 torchcast 同一条判据 —— 只能以 `noncausal_full` 出货。

权重的准备约定：发布包的 `config.json` 留着整块 `vision_config`，而**塔的权重也在包里**
（297 个 `visual.*`；`visual.blocks.0.attn.qkv.weight` 实测 BF16 `[3072, 1024]`），几何与
torchcast 逐项相同。所以这里照读完整 config：描述符会声明这座塔，`split` 装配据此建塔，图片
走 `MudePrompter` 的 content parts 与 `models/vision/merge` 的展开/位置表。早先的
`qwen35.text_only` 是误判（把 d3 当成了 mude 那种"塔继承但没用"的检查点）—— 那会让每个带图
请求都 422，而 d3-mini 的发布卡片把读图列为核心能力。视频仍没接：塔只做单帧
（`devices/apple/vision.py` 的 `patchify` 把单帧沿 temporal 复制），没有时间轴与抽帧。
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

# 4B 决策检查点与 mude 用同一套容器布局：`language_model.layers.N…`。
PREFIX = "language_model"
# 与 mude / h2o / torchcast 同一套 Qwen3.5 结构：norm 是零中心存的，真值 `w + 1`。
NORM_OFFSET = 1.0

# 只训练过整段双向。声明 `causal` 会让人以为换掩码是受支持的，而 §8.8 的读数正好相反。
MODES = frozenset({MaskMode.NONCAUSAL_FULL})


def parse_config(model_dir: Path) -> ModelDescriptor:
    """按 d3 线格式读描述符：温度随 checkpoint 走，掩码只有双向那一种。"""
    config = qwen35.load_json(model_dir / "config.json")
    decision = qwen35.load_json(model_dir / "decision_config.json")
    temperature = decision.get("temperature")
    return qwen35.descriptor_from(
        config,
        family="d3",
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
    """张量前缀、norm 偏移与 RoPE 角基；读出表名由 `parse_config` 的 `ReadoutPlan` 给。"""
    config = qwen35.load_json(Path(model_dir) / "config.json")
    return WeightPlan(
        descriptor=parse_config(Path(model_dir)),
        prefix=PREFIX,
        norm_offset=NORM_OFFSET,
        rope_theta=qwen35.rope_theta(config),
    )


# 与 mude 共用同一份配方：同一线格式的两份检查点，题面/码表/置信度没有第二套定义。
RECIPE = mude.RECIPE

register_family("d3", parse_config, build_layer_plan,
                strategy="split", weight_plan=weight_plan, recipe=RECIPE, source=__name__)
