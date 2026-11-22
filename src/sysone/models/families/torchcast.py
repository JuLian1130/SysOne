"""Torchcast Decision 4B 家族：解绑 `lm_head`，只训练过整段双向。

这条路径与 h2o 同一套装配（`pooled`）：前向走我们的 ANE 链，题面与判答走仓库内的
`vendors/torchcast_serve.py`（厂商 `render.py` / `model.py` 的移植）。厂商自己的 torch 引擎仍
留在 `open_bridge` 里，但只作对拍基线 —— 它要装 ≈9.3 GB bf16 底座与 torch。

两条**物理边界**，不在装配层：

- 只训练过 `noncausal_full_attention`，而全双向没有前缀可复用（`use_prefix=False`），所以
  `prefix_noncausal` 那条"前缀一遍、每题一条尾巴"的收益它拿不到；
- 读出在 `lm_head`（解绑），不在词表上：同一份 `pooled` 装配按 `ReadoutPlan.weight` 取那 26
  行，`tied_embeddings=False` 是这份 checkpoint 的事实。
"""

from __future__ import annotations

from pathlib import Path

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ..descriptor import ModelDescriptor, ReadoutPlan
from ...weights.checkpoint import stored_quant
from ..layer_plan import LayerPlan, WeightPlan
from ..recipe import PooledRecipe
from ..registry import register_family
from . import qwen35

# 与 h2o 同形：`int8_manifest.json` 里的层名是 `model.language_model.layers.N…`，词表是
# `model.language_model.embed_tokens.weight`（顶层另有一个解绑的 `lm_head.weight`）。
PREFIX = "model.language_model"
# 这份 checkpoint 的 norm 也是零中心存的（真值 `w + 1`），与 h2o / mude 同一套 Qwen3.5 结构。
NORM_OFFSET = 1.0

MODES = frozenset({MaskMode.NONCAUSAL_FULL})


def parse_config(model_dir: Path) -> ModelDescriptor:
    """读出方案是 `lm_head` 的 A..Z 行；没有因果读法（引擎里也没有开关）。

    `quant` 报的是**这份目录存成什么**（`int8_manifest.json` ⇒ row-int8），与 h2o 同一条判据：
    不填的话 `Provenance` 与 `inspect` 会声称"未量化"，而机器上跑的正是 int8 权重。
    """
    config = qwen35.load_json(model_dir / "config.json")
    return qwen35.descriptor_from(
        config,
        family="torchcast",
        model_id=model_dir.name,
        readout=ReadoutPlan(kind="letter_rows", weight="lm_head.weight"),
        supported_modes=MODES,
        trained_modes=MODES,
        default_mode=MaskMode.NONCAUSAL_FULL,
        tied_embeddings=False,
        quant=stored_quant(model_dir),
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


def weight_plan(model_dir: Path) -> WeightPlan:
    """交给设备层的四项：前缀、norm 偏移、rope θ；读出表名由 `parse_config` 的 ReadoutPlan 给。"""
    config = qwen35.load_json(Path(model_dir) / "config.json")
    return WeightPlan(
        descriptor=parse_config(Path(model_dir)),
        prefix=PREFIX,
        norm_offset=NORM_OFFSET,
        rope_theta=qwen35.rope_theta(config),
    )


def open_serve(model_dir: str | Path, tokenizer: object | None = None) -> object:
    """答面：`jev_render_v3` 题面、按题型换温度、noul 的地板、两种置信度。

    惰性 import 与 h2o 同一个理由：注册家族发生在启动路径上，而这条面要读分词器模板。
    """
    from ..vendors.torchcast_serve import Serve

    return Serve.open(model_dir, tokenizer)


def open_bridge(model_dir: Path) -> object:
    """厂商桥：模板、前向与读出全归厂商包，我们只提供权重（对拍基线，不再是默认路径）。

    惰性 import 是必须的：`torchcast_bridge` 在模块级 import torch，而注册家族发生在启动
    路径上（核心面不依赖 torch，`tests/unit/test_import_light.py` 盯着这条线）。
    """
    from ..vendors.torchcast_bridge import TorchcastBridge

    return TorchcastBridge(model_dir)


# 交给 pooled 策略的接入件。形状与张量前缀仍由 `weight_plan` 给，两者各管一半。
RECIPE = PooledRecipe(serve=open_serve)
# 厂商直通不挂 CLI：`vendor_runner` 与 `TorchcastBridge` 留着作对拍基线。
BASELINE = "torchcast"

# 参数量按发布包的三个分片实测：5 174 964 736（同一个 4B 底座，多一份解绑的 `lm_head`）。
register_family("torchcast", parse_config, build_layer_plan, strategy="pooled",
                weight_plan=weight_plan, recipe=RECIPE, params="5.17B", source=__name__)
