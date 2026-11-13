"""mude 1.0 家族：Qwen3.5-2B 剪到 20 层 + 255 路码表读出。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ...config import EngineConfig
from ...domain.attention import AttentionRole, MaskMode
from ...domain.work import ffn_parts
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan
from ..registry import register_family
from . import qwen35

# `d3_runtime.py` 的 MAX_OPTIONS；readout 表是 [255, hidden]。
CODE_ROWS = 255

# 这份 checkpoint 的 norm 权重是零中心存的，真值是 `w + 1`（docs/ANE/README.md §10.22）。
# 不补 +1 时第 0 层的 cos 只有 0.53，看上去像引擎结构错了。
NORM_OFFSET = 1.0


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


def normalize_norm(weight: np.ndarray) -> np.ndarray:
    """norm 权重补 `+1`（`NORM_OFFSET` 的说明见上）。"""
    return np.asarray(weight, dtype=np.float32) + NORM_OFFSET


def ane_layer_weights(
    read: Callable[[str], np.ndarray],
    *,
    layer: int,
    layer_type: AttentionRole,
    prefix: str = "language_model",
) -> dict[str, np.ndarray]:
    """一层的 ANE 权重（`layer_mil` 的 `entries`）：三段计算各自的权重。

    投影是**整块行拼接**，这是旧引擎 `COUT` 的来历，也是 20 层链要复用的同一事实：
    - 全注意力层 5120 = `q_proj` 4096（前 2048 是 q、后 2048 是门控）+ `k_proj` 512 + `v_proj` 512；
    - 线性注意力层 8224 = `in_proj_qkv` 6144（q丨k丨v 各 2048）+ `in_proj_z` 2048 + `in_proj_b` 16 + `in_proj_a` 16。

    线性层的 `b` 在 `a` 之前，是 Metal 核的约定（`k_gdn` 从 `COUT_GDN-32` 起读 `b`、从
    `COUT_GDN-16` 起读 `a`，见 `docs/ANE/anebench/e8_common.h:29`）；numpy 链按同一顺序读。

    FFN 的 `down_proj` 是 `[hidden, ffn]`，按 `hidden` 切 `ffn_parts()` 段；`out` 是
    `out_proj` / `o_proj`（`[hidden, hidden]`，与 `layer_mil` 的 p1 一致 —— 残差在 program 里加）。
    """
    base = f"{prefix}.layers.{layer}"
    if layer_type is AttentionRole.FULL:
        attn = f"{base}.self_attn"
        proj = np.concatenate(
            [read(f"{attn}.q_proj.weight"), read(f"{attn}.k_proj.weight"),
             read(f"{attn}.v_proj.weight")],
            axis=0,
        )
        out = read(f"{attn}.o_proj.weight")
    else:
        attn = f"{base}.linear_attn"
        proj = np.concatenate(
            [read(f"{attn}.in_proj_qkv.weight"), read(f"{attn}.in_proj_z.weight"),
             read(f"{attn}.in_proj_b.weight"), read(f"{attn}.in_proj_a.weight")],
            axis=0,
        )
        out = read(f"{attn}.out_proj.weight")

    gate = read(f"{base}.mlp.gate_proj.weight")
    up = read(f"{base}.mlp.up_proj.weight")
    down = read(f"{base}.mlp.down_proj.weight")
    hidden, ffn = int(down.shape[0]), int(down.shape[1])
    parts = ffn_parts(hidden, ffn)

    entries: dict[str, np.ndarray] = {"proj": proj, "o": out, "gate": gate, "up": up}
    for i in range(parts):
        entries[f"down{i}"] = down[:, i * hidden:(i + 1) * hidden]
    return entries


register_family("mude", parse_config, build_layer_plan, source=__name__)
