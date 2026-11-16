"""Qwen3.5 家族共用部分：config 解析与描述符构造（不注册自己）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ...config import EngineConfig
from ...domain.attention import AttentionRole, MaskMode
from ...errors import ConfigError
from ...domain.quant import QuantSpec
from ..descriptor import ModelDescriptor, ReadoutPlan, VisionSpec
from ..layer_plan import LayerPlan, plan_layers


def load_json(path: Path) -> dict[str, object]:
    """读一个 JSON 文件；缺失或语法错都归为 `ConfigError`。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise ConfigError("模型目录缺少配置文件", path=str(path)) from None
    except json.JSONDecodeError:
        raise ConfigError("配置文件不是合法 JSON", path=str(path)) from None


def text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    """多模态 config 的 text_config；纯文本 config 则返回自身。"""
    nested = config.get("text_config")
    return nested if isinstance(nested, Mapping) else config  # type: ignore[return-value]


def layer_types(text: Mapping[str, object]) -> tuple[AttentionRole, ...]:
    """优先读显式 `layer_types`，退回 `full_attention_interval`。"""
    count = int(text["num_hidden_layers"])  # type: ignore[arg-type]
    raw = text.get("layer_types")
    if raw is not None:
        if len(raw) != count:  # type: ignore[arg-type]
            raise ConfigError(
                "layer_types 长度与 num_hidden_layers 不一致",
                layers=len(raw),  # type: ignore[arg-type]
                num_layers=count,
            )
        roles: list[AttentionRole] = []
        for item in raw:  # type: ignore[union-attr]
            if item == "full_attention":
                roles.append(AttentionRole.FULL)
            elif item == "linear_attention":
                roles.append(AttentionRole.LINEAR)
            else:
                raise ConfigError("未知的 layer_types 取值", value=str(item))
        return tuple(roles)
    interval = text.get("full_attention_interval")
    if interval:
        step = int(interval)  # type: ignore[arg-type]
        return tuple(
            AttentionRole.FULL if (index + 1) % step == 0 else AttentionRole.LINEAR
            for index in range(count)
        )
    raise ConfigError("config 缺少 layer_types 与 full_attention_interval")


def geometry(text: Mapping[str, object]) -> dict[str, int]:
    """取出与设备无关的结构数字。"""
    heads = int(text["num_attention_heads"])  # type: ignore[arg-type]
    hidden = int(text["hidden_size"])  # type: ignore[arg-type]
    return {
        "hidden": hidden,
        "ffn": int(text["intermediate_size"]),  # type: ignore[arg-type]
        "heads": heads,
        "kv_heads": int(text.get("num_key_value_heads", heads)),  # type: ignore[arg-type]
        "head_dim": int(text.get("head_dim") or hidden // heads),  # type: ignore[arg-type]
        "vocab": int(text["vocab_size"]),  # type: ignore[arg-type]
        "max_position": int(text.get("max_position_embeddings", 0)),  # type: ignore[arg-type]
        "num_layers": int(text["num_hidden_layers"]),  # type: ignore[arg-type]
    }


def _opt_int(value: object) -> int | None:
    """缺项留 `None`，不要拿 `int(value or 0)` 压成一个"看起来合理"的数。

    设备层（`Geometry.from_descriptor`）靠 `None` 认出缺项并报错；在这里压成 0 只是把
    错误推迟到分配缓冲那一步。
    """
    return None if value is None else int(value)  # type: ignore[arg-type]


def partial_rotary(text: Mapping[str, object]) -> float:
    """`partial_rotary_factor`：Qwen3.5 写在 `rope_parameters` 里（0.25），也有顶层再抄一份
    的 config；都没写就是全旋转。设备层用它算 `rotary_dim = head_dim × factor`。"""
    rope = text.get("rope_parameters")
    value = rope.get("partial_rotary_factor") if isinstance(rope, Mapping) else None
    if value is None:
        value = text.get("partial_rotary_factor")
    return 1.0 if value is None else float(value)  # type: ignore[arg-type]


def vision_spec(config: Mapping[str, object]) -> VisionSpec | None:
    """视觉塔几何；M0 只记录，不加载。"""
    nested = config.get("vision_config")
    if not isinstance(nested, Mapping):
        return None
    return VisionSpec(
        model_type=str(nested.get("model_type", "")),
        depth=int(nested.get("depth", 0)),  # type: ignore[arg-type]
        hidden=int(nested.get("hidden_size", 0)),  # type: ignore[arg-type]
        patch_size=int(nested.get("patch_size", 0)),  # type: ignore[arg-type]
        spatial_merge=int(nested.get("spatial_merge_size", 2)),  # type: ignore[arg-type]
    )


def rope_theta(config: Mapping[str, object]) -> float:
    """`rope_parameters.rope_theta`：RoPE 的角基。缺了就报错，不替它猜一个。

    这个数是**数值**约定（不是形状），错了不会崩，只会让所有注意力位置角整体缩放 ——
    正是最难发现的那类错。两个出货模型都是 1e7。
    """
    text = text_config(config)
    rope = text.get("rope_parameters") or {}
    if not isinstance(rope, Mapping) or "rope_theta" not in rope:
        raise ConfigError(
            "config 缺少 rope_parameters.rope_theta",
            hint="qwen3.5 的 rope 参数在 text_config.rope_parameters 下",
        )
    return float(rope["rope_theta"])  # type: ignore[arg-type]


def descriptor_from(
    config: Mapping[str, object],
    *,
    family: str,
    model_id: str,
    readout: ReadoutPlan,
    supported_modes: frozenset[MaskMode],
    trained_modes: frozenset[MaskMode],
    default_mode: MaskMode,
    tied_embeddings: bool | None = None,
    quant: QuantSpec | None = None,
) -> ModelDescriptor:
    """把一份 Qwen3.5 风格 config 拼成 `ModelDescriptor`。

    `quant` 是**检查点目录里存的那一种**（`weights.checkpoint.stored_quant`），不是 config
    里的字段：它由容器形态决定，所以由家族在读目录时给。不给就按未量化报（`QuantSpec()`
    的默认值）—— 那是"没告诉我"，不是"猜它是 bf16"。
    """
    text = text_config(config)
    numbers = geometry(text)
    descriptor = ModelDescriptor(
        family=family,
        model_id=model_id,
        hidden=numbers["hidden"],
        num_layers=numbers["num_layers"],
        layer_types=layer_types(text),
        heads=numbers["heads"],
        kv_heads=numbers["kv_heads"],
        head_dim=numbers["head_dim"],
        # 线性层的 k/v 几何分开填：4B 的 v 头数是 k 的两倍。config 没写就留 `None`，
        # 由 `validate()` 报错——不要拿 k 的四个数顶替 v（那正是 4B 会算错的地方）。
        linear_key_heads=_opt_int(text.get("linear_num_key_heads")),
        linear_key_dim=_opt_int(text.get("linear_key_head_dim")),
        linear_value_heads=_opt_int(text.get("linear_num_value_heads")),
        linear_value_dim=_opt_int(text.get("linear_value_head_dim")),
        conv_kernel=int(text.get("linear_conv_kernel_dim") or 4),
        partial_rotary_factor=partial_rotary(text),
        ffn=numbers["ffn"],
        vocab=numbers["vocab"],
        max_position=numbers["max_position"],
        readout=readout,
        supported_modes=supported_modes,
        trained_modes=trained_modes,
        default_mode=default_mode,
        quant=quant if quant is not None else QuantSpec(),
        tied_embeddings=(
            bool(text.get("tie_word_embeddings", True))
            if tied_embeddings is None
            else tied_embeddings
        ),
        mtp_layers=int(text.get("mtp_num_hidden_layers", 0)),  # type: ignore[arg-type]
        rms_eps=float(text.get("rms_norm_eps", 1e-6)),  # type: ignore[arg-type]
        vision=vision_spec(config),
    )
    descriptor.validate()
    return descriptor


def layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    """按 ring 切块（三个家族一样：Qwen3.5 的层布局相同）。"""
    return plan_layers(descriptor.layer_types, config.scheduler.ring)
