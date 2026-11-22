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


# 视觉塔的**形状**键。缺一项就报错：拿 0 顶上的话，错误会推迟到装塔时变成"某个维度是 0"的
# 缓冲，那时候已经看不出是 config 少了一项。
_VISION_SHAPE_KEYS = (
    "depth",
    "hidden_size",
    "intermediate_size",
    "num_heads",
    "out_hidden_size",
    "patch_size",
    "spatial_merge_size",
    "temporal_patch_size",
    "num_position_embeddings",
)


def vision_spec(config: Mapping[str, object]) -> VisionSpec | None:
    """视觉塔的几何；没有 `vision_config` 的模型返回 `None`。

    `model_type` 只记录、不分支：h2o 的 `vision_config` 写的是 `qwen3_5`（mude 写的是
    `qwen3_5_vision`），按它挑构造器会在同一个塔上分叉出两条路径，而两个模型的塔几何逐项相同
    （只有 `out_hidden_size` 不同：2048 对 2560）。

    `rope_parameters` 是唯一的可选段：h2o 的 `vision_config` 没有它，而厂商
    `Qwen3_5VisionConfig` 的默认值就是 `axial` / θ=1e4（mude 显式写的正是同一个值）。
    """
    nested = config.get("vision_config")
    if not isinstance(nested, Mapping):
        return None
    missing = [key for key in _VISION_SHAPE_KEYS if nested.get(key) is None]
    if missing:
        raise ConfigError(
            "vision_config 缺少视觉塔的几何",
            missing=missing,
            hint="设备层按这些数字编 program；猜 0 只会在装塔时分到一个空缓冲",
        )
    rope = nested.get("rope_parameters")
    if not isinstance(rope, Mapping):
        rope = {}
    return VisionSpec(
        model_type=str(nested.get("model_type", "")),
        depth=int(nested["depth"]),  # type: ignore[arg-type]
        hidden=int(nested["hidden_size"]),  # type: ignore[arg-type]
        patch_size=int(nested["patch_size"]),  # type: ignore[arg-type]
        spatial_merge=int(nested["spatial_merge_size"]),  # type: ignore[arg-type]
        intermediate_size=int(nested["intermediate_size"]),  # type: ignore[arg-type]
        num_heads=int(nested["num_heads"]),  # type: ignore[arg-type]
        out_hidden_size=int(nested["out_hidden_size"]),  # type: ignore[arg-type]
        temporal_patch_size=int(nested["temporal_patch_size"]),  # type: ignore[arg-type]
        num_position_embeddings=int(nested["num_position_embeddings"]),  # type: ignore[arg-type]
        rope_theta=float(rope.get("rope_theta", 10000.0)),  # type: ignore[arg-type]
        rope_type=str(rope.get("rope_type", "axial")),
    )


def mrope_options(text: Mapping[str, object]) -> tuple[tuple[int, ...] | None, bool]:
    """M-RoPE 的两项：频率分段与是否交错。Qwen3.5 把它们写在 `rope_parameters` 下。"""
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        return None, False
    section = rope.get("mrope_section")
    sections = None if section is None else tuple(int(item) for item in section)  # type: ignore[union-attr]
    return sections, bool(rope.get("mrope_interleaved", False))


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
    mrope_section, mrope_interleaved = mrope_options(text)
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
        # 占位 token 的 id 在 config **顶层**（不在 text_config 下）；M-RoPE 在
        # text_config.rope_parameters 下。纯文本 config 这两处都没有，留 `None`。
        image_token_id=_opt_int(config.get("image_token_id")),
        video_token_id=_opt_int(config.get("video_token_id")),
        vision_start_token_id=_opt_int(config.get("vision_start_token_id")),
        vision_end_token_id=_opt_int(config.get("vision_end_token_id")),
        mrope_section=mrope_section,
        mrope_interleaved=mrope_interleaved,
    )
    descriptor.validate()
    return descriptor


def layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    """按 ring 切块（三个家族一样：Qwen3.5 的层布局相同）。"""
    return plan_layers(descriptor.layer_types, config.scheduler.ring)
