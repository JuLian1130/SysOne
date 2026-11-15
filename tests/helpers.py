"""测试辅助：造 tiny 权重、写最小 safetensors、构造请求。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from sysone.domain.attention import MaskMode
from sysone.domain.decision import DecisionRequest, Question
from sysone.models.descriptor import ModelDescriptor
from sysone.weights.packing import pack_tensors
from sysone.weights.store import TensorStore, write_file

FIXTURES = Path(__file__).parent / "fixtures"
OPTIONS = ("A", "B", "C", "D")


def load_tiny_descriptor() -> ModelDescriptor:
    payload = json.loads(
        (FIXTURES / "reference_tiny" / "descriptor.json").read_text(encoding="utf-8")
    )
    return ModelDescriptor.from_mapping(payload)


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> Path:
    """写一个只含 F32 张量的最小 safetensors 文件（与读取端对齐）。"""
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        array = np.ascontiguousarray(np.asarray(tensors[name], dtype=np.float32))
        data = array.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))
    return path



def write_row_int8_checkpoint(
    directory: Path, tensors: dict[str, np.ndarray], *, min_elems: int = 1
) -> Path:
    """把一组张量按 row-int8 容器写到目录：`__q` / `__s` 分片加 `int8_manifest.json`。

    `min_elems` 决定哪些二维张量被量化；默认全量化，因为测试里的张量都远小于真实检查点
    的 1<<20 门限，否则一条 int8 通路都走不到。
    """
    packed, manifest = pack_tensors(tensors, min_elems=min_elems)
    write_file(directory / "model.safetensors", packed)
    payload = {
        "tensors": {
            name: {"kind": entry.kind, "shape": list(entry.shape)}
            for name, entry in manifest.entries.items()
        },
        "source_bf16_bytes": manifest.source_bf16_bytes,
        "int8_bytes": manifest.int8_bytes,
    }
    (directory / "int8_manifest.json").write_text(json.dumps(payload), "utf-8")
    return directory



def make_mude_checkpoint(
    directory: Path,
    *,
    hidden: int = 256,
    ffn: int = 768,
    q_heads: int = 2,
    kv_heads: int = 1,
    head_dim: int = 256,
    l_k_heads: int = 2,
    l_k_dim: int = 128,
    l_v_heads: int = 4,
    l_v_dim: int = 128,
    conv_k: int = 4,
    vocab: int = 512,
    layers: tuple[str, ...] = ("linear_attention", "full_attention"),
    code_rows: int = 255,
    seed: int = 0,
    tensor_prefix: str = "language_model",
    with_readout: bool = True,
) -> Path:
    """造一个 mude 链形状的检查点目录（`config.json` + 权重 + 题面配置 + 读出表）。

    默认几何刻意选成 **4B 的形状关系**：`qk_dim = 256`、`value_dim = 512`（v 头 4 个对
    k 头 2 个），`core_out = q_heads × head_dim = 2 × 256` 等于 `value_dim` —— 与 h2o 一样
    是 `2 × qk + value` 的卷积宽度，且 **v 头数是 k 头数的两倍**（`repeat_interleave` 那条
    路才真的被走到）。`core_out == value_dim` 是几何的硬约束（两种层共用一套 out program）。

    `tensor_prefix` / `with_readout` 用来造**另一种布局**的检查点：h2o 的张量叫
    `model.language_model.*`、读出是厂商的 pooled（目录里没有 `readout.safetensors`），
    所以"整链能不能装 h2o"这条要靠这两项才测得出来（形状仍用上面的小模型）。

    两个头维不能随便改，两个核都按 128 / 256 排过：
    - GDN 的 `l_k_dim` / `l_v_dim` 必须是 128（`e8_gdn.metal:109`，线程几何按 128 排）；
    - 全注意力的 `head_dim` 必须是 256（`e8_attn.metal` 的 `q_0..q_31` 分片偏移按 256 排）。
    mude 与 h2o 两两都满足，所以这不是缺口，但夹具换个头维就会炸（或更糟：静默算错）。
    """
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    assert q_heads * head_dim == l_v_heads * l_v_dim, "core_out 必须等于 value_dim"

    def matrix(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.05).astype(np.float16)

    def vector(size: int, *, positive: bool = False) -> np.ndarray:
        values = rng.standard_normal(size) * 0.05
        return (np.abs(values) if positive else values).astype(np.float16)

    qk, value = l_k_heads * l_k_dim, l_v_heads * l_v_dim
    prefix = tensor_prefix
    weights: dict[str, np.ndarray] = {
        f"{prefix}.embed_tokens.weight": matrix(vocab, hidden),
        # norm 是零中心存的（家族递 `NORM_OFFSET = 1.0`），所以这里给 0 附近的值。
        f"{prefix}.norm.weight": vector(hidden),
    }
    for index, kind in enumerate(layers):
        base = f"{prefix}.layers.{index}"
        weights[f"{base}.input_layernorm.weight"] = vector(hidden)
        weights[f"{base}.post_attention_layernorm.weight"] = vector(hidden)
        weights[f"{base}.mlp.gate_proj.weight"] = matrix(ffn, hidden)
        weights[f"{base}.mlp.up_proj.weight"] = matrix(ffn, hidden)
        weights[f"{base}.mlp.down_proj.weight"] = matrix(hidden, ffn)
        if kind == "linear_attention":
            attn = f"{base}.linear_attn"
            weights[f"{attn}.in_proj_qkv.weight"] = matrix(2 * qk + value, hidden)
            weights[f"{attn}.in_proj_z.weight"] = matrix(value, hidden)
            weights[f"{attn}.in_proj_b.weight"] = matrix(l_v_heads, hidden)
            weights[f"{attn}.in_proj_a.weight"] = matrix(l_v_heads, hidden)
            weights[f"{attn}.out_proj.weight"] = matrix(hidden, value)
            weights[f"{attn}.conv1d.weight"] = (
                rng.standard_normal((2 * qk + value, 1, conv_k)) * 0.05
            ).astype(np.float16)
            weights[f"{attn}.A_log"] = rng.standard_normal(l_v_heads).astype(np.float32)
            weights[f"{attn}.dt_bias"] = vector(l_v_heads)
            weights[f"{attn}.norm.weight"] = vector(l_v_dim, positive=True) + 0.5
        else:
            attn = f"{base}.self_attn"
            weights[f"{attn}.q_proj.weight"] = matrix(2 * q_heads * head_dim, hidden)
            weights[f"{attn}.k_proj.weight"] = matrix(kv_heads * head_dim, hidden)
            weights[f"{attn}.v_proj.weight"] = matrix(kv_heads * head_dim, hidden)
            weights[f"{attn}.o_proj.weight"] = matrix(hidden, q_heads * head_dim)
            weights[f"{attn}.q_norm.weight"] = vector(head_dim)
            weights[f"{attn}.k_norm.weight"] = vector(head_dim)

    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, array in weights.items():
        data = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16"}[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (directory / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)
    )

    config = {
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": ffn,
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "num_hidden_layers": len(layers),
            "layer_types": list(layers),
            "linear_num_key_heads": l_k_heads,
            "linear_key_head_dim": l_k_dim,
            "linear_num_value_heads": l_v_heads,
            "linear_value_head_dim": l_v_dim,
            "linear_conv_kernel_dim": conv_k,
            "rope_parameters": {"rope_theta": 10_000.0, "partial_rotary_factor": 0.25},
            "max_position_embeddings": 4096,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True,
        }
    }
    (directory / "config.json").write_text(json.dumps(config), "utf-8")
    (directory / "decision_config.json").write_text(json.dumps({"temperature": 1.0}), "utf-8")
    if with_readout:
        write_safetensors(
            directory / "readout.safetensors",
            {"weight": (rng.standard_normal((code_rows, hidden)) * 0.05).astype(np.float32)},
        )
    return directory


def tiny_weights(descriptor: ModelDescriptor, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return rng.normal(0.0, 0.35, (rows, cols)).astype(np.float32)

    hidden = descriptor.hidden
    weights: dict[str, np.ndarray] = {
        "wte": matrix(descriptor.vocab, hidden),
        "norm": np.ones(hidden, dtype=np.float32),
        "readout": matrix(descriptor.readout.rows or 4, hidden),
    }
    q_dim = descriptor.heads * descriptor.head_dim
    kv_dim = descriptor.kv_heads * descriptor.head_dim
    for index in range(descriptor.num_layers):
        prefix = f"layers.{index}"
        weights.update(
            {
                f"{prefix}.norm1": np.ones(hidden, dtype=np.float32),
                f"{prefix}.norm2": np.ones(hidden, dtype=np.float32),
                f"{prefix}.attn.q": matrix(q_dim, hidden),
                f"{prefix}.attn.k": matrix(kv_dim, hidden),
                f"{prefix}.attn.v": matrix(kv_dim, hidden),
                f"{prefix}.attn.o": matrix(hidden, q_dim),
                f"{prefix}.ffn.gate": matrix(descriptor.ffn, hidden),
                f"{prefix}.ffn.up": matrix(descriptor.ffn, hidden),
                f"{prefix}.ffn.down": matrix(hidden, descriptor.ffn),
            }
        )
    return weights


def make_tiny(tmp_path: Path, seed: int = 0) -> tuple[ModelDescriptor, TensorStore]:
    descriptor = load_tiny_descriptor()
    path = write_safetensors(tmp_path / "tiny.safetensors", tiny_weights(descriptor, seed))
    return descriptor, TensorStore([path])


def make_request(*, mode: MaskMode | None = None, questions: int = 1, state: str = "Q: 选哪个？") -> DecisionRequest:
    return DecisionRequest(
        state=state,
        questions=tuple(
            Question(text=f"item {index}", options=OPTIONS, kind="choice")
            for index in range(questions)
        ),
        attention_mode=mode,
        request_id=f"req{questions}",
    )
