"""测试辅助：造 tiny 权重、写最小 safetensors、构造请求。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from sysone.domain.attention import MaskMode
from sysone.domain.decision import DecisionRequest, Question
from sysone.models.descriptor import ModelDescriptor
from sysone.weights.store import TensorStore

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
