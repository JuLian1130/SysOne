"""参考后端（numpy）：M0 的语义裁判，不追求性能。

只实现全注意力层与 `code_rows` 读出；GDN 参考核、4B 的厂商读法都在 M1。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ...domain.attention import AttentionRole, MaskMode
from ...domain.decision import Answer, RunnerOutput
from ...domain.quant import DType, QuantSpec, Scheme
from ...domain.work import CompilationTimes, StageOp, WorkPlan
from ...errors import ConfigError, MissingTensorError, ReadoutError, UnsupportedPrecisionError
from ...models.descriptor import ModelDescriptor
from ...weights.store import TensorStore
from ..base import Capabilities, CompiledSet, MemoryInfo, Placement, Shape


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


@dataclass(frozen=True)
class _EmptyCompiled:
    """参考后端没有编译产物；占位以便 `Platform.compile` 有返回值。"""

    op: StageOp = StageOp.COPY
    s_bucket: int = 0

    def release(self) -> None:
        return None


class ReferencePlatform:
    """纯 CPU、无编译、无并行的平台声明。"""

    name = "reference"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            dtypes=frozenset({DType.FP32}),
            quant=frozenset({Scheme.NONE}),
            modes=frozenset(MaskMode),
            max_seq=0,
            max_tile=0,
            zero_copy=False,
        )

    def verify_model(self, descriptor: ModelDescriptor) -> None:
        for index, role in enumerate(descriptor.layer_types):
            if role is not AttentionRole.FULL:
                raise ConfigError(
                    "参考后端只支持全注意力层（GDN 参考核在 M1）",
                    layer=index,
                    role=role.value,
                )

    def verify_quant(self, quant: QuantSpec) -> None:
        if quant.is_quantized:
            raise UnsupportedPrecisionError("参考后端只接受未量化权重", quant=str(quant))

    def memory(self) -> MemoryInfo:
        return MemoryInfo(total=0, current=0, usable=0)

    def default_placement(self, descriptor: ModelDescriptor) -> Placement:
        return Placement("cpu")

    def backend(self, op: StageOp, shape: Shape) -> None:
        return None

    def compile(self, programs: object) -> CompiledSet:
        return _EmptyCompiled()

    def sync(self) -> None:
        return None


class ReferenceRunner:
    """单线程 float32 前向 + 码表读出。"""

    def __init__(self, descriptor: ModelDescriptor, store: TensorStore) -> None:
        self._descriptor = descriptor
        self._store = store
        self._weights: dict[str, np.ndarray] = {}

    # ---- 生命周期 -----------------------------------------------------
    def init_device(self) -> None:
        return None

    def load_weights(self) -> None:
        expected = list(self._expected())
        missing = self._store.missing(expected)
        if missing:
            raise MissingTensorError("参考实现的权重不完整", missing=sorted(missing))
        self._weights = {
            name: self._store.read(name).astype(np.float32, copy=False) for name in expected
        }

    def artifact_bytes(self) -> int:
        """参考后端没有编译产物。"""
        return 0

    def compile_or_warm_up(self) -> CompilationTimes:
        return CompilationTimes(load_s=0.0, compile_s=0.0, warmup_s=0.0)

    def check_health(self) -> bool:
        return True

    def shutdown(self) -> None:
        self._weights = {}

    # ---- 执行 ---------------------------------------------------------
    def execute(self, plan: WorkPlan, session: object) -> RunnerOutput:
        if len(plan.questions) != 1:
            raise ConfigError("参考实现每份计划只读一个问题", questions=len(plan.questions))
        hidden = self._forward(plan)
        started = time.perf_counter()
        answer = self._read(plan, hidden)
        return RunnerOutput(answer=answer, readout_s=time.perf_counter() - started)

    # ---- 前向 ---------------------------------------------------------
    def _forward(self, plan: WorkPlan) -> np.ndarray:
        tokens = np.asarray(plan.tokens, dtype=np.int64)
        hidden = self._weights["wte"][tokens]
        mask = plan.mask.as_reference()
        for index, role in enumerate(self._descriptor.layer_types):
            if role is not AttentionRole.FULL:
                raise ConfigError("参考实现不支持线性注意力层", layer=index)
            hidden = hidden + self._attention(hidden, index, mask)
            hidden = hidden + self._feed_forward(hidden, index)
        return self._rms(hidden, "norm")

    def _attention(self, hidden: np.ndarray, index: int, mask: np.ndarray) -> np.ndarray:
        descriptor = self._descriptor
        normed = self._rms(hidden, f"layers.{index}.norm1")
        length = normed.shape[0]
        prefix = f"layers.{index}.attn"
        query = (normed @ self._weights[f"{prefix}.q"].T).reshape(length, descriptor.heads, descriptor.head_dim)
        key = (normed @ self._weights[f"{prefix}.k"].T).reshape(length, descriptor.kv_heads, descriptor.head_dim)
        value = (normed @ self._weights[f"{prefix}.v"].T).reshape(length, descriptor.kv_heads, descriptor.head_dim)
        if descriptor.kv_heads != descriptor.heads:
            repeat = descriptor.heads // descriptor.kv_heads
            key = np.repeat(key, repeat, axis=1)
            value = np.repeat(value, repeat, axis=1)
        scores = np.einsum("lhd,khd->hlk", query, key) / np.sqrt(descriptor.head_dim)
        scores = np.where(mask[None, :, :], scores, -np.inf)
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        out = np.einsum("hlk,khd->lhd", probabilities, value).reshape(length, descriptor.heads * descriptor.head_dim)
        return out @ self._weights[f"{prefix}.o"].T

    def _feed_forward(self, hidden: np.ndarray, index: int) -> np.ndarray:
        normed = self._rms(hidden, f"layers.{index}.norm2")
        prefix = f"layers.{index}.ffn"
        gate = normed @ self._weights[f"{prefix}.gate"].T
        up = normed @ self._weights[f"{prefix}.up"].T
        return (_silu(gate) * up) @ self._weights[f"{prefix}.down"].T

    def _rms(self, hidden: np.ndarray, name: str) -> np.ndarray:
        variance = np.mean(hidden * hidden, axis=-1, keepdims=True) + self._descriptor.rms_eps
        return (hidden / np.sqrt(variance)) * self._weights[name]

    # ---- 读出 ---------------------------------------------------------
    def _read(self, plan: WorkPlan, hidden: np.ndarray) -> Answer:
        question = plan.questions[0]
        plan_spec = self._descriptor.readout
        if plan_spec.kind != "code_rows":
            raise ConfigError("参考实现只实现 code_rows 读出", kind=plan_spec.kind)
        rows = self._weights[self._readout_name()]
        labels = question.labels
        if not labels:
            raise ReadoutError("题目没有候选标签", lane=plan.lane)
        known = plan_spec.labels
        if known:
            missing = [label for label in labels if label not in known]
            if missing:
                raise ReadoutError("标签不在读出表内", labels=list(labels), missing=missing)
            if rows.shape[0] < len(known):
                raise ConfigError("读出表行数少于标签数", rows=int(rows.shape[0]), labels=len(known))
            indices = [known.index(label) for label in labels]
        else:
            if len(labels) > rows.shape[0]:
                raise ReadoutError("候选标签多于读出表行数", labels=len(labels), rows=int(rows.shape[0]))
            indices = list(range(len(labels)))
        temperature = plan_spec.temperature if plan_spec.temperature is not None else 1.0
        logits = rows[np.asarray(indices)] @ hidden[question.end - 1] / temperature
        probabilities = _softmax(logits.astype(np.float64))
        best = int(np.argmax(probabilities))
        return Answer(
            question_index=question.index,
            kind=question.kind,
            value=labels[best],
            probs=tuple(float(value) for value in probabilities),
            labels=labels,
            confidence=float(probabilities[best]),
        )

    def _readout_name(self) -> str:
        plan = self._descriptor.readout
        if plan.file:
            return f"{plan.file.rsplit('.', 1)[0]}.{plan.weight}"
        return str(plan.weight)

    def _expected(self) -> set[str]:
        names = {"wte", "norm", self._readout_name()}
        for index in range(self._descriptor.num_layers):
            prefix = f"layers.{index}"
            names.update(
                {
                    f"{prefix}.norm1",
                    f"{prefix}.norm2",
                    f"{prefix}.attn.q",
                    f"{prefix}.attn.k",
                    f"{prefix}.attn.v",
                    f"{prefix}.attn.o",
                    f"{prefix}.ffn.gate",
                    f"{prefix}.ffn.up",
                    f"{prefix}.ffn.down",
                }
            )
        return names
