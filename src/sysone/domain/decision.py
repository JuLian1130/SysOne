"""请求与回包：服务层与引擎之间的唯一数据契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .attention import MaskMode
from .quant import QuantSpec


@dataclass(frozen=True)
class MediaRef:
    """图片/视频引用；M0 只记录，不解码（`UnsupportedMediaError`）。"""

    kind: str
    locator: str
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class Question:
    """一道题：题干文本 + 候选标签（choice）。

    `options` 是线格式渲染好的选项行 `<键>: <描述>`：choice 的键是选项键，noul 的键是
    `true` / `false`（判据描述，可缺省），score 的键是档位序号。`noul` 的答案标签固定是
    ("yes", "no")，与 `options` 无关；mude 与参考编码器都不看 noul 的 `options`。
    """

    text: str
    options: tuple[str, ...] = ()
    kind: str = "choice"


@dataclass(frozen=True)
class DecisionRequest:
    """一次决策请求：共享前缀 `state` + 若干问题 + 可选模态。"""

    state: str
    questions: tuple[Question, ...]
    media: tuple[MediaRef, ...] = ()
    attention_mode: MaskMode | None = None
    request_id: str = ""


@dataclass(frozen=True)
class Answer:
    """单题结果。`probs` 与 `labels` 一一对应，且只在候选标签上归一化。"""

    question_index: int
    kind: str
    value: str
    probs: tuple[float, ...]
    labels: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class Usage:
    """用量：输入 token 数与前向调用次数。"""

    input_tokens: int
    prefill_calls: int


@dataclass(frozen=True)
class Provenance:
    """结果来源：家族、产物键、量化、注意力模式及其是否经过训练。"""

    family: str
    artifact_id: str
    quant: QuantSpec
    attention_mode: MaskMode
    attention_trained: bool
    vendor_bridge: str | None = None


@dataclass(frozen=True)
class Timings:
    """分阶段耗时（秒）。"""

    prepare: float
    prefill: float
    readout: float
    total: float


@dataclass(frozen=True)
class DecisionResult:
    """一次请求的完整回包。"""

    answers: tuple[Answer, ...]
    usage: Usage
    provenance: Provenance
    timings: Timings | None = None
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedPrompt:
    """编码器输出：共享前缀 + 每题 tail 的 token、标签与题型。

    `tails` 与 `labels` 与请求里的问题一一对应；空 tail 由编码器补齐（不能为空序列）。
    """

    prefix_tokens: tuple[int, ...]
    tails: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[str, ...], ...]
    kinds: tuple[str, ...]


class PromptEncoder(Protocol):
    """把请求编码成 token 的组件（参考实现按码点，M1 起用厂商分词器）。"""

    def encode(self, request: DecisionRequest) -> EncodedPrompt: ...


@dataclass(frozen=True)
class RunnerOutput:
    """模型执行体的返回值：答案 + 读出阶段耗时（秒）。"""

    answer: Answer
    readout_s: float = 0.0
