"""厂商侧的两份公共契约：厂商桥，与答面。

厂商桥（`VendorBridge`）只负责两件事：把「题面 + 选项」渲染成它自己训练时的线格式，以及在
它自己的前向与读出代码上跑一次。权重装载、注意力模式和预算由我们给，其余一律交给厂商代码，
这样对拍时只可能是权重或模式不同，不会是模板不同。

答面（`AnswerSurface`）把同一件事拆成两半：`prepare` 渲染题面并记下这一题判答案要用的量，
`answer` 吃一个 logits 向量判答案。前向归谁不影响它 —— 厂商桥在自己的 torch 模型上跑，
`pooled` 策略在我们的 ANE 链上跑，两条路共用同一份答面，所以判词差异不可能是答面造成的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ...domain.attention import MaskMode
from ...domain.decision import Question
from ...domain.work import CompilationTimes


@dataclass(frozen=True)
class VendorAnswer:
    """厂商线格式读出后的一题答案：标签顺序即厂商给的顺序。"""

    value: str
    labels: tuple[str, ...]
    probs: tuple[float, ...]
    input_tokens: int
    confidence: float = 0.0
    readout_s: float = 0.0


class VendorBridge(Protocol):
    """一个厂商模型的桥；生命周期由 VendorRunner 驱动。"""

    name: str

    def load(self) -> None:
        """装载权重并自检；失败要抛 sysone.errors 里的具体错误。"""

    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        """按厂商自己的模板与前向跑一道题。"""

    def compile_or_warm_up(self) -> CompilationTimes:
        """跑一次预热前向并校验它的概率；不合法要抛 `NotReadyError`。"""

    def artifact_bytes(self) -> int:
        """占用磁盘的编译产物字节数；没有产物就返回 0。"""

    def close(self) -> None:
        """释放权重。"""


@dataclass(frozen=True)
class Prepared:
    """一道题在答面上的全部事实：喂给前向的 token，与判答案要用的东西。

    `label_ids[k]` 对应 `labels[k]`、`names[k]`，也对应 `answer()` 收到的 `logits[k]` ——
    三处顺序都是厂商题面给的顺序，不许重排。
    """

    ids: tuple[int, ...]
    label_ids: tuple[int, ...]
    labels: tuple[str, ...]
    names: tuple[str, ...]
    descs: tuple[str, ...]
    kind: str
    temperature: float
    threshold: float | None


class AnswerSurface(Protocol):
    """答面：题面怎么渲染、答案怎么判。

    `serve(model_dir, tokenizer)` 交出一个实现它的对象；`pooled` 策略只认这两个方法与它，所以
    它不认识任何具体家族 —— 换一家接进来，改的是那一家自己的适配文件。

    `supports_images` 是"这份答面会不会把图块渲染进题面"。`False` 时装配**不建视觉塔**：给一个
    收不了图的模型编 26 份塔产物，是每次启动几秒与一块常驻（票据 21）。
    """

    supports_images: bool

    def prepare(self, state: str, spec: dict, *, images: int = 0) -> Prepared:
        """题面 → 整条 prompt 的 token 与回答槽上的读出位置。"""

    def answer(
        self, prepared: Prepared, logits: Sequence[float]
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], float]:
        """回答槽上的 logits（未除温度）→ (值, 标签, 概率, 置信度)。"""


# 协议层把选项拼成 `"<key>: <描述>"`（`service/protocol._wire_question`），而厂商的线格式要的是
# **两个字段**：选项键（它自己排 A/B/C 字母）与判据描述。`question_spec` 只拿到拼好的那一个
# 字符串，所以这里按第一个 `": "` 拆回去 —— 不拆的话厂商看到的是「键: 键」，判据描述整个丢掉。
_OPTION_SEPARATOR = ": "


def question_spec(question: Question) -> dict:
    """把 domain 的 Question 转成厂商线格式的问题描述。

    厂商自己的 `question_prompt` 会按 type 选模板；criteria 是「选项键 → 判据描述」的映射
    （noul 的键是 `true` / `false`，由协议层渲染成两行，这里原样拆回）。
    **score 例外**：它的 criteria 是**有序档位数组**（顺序即档位高低，厂商按 `Σ i·p_i` 出
    期望值）。协议层给 score 渲染的也是 `"<序号>: <档位>"` 的行，这里按序拆回档位本身。

    拆分的必要性是实测的（票据 12）：只给键不给描述时，h2o 在真实题集上与厂商自己的 harness
    有 7/105 条判词不同（torchcast 6/105）；把厂商原装的 criteria 喂回去，这 13 条**全部**
    复现 —— 也就是说差的不是权重、不是掩码、也不是 dtype，是选项行里少了判据描述。
    """
    if question.kind == "score":
        return {
            "type": "score",
            "instructions": question.text,
            "criteria": [_description(option) for option in question.options or ()],
        }
    criteria: dict[str, str] = {}
    for option in question.options or ():
        key, separator, description = option.partition(_OPTION_SEPARATOR)
        criteria[key if separator else option] = description if separator else option
    return {
        "type": question.kind,
        "instructions": question.text,
        "criteria": criteria,
    }


def _description(option: str) -> str:
    """选项行 `"<键>: <描述>"` 的描述那一段；没有分隔符时整行就是描述。"""
    _key, separator, description = option.partition(_OPTION_SEPARATOR)
    return description if separator else option


def normalize(answer: VendorAnswer, question: Question) -> tuple[str, tuple[float, ...], tuple[str, ...]]:
    """把厂商返回的标签/概率对齐到本题的候选顺序。"""
    if not answer.labels:
        raise ValueError("厂商没有返回候选标签")
    probs = tuple(float(value) for value in answer.probs)
    if len(probs) != len(answer.labels):
        raise ValueError("厂商返回的标签与概率长度不一致")
    return answer.value, probs, answer.labels


def wire_answer(result: dict) -> tuple[str, tuple[str, ...], tuple[float, ...], float]:
    """把厂商线格式（choice / noul / score）转成 (值, 标签, 概率, 置信度)。

    置信度取厂商答面**自己**算的那个数：两条前向共用这一份判答案的代码，这个口径就不该由
    我们再算一遍（mude 的码表读出是另一种口径，见 `models/families/mude.confidence`）。
    noul 的答面不带这个字段 —— 厂商只给 P(true) —— 这里取它 `served_top` 的口径，即较大的
    那一侧概率。
    """
    kind = result.get("type")
    if kind == "noul":
        p = float(result.get("noul", 0.0))
        decision = result.get("decision")
        value = "yes" if (decision if decision is not None else p >= 0.5) else "no"
        probs = (p, 1.0 - p)
        return value, ("yes", "no"), probs, max(probs)
    if kind == "score":
        probabilities = result.get("probabilities") or {}
        labels = tuple(str(key) for key in probabilities)
        probs = tuple(float(value) for value in probabilities.values())
        if not labels:
            return f"{float(result.get('score', 0.0)):.3f}", (), (), 0.0
        score = float(result.get("score", 0.0))
        return f"{score:.3f}", labels, probs, _confidence(result, probs)
    probabilities = result.get("probabilities") or {}
    labels = tuple(str(key) for key in probabilities)
    probs = tuple(float(value) for value in probabilities.values())
    if not labels:
        raise ValueError("厂商的 choice 答案没有概率表")
    return str(result.get("choice", labels[0])), labels, probs, _confidence(result, probs)


def _confidence(result: dict, probs: Sequence[float]) -> float:
    """厂商答面给的置信度；没给时退回最大概率（只为让类型定下来，不该走到）。"""
    given = result.get("confidence")
    if isinstance(given, (int, float)):
        return float(given)
    return max(probs) if probs else 0.0
