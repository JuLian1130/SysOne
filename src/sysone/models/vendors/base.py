"""厂商桥的公共契约。

厂商桥只负责两件事：把「题面 + 选项」渲染成它自己训练时的线格式，以及在
它自己的前向与读出代码上跑一次。权重装载、注意力模式和预算由我们给，
其余一律交给厂商代码，这样对拍时只可能是权重或模式不同，不会是模板不同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ...domain.attention import MaskMode
from ...domain.decision import Question


@dataclass(frozen=True)
class VendorAnswer:
    """厂商线格式读出后的一题答案：标签顺序即厂商给的顺序。"""

    value: str
    labels: tuple[str, ...]
    probs: tuple[float, ...]
    input_tokens: int
    readout_s: float = 0.0


class VendorBridge(Protocol):
    """一个厂商模型的桥；生命周期由 VendorRunner 驱动。"""

    name: str

    def load(self) -> None:
        """装载权重并自检；失败要抛 sysone.errors 里的具体错误。"""

    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        """按厂商自己的模板与前向跑一道题。"""

    def artifact_bytes(self) -> int:
        """占用磁盘的编译产物字节数；没有产物就返回 0。"""

    def close(self) -> None:
        """释放权重。"""


def question_spec(question: Question) -> dict:
    """把 domain 的 Question 转成厂商线格式的问题描述。

    厂商自己的 `question_prompt` 会按 type 选模板；criteria 用「选项→判据」
    的自然语言映射，选项本身即判据（与旧实现一致）。
    """
    options = question.options or ()
    return {
        "type": question.kind,
        "instructions": question.text,
        "criteria": {option: option for option in options},
    }


def normalize(answer: VendorAnswer, question: Question) -> tuple[str, tuple[float, ...], tuple[str, ...]]:
    """把厂商返回的标签/概率对齐到本题的候选顺序。"""
    if not answer.labels:
        raise ValueError("厂商没有返回候选标签")
    probs = tuple(float(value) for value in answer.probs)
    if len(probs) != len(answer.labels):
        raise ValueError("厂商返回的标签与概率长度不一致")
    return answer.value, probs, answer.labels


def wire_answer(result: dict) -> tuple[str, tuple[str, ...], tuple[float, ...]]:
    """把厂商线格式（choice / noul / score）转成 (值, 标签, 概率)。"""
    kind = result.get("type")
    if kind == "noul":
        p = float(result.get("noul", 0.0))
        decision = result.get("decision")
        value = "yes" if (decision if decision is not None else p >= 0.5) else "no"
        return value, ("yes", "no"), (p, 1.0 - p)
    if kind == "score":
        probabilities = result.get("probabilities") or {}
        labels = tuple(str(key) for key in probabilities)
        probs = tuple(float(value) for value in probabilities.values())
        if not labels:
            return f"{float(result.get('score', 0.0)):.3f}", (), ()
        score = float(result.get("score", 0.0))
        return f"{score:.3f}", labels, probs
    probabilities = result.get("probabilities") or {}
    labels = tuple(str(key) for key in probabilities)
    probs = tuple(float(value) for value in probabilities.values())
    if not labels:
        raise ValueError("厂商的 choice 答案没有概率表")
    return str(result.get("choice", labels[0])), labels, probs
