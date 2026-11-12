"""参考编码器：按字符码点把文本变成 token（确定、无分词器依赖）。"""

from __future__ import annotations

from dataclasses import dataclass

from ...domain.decision import DecisionRequest, EncodedPrompt
from ...errors import ConfigError


@dataclass(frozen=True)
class CharEncoder:
    """一个最小可用的 `PromptEncoder`：`token = ord(ch) % vocab`。

    它不模拟厂商模板，只保证 Request→Token 的语义在 CI 里可复现；
    M1 的 vendor 策略会换成厂商自己的渲染与分词器。
    """

    vocab: int

    def encode(self, request: DecisionRequest) -> EncodedPrompt:
        if self.vocab < 1:
            raise ConfigError("vocab 必须为正", vocab=self.vocab)
        prefix = self._tokens(request.state)
        tails: list[tuple[int, ...]] = []
        labels: list[tuple[str, ...]] = []
        kinds: list[str] = []
        for index, question in enumerate(request.questions):
            tail = self._tokens(question.text) or (0,)
            if question.kind == "choice":
                if not question.options:
                    raise ConfigError("choice 题缺少候选标签", question=index)
                labels.append(tuple(question.options))
            elif question.kind == "noul":
                labels.append(("yes", "no"))
            else:
                raise ConfigError("未知题型", kind=question.kind, question=index)
            tails.append(tail)
            kinds.append(question.kind)
        return EncodedPrompt(
            prefix_tokens=prefix,
            tails=tuple(tails),
            labels=tuple(labels),
            kinds=tuple(kinds),
        )

    def _tokens(self, text: str) -> tuple[int, ...]:
        return tuple((ord(char) + 1) % self.vocab for char in text)
