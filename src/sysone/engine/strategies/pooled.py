"""pooled 策略：整链 hidden 走我们的 ANE 链，读出与答面格式交回厂商的 `Serve`。

与 split 策略的差别不在"谁做前向"，而在**读出形态**：

- split / mude 是固定的一张码表，`forward(options=n)` 就够表达"读前 n 行"；
- pooled / h2o 的读出是"题面在回答槽上落成哪个标签 token"，这些行随题变（标签能不能落在
  一个 token 上、落在哪个 id 上，取决于题面末尾那几个字符），而且按题型换温度、noul 有地板
  与阈值、score 要出期望值 —— 这些都是厂商 `format_answer` 的事。

所以这条路的边界是：我们算 `hidden`，**再算"哪些词表行 × hidden"**（这一步是设备层的
真相），把 logits 原样交给 `Serve.answer`，由厂商代码判答案。概率不会在中间被我们自己
归一化一次再交给它 —— 那会变成"我们的温度 + 厂商的温度"。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from ...domain.attention import MaskMode
from ...domain.decision import Answer, DecisionRequest, Question
from ...domain.work import CompilationTimes
from ...errors import (
    ConfigError,
    NotReadyError,
    RequestTooLongError,
    UnsupportedModeError,
)
from ...models.vendors.base import question_spec
from ...models.vendors.h2o_serve import Prepared, Serve

# h2o 训练过两种读法、也只声明了两种：整段因果，或整段双向（没有 prefix 语义）。
_SUPPORTED_MODES = (MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL)

_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = Question(text="Which option is correct?", options=("This one", "That one"))


class PooledChain(Protocol):
    """这条策略只需要链的两件事：整条 prompt 的最终 hidden 与卸载。"""

    def hidden(self, tokens: Sequence[int], *, causal: bool = False) -> np.ndarray: ...

    def close(self) -> None: ...


class PooledRunner:
    """实现 `ModelRunner` 的 pooled 路径：一题一次整链前向 + 一次标签行读出。

    `rows` 把"标签 token id"换成"词表里的那几行"（tied 模型上就是 `embed_tokens`），由调用
    方注入：设备层不加一个只为这份 checkpoint 存在的读出方式，链也不需要认识 tokenizer。
    """

    def __init__(
        self,
        load_chain: Callable[[], PooledChain],
        serve: Serve,
        rows: Callable[[Sequence[int]], np.ndarray],
        *,
        max_len: int,
    ) -> None:
        if max_len < 1:
            raise ConfigError("max_len 必须为正", max_len=max_len)
        self._load_chain = load_chain
        self._serve = serve
        self._rows = rows
        self._max_len = int(max_len)
        self._chain: PooledChain | None = None
        self._load_s = 0.0
        self._warm_ok = False

    # ------------------------------------------------------------------ 生命周期
    def init_device(self) -> None:
        """ANE runtime 由链的工厂自己建，这里没有要先做的事。"""
        return None

    def load_weights(self) -> None:
        started = time.perf_counter()
        self._chain = self._load_chain()
        self._load_s = time.perf_counter() - started

    def artifact_bytes(self) -> int:
        """编译产物落在链自己的 scratch 目录；这里不计。"""
        return 0

    def compile_or_warm_up(self) -> CompilationTimes:
        if self._chain is None:
            raise ConfigError("pooled 策略尚未装载权重", strategy="pooled")
        started = time.perf_counter()
        answers, _, _ = self.answers_for(
            DecisionRequest(state=_WARMUP_STATE, questions=(_WARMUP_QUESTION,)),
            MaskMode.CAUSAL,
        )
        probs = answers[0].probs
        if not probs or not all(math.isfinite(value) for value in probs):
            raise NotReadyError("pooled 预热前向产生了非法概率", strategy="pooled",
                                probs=list(probs))
        if abs(sum(probs) - 1.0) > 1e-3:
            raise NotReadyError("pooled 预热前向概率未归一", strategy="pooled",
                                total=sum(probs))
        self._warm_ok = True
        return CompilationTimes(
            load_s=self._load_s,
            compile_s=0.0,
            warmup_s=time.perf_counter() - started,
        )

    def check_health(self) -> bool:
        return self._warm_ok

    def execute(self, plan: object, session: object) -> object:
        raise ConfigError("pooled 策略按请求整批执行，不走单题 execute")

    def shutdown(self) -> None:
        if self._chain is not None:
            self._chain.close()
            self._chain = None
        self._warm_ok = False

    # ------------------------------------------------------------------ 前向
    def answers_for(
        self, request: DecisionRequest, mode: MaskMode
    ) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 读出耗时)。"""
        chain = self._require_chain()
        if mode not in _SUPPORTED_MODES:
            raise UnsupportedModeError(
                "h2o 只有 causal / noncausal_full 两种读法",
                mode=mode.value,
                strategy="pooled",
                supported=sorted(item.value for item in _SUPPORTED_MODES),
            )
        causal = mode is MaskMode.CAUSAL
        started = time.perf_counter()
        answers: list[Answer] = []
        tokens = 0
        for index, question in enumerate(request.questions):
            prepared = self._serve.prepare(request.state, question_spec(question))
            length = len(prepared.ids)
            if length > self._max_len:
                raise RequestTooLongError(
                    "prompt 超过链的缓冲宽度",
                    length=length,
                    max_len=self._max_len,
                    question=index,
                )
            answers.append(self._answer(chain, prepared, question, causal, index))
            tokens += length
        return tuple(answers), tokens, time.perf_counter() - started

    def _answer(
        self,
        chain: PooledChain,
        prepared: Prepared,
        question: Question,
        causal: bool,
        index: int,
    ) -> Answer:
        hidden = chain.hidden(prepared.ids, causal=causal)
        if hidden.shape != (len(prepared.ids), hidden.shape[1]):
            raise ConfigError(
                "链返回的 hidden 行数与 prompt 长度不一致",
                rows=int(hidden.shape[0]),
                length=len(prepared.ids),
            )
        # 读出只取最后一行的**最终归一化** hidden（厂商 `last_hidden_state` 就是它，见
        # `Qwen3_5TextModel.forward` 末尾的 `self.norm`），再点积标签那几行词表权重。
        logits = self._rows(prepared.label_ids) @ hidden[-1]
        value, labels, probs = self._serve.answer(prepared, logits)
        return Answer(
            question_index=index,
            kind=question.kind,
            value=value,
            probs=tuple(probs),
            labels=tuple(labels),
            confidence=max(probs) if probs else 0.0,
        )

    def _require_chain(self) -> PooledChain:
        if self._chain is None:
            raise ConfigError("pooled 策略尚未装载权重", strategy="pooled")
        return self._chain


# ---------------------------------------------------------------------- 装配


def h2o_runner(
    model_dir: str | Path,
    *,
    bucket: int | None = None,
    width: int | None = None,
    quantize: bool = True,
    scratch_dir: str | Path | None = None,
    tokenizer: object | None = None,
) -> PooledRunner:
    """装一条 h2o 的 pooled 链：厂商答面 + 词表行按需取 + ANE 链。

    `bucket` 是 ANE program 的静态列宽、`width` 是缓冲宽度（默认都取 `S_NATIVE`）。h2o 的
    `ffn` 是 9216、`hidden` 是 2560，两个 program 都比 mude 大，所以这两项由调用方说清楚，
    不在这里猜。

    今天**不走 Metal**：`metal=True` 那条路（注意力/GDN 上 GPU）只在 mude 的
    `(2048, 6144)` 上做过逐位对拍，h2o 的 `(2560, 9216)` 还没有证据，所以这条路先用
    numpy 那侧算注意力与 GDN —— 与票据 15 §20 里对过拍的形态一致。因此这里**也不收
    `resident`**：待命窗口是 Metal 流水线换窗的机制，在 numpy 路上传它只会在
    `SplitChain.load` 里换来一句"常驻窗口只在 Metal 流水线上有意义"。
    """
    from ...devices.apple.chain import SplitChain
    from ...devices.apple.compile import S_NATIVE
    from ...models.families import h2o as h2o_family
    from ...weights.checkpoint import open_checkpoint

    serve = Serve.open(model_dir, tokenizer)
    store = open_checkpoint(Path(model_dir))
    embed_name = f"{h2o_family.PREFIX}.embed_tokens.weight"
    weights = h2o_family.weight_plan(Path(model_dir))
    if not weights.descriptor.tied_embeddings:
        raise ConfigError(
            "h2o 的读出权重来自词表（tied），这份 checkpoint 却不是 tied",
            model=str(model_dir),
        )
    bucket_width = int(S_NATIVE if bucket is None else bucket)
    buffer_width = int(bucket_width if width is None else width)

    def load_chain() -> PooledChain:
        return SplitChain.load(
            model_dir,
            weights=weights,
            s=bucket_width,
            width=buffer_width,
            quantize=quantize,
            metal=False,
            scratch_dir=scratch_dir,
        )

    def rows(label_ids: Sequence[int]) -> np.ndarray:
        return np.asarray(store.read_rows(embed_name, list(label_ids)), np.float32)

    return PooledRunner(load_chain, serve, rows, max_len=buffer_width)


__all__ = ["PooledRunner", "h2o_runner"]
