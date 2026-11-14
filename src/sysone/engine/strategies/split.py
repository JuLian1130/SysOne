"""split 策略：投影/FFN 走 ANE program，注意力与 GDN 走 Metal 核。

与 vendor 的差别只在"谁来做前向"：题面照家族的线格式自己渲染（`MudePrompter`），
投影与 FFN 交给 ANE、注意力/GDN 交给 GPU，掩码由链自己实现；vendor 连模板与前向
都交给厂商代码。两者都不排 `MaskPlan`，所以 `execute()` 与 vendor 一样不可用。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

from ...domain.attention import MaskMode
from ...domain.decision import Answer, DecisionRequest, Question
from ...domain.work import CompilationTimes
from ...errors import ConfigError, NotReadyError, ReadoutError, RequestTooLongError, UnsupportedModeError

# 三种模式这条链都能表达：双向（`prefix_noncausal` 的前缀窗 + `noncausal_full` 的全可见）
# 与三角（`causal`，M3 的核改动）。声明在这里而不在 descriptor 里：descriptor 说模型能跑
# 什么，这里说这条实现做到了什么。
_SUPPORTED_MODES = (MaskMode.PREFIX_NONCAUSAL, MaskMode.NONCAUSAL_FULL, MaskMode.CAUSAL)

# 预热用的题：真正的冷启动失败（program 没装上、面没绑好）在第一题就会炸，
# 这一题让它在 `Engine.start()` 里就炸，而不是让第一道真实请求撞上。
_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = Question(text="Which option is correct?", options=("This one", "That one"))


class ModelChain(Protocol):
    """`MudeChain` 只需要这两件事：一次整链前向与卸载。"""

    def forward(self, tokens: Sequence[int], *, options: int, prefix: int = 0,
                causal: bool = False) -> object: ...

    def close(self) -> None: ...


class Prompter(Protocol):
    """题面 → (整条 prompt 的 id, 前缀长度, 答案标签, 标签对应的选项序号)。"""

    def encode(
        self, state: str, question: Question
    ) -> tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]: ...


class SplitRunner:
    """实现 `ModelRunner` 的 split 路径：一题一次整链前向。

    `load_chain` 延迟到 `load_weights()` 才调用，所以构造它不碰 ANE；`max_len` 是链的
    缓冲宽度（prompt 超过它就是 422，而不是让链里抛裸 `ValueError`）。
    """

    def __init__(
        self,
        load_chain: Callable[[], ModelChain],
        prompter: Prompter,
        *,
        max_len: int,
    ) -> None:
        if max_len < 1:
            raise ConfigError("max_len 必须为正", max_len=max_len)
        self._load_chain = load_chain
        self._prompter = prompter
        self._max_len = int(max_len)
        self._chain: ModelChain | None = None
        self._load_s = 0.0
        self._warm_ok = False

    # ------------------------------------------------------------------ 生命周期

    def init_device(self) -> None:
        """ANE runtime 与 Metal 流水线由链的工厂自己建，这里没有要先做的事。"""
        return None

    def load_weights(self) -> None:
        started = time.perf_counter()
        self._chain = self._load_chain()
        self._load_s = time.perf_counter() - started

    def artifact_bytes(self) -> int:
        """编译产物落在链自己的 scratch 目录；这里不计（产物预算默认不设上限）。"""
        return 0

    def compile_or_warm_up(self) -> CompilationTimes:
        if self._chain is None:
            raise ConfigError("split 策略尚未装载权重", strategy="split")
        started = time.perf_counter()
        self._warm()
        return CompilationTimes(
            load_s=self._load_s,
            compile_s=0.0,
            warmup_s=time.perf_counter() - started,
        )

    def check_health(self) -> bool:
        return self._warm_ok

    def execute(self, plan: object, session: object) -> object:
        raise ConfigError("split 策略按请求整批执行，不走单题 execute")

    def shutdown(self) -> None:
        if self._chain is not None:
            self._chain.close()
            self._chain = None
        self._warm_ok = False

    # ------------------------------------------------------------------ 前向

    def answers_for(
        self, request: DecisionRequest, mode: MaskMode
    ) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 前向耗时)。

        一个请求里所有题共享同一个 `state`，所以前缀天然相同：**两题以上**且链提供共享路径
        时，前缀只算一遍、每题只跑自己那条尾巴（`SplitRunner._answer_shared`）。只有一题、
        或分词边界让某题的前缀与别人不同、或链是 numpy 形态（没有零拷贝流水线）时，退回
        逐题整趟。两条路径的答案在这套实现里应当逐位相同（信息量守恒，不做近似）。
        """
        chain = self._require_chain()
        if mode not in _SUPPORTED_MODES:
            raise UnsupportedModeError(
                "mude 链没有这个掩码",
                mode=mode.value,
                strategy="split",
                supported=sorted(item.value for item in _SUPPORTED_MODES),
            )
        # `noncausal_full` 就是把前缀当 0：掩码 `(q >= 0) | (k < 0)` 恒真。
        # `causal` 的共享前缀照样成立：前缀行的掩码在相内自己成三角，尾巴行看 `[0, q]` 包含
        # 整个前缀 —— 两条路（共享 / 逐题）的可见集合一样，所以批尾巴的槽位规则也不用改。
        causal = mode is MaskMode.CAUSAL
        use_prefix = mode is not MaskMode.NONCAUSAL_FULL
        started = time.perf_counter()
        # 先把所有题编码完：共享前缀要先知道每题的尾巴，长度检查也按整条 prompt 做。
        encoded: list[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]] = []
        for index, question in enumerate(request.questions):
            encoded.append(self._prompter.encode(request.state, question))
            if len(encoded[-1][0]) > self._max_len:
                raise RequestTooLongError(
                    "prompt 超过链的缓冲宽度",
                    length=len(encoded[-1][0]),
                    max_len=self._max_len,
                    question=index,
                )
        shared = _shared_prefix(encoded) if use_prefix else None
        forward_shared = getattr(chain, "forward_shared", None)
        if shared is not None and len(encoded) > 1 and callable(forward_shared):
            answers = _answer_shared(forward_shared, shared, encoded, request, causal)
        else:
            answers = _answer_serial(chain, encoded, use_prefix, request, causal)
        tokens = sum(len(sequence) for sequence, *_ in encoded)
        return tuple(answers), tokens, time.perf_counter() - started

    def _warm(self) -> None:
        answers, _, _ = self.answers_for(
            DecisionRequest(state=_WARMUP_STATE, questions=(_WARMUP_QUESTION,)),
            MaskMode.PREFIX_NONCAUSAL,
        )
        answer = answers[0]
        if not answer.probs or not all(math.isfinite(value) for value in answer.probs):
            raise NotReadyError("split 预热前向产生了非法概率", strategy="split", probs=list(answer.probs))
        if abs(sum(answer.probs) - 1.0) > 1e-3:
            raise NotReadyError("split 预热前向概率未归一", strategy="split", total=sum(answer.probs))
        self._warm_ok = True

    def _require_chain(self) -> ModelChain:
        if self._chain is None:
            raise ConfigError("split 策略尚未装载权重", strategy="split")
        return self._chain

    @staticmethod
    def _shares(raw: object) -> tuple[float, ...]:
        values = tuple(float(value) for value in raw)  # type: ignore[union-attr]
        if not values:
            raise ReadoutError("链没有给出概率")
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ReadoutError("链给出的概率非法", probs=list(values))
        total = sum(values)
        if not total > 0.0:
            raise ReadoutError("链给出的概率之和为 0", probs=list(values))
        return values


def _shared_prefix(
    encoded: Sequence[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]],
) -> tuple[int, ...] | None:
    """所有题共用同一条前缀时返回它，否则 `None`。

    前缀只由 `state` 决定（`MudePrompter` 按 state 缓存它），所以一次请求里天然一致；仍然
    逐题比对：分词边界可能把某题的尾巴和前缀粘成一个 token，那时 `prefix_len` 或前缀内容
    就不一样，不能复用（复用会改变那一题的数值）。
    """
    length = encoded[0][1]
    if length <= 0:
        return None
    prefix = encoded[0][0][:length]
    for sequence, prefix_len, _, _ in encoded[1:]:
        if prefix_len != length or sequence[:length] != prefix:
            return None
    return prefix


def _answer_shared(
    forward_shared: Callable[..., Sequence[object]],
    prefix: tuple[int, ...],
    encoded: Sequence[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]],
    request: DecisionRequest,
    causal: bool = False,
) -> list[Answer]:
    """共享前缀的一条路：`prefix` + 每题的尾巴，一次调用拿到所有题的概率。"""
    tails = [sequence[len(prefix):] for sequence, *_ in encoded]
    options = [len(labels) for _, _, labels, _ in encoded]
    raw = forward_shared(prefix, tails, options=options, causal=causal)
    answers: list[Answer] = []
    for index, (_, _, labels, indexes) in enumerate(encoded):
        probs = SplitRunner._shares(raw[index])
        values = tuple(probs[item] for item in indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=request.questions[index].kind,
                value=labels[best],
                probs=values,
                labels=labels,
                confidence=values[best],
            )
        )
    return answers


def _answer_serial(
    chain: ModelChain,
    encoded: Sequence[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]],
    use_prefix: bool,
    request: DecisionRequest,
    causal: bool = False,
) -> list[Answer]:
    """逐题整趟（今天的行为；单题与不可复用时走这条）。"""
    answers: list[Answer] = []
    for index, (sequence, prefix_len, labels, indexes) in enumerate(encoded):
        probs = SplitRunner._shares(
            chain.forward(sequence, options=len(labels),
                          prefix=prefix_len if use_prefix else 0, causal=causal))
        values = tuple(probs[item] for item in indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=request.questions[index].kind,
                value=labels[best],
                probs=values,
                labels=labels,
                confidence=values[best],
            )
        )
    return answers


# ---------------------------------------------------------------------- 装配


def mude_runner(
    model_dir: str | Path,
    *,
    bucket: int | None = None,
    width: int | None = None,
    metal: bool = True,
    quantize: bool = True,
    scratch_dir: str | Path | None = None,
    resident: int | None = None,
) -> SplitRunner:
    """装一条 mude split 链：分词器 + 回答码 + ANE/Metal 链。

    `bucket` 是 ANE program 的静态列宽（默认 `S_NATIVE`），`width` 是 GPU 侧缓冲的列数
    （默认等于 bucket，prompt 更长时必须给更大的值）。两者都决定内存，所以由调用方说。

    `resident` 是待命常驻的层数，`None` 表示全装；窗口的具体病理见
    `devices/apple/residency.py`（它是这张卡上最大的一笔内存，所以由调用方决定让不让）。
    """
    from ...devices.apple.chain import MudeChain
    from ...devices.apple.compile import S_NATIVE
    from ...models.families import mude as mude_family

    codes = mude_family.read_codes(model_dir)
    prompter = mude_family.MudePrompter(_load_tokenizer(model_dir), codes)
    bucket_width = int(S_NATIVE if bucket is None else bucket)
    buffer_width = int(bucket_width if width is None else width)

    def load_chain() -> ModelChain:
        return MudeChain.load(
            model_dir,
            s=bucket_width,
            width=buffer_width,
            quantize=quantize,
            metal=metal,
            scratch_dir=scratch_dir,
            resident=resident,
        )

    return SplitRunner(load_chain, prompter, max_len=buffer_width)


def _load_tokenizer(model_dir: str | Path) -> object:
    """模型目录的分词器：`tokenizers` 直连 + `jinja2` 模板，不 import transformers。"""
    from ...models.tokenizer import ChatTokenizer

    try:
        return ChatTokenizer.load(model_dir)
    except ImportError as exc:  # 装了模型但没装这层的依赖：环境未就绪，不是模型问题
        raise NotReadyError(
            "split 策略需要 tokenizers 与 jinja2 的分词器",
            strategy="split",
            reason=str(exc),
        ) from exc


__all__ = ["ModelChain", "Prompter", "SplitRunner", "mude_runner"]
