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
    UnsupportedMediaError,
    UnsupportedModeError,
)
from ...models.vision import merge
from ...models.vendors.base import question_spec
from ...models.vendors.h2o_serve import Prepared, Serve
from .vision import VisionRunner

# h2o 训练过两种读法、也只声明了两种：整段因果，或整段双向（没有 prefix 语义）。
_SUPPORTED_MODES = (MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL)

_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = Question(text="Which option is correct?", options=("This one", "That one"))


class PooledChain(Protocol):
    """这条策略只需要链的两件事：末位 token 的最终 hidden 与卸载。"""

    def last_hidden(self, tokens: Sequence[int], *, causal: bool = False,
                    mm: object | None = None) -> np.ndarray: ...

    def close(self) -> None: ...


class PooledRunner(VisionRunner):
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
        image_token_id: int | None = None,
        load_tower: Callable[[object], object] | None = None,
    ) -> None:
        if max_len < 1:
            raise ConfigError("max_len 必须为正", max_len=max_len)
        self._load_chain = load_chain
        self._serve = serve
        self._rows = rows
        self._max_len = int(max_len)
        self._image_token = image_token_id
        # 视觉塔的工厂（`chain → VisionSlot`）：塔要复用链的权重容器与运行时，所以只能在
        # `load_weights()` 之后建。`None` 表示这条装配不做图（`--vision off` 或没有塔）。
        self._load_tower = load_tower
        self._vision: object | None = None
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
        if self._load_tower is not None:
            self._vision = self._load_tower(self._chain)
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
        self._release_vision()
        if self._chain is not None:
            self._chain.close()
            self._chain = None
        self._warm_ok = False

    # ------------------------------------------------------------------ 前向
    def answers_for(
        self, request: DecisionRequest, mode: MaskMode, *, media: object | None = None
    ) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 读出耗时)。

        `media` 是给门与替身用的旁路；**请求自带的图**（`request.media`）走真路径（解码 →
        像素管线 → 塔 → `Media`），两条路在 `_answers` 汇合。
        """
        if media is None and request.media:
            return self._answers_with_images(request, mode)
        return self._answers(request, mode, media)

    def _answers(
        self, request: DecisionRequest, mode: MaskMode, media: object | None
    ) -> tuple[tuple[Answer, ...], int, float]:
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
        images = 0 if media is None else int(media.images)
        if images and self._image_token is None:
            raise UnsupportedMediaError("这个模型没有视觉塔，收不了图", images=images)
        for index, question in enumerate(request.questions):
            prepared = self._serve.prepare(request.state, question_spec(question),
                                           images=images)
            ids, placement = prepared.ids, None
            if media is not None:
                # 与 mude 同一条路：厂商给的 prompt 里一张图一个占位符，展开与定位都在合流里。
                ids, placement = merge.attach(prepared.ids, media,
                                              image_token_id=self._image_token)
            if len(ids) > self._max_len:
                raise RequestTooLongError(
                    "prompt 超过链的缓冲宽度",
                    length=len(ids),
                    max_len=self._max_len,
                    question=index,
                )
            answers.append(self._answer(chain, prepared, question, causal, index,
                                        ids=ids, mm=placement))
            tokens += len(ids)
        return tuple(answers), tokens, time.perf_counter() - started

    def _answer(
        self,
        chain: PooledChain,
        prepared: Prepared,
        question: Question,
        causal: bool,
        index: int,
        *,
        ids: Sequence[int] | None = None,
        mm: object | None = None,
    ) -> Answer:
        # 读出只取最后一行的**最终归一化** hidden（厂商 `last_hidden_state` 就是它，见
        # `Qwen3_5TextModel.forward` 末尾的 `self.norm`），再点积标签那几行词表权重。
        # 链只读回这一行：整段读回 + 归一在 16k 上是 87 ms，末行 0.02 ms，而这里只需要末行。
        logits = self._rows(prepared.label_ids) @ chain.last_hidden(
            prepared.ids if ids is None else ids, causal=causal, mm=mm)
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
    metal: bool = True,
    resident: int | None = None,
    vision: bool | None = None,
    vision_idle: float = 60.0,
    vision_bucket: int | None = None,
) -> PooledRunner:
    """装一条 h2o 的 pooled 链：厂商答面 + 词表行按需取 + ANE 链。

    `bucket` 是 ANE program 的静态列宽、`width` 是缓冲宽度（默认都取 `S_NATIVE`）。h2o 的
    `ffn` 是 9216、`hidden` 是 2560，两个 program 都比 mude 大，所以这两项由调用方说清楚，
    不在这里猜。

    `metal=True` 默认开：h2o 的 `(2560, 9216)` 已经做过 numpy vs Metal 的逐位对拍
    （票据 15 §24：桶 256、64/128 token、两种读法，hidden relL2 1.3e-3、logits 3e-4 量级、
    argmax 全同），注意力与 GDN 走 GPU 快 3~10 倍。

    `resident` 是待命常驻的层数（`None` = 32 层全装）。h2o 的 32 层 blob 合计 ≈3.33 GiB，
    是这条链最大的一笔常驻；给 4 就是"一个块"，换窗代价见票据 15 §25。它只在 `metal=True`
    时才有意义（窗口是流水线的机制），`metal=False` 传它会退化成"全装"。

    `vision` 是 `None`（自动）/ `False`（`--vision off`，纯文本服务：不建塔、不读塔权重）。
    h2o 的塔与 mude **同一套几何**，只差 merger 的输出宽度（2560 vs 2048），所以这里不需要
    第二份实现；`vision_idle` 是塔的空闲卸载秒数（`0` = 每请求后即卸）。
    """
    from ...devices.apple.chain import SplitChain
    from ...devices.apple.compile import S_NATIVE
    from ...devices.apple.platform import warn_if_unmeasured
    from ...models.families import h2o as h2o_family
    from ...weights.checkpoint import open_checkpoint

    # 形状检查放在开答面与检查点之前：它只读 config，却要在任何"看起来像结果"的运行之前
    # 先出声。告警不拦（`platform.py` 顶部写明理由）—— ANE 对形状很宽容，未测几何也会
    # 编译、会跑出数字，所以日志里必须有这条记录。
    weights = h2o_family.weight_plan(Path(model_dir))
    warn_if_unmeasured(weights.descriptor, strategy="pooled")
    serve = Serve.open(model_dir, tokenizer)
    store = open_checkpoint(Path(model_dir))
    embed_name = f"{h2o_family.PREFIX}.embed_tokens.weight"
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
            metal=metal,
            scratch_dir=scratch_dir,
            # numpy 那条路没有流水线，也就没有块边界可换窗：`resident` 只对 Metal 有意义。
            resident=resident if metal else None,
        )

    def rows(label_ids: Sequence[int]) -> np.ndarray:
        return np.asarray(store.read_rows(embed_name, list(label_ids)), np.float32)

    # 塔自己的列宽（`None` = 跟随链的桶）；4 的倍数是 merger 的 2×2 合并要求的。
    tower_s = bucket_width if vision_bucket is None else int(vision_bucket)
    if tower_s % 4:
        raise ConfigError("视觉塔的桶宽必须是 4 的倍数（merger 的 2×2 合并）", vision_bucket=tower_s)

    def load_tower(chain: object) -> object:
        from ...devices.apple.vision import VisionSlot, VisionTower

        return VisionSlot(
            lambda: VisionTower(store=chain.store, descriptor=weights.descriptor,  # type: ignore[attr-defined]
                                s=tower_s, runtime=chain.runtime,                # type: ignore[attr-defined]
                                metal=chain.metal_runtime,                       # type: ignore[attr-defined]
                                artifacts_dir=scratch_dir),
            spec=weights.descriptor.vision,
            idle=vision_idle,
        )

    load = load_tower if (vision is not False and weights.descriptor.vision is not None) else None
    return PooledRunner(load_chain, serve, rows, max_len=buffer_width,
                        image_token_id=weights.descriptor.image_token_id, load_tower=load)


__all__ = ["PooledRunner", "h2o_runner"]
