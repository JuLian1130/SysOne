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
from .vision import VisionRunner
from ...errors import (
    ConfigError,
    NotReadyError,
    ReadoutError,
    RequestTooLongError,
    UnsupportedMediaError,
    UnsupportedModeError,
)
from ...models.vision import merge

# 三种模式这条链都能表达：双向（`prefix_noncausal` 的前缀窗 + `noncausal_full` 的全可见）
# 与三角（`causal`，M3 的核改动）。声明在这里而不在 descriptor 里：descriptor 说模型能跑
# 什么，这里说这条实现做到了什么。
_SUPPORTED_MODES = (MaskMode.PREFIX_NONCAUSAL, MaskMode.NONCAUSAL_FULL, MaskMode.CAUSAL)

# 预热用的题：真正的冷启动失败（program 没装上、面没绑好）在第一题就会炸，
# 这一题让它在 `Engine.start()` 里就炸，而不是让第一道真实请求撞上。
_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = Question(text="Which option is correct?", options=("This one", "That one"))


class ModelChain(Protocol):
    """`SplitChain` 只需要这几件事：一次整链前向、卸载，以及装载耗时拆分。"""

    def forward(self, tokens: Sequence[int], *, options: int, prefix: int = 0,
                causal: bool = False, mm: object | None = None) -> object: ...

    def close(self) -> None: ...

    @property
    def load_s(self) -> float:
        """装载耗时（秒），不含产物编译。"""

    @property
    def compile_s(self) -> float:
        """产物编译耗时（秒）；产物全部命中缓存时是 0。"""


class Prompter(Protocol):
    """题面 → (整条 prompt 的 id, 前缀长度, 答案标签, 标签对应的选项序号)。

    `images` 是这道题题面里的图数；给出的 id 里一张图仍然只占**一个**占位符。
    """

    def encode(
        self, state: str, question: Question, *, images: int = 0
    ) -> tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]: ...


class SplitRunner(VisionRunner):
    """实现 `ModelRunner` 的 split 路径：一题一次整链前向。

    `load_chain` 延迟到 `load_weights()` 才调用，所以构造它不碰 ANE；`max_len` 是链的
    缓冲宽度（prompt 超过它就是 422，而不是让链里抛裸 `ValueError`）。

    `load_tower` 是视觉塔的工厂（`chain → VisionSlot`）：塔要复用链已经打开的权重容器与
    运行时（第二个 `AneRuntime` 能建、但第二个 `MetalRuntime` 会按另一套几何编译失败），
    所以它必须等到 `load_weights()` 之后才能调。`None` 表示这条装配不做图（`--vision off`，
    或者模型本身没有塔）。
    """

    def __init__(
        self,
        load_chain: Callable[[], ModelChain],
        prompter: Prompter,
        *,
        max_len: int,
        image_token_id: int | None = None,
        load_tower: Callable[[object], object] | None = None,
    ) -> None:
        if max_len < 1:
            raise ConfigError("max_len 必须为正", max_len=max_len)
        self._load_chain = load_chain
        self._prompter = prompter
        self._max_len = int(max_len)
        # 占位 token 的 id 来自 descriptor（`image_token_id`）。没有任何模型的 `validate()`
        # 允许"带塔却不说 id"，所以这里的 `None` 只出现在纯文本装配里。
        self._image_token = image_token_id
        self._load_tower = load_tower
        self._vision: object | None = None
        self._chain: ModelChain | None = None
        self._warm_ok = False

    # ------------------------------------------------------------------ 生命周期

    def init_device(self) -> None:
        """ANE runtime 与 Metal 流水线由链的工厂自己建，这里没有要先做的事。"""
        return None

    def load_weights(self) -> None:
        self._chain = self._load_chain()
        if self._load_tower is not None:
            self._vision = self._load_tower(self._chain)

    def artifact_bytes(self) -> int:
        """编译产物落在链自己的 scratch 目录；这里不计（产物预算默认不设上限）。"""
        return 0

    def compile_or_warm_up(self) -> CompilationTimes:
        if self._chain is None:
            raise ConfigError("split 策略尚未装载权重", strategy="split")
        started = time.perf_counter()
        self._warm()
        # 装载与编译的拆分由链自己量（只有它知道哪几层是从头构建的）；预热是这里量的。
        return CompilationTimes(
            load_s=self._chain.load_s,
            compile_s=self._chain.compile_s,
            warmup_s=time.perf_counter() - started,
        )

    def check_health(self) -> bool:
        return self._warm_ok

    def execute(self, plan: object, session: object) -> object:
        raise ConfigError("split 策略按请求整批执行，不走单题 execute")

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
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 前向耗时)。

        `media` 是给门与替身用的旁路：给了就原样用。**请求自带的图**（`request.media`）在这里
        走真路径 —— 解码、像素管线、塔、`Media` —— 然后与 `media=` 那条路汇合。两条都在
        同一个 `finally` 里释放塔（`idle = 0` 时"每请求后即卸"就是靠它）。
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
        images = 0 if media is None else int(media.images)
        if images and self._image_token is None:
            # 图块没有 token id 的模型（纯文本装配）收到图：这是**请求**的问题，不是我们的配置
            # 坏了 —— 客户端据此知道不要重试。前门会先用 `image_contract()` 拒一次，这里是最后
            # 一道，不能让 `image_token_id=None` 变成一个诡异的"占位符数不对"。
            raise UnsupportedMediaError("这个模型没有视觉塔，收不了图", images=images)
        # 先把所有题编码完：共享前缀要先知道每题的尾巴，长度检查也按整条 prompt 做。
        encoded: list[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]] = []
        for index, question in enumerate(request.questions):
            encoded.append(self._prompter.encode(request.state, question, images=images))
            if len(encoded[-1][0]) > self._max_len:
                raise RequestTooLongError(
                    "prompt 超过链的缓冲宽度",
                    length=len(encoded[-1][0]),
                    max_len=self._max_len,
                    question=index,
                )
        # 展开在**编码之后**：一张图一个占位符 → `counts[i]` 个，图块的起点要等前面那些图
        # 各自展开了多少才知道（所以是 span，不是一个偏移）。展开之后长度才是真的，所以
        # 越界检查放在展开之后。
        placements: list[object] = []
        if media is not None:
            placed: list[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]] = []
            for index, (sequence, prefix_len, labels, indexes) in enumerate(encoded):
                expanded, placement = merge.attach(sequence, media,
                                                   image_token_id=self._image_token)
                if len(expanded) > self._max_len:
                    raise RequestTooLongError(
                        "展开图像占位符之后超过链的缓冲宽度",
                        length=len(expanded), max_len=self._max_len, question=index)
                # 前缀里也有图块，所以前缀长度要跟着展开（`_prefix_ids` 量的是未展开的那串）。
                prefix_ids, _ = merge.expand_image_tokens(
                    sequence[:prefix_len], media.counts, image_token_id=self._image_token)
                placed.append((tuple(expanded), len(prefix_ids), labels, indexes))
                placements.append(placement)
            encoded = placed
        shared = _shared_prefix(encoded) if use_prefix else None
        forward_shared = getattr(chain, "forward_shared", None)
        mm = placements[0] if placements else None
        if shared is not None and len(encoded) > 1 and callable(forward_shared):
            answers = _answer_shared(forward_shared, shared, encoded,
                                     [item.kind for item in request.questions], causal, mm=mm)
        else:
            answers = _answer_serial(chain, encoded, use_prefix,
                                     [item.kind for item in request.questions], causal, mm=mm)
        tokens = sum(len(sequence) for sequence, *_ in encoded)
        return tuple(answers), tokens, time.perf_counter() - started

    def answers_batch(
        self, requests: Sequence[DecisionRequest], mode: MaskMode
    ) -> list[tuple[tuple[Answer, ...], int, float]]:
        """多条**互不相干**的请求走一个 pass，返回每条各自的 `(答案, token 数, 前向秒)`。

        两条请求合成一个 pass 的前提是它们**共用同一条前缀** —— 在决策负载里这就是"同一个
        state、不同的问题"，`split` 本来就在一次请求内用 `forward_shared` 批尾巴，这里只是把
        批的原料从"一条请求的多道题"扩到"多条请求的所有题"。实测收益（票据 07 §7.25j，
        `bucket=256`）：2 条 1.53×、4 条 2.01×、1k 的请求 1.31×。

        三条约束，调用方（`Engine.batch`）负责满足：**都没有图**（图的塔与像素管线按请求走，
        塞进批里会把调度搅乱）、**同一个 mode**（掩码不同就打不到一起）、**非空**。
        共用前缀拿不到时退回逐题整趟（`_answer_serial`），结果与逐条调用**逐位相同** ——
        这只是没有收益，不是另一种答案。
        """
        if not requests:
            return []
        chain = self._require_chain()
        if mode not in _SUPPORTED_MODES:
            raise UnsupportedModeError(
                "mude 链没有这个掩码",
                mode=mode.value,
                strategy="split",
                supported=sorted(item.value for item in _SUPPORTED_MODES),
            )
        if any(request.media for request in requests):
            raise ConfigError("批里不能有带图的请求（图片按请求走塔）",
                              images=sum(1 for r in requests if r.media))
        causal = mode is MaskMode.CAUSAL
        use_prefix = mode is not MaskMode.NONCAUSAL_FULL
        started = time.perf_counter()
        encoded: list[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]] = []
        kinds: list[str] = []
        # 每条请求的题在扁平列表里的区间，用来把答案切回去。
        spans: list[tuple[int, int]] = []
        for request in requests:
            first = len(encoded)
            for index, question in enumerate(request.questions):
                encoded.append(self._prompter.encode(request.state, question, images=0))
                if len(encoded[-1][0]) > self._max_len:
                    raise RequestTooLongError(
                        "prompt 超过链的缓冲宽度",
                        length=len(encoded[-1][0]),
                        max_len=self._max_len,
                        question=index,
                    )
                kinds.append(question.kind)
            spans.append((first, len(encoded)))
        shared = _shared_prefix(encoded) if use_prefix else None
        forward_shared = getattr(chain, "forward_shared", None)
        forward_batched = getattr(chain, "forward_batched", None)
        if shared is not None and len(encoded) > 1 and callable(forward_shared):
            answers = _answer_shared(forward_shared, shared, encoded, kinds, causal)
        elif len(encoded) > 1 and callable(forward_batched) and _batchable(encoded, chain):
            # 没有共享前缀（不同 state 的独立请求）时走块对角：每条序列一个瓦片、互不可见。
            answers = _answer_independent(forward_batched, encoded, kinds, causal)
        else:
            answers = _answer_serial(chain, encoded, use_prefix, kinds, causal)
        elapsed = time.perf_counter() - started
        out: list[tuple[tuple[Answer, ...], int, float]] = []
        for request, (first, last) in zip(requests, spans):
            own = tuple(answers[first:last])
            tokens = sum(len(encoded[index][0]) for index in range(first, last))
            out.append((own, tokens, elapsed))
        return out

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
    kinds: Sequence[str],
    causal: bool = False,
    mm: object | None = None,
) -> list[Answer]:
    """共享前缀的一条路：`prefix` + 每题的尾巴，一次调用拿到所有题的概率。"""
    tails = [sequence[len(prefix):] for sequence, *_ in encoded]
    options = [len(labels) for _, _, labels, _ in encoded]
    raw = forward_shared(prefix, tails, options=options, causal=causal, mm=mm)
    answers: list[Answer] = []
    for index, (_, _, labels, indexes) in enumerate(encoded):
        probs = SplitRunner._shares(raw[index])
        values = tuple(probs[item] for item in indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=kinds[index],
                value=labels[best],
                probs=values,
                labels=labels,
                confidence=values[best],
            )
        )
    return answers


def _batchable(
    encoded: Sequence[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]],
    chain: object,
) -> bool:
    """这批能不能走块对角合批：每条序列 ≤ 桶宽、总列数 ≤ 缓冲宽度。

    判据与 `Pipeline.run_batched` 的入参约束一致；这里先判一次是为了**静默退回逐条**而不是
    让一次超长的合批在流水线里抛异常。

    **这个退回没有计数器**，`metrics()` 里也读不到，只能从墙时看出来。这不是小毛病：生产默认
    `bucket = 1024`，而票据 07 §7.25p 量到的真实决策 prompt 是 1–2k —— 超过 1024 的那一半一律
    退成逐条，"批处理已开"在它们身上等于没开（实测 1.02×）。
    """
    bucket = int(getattr(chain, "bucket", 0) or 0)
    width = int(getattr(chain, "width", 0) or 0)
    if bucket <= 0 or width <= 0:
        return False
    lengths = [len(sequence) for sequence, *_ in encoded]
    if max(lengths) > bucket:
        return False
    return (len(lengths) - 1) * bucket + max(lengths) <= width


def _answer_independent(
    forward_batched: Callable[..., Sequence[object]],
    encoded: Sequence[tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...]]],
    kinds: Sequence[str],
    causal: bool = False,
) -> list[Answer]:
    """块对角合批：N 条**独立**序列并排跑一个 pass（各段互不可见、各段状态从零起）。"""
    sequences = [sequence for sequence, *_ in encoded]
    options = [len(labels) for _, _, labels, _ in encoded]
    # 每条序列自带它自己的 `prefix_noncausal` 前缀长度：块对角只隔离序列之间的可见性，
    # **不改变一条序列内部**的前缀语义（前缀行只看前缀、尾部行看整段）。
    prefixes = [prefix_len for _, prefix_len, _, _ in encoded]
    raw = forward_batched(sequences, options=options, prefixes=prefixes, causal=causal)
    answers: list[Answer] = []
    for index, (_, _, labels, indexes) in enumerate(encoded):
        probs = SplitRunner._shares(raw[index])
        values = tuple(probs[item] for item in indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=kinds[index],
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
    kinds: Sequence[str],
    causal: bool = False,
    mm: object | None = None,
) -> list[Answer]:
    """逐题整趟（今天的行为；单题与不可复用时走这条）。"""
    answers: list[Answer] = []
    for index, (sequence, prefix_len, labels, indexes) in enumerate(encoded):
        probs = SplitRunner._shares(
            chain.forward(sequence, options=len(labels),
                          prefix=prefix_len if use_prefix else 0, causal=causal, mm=mm))
        values = tuple(probs[item] for item in indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=kinds[index],
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
    vision: bool | None = None,
    vision_idle: float = 60.0,
    vision_bucket: int | None = None,
) -> SplitRunner:
    """装一条 mude split 链：分词器 + 回答码 + ANE/Metal 链。

    `bucket` 是 ANE program 的静态列宽（默认 `S_NATIVE`），`width` 是 GPU 侧缓冲的列数
    （默认等于 bucket，prompt 更长时必须给更大的值）。两者都决定内存，所以由调用方说。

    `resident` 是待命常驻的层数，`None` 表示全装；窗口的具体病理见
    `devices/apple/residency.py`（它是这张卡上最大的一笔内存，所以由调用方决定让不让）。

    `vision` 是 `None`（自动：有塔且图能解就收图）/ `False`（`--vision off`：纯文本服务，
    不建塔、不读塔权重）。`vision_idle` 是塔的空闲卸载秒数（`0` = 每请求后即卸）。**都不预热**：
    建塔发生在第一次带图请求（`VisionSlot`）。
    """
    from ...devices.apple.chain import SplitChain
    from ...devices.apple.compile import S_NATIVE
    from ...devices.apple.platform import warn_if_unmeasured
    from ...models.families import mude as mude_family

    # 形状检查放在分词器与检查点之前：它只读 config，却要在任何"看起来像结果"的运行之前
    # 先出声。告警不拦（`platform.py` 顶部写明理由）。
    weights = mude_family.weight_plan(Path(model_dir))
    warn_if_unmeasured(weights.descriptor, strategy="split")
    codes = mude_family.read_codes(model_dir)
    prompter = mude_family.MudePrompter(_load_tokenizer(model_dir), codes)
    bucket_width = int(S_NATIVE if bucket is None else bucket)
    buffer_width = int(bucket_width if width is None else width)

    def load_chain() -> ModelChain:
        return SplitChain.load(
            model_dir,
            # 形状 / 张量名前缀 / norm 的存法偏移 / RoPE 角基都在这份计划里（家族给的纯数据），
            # 设备层不认识任何具体 checkpoint。
            weights=weights,
            s=bucket_width,
            width=buffer_width,
            quantize=quantize,
            metal=metal,
            scratch_dir=scratch_dir,
            resident=resident,
        )

    # 塔自己的列宽（`None` = 跟随链的桶）。它决定"每层几个瓦片"：桶越窄瓦片越多、交错越好，
    # 但 ANE 的每 token 成本上升。实测（1024²，票据 07 §7.17）：256/512/1024/2048/4096 =
    # 959/902/**781**/794/849 ms —— 1024 是拐点，所以默认跟随链（`S_NATIVE` = 1024）。
    tower_s = bucket_width if vision_bucket is None else int(vision_bucket)
    if tower_s % 4:
        raise ConfigError("视觉塔的桶宽必须是 4 的倍数（merger 的 2×2 合并）", vision_bucket=tower_s)

    def load_tower(chain: object) -> object:
        # 塔复用链已经打开的权重容器与两个运行时：`MetalRuntime` 是进程单例、按几何编译，
        # 塔再要一份的话，先建塔后建链（或反过来）会让第二次构造撞上"已经按另一套几何编译过"。
        from ...devices.apple.vision import VisionSlot, VisionTower

        slot = VisionSlot(
            lambda: VisionTower(store=chain.store, descriptor=weights.descriptor,  # type: ignore[attr-defined]
                                s=tower_s, runtime=chain.runtime,                # type: ignore[attr-defined]
                                metal=chain.metal_runtime,                       # type: ignore[attr-defined]
                                artifacts_dir=scratch_dir),
            spec=weights.descriptor.vision,
            idle=vision_idle,
        )
        return slot

    load = load_tower if (vision is not False and weights.descriptor.vision is not None) else None
    return SplitRunner(load_chain, prompter, max_len=buffer_width,
                       image_token_id=weights.descriptor.image_token_id, load_tower=load)


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
