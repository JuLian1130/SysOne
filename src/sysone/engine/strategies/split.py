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
from .decision import DecisionRunner, Encoded, shared_prefix
from .vision import VisionRunner
from ...errors import (
    ConfigError,
    NotReadyError,
    ReadoutError,
    UnsupportedModeError,
)

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


class SplitRunner(VisionRunner, DecisionRunner):
    """实现 `ModelRunner` 的 split 路径：投影/FFN 走 ANE program、注意力/GDN 走 Metal。

    **它不再自己编排**：一题一次整链前向还是"前缀一次 + 每条尾巴一次"由基类
    （`DecisionRunner._answers`）按前缀与能力位决定，这里只提供四个原语 —— `encode`
    （复用 `MudePrompter`）、`forward`、`forward_shared`、`answer`（码表读出）。

    `load_chain` 延迟到 `load_weights()` 才调用，所以构造它不碰 ANE；`max_len` 是链的
    缓冲宽度（prompt 超过它就是 422，而不是让链里抛裸 `ValueError`）。

    `load_tower` 是视觉塔的工厂（`chain → VisionSlot`）：塔要复用链已经打开的权重容器与
    运行时（第二个 `AneRuntime` 能建、但第二个 `MetalRuntime` 会按另一套几何编译失败），
    所以它必须等到 `load_weights()` 之后才能调。`None` 表示这条装配不做图（`--vision off`，
    或者模型本身没有塔）。
    """

    # 能力按**这条链上有没有那条原语**问，而不是无条件声明：`ModelChain` 的协议只要求
    # `forward` / `close`，替身链与将来的 numpy 路径都可能没有 `forward_shared`。声明了却调不到
    # 是一条 500，退回串行只是一次没有收益的前向。
    # 声明也不等于每次都走：前缀不同、单题、`noncausal_full` 都走逐题（见基类的 `_answers`）。

    @property
    def supports_shared(self) -> bool:
        return callable(getattr(self._chain, "forward_shared", None))

    @property
    def supports_batch(self) -> bool:
        return callable(getattr(self._chain, "forward_batched", None))

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

    # ------------------------------------------------------------------ 原语

    def encode(self, state: str, question: Question, *, images: int = 0) -> Encoded:
        """题面 → 整条 prompt / 前缀边界 / 读出标签与位置（复用家族的 `MudePrompter`）。

        `images` 是这道题里的图数：一张图在题面里仍只占一个占位符，展开在基类
        （`DecisionRunner.encode_request`）里做，因为"一图多占位"是协议层的事实，不是模型特征。
        """
        sequence, prefix_len, labels, indexes = self._prompter.encode(
            state, question, images=images)
        return Encoded(ids=sequence, prefix_len=prefix_len, labels=labels, indexes=indexes)

    def forward(self, encoded: Encoded, *, prefix: int, causal: bool = False,
                mm: object | None = None) -> Sequence[float]:
        """单题一整趟：链给出该题的候选概率。"""
        return self._require_chain().forward(
            encoded.ids, options=len(encoded.labels), prefix=prefix, causal=causal, mm=mm)

    def forward_shared(self, prefix: Sequence[int], tails: Sequence[Sequence[int]],
                       *, options: Sequence[int], causal: bool = False,
                       mm: object | None = None) -> Sequence[Sequence[float]]:
        """前缀 + 每条尾巴一次前向：链给出每题一份候选概率。"""
        chain = self._require_chain()
        return chain.forward_shared(  # type: ignore[attr-defined]
            prefix, tails, options=options, causal=causal, mm=mm)

    def answer(self, encoded: Encoded, raw: object, *, index: int, kind: str) -> Answer:
        """概率 → `Answer`：按标签序号取候选，最大的那一项胜出（码表读出的形状）。"""
        shares = self._shares(raw)
        values = tuple(shares[position] for position in encoded.indexes)
        best = max(range(len(values)), key=values.__getitem__)
        return Answer(
            question_index=index,
            kind=kind,
            value=encoded.labels[best],
            probs=values,
            labels=encoded.labels,
            confidence=values[best],
        )

    def answers_batch(
        self, requests: Sequence[DecisionRequest], mode: MaskMode
    ) -> list[tuple[tuple[Answer, ...], int, float]]:
        """多条**互不相干**的请求走一个 pass，返回每条各自的 `(答案, token 数, 前向秒)`。

        合成一个 pass 有两条路：**前缀相同**（决策负载里就是"同一个 state、不同的问题"）走
        `forward_shared` 批尾巴；**前缀不同**走块对角（一条序列一个槽，每行只看自己那一槽）。
        两条都拿不到才退回逐题整趟（基类的 `serial_answers`），结果与逐条调用**逐位相同** ——
        这只是没有收益，不是另一种答案。

        收益按**瓦片数**算，不按列：program 的静态宽度就是桶宽，一格（一片瓦片）里放 118 列
        还是 512 列，这一趟的成本几乎一样。所以 N 条短序列全挤进**一格**时倍率 ≈ N（票据 07
        §7.25o，`bucket=512`：4 × 120 token 2.23× → **3.94×**、2 条 1.79×、3 条 2.52×）；
        溢出到第二格就掉下来（4 × 100–130 token 的槽宽是 256 ⇒ 2 格 ⇒ 2.64×）；瓦片数完全
        不降时只剩每趟固定项可省（§7.25j 的宽度标定臂：2 条 1.53×、4 条 2.01×）。上限还受
        两张表约束：`nkw` / `pcw` 只有 8 槽，槽宽按 128 向上取整、由**最长**那条决定。

        三条约束，调用方（`Engine.batch`）负责满足：**都没有图**（图的塔与像素管线按请求走，
        塞进批里会把调度搅乱）、**同一个 mode**（掩码不同就打不到一起）、**非空**。
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
        # 编码、长度检查与"一图多占位"的展开都在基类（批里没有图，所以落点一定是空的）。
        encoded: list[Encoded] = []
        kinds: list[str] = []
        # 每条请求的题在扁平列表里的区间，用来把答案切回去。
        spans: list[tuple[int, int]] = []
        for request in requests:
            first = len(encoded)
            own, _ = self.encode_request(request, None)
            encoded.extend(own)
            kinds.extend(question.kind for question in request.questions)
            spans.append((first, len(encoded)))
        shared = shared_prefix(encoded) if use_prefix else None
        forward_batched = getattr(chain, "forward_batched", None)
        if shared is not None and len(encoded) > 1 and self.supports_shared:
            self.decisions.batch_hit += 1
            answers = self.shared_answers(shared, encoded, kinds, causal=causal)
        elif len(encoded) > 1 and callable(forward_batched) and _batchable(encoded, chain):
            # 没有共享前缀（不同 state 的独立请求）时走块对角：每条序列一个瓦片、互不可见。
            self.decisions.batch_hit += 1
            answers = _answer_independent(forward_batched, encoded, kinds, causal)
        else:
            # 一条序列（引擎把单条也送进这条路）没有可合的东西；≥2 条才是有机会却没合上。
            if len(encoded) > 1:
                self.decisions.batch_fallback += 1
            else:
                self.decisions.batch_miss += 1
            answers = self.serial_answers(encoded, kinds, use_prefix=use_prefix, causal=causal)
        elapsed = time.perf_counter() - started
        out: list[tuple[tuple[Answer, ...], int, float]] = []
        for request, (first, last) in zip(requests, spans):
            own = tuple(answers[first:last])
            tokens = sum(len(encoded[index].ids) for index in range(first, last))
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

    def _require_ready(self) -> None:
        """基类在编码之前调它：没装权重就当场 `ConfigError`，别先记一笔不会发生的 miss。"""
        self._require_chain()

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


def _batchable(
    encoded: Sequence[Encoded],
    chain: object,
) -> bool:
    """这批能不能走块对角合批：每条序列 ≤ 桶宽、总列数 ≤ 缓冲宽度。

    判据与 `Pipeline.run_batched` 的入参约束一致；这里先判一次是为了**静默退回逐条**而不是
    让一次超长的合批在流水线里抛异常。

    退回现在记在 `DecisionMetrics.batch_fallback` 里：以前它没有任何计数，只能从墙时看出来。
    这不是小毛病 —— 生产默认 `bucket = 1024`，而票据 07 §7.25p 量到的真实决策 prompt 是
    1–2k，超过 1024 的那一半一律退成逐条，"批处理已开"在它们身上等于没开（实测 1.02×）。
    """
    bucket = int(getattr(chain, "bucket", 0) or 0)
    width = int(getattr(chain, "width", 0) or 0)
    if bucket <= 0 or width <= 0:
        return False
    lengths = [len(item.ids) for item in encoded]
    if max(lengths) > bucket:
        return False
    return (len(lengths) - 1) * bucket + max(lengths) <= width


def _answer_independent(
    forward_batched: Callable[..., Sequence[object]],
    encoded: Sequence[Encoded],
    kinds: Sequence[str],
    causal: bool = False,
) -> list[Answer]:
    """块对角合批：N 条**独立**序列并排跑一个 pass（各段互不可见、各段状态从零起）。"""
    sequences = [item.ids for item in encoded]
    options = [len(item.labels) for item in encoded]
    # 每条序列自带它自己的 `prefix_noncausal` 前缀长度：块对角只隔离序列之间的可见性，
    # **不改变一条序列内部**的前缀语义（前缀行只看前缀、尾部行看整段）。`None` 只出现在量不出
    # 边界的链上（它们只开因果共享，那时这个长度不决定掩码），按 0 处理。
    prefixes = [0 if item.prefix_len is None else item.prefix_len for item in encoded]
    raw = forward_batched(sequences, options=options, prefixes=prefixes, causal=causal)
    answers: list[Answer] = []
    for index, item in enumerate(encoded):
        probs = SplitRunner._shares(raw[index])
        values = tuple(probs[position] for position in item.indexes)
        best = max(range(len(values)), key=values.__getitem__)
        answers.append(
            Answer(
                question_index=index,
                kind=kinds[index],
                value=item.labels[best],
                probs=values,
                labels=item.labels,
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
