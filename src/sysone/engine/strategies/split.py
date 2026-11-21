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
from .decision import DecisionRunner, Encoded
from .vision import VisionRunner
from ...errors import (
    ConfigError,
    NotReadyError,
    ReadoutError,
)

# 块对角的一条序列至少占一槽，而槽宽是注意力 query 块（`devices/apple/pipeline.py` 的
# `AT_BM` = 128）的整数倍 —— 所以一趟缓冲最多并排 `width // 128` 条序列。那个数不 import：
# 这个模块在没有 Metal 的环境里也要能 import（设备层的 import 都在 `mude_runner` 里）。
_MIN_SLOT = 128

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

    **它不再自己编排**：一题一次整链前向还是"前缀一次 + 每条尾巴一次"、多条请求能不能挤进
    一个 pass，都由基类（`DecisionRunner`）按前缀、几何与能力位决定，这里只提供五个原语 ——
    `encode`（复用 `MudePrompter`）、`forward`、`forward_shared`、`forward_batched`、
    `answer`（码表读出）。

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

    @property
    def batch_limit(self) -> int:
        """一个 pass 最多装几条请求：缓冲宽度 ÷ 最短槽宽（`_MIN_SLOT`）。

        这是**形状**上限 —— `run_batched` 一条序列至少占一槽、槽宽按 128 向上取整，所以条数
        不会超过 `width // 128`；它不保证这一批真能合上（每条还要 ≤ 桶宽、打包区要装得下），
        那由 `forward_batched` 判。两条原语都没有时给 1：没有可合的东西，别让引擎去走批这条路。
        """
        if not (self.supports_shared or self.supports_batch):
            return 1
        return max(1, self._max_len // _MIN_SLOT)

    def __init__(
        self,
        load_chain: Callable[[], ModelChain],
        prompter: Prompter,
        *,
        max_len: int,
        image_token_id: int | None = None,
        load_tower: Callable[[object], object] | None = None,
        vision_state: Callable[[object], dict[str, object]] | None = None,
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
        self._vision_state = vision_state
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
        # 视觉产物与链的产物同一条启动路径：首启补齐，全命中时不动塔（见 `prepare_vision`）。
        self.prepare_vision()
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

    def forward_batched(self, encoded: Sequence[Encoded], *, causal: bool = False,
                        use_prefix: bool = True) -> Sequence[Sequence[float]] | None:
        """N 条独立序列并排一个 pass：链给出每题一份候选概率；装不下就 `None`。

        块对角只隔离**序列之间**的可见性，一条序列内部的前缀语义不变：每条序列自带它的
        `prefix_noncausal` 前缀长度（`None` 只出现在量不出边界的链上，按 0 传）。

        **形状由链自己判**（每条 ≤ 桶宽、槽宽整除桶宽、打包区 ≤ 缓冲宽度）：`None` 就是
        "这批没有一次前向的解法"，编排拿到它退回逐题。策略层不抄第二份预判 —— 抄出来的那份
        曾经漏掉"槽宽整除桶宽"，于是它判能批、链拒，一次本该静默回退的请求变成 500。

        `use_prefix=False`（`noncausal_full`）时前缀长度传 0：那条掩码没有前缀行，带进去会
        让批里的序列按另一种掩码收行。
        """
        batch = getattr(self._require_chain(), "forward_batched", None)
        if not callable(batch):
            return None
        return batch(
            [item.ids for item in encoded],
            options=[len(item.labels) for item in encoded],
            prefixes=[0 if not use_prefix or item.prefix_len is None else item.prefix_len
                      for item in encoded],
            causal=causal,
        )

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

    def tower_state(chain: object) -> dict[str, object]:
        from ...devices.apple import vision as vision_device

        state = vision_device.artifact_state(
            chain.store, descriptor=weights.descriptor, s=tower_s, artifacts_dir=scratch_dir,
        )
        return {"directory": str(state.directory), "reusable": state.hits, "total": state.total}

    load = load_tower if (vision is not False and weights.descriptor.vision is not None) else None
    return SplitRunner(load_chain, prompter, max_len=buffer_width,
                       image_token_id=weights.descriptor.image_token_id, load_tower=load,
                       vision_state=tower_state if load is not None else None)


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
