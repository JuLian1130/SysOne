"""pooled 策略：整链 hidden 走我们的 ANE 链，读出与答面格式交回厂商的答面。

两家居于这条路上：h2o（tied，读出取 `embed_tokens` 的标签行）与 torchcast（untied，读出取
`lm_head` 的字母行）。差别全在"读出张量叫什么、题面怎么渲染、答案怎么判"，由家族的
`weight_plan` 与 `PooledRecipe` 给；这条策略只认 `AnswerSurface` 那两件事。

与 split 策略的差别不在"谁做前向"，而在**读出形态**：

- split / mude 是固定的一张码表，`forward(options=n)` 就够表达"读前 n 行"；
- pooled / h2o 的读出是"题面在回答槽上落成哪个标签 token"，这些行随题变（标签能不能落在
  一个 token 上、落在哪个 id 上，取决于题面末尾那几个字符），而且按题型换温度、noul 有地板
  与阈值、score 要出期望值 —— 这些都是厂商 `format_answer` 的事。

所以这条路的边界是：我们算 `hidden`，**再算"哪些词表行 × hidden"**（这一步是设备层的
真相），把 logits 原样交给 `Serve.answer`，由厂商代码判答案。概率不会在中间被我们自己
归一化一次再交给它 —— 那会变成"我们的温度 + 厂商的温度"。

编排（单题整趟 / 一次请求内的共享前缀 / 跨请求合批）**不在这个模块里**：那是
`DecisionRunner` 的事（`docs/sysone/decision-runner.md`），这里只给原语。其中"前缀一遍、
每题一条尾巴"是这条策略唯一的真实收益 —— 真实评测流量 34 条请求全是 4–5 题共享同一个
state，理想共享比 3.8–3.9×。
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
from ...errors import ConfigError, NotReadyError, UnsupportedModeError
from ...models.layer_plan import WeightPlan
from ...models.recipe import PooledRecipe
from ...models.vendors.base import AnswerSurface, Prepared, question_spec
from .decision import MIN_SLOT, DecisionRunner, Encoded
from .vision import VisionRunner

# 读出与答面都归厂商；这里只保留"池化读出"这件事的共性。能读哪几种掩码由每份 checkpoint
# 的 descriptor 声明（`supported_modes`），策略不再写死一份 —— 写死会把下一家挡在门外。

_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = Question(text="Which option is correct?", options=("This one", "That one"))


class PooledChain(Protocol):
    """这条策略只需要链的两件事：末位 token 的最终 hidden 与卸载。

    另外两条是**可选的**：`last_hidden_shared`（"前缀一遍、每题一条尾巴"）与
    `last_hidden_batched`（"多条独立序列并排一个 pass"）。有就走，没有就退回逐题 ——
    `ModelChain` 的协议不强制它们，替身链与 numpy 形态的链都可能没有。

    `load_s` / `compile_s` 也是可选的：链自己量那两段（`SplitChain` 有），没有就按
    "装载即全部、编译 0"报。
    """

    def last_hidden(self, tokens: Sequence[int], *, prefix: int = 0, causal: bool = False,
                    mm: object | None = None) -> np.ndarray: ...

    def close(self) -> None: ...


class PooledRunner(VisionRunner, DecisionRunner):
    """实现 `ModelRunner` 的 pooled 路径：一题一次整链前向 + 一次标签行读出。

    `rows` 把"标签 token id"换成"读出张量里的那几行"（tied 模型上是 `embed_tokens`，解绑的是
    `lm_head`），由调用方注入：设备层不加一个只为这份 checkpoint 存在的读出方式，链也不需要
    认识 tokenizer。

    **它不自己编排**：一题整趟还是"state 前缀只算一遍"，由基类（`DecisionRunner`）按前缀、
    题数与能力位决定，这里只提供原语 —— `encode`（厂商答面渲染）、`forward`（末行 hidden）、
    `forward_shared`（前缀一遍 + 每条尾巴的末行）、`answer`（标签行 × hidden 交回厂商）。

    跨请求合批与 mude 走**同一套编排与同两条上限**：`batch_limit` 报形状上限（缓冲宽 ÷ 128），
    策略上限是服务层的 `--batch`（默认 4），两边取小。合不合得上仍由设备层判 —— 每条序列要
    ≤ 桶宽、槽宽要整除桶宽、打包区要装得下，装不下就静默退回逐条（`Pipeline._slot_layout`）。
    """

    @property
    def supports_batch(self) -> bool:
        """这条链能不能"多条独立序列一个 pass"：`last_hidden_batched` 与零拷贝流水线两样都要。

        与 `supports_shared` 同一个口径：`SplitChain` 上可能有这个名字，但 numpy 形态没有打包区
        的落脚点（`can_share_prefix` 为假）。声明了却调不到是一条 500；退回串行只是一次没有
        收益的前向。声明也不等于每次都走：装不下、带图、单条都按原样退回逐题。

        这一位只回答"有没有原语"；**开不开、一次装几条**由 `batch_limit`（形状上限）与服务层
        `--batch`（策略上限）共同决定，两边由 `Engine.batch_limit` 取小。
        """
        chain = self._chain
        return (callable(getattr(chain, "last_hidden_batched", None))
                and bool(getattr(chain, "can_share_prefix", False)))

    @property
    def batch_limit(self) -> int:
        """一个 pass 最多装几条请求：缓冲宽度 ÷ 最短槽宽（`MIN_SLOT`）。

        与 `SplitRunner` 同一个算法，是**形状**上限：`run_batched` 一条序列至少占一槽、槽宽按
        128 向上取整，所以条数不会超过 `width // 128`。它不保证这一批真能合上 —— 每条还要
        ≤ 桶宽、槽宽要整除桶宽，那由 `forward_batched` 判（判据不在这里抄第二份）。

        真正的**策略**上限是服务层的 `--batch`（`SchedulerConfig.batch`），两边由
        `Engine.batch_limit` 取小；`--batch 1` 就是关掉。链没有块对角原语时这里给 1，别让引擎
        白攒一趟必退回的批。
        """
        if not self.supports_batch:
            return 1
        return max(1, self._max_len // MIN_SLOT)

    def __init__(
        self,
        load_chain: Callable[[], PooledChain],
        serve: AnswerSurface,
        rows: Callable[[Sequence[int]], np.ndarray],
        *,
        supported_modes: frozenset[MaskMode],
        default_mode: MaskMode,
        max_len: int,
        image_token_id: int | None = None,
        load_tower: Callable[[object], object] | None = None,
        vision_state: Callable[[object], dict[str, object]] | None = None,
    ) -> None:
        if max_len < 1:
            raise ConfigError("max_len 必须为正", max_len=max_len)
        self._load_chain = load_chain
        self._serve = serve
        self._rows = rows
        # 训练过哪些读法是**这份 checkpoint** 的事实，由家族写进 descriptor（`parse_config`），
        # 装配层转交进来。策略不自己另写一份：写死的话，下一个 pooled 家族会被前一家的
        # 掩码集合挡住。
        self._modes = frozenset(supported_modes)
        if not self._modes:
            raise ConfigError("pooled 策略需要 checkpoint 声明支持的读法", strategy="pooled")
        # 预热要用**这份 checkpoint 的默认读法**：torchcast 只训练过全双向，用因果去热一块
        # 它不认的 program 会白编一次、也可能在掩码边界上读出另一种结果。
        self._default_mode = default_mode
        self._max_len = int(max_len)
        self._image_token = image_token_id
        # 视觉塔的工厂（`chain → VisionSlot`）：塔要复用链的权重容器与运行时，所以只能在
        # `load_weights()` 之后建。`None` 表示这条装配不做图（`--vision off` 或没有塔）。
        self._load_tower = load_tower
        self._vision_state = vision_state
        self._vision: object | None = None
        self._chain: PooledChain | None = None
        self._load_s = 0.0
        self._warm_ok = False

    # ------------------------------------------------------------------ 能力
    @property
    def supports_shared(self) -> bool:
        """这条链能不能"前缀一遍"：`last_hidden_shared` 与零拷贝流水线两样都要。

        `last_hidden_shared` 在 `SplitChain` 上总有这个名字，但 numpy 形态没有前缀 K/V 的
        落脚点（`can_share_prefix` 为假）。声明了却调不到是一条 500；退回串行只是一次没有
        收益的前向。声明也不等于每次都走：单题、前缀不同、`noncausal_full` 都走逐题。
        """
        chain = self._chain
        return (callable(getattr(chain, "last_hidden_shared", None))
                and bool(getattr(chain, "can_share_prefix", False)))

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
            self._default_mode,
        )
        probs = answers[0].probs
        if not probs or not all(math.isfinite(value) for value in probs):
            raise NotReadyError("pooled 预热前向产生了非法概率", strategy="pooled",
                                probs=list(probs))
        if abs(sum(probs) - 1.0) > 1e-3:
            raise NotReadyError("pooled 预热前向概率未归一", strategy="pooled",
                                total=sum(probs))
        self.prepare_vision()
        self._warm_ok = True
        # 装载与编译的拆分由链自己量：只有它知道哪几层是从头构建的。这里拿不到那两段时
        # （替身链）才退回"装载即全部、编译 0"—— 但那会让冷启动的 `compile` 报成 0.00 s，
        # 与"产物全命中"分不开，所以真链一律走前一条。
        chain = self._chain
        return CompilationTimes(
            load_s=float(getattr(chain, "load_s", self._load_s)),
            compile_s=float(getattr(chain, "compile_s", 0.0)),
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

    # ------------------------------------------------------------------ 请求入口
    def answers_for(
        self, request: DecisionRequest, mode: MaskMode, *, media: object | None = None
    ) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求的所有题：先过这份 checkpoint 的读法门，再交给基类的编排。

        读法的门留在这里而不是基类：能读哪几种掩码是**这个 checkpoint** 训练出来的事实，
        由家族在 descriptor 里声明、装配层传进来；一个没验证过的读法（比如 h2o 上的
        `prefix_noncausal`）会算出一个"看起来正常"的答案，所以它必须当场拒绝，而不是等
        数值对不上才发现。

        `media` 是给门与替身用的旁路（塔已经算完的图像侧）；请求自带的图仍走真路径，两条路
        在基类汇合。跨请求合批那条入口不需要这道门：`batch_limit` 是基类的 1，引擎据此既不
        攒批、也不把单条送进批路（`supports_batch` 只回答"链有没有那条原语"）。
        """
        if mode not in self._modes:
            raise UnsupportedModeError(
                "这个 checkpoint 没有声明所选读法",
                mode=mode.value,
                strategy="pooled",
                supported=sorted(item.value for item in self._modes),
            )
        return super().answers_for(request, mode, media=media)

    # ------------------------------------------------------------------ 原语
    def encode(self, state: str, question: Question, *, images: int = 0) -> Encoded:
        """题面 → 整条 prompt 与读出要的全部东西（厂商答面渲染的那一半）。

        `prefix_len=None`：厂商答面给不出"state 在哪结束"的词元边界，所以共享前缀由基类用
        公共前缀（LCP）判。那只对因果共享成立，而这条装配只有 causal / 全双向两种读法
        （全双向不共享），所以这里没有第二种切法要担心。

        `payload` 带厂商的 `Prepared`：答案不是"取最大概率的那个码"，而是厂商按题型判的
        （温度、noul 的地板与阈值、score 的期望值），这些量只有 `Prepared` 里有。
        """
        prepared = self._serve.prepare(state, question_spec(question), images=images)
        return Encoded(
            ids=prepared.ids,
            prefix_len=None,
            labels=prepared.labels,
            indexes=prepared.label_ids,
            payload=prepared,
        )

    def forward(self, encoded: Encoded, *, prefix: int, causal: bool = False,
                mm: object | None = None) -> np.ndarray:
        """单题一整趟：链只交最后一条 token 的最终归一化 hidden（不整段读回）。

        `prefix` 是 `prefix_noncausal` 的公共前缀长度；这条装配没有那个读法，基类按 0 传。
        """
        return self._require_chain().last_hidden(
            encoded.ids, prefix=prefix, causal=causal, mm=mm)

    def forward_shared(self, prefix: Sequence[int], tails: Sequence[Sequence[int]],
                       *, options: Sequence[int], causal: bool = False,
                       mm: object | None = None) -> Sequence[np.ndarray]:
        """前缀 + 每条尾巴一次前向：链给每题一份末行 hidden。

        `options` 只服务码表类的读出（"读前 n 个码"）；这里的候选集是标签行，由 `answer`
        按 `encoded.indexes` 取，所以不用它 —— 但它是基类的原语契约，得收下。
        """
        chain = self._require_chain()
        return chain.last_hidden_shared(  # type: ignore[attr-defined]
            prefix, tails, causal=causal, mm=mm)

    def forward_batched(self, encoded: Sequence[Encoded], *, causal: bool = False,
                        use_prefix: bool = True) -> Sequence[np.ndarray] | None:
        """N 条独立序列并排一个 pass：链给每题一份末行 hidden；装不下就 `None`。

        与 `forward_shared` 的分工是**没有共享前缀**：每条序列从第 0 列起，GDN 状态每槽一份
        零值、注意力按块对角互不可见。所以它合的是"不同 state 的独立请求"。

        `use_prefix=False`（`noncausal_full`）时前缀长度传 0：那条掩码没有前缀行，带进去会让
        批里的序列按另一种掩码收行。h2o 的 `prefix_len` 恒为 `None`（厂商答面给不出切点），
        所以 `prefixes` 实际上总是 0 —— 保留这条口径是为了与 split 同构。

        形状由链自己判（`Pipeline._slot_layout` 一处）：`None` 就是"这批没有一次前向的解法"，
        编排拿到它退回逐题，结果与逐条调用逐位相同。
        """
        chain = self._require_chain()
        batch = getattr(chain, "last_hidden_batched", None)
        if not callable(batch):
            return None
        return batch(
            [item.ids for item in encoded],
            prefixes=tuple(
                0 if not use_prefix or item.prefix_len is None else item.prefix_len
                for item in encoded),
            causal=causal,
        )

    def answer(self, encoded: Encoded, raw: object, *, index: int, kind: str) -> Answer:
        """末行 hidden × 标签行 → logits → 厂商判答案。

        点积只算标签那几行（不是整张码表）：整段读回 + 归一在 16k 上是 87 ms，末行 0.02 ms，
        而这里只需要这一行。概率**不在这里归一化**：厂商按题型换温度、noul 有地板与阈值，
        我们归一化一次再交过去就成了两套温度。
        """
        prepared = encoded.payload
        if not isinstance(prepared, Prepared):
            raise ConfigError(
                "pooled 的读出载体不是厂商答面（编码与读出不是同一条路）", kind=kind)
        logits = self._rows(encoded.indexes) @ np.asarray(raw, dtype=np.float32)
        value, labels, probs, confidence = self._serve.answer(prepared, logits)
        return Answer(
            question_index=index,
            kind=kind,
            value=value,
            probs=tuple(probs),
            labels=tuple(labels),
            confidence=confidence,
        )

    def _require_ready(self) -> None:
        """基类在编码之前调它：没装权重就当场 `ConfigError`，别先记一笔不会发生的 miss。"""
        self._require_chain()

    def _require_chain(self) -> PooledChain:
        if self._chain is None:
            raise ConfigError("pooled 策略尚未装载权重", strategy="pooled")
        return self._chain


# ---------------------------------------------------------------------- 装配


def pooled_runner(
    model_dir: str | Path,
    *,
    recipe: PooledRecipe,
    weights: WeightPlan | None = None,
    bucket: int | None = None,
    width: int | None = None,
    quantize: bool = True,
    scratch_dir: str | Path | None = None,
    tokenizer: object | None = None,
    metal: bool = True,
    resident: int | None = None,
    vision: bool | None = None,
    vision_idle: float = 0.0,
    vision_bucket: int | None = None,
) -> PooledRunner:
    """装一条 pooled 链：家族给的厂商答面 + 词表行按需取 + ANE 链。

    这条工厂**不认识任何家族**：答面（题面、温度、地板、判答）从 `recipe` 来，形状与张量名
    从 `weights` 来（家族声明的 `weight_plan`，装配层解析好传进来）。换个 Qwen3.5 决策模型
    接进来，改动只在它自己的家族模块里。

    `bucket` 是 ANE program 的静态列宽、`width` 是缓冲宽度（默认都取 `S_NATIVE`）。4B 的
    `ffn` 是 9216、`hidden` 是 2560，两个 program 都比 2B 大，所以这两项由调用方说清楚，
    不在这里猜。

    读出表名来自 `weights.descriptor.readout.weight`：h2o 没声明，退回 tied 的 `embed_tokens`；
    torchcast 声明 `lm_head.weight`（在文本前缀之外，完整名，不拼前缀）。

    `metal=True` 默认开：h2o 的 `(2560, 9216)` 已经做过 numpy vs Metal 的逐位对拍
    （票据 15 §24：桶 256、64/128 token、两种读法，hidden relL2 1.3e-3、logits 3e-4 量级、
    argmax 全同），注意力与 GDN 走 GPU 快 3~10 倍。

    `resident` 是待命常驻的层数（`None` = 32 层全装）。h2o 的 32 层 blob 合计 ≈3.33 GiB，
    是这条链最大的一笔常驻；给 4 就是"一个块"，换窗代价见票据 15 §25。它只在 `metal=True`
    时才有意义（窗口是流水线的机制），`metal=False` 传它会退化成"全装"。

    `vision` 是 `None`（自动）/ `False`（`--vision off`，纯文本服务：不建塔、不读塔权重）。
    h2o 的塔与 mude **同一套几何**，只差 merger 的输出宽度（2560 vs 2048），所以这里不需要
    第二份实现；`vision_idle` 是塔的空闲卸载秒数，默认 `0`（每请求后即卸，闲时 0 常驻）。
    """
    from ...devices.apple.chain import SplitChain
    from ...devices.apple.compile import S_NATIVE
    from ...devices.apple.platform import warn_if_unmeasured
    from ...weights.checkpoint import open_checkpoint

    if weights is None:
        raise ConfigError(
            "pooled 策略需要家族提供 weight_plan",
            strategy="pooled",
            hint="在 register_family(..., weight_plan=...) 里给出这份检查点的前缀与 norm/rope 事实",
        )
    # 形状检查放在开答面与检查点之前：它只读 config，却要在任何"看起来像结果"的运行之前
    # 先出声。告警不拦（`platform.py` 顶部写明理由）—— ANE 对形状很宽容，未测几何也会
    # 编译、会跑出数字，所以日志里必须有这条记录。
    warn_if_unmeasured(weights.descriptor, strategy="pooled")
    serve = recipe.serve(model_dir, tokenizer)
    store = open_checkpoint(Path(model_dir))
    # 读出张量按 descriptor 取，不按"tied 就用词表"猜：torchcast 解绑了 `lm_head`，而且它在
    # `model.language_model` 之外 —— 用前缀去拼 `lm_head` 会拼出一个不存在的张量名。
    readout_name = weights.descriptor.readout.weight
    if readout_name is None:
        if not weights.descriptor.tied_embeddings:
            raise ConfigError(
                "这份 checkpoint 的读出不在词表上（untied），descriptor 却没声明读出张量",
                model=str(model_dir),
                hint="在 ReadoutPlan(weight=...) 里写出读出表的名字",
            )
        readout_name = f"{weights.prefix}.embed_tokens.weight"
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
        return np.asarray(store.read_rows(readout_name, list(label_ids)), np.float32)

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

    def tower_state(chain: object) -> dict[str, object]:
        from ...devices.apple import vision as vision_device

        state = vision_device.artifact_state(
            chain.store, descriptor=weights.descriptor, s=tower_s, artifacts_dir=scratch_dir,
        )
        return {"directory": str(state.directory), "reusable": state.hits, "total": state.total}

    # 答面收不了图时不建塔（今天两家都能收，这一位留给下一个 pooled 家族）：编 26 份塔产物是
    # 每次启动几秒与一块常驻，而请求会在编码前就 422。`image_token_id=None` 让 `VisionRunner`
    # 报出**真实**原因（"这个模型没有视觉塔"），而不是错误的建议"给 --vision auto"。
    images_ok = bool(getattr(serve, "supports_images", True))
    load = (load_tower
            if (images_ok and vision is not False and weights.descriptor.vision is not None)
            else None)
    return PooledRunner(
        load_chain, serve, rows, supported_modes=weights.descriptor.supported_modes,
        default_mode=weights.descriptor.default_mode, max_len=buffer_width,
        image_token_id=weights.descriptor.image_token_id if images_ok else None, load_tower=load,
        vision_state=tower_state if load is not None else None,
    )


__all__ = ["PooledRunner", "pooled_runner"]
