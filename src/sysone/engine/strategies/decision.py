"""决策 runner 的共性编排：编码循环、共享前缀、跨请求合批与串行回退。

这是 `docs/sysone/decision-runner.md` 的 Stage 1 与 Stage 2：两层调度都由这里决定 ——
**一个请求内** 1 个 state + N 个题走共享前缀还是逐题整趟（`answers_for`），**多个请求之间**
能不能挤进一个 pass（`answers_batch`）。模型只提供五个薄原语（`encode` / `forward` /
`forward_shared` / `forward_batched` / `answer`），外加自己的几何与能力位。

为什么这两层都不属于任何一条模型：协议天然是 1 state + N questions，合批的前提是"同一个
掩码、都没有图、槽位装得下"。"那一段能不能复用"只取决于每题的前缀 tokens 是否相同，
"几条请求能不能并排"只取决于形状与掩码 —— 都与模型结构无关。它们写在 `SplitRunner` 里的
那些天，h2o 与 vendor 就各写了一遍串行，谁都没共享 —— 真实评测流量 34 条请求全是 4–5 题、
理想共享比 3.8–3.9×，而 `/metrics` 里 `shared_miss` 那一格才把这件事量出来。

六个计数的语义（互不重叠，一个机会只记一次）：

| 计数 | 记什么 |
|---|---|
| `shared_hit` | 一次**请求内**（`answers_for`）多题真的复用了同一段前缀 |
| `shared_miss` | 一次请求内多题**本可以**共享却没共享（前缀不同 / 没这个原语） |
| `batch_hit` | 一次**跨请求**合批（`answers_batch`）真的进了同一个 pass |
| `batch_miss` | `answers_batch` 只拿到一条**序列**：批这条路走了，但没有可合的东西 |
| `batch_fallback` | `answers_batch` 拿到 ≥2 条，却退回逐题整趟（前缀与块对角都不成立） |
| `media_skip` | 带图请求过了视觉那条路，共享/合批对它无效 |

单题请求不计 `shared_*`：没有"与谁共享"这件事，记成 miss 会把最常见的负载涂成失败。
`shared_miss` 也只在**有前缀语义**的模式下记（`noncausal_full` 的双向前向里，前缀行与题面
耦合，本来就不能复用）。vendor 桥不透明、永远串行，也不记 —— 那些数字会永远不动，读的人
会以为有机会没做。

`batch_limit` 是"这条执行路径一个 pass 最多装几条请求"，基类给 1。它是**形状**上限，不是
策略上限：策略上限是服务层的 `--batch`，两边由 `Engine.batch_limit` 取小。合批的收益按瓦片
数算（见 `answers_batch` 的 docstring）；"这一批装不装得下"的判断留在 `forward_batched` 里
—— 只有执行器知道那条链的桶宽与缓冲宽度。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Sequence

from ...domain.attention import MaskMode
from ...domain.decision import Answer, DecisionRequest, Question
from ...errors import ConfigError, RequestTooLongError, UnsupportedMediaError
from ...models.vision import merge


@dataclass(frozen=True)
class Encoded:
    """一题编码后的共性部分：整条 prompt、共享前缀边界、读出标签与它们的位置。

    `prefix_len` 是"state 前缀"的**词元边界**：`None` 表示这条链量不出边界，由
    `shared_prefix` 退到公共前缀（LCP）。带前缀语义的模式（`prefix_noncausal`）必须给 int：
    边界换一个位置，前缀行的可见集合就变了，答案也会跟着变。

    `labels` / `indexes` 是读出要用的两样东西（mude 是码表里的码与选项序号，h2o 是答面标签
    与它们在 logits 里的位置）。词表行之类的 runner 私有事实不进这里：它们是读出形态的一部分，
    由各 runner 自己带着。

    `payload` 是**执行器私有的读出载体**。有码表的链（mude）用不上：`labels` / `indexes`
    就够读出。没有码表的链（h2o 的 pooled）要把厂商答面给出的 `Prepared` 带到 `answer` ——
    温度、noul 的地板与阈值、score 的期望值都在那里，`indexes` 只装得下标签 id。基类不碰
    它、也不认识它的类型：这里只是"编码"到"读出"之间搭的一趟便车。
    """

    ids: tuple[int, ...]
    prefix_len: int | None
    labels: tuple[str, ...]
    indexes: tuple[int, ...]
    payload: object | None = None


def shared_prefix(encoded: Sequence[Encoded]) -> tuple[int, ...] | None:
    """所有题共用同一条前缀时返回它，否则 `None`。

    两条判据，按 `prefix_len` 分：

    - **都给了 int**：必须都等于同一个正整数，且逐词元相同。这是 `prefix_noncausal` 的
      `common_prefix` 那条路（mude）—— 分词边界可能把某题的尾巴和前缀粘成一个词元，那时
      长度或内容就有一处不一样，复用会改变那一题的数值。
    - **都给 `None`**：取公共前缀（LCP），尾巴至少留一个词元。**只对因果共享成立**：
      因果掩码下"在哪切开"不改变任何一行的可见集合（前缀行的三角在相内自己成立，尾巴行看
      `[0, q]`），所以任意切点都等价；双向共享不能这样切。
    - **混着给**：没有可证的公共边界，返回 `None`。

    **单题与空序列不算共享**：这个函数只回答"是不是同一条前缀"，调用方要自己要求
    `len(encoded) > 1`（编排在 `DecisionRunner._answers` 与 `answers_batch` 里）。
    """
    if not encoded:
        return None
    declared = [item.prefix_len for item in encoded]
    if all(length is None for length in declared):
        return _common_prefix(encoded)
    if any(length is None for length in declared):
        return None
    length = int(declared[0])  # type: ignore[arg-type] - 上面已经排除了 None
    if length <= 0 or any(other != declared[0] for other in declared[1:]):
        return None
    if any(len(item.ids) < length for item in encoded):
        return None
    prefix = encoded[0].ids[:length]
    if any(item.ids[:length] != prefix for item in encoded[1:]):
        return None
    return prefix


def _common_prefix(encoded: Sequence[Encoded]) -> tuple[int, ...] | None:
    """LCP；上限是 `min(len)-1`（读出位置永远不在前缀里，尾巴也不能空）。"""
    first = encoded[0].ids
    limit = min(len(item.ids) for item in encoded) - 1
    count = 0
    while count < limit and all(item.ids[count] == first[count] for item in encoded[1:]):
        count += 1
    return first[:count] if count > 0 else None


@dataclass
class DecisionMetrics:
    """共享与合批的计数（互不重叠）；编排按能力位分派，计数记在分派处。"""

    shared_hit: int = 0
    shared_miss: int = 0
    batch_hit: int = 0
    batch_miss: int = 0
    batch_fallback: int = 0
    media_skip: int = 0

    def snapshot(self) -> dict[str, int]:
        return asdict(self)


class DecisionRunner:
    """决策 runner 的共性骨架：编排 + 能力声明 + 计数。

    `answers_for` 是一次请求的全部编排：编码 → 判前缀 → 共享或逐题 → 读出；`answers_batch`
    是多个请求的全部编排：编码 → 判前缀 → 同前缀共享 / 块对角合批 / 逐题回退 → 读出。子类只需
    实现五个原语，并让 `_max_len` / `_image_token` 就位（后者与 `VisionRunner` 是同一个字段：
    两条装配共用"一张图一个占位符"这个语义）。

    `supports_shared` / `supports_batch` **不是"想不想共享"，而是"这条链现在有没有那条原语"**：
    `ModelChain` 的协议只要求 `forward` / `close`，不强制 `forward_shared` / `forward_batched`。
    所以 `SplitRunner` 把它们实现成按链问的属性 —— 一条没有 `forward_shared` 的链在这里就退回
    串行，而不是等 `AttributeError` 把它变成一次 500。声明能力也不等于每次都用：前缀不同、
    单题、`noncausal_full` 都走串行。
    """

    # 有没有"多题共用一段前缀"与"多请求挤进一个 pass"的原语。默认没有：一个没声明的模型
    # 不该被当成能共享，宁可退回串行。
    supports_shared: bool = False
    supports_batch: bool = False

    #: 这条执行路径的缓冲宽度（每题整条 prompt 的上限）。子类在 `__init__` 里设。
    _max_len: int
    #: 图像占位符的 id；`None` = 这个装配收不了图。
    _image_token: int | None
    _decisions: DecisionMetrics | None = None

    # ------------------------------------------------------------------ 编排

    @property
    def batch_limit(self) -> int:
        """一个 pass 最多装几条请求；1 = 这条执行路径不进合批。

        基类没有合批原语，所以是 1。子类按自己的**形状**给上限（`SplitRunner` 用缓冲宽度与
        最短槽宽推），服务层的 `--batch` 是**策略**上限，两边由 `Engine.batch_limit` 取小。
        """
        return 1

    def answers_for(
        self, request: DecisionRequest, mode: MaskMode, *, media: object | None = None
    ) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 前向耗时)。

        `media` 是给门与替身用的旁路（塔已经算完的图像侧），给了就原样用；**请求自带的图**
        （`request.media`）走真路径 —— 解码、像素管线、塔、`Media` —— 然后与 `media=` 那条路
        汇合。两条都在同一个 `finally` 里释放塔（`idle = 0` 时"每请求后即卸"就是靠它）。
        """
        if media is None and request.media:
            # 视觉半边在 `VisionRunner`（两条装配共用这一份）；它回来时调用本类的 `_answers`。
            return self._answers_with_images(request, mode)  # type: ignore[attr-defined]
        return self._answers(request, mode, media)

    def _answers(
        self, request: DecisionRequest, mode: MaskMode, media: object | None
    ) -> tuple[tuple[Answer, ...], int, float]:
        """编码 + 分派 + 读出。`encode_request` 与两条 `*_answers` 也是 `answers_batch` 用的。"""
        self._require_ready()
        # `noncausal_full` 就是把前缀当 0：掩码 `(q >= 0) | (k < 0)` 恒真，前缀行与题面耦合，
        # 复用那一段会改答案 —— 所以它不是"漏了共享"，是没这个语义。
        # `causal` 的共享前缀照样成立：前缀行的掩码在相内自己成三角，尾巴行看 `[0, q]` 包含
        # 整个前缀 —— 两条路（共享 / 逐题）的可见集合一样，所以批尾巴的槽位规则也不用改。
        causal = mode is MaskMode.CAUSAL
        use_prefix = mode is not MaskMode.NONCAUSAL_FULL
        started = time.perf_counter()
        encoded, placements = self.encode_request(request, media)
        kinds = [question.kind for question in request.questions]
        # 共享与串行两条路都只吃一份行落点（这是既有行为）：同一请求里的题共用同一批图时，
        # 一图一块的位置在每题的整条序列上不同，但链只按第一题那份把塔的行盖进去。
        mm = placements[0] if placements else None
        shared = shared_prefix(encoded) if use_prefix else None
        if shared is not None and len(encoded) > 1 and self.supports_shared:
            self.decisions.shared_hit += 1
            answers = self.shared_answers(shared, encoded, kinds, causal=causal, mm=mm)
        else:
            if len(encoded) > 1 and use_prefix:
                # 有前缀语义的多题却没共享：前缀不同、链没有 `forward_shared`，或分词把它们的
                # 前缀切成了不同的串。都记 miss —— 它就是共享的收益没拿到。
                self.decisions.shared_miss += 1
            answers = self.serial_answers(
                encoded, kinds, use_prefix=use_prefix, causal=causal, mm=mm)
        tokens = sum(len(item.ids) for item in encoded)
        return tuple(answers), tokens, time.perf_counter() - started

    def answers_batch(
        self, requests: Sequence[DecisionRequest], mode: MaskMode
    ) -> list[tuple[tuple[Answer, ...], int, float]]:
        """多条**互不相干**的请求走一个 pass，返回每条各自的 `(答案, token 数, 前向秒)`。

        合成一个 pass 有两条路：**前缀相同**（决策负载里就是"同一个 state、不同的问题"）走
        `forward_shared` 批尾巴；**前缀不同**走块对角（一条序列一个槽，每行只看自己那一槽）。
        两条都拿不到才退回逐题整趟（`serial_answers`），结果与逐条调用**逐位相同** ——
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
        self._require_ready()
        if any(request.media for request in requests):
            raise ConfigError("批里不能有带图的请求（图片按请求走塔）",
                              images=sum(1 for r in requests if r.media))
        causal = mode is MaskMode.CAUSAL
        use_prefix = mode is not MaskMode.NONCAUSAL_FULL
        started = time.perf_counter()
        # 编码、长度检查与"一图多占位"的展开都在 `encode_request`（批里没有图，落点一定是空的）。
        # 多条请求的题摊平成一张表，所以每条记一个区间，答案最后按区间切回去。
        encoded: list[Encoded] = []
        kinds: list[str] = []
        spans: list[tuple[int, int]] = []
        for request in requests:
            first = len(encoded)
            own, _ = self.encode_request(request, None)
            encoded.extend(own)
            kinds.extend(question.kind for question in request.questions)
            spans.append((first, len(encoded)))
        shared = shared_prefix(encoded) if use_prefix else None
        answers: list[Answer] | None = None
        if shared is not None and len(encoded) > 1 and self.supports_shared:
            self.decisions.batch_hit += 1
            answers = self.shared_answers(shared, encoded, kinds, causal=causal)
        elif len(encoded) > 1 and self.supports_batch:
            # 没有共享前缀（不同 state 的独立请求）时走块对角：一条序列一个槽、互不可见。
            # `None` = 这一批的形状挤不进一趟（判据在 `forward_batched` 里，只有它知道几何）。
            answers = self.batched_answers(
                encoded, kinds, causal=causal, use_prefix=use_prefix)
            if answers is not None:
                self.decisions.batch_hit += 1
        if answers is None:
            # 一条序列（引擎把单条也送进这条路）没有可合的东西；≥2 条才是有机会却没合上。
            if len(encoded) > 1:
                self.decisions.batch_fallback += 1
            else:
                self.decisions.batch_miss += 1
            answers = self.serial_answers(encoded, kinds, use_prefix=use_prefix, causal=causal)
        elapsed = time.perf_counter() - started
        out: list[tuple[tuple[Answer, ...], int, float]] = []
        for first, last in spans:
            # 摊平编码之后，答案的 `question_index` 是**全批**的序号；每条请求的结果要报"这条
            # 请求里的第几题"（逐条调用时就是这个数），所以切片时按请求内序号重编 —— 只是这
            # 份数据对齐，没有另一种答案。
            own = tuple(
                replace(answer, question_index=index)
                for index, answer in enumerate(answers[first:last])
            )
            tokens = sum(len(encoded[index].ids) for index in range(first, last))
            out.append((own, tokens, elapsed))
        return out

    def encode_request(
        self, request: DecisionRequest, media: object | None
    ) -> tuple[list[Encoded], list[object]]:
        """把请求里的每道题编码成 `Encoded`；带图时展开占位符并返回每题的行落点。

        `media` 只说明"这一趟带图"：具体哪张图落在哪儿由 `merge.attach` 按占位符位置去对，
        返回的落点与题一一对应（调用方只把第一份交给链，见 `_answers`）。

        **展开在编码之后**：一张图一个占位符 → `counts[i]` 个，图块的起点要等前面那些图各自
        展开了多少才知道（所以是 span，不是一个偏移）。展开之后长度才是真的，越界检查也放在
        展开之后 —— 否则一个"刚好塞进桶"的题会展开成两倍长，在链里抛裸 `ValueError`。
        """
        images = 0 if media is None else int(media.images)  # type: ignore[attr-defined]
        if images and self._image_token is None:
            # 图块没有 token id 的模型（纯文本装配）收到图：这是**请求**的问题，不是我们的配置
            # 坏了 —— 客户端据此知道不要重试。前门会先用 `image_contract()` 拒一次，这里是最后
            # 一道，不能让 `image_token_id=None` 变成一个诡异的"占位符数不对"。
            raise UnsupportedMediaError("这个模型没有视觉塔，收不了图", images=images)
        # 先把所有题编码完：共享前缀要先知道每题的尾巴，长度检查也按整条 prompt 做。
        encoded: list[Encoded] = []
        for index, question in enumerate(request.questions):
            item = self.encode(request.state, question, images=images)
            if len(item.ids) > self._max_len:
                raise RequestTooLongError(
                    "prompt 超过链的缓冲宽度",
                    length=len(item.ids),
                    max_len=self._max_len,
                    question=index,
                )
            encoded.append(item)
        if media is None:
            return encoded, []
        # 带图请求不共享也不合批；记一笔，否则"图一多，优化就静默失效"看不出来。
        self.decisions.media_skip += 1
        placements: list[object] = []
        placed: list[Encoded] = []
        for index, item in enumerate(encoded):
            expanded, placement = merge.attach(
                item.ids, media, image_token_id=self._image_token)  # type: ignore[arg-type]
            if len(expanded) > self._max_len:
                raise RequestTooLongError(
                    "展开图像占位符之后超过链的缓冲宽度",
                    length=len(expanded), max_len=self._max_len, question=index)
            # 前缀里也有图块，所以前缀长度要跟着展开（模型量的是未展开的那串）。
            # `prefix_len is None` 只出现在量不出边界的链上：按 0 处理等于退回逐题，
            # 不会假装共享。
            cut = 0 if item.prefix_len is None else item.prefix_len
            prefix_ids, _ = merge.expand_image_tokens(
                item.ids[:cut], media.counts, image_token_id=self._image_token)  # type: ignore[arg-type]
            placed.append(replace(item, ids=tuple(expanded), prefix_len=len(prefix_ids)))
            placements.append(placement)
        return placed, placements

    def shared_answers(
        self, prefix: tuple[int, ...], encoded: Sequence[Encoded], kinds: Sequence[str],
        *, causal: bool = False, mm: object | None = None,
    ) -> list[Answer]:
        """共享前缀的一条路：`prefix` + 每题的尾巴，一次前向拿到所有题的读数。

        尾巴是"整条 prompt 去掉前缀"，切点就是 `shared_prefix` 判出来的那个数 —— 判定与切分
        用同一个值，避免"判的是 A 段、切的是 B 段"这种只有数值差、没有报错的错。
        """
        tails = [item.ids[len(prefix):] for item in encoded]
        options = [len(item.labels) for item in encoded]
        raw = self.forward_shared(prefix, tails, options=options, causal=causal, mm=mm)
        return [
            self.answer(item, raw[index], index=index, kind=kinds[index])
            for index, item in enumerate(encoded)
        ]

    def batched_answers(
        self, encoded: Sequence[Encoded], kinds: Sequence[str], *, causal: bool = False,
        use_prefix: bool = True,
    ) -> list[Answer] | None:
        """块对角的一条路：N 条互不相干的序列并排跑一个 pass；装不下就是 `None`。

        `None` 不是错误，是"这一批的形状挤不进一趟"（每条 ≤ 桶宽、总列数 ≤ 缓冲宽度）。
        判断留在 `forward_batched` 里，因为只有它知道那条链的几何；编排拿到 `None` 就退回逐题，
        结果与逐条调用逐位相同 —— 这只是没有收益，不是另一种答案。

        `use_prefix` 与 `_answers` 同一个口径：`noncausal_full` 没有前缀语义，前缀长度一律传 0
        （见 `forward_batched`）。
        """
        raw = self.forward_batched(encoded, causal=causal, use_prefix=use_prefix)
        if raw is None:
            return None
        return [
            self.answer(item, raw[index], index=index, kind=kinds[index])
            for index, item in enumerate(encoded)
        ]

    def serial_answers(
        self, encoded: Sequence[Encoded], kinds: Sequence[str],
        *, use_prefix: bool, causal: bool = False, mm: object | None = None,
    ) -> list[Answer]:
        """逐题整趟（单题、前缀不同、无前缀语义、链没有那条原语时都走这条）。"""
        answers: list[Answer] = []
        for index, item in enumerate(encoded):
            # `prefix_len is None` 只出现在量不出边界的链上：按 0 处理等于按"整条序列一个窗口"
            # 跑，而不是假装有个前缀。没有前缀语义的模式（`noncausal_full`）同样按 0 传。
            prefix = 0 if not use_prefix or item.prefix_len is None else item.prefix_len
            raw = self.forward(item, prefix=prefix, causal=causal, mm=mm)
            answers.append(self.answer(item, raw, index=index, kind=kinds[index]))
        return answers

    def _require_ready(self) -> None:
        """跑一次请求之前的就绪检查。基类没有共享状态；子类在链还没装时报 `ConfigError`。

        放在**编码之前**：一次不会发生的请求不该把 `shared_miss` 记上一笔。
        """
        return None

    # ------------------------------------------------------------------ 原语

    def encode(self, state: str, question: Question, *, images: int = 0) -> Encoded:
        """题面 → 整条 prompt 的 id、前缀边界、读出标签与位置。**模型相关的那一半**。"""
        raise NotImplementedError("这条执行路径没有声明 encode")

    def forward(self, encoded: Encoded, *, prefix: int, causal: bool = False,
                mm: object | None = None) -> object:
        """单题一整趟前向；返回值原样交给 `answer()`（概率或末行 hidden 由执行器定）。"""
        raise NotImplementedError("这条执行路径没有声明单题前向")

    def forward_shared(self, prefix: Sequence[int], tails: Sequence[Sequence[int]],
                       *, options: Sequence[int], causal: bool = False,
                       mm: object | None = None) -> Sequence[object]:
        """前缀 + 每条尾巴一次前向；每题一份读数，顺序与 `tails` 一致。"""
        raise NotImplementedError("这条执行路径没有声明共享前缀前向")

    def forward_batched(self, encoded: Sequence[Encoded], *, causal: bool = False,
                        use_prefix: bool = True) -> Sequence[object] | None:
        """N 条独立序列一次前向；每题一份读数，顺序与 `encoded` 一致。

        装不进一趟返回 `None`（编排退回逐题）：形状判据是几何的一部分，只有执行器知道。

        `use_prefix=False`（`noncausal_full`）时每条序列的前缀长度传 0：那条掩码里没有"前缀
        行"，把 `prefix_len` 带进批会让流水线按 `prefix_noncausal` 收行 —— 同一个 mode 的批与
        逐条就会得到不同的掩码（曾经如此，见 Stage 2 的对拍）。
        """
        raise NotImplementedError("这条执行路径没有声明块对角前向")

    def answer(self, encoded: Encoded, raw: object, *, index: int, kind: str) -> Answer:
        """读数 → `Answer`（标签映射、温度、归一化、合法性检查都在这里）。"""
        raise NotImplementedError("这条执行路径没有声明读出")

    # ------------------------------------------------------------------ 观测

    @property
    def decisions(self) -> DecisionMetrics:
        """惰性建一次：三条策略的 `__init__` 各写一份会漂开，基类也不去抢它们的构造。"""
        if self._decisions is None:
            self._decisions = DecisionMetrics()
        return self._decisions

    def decision_stats(self) -> dict[str, int | bool]:
        """`/metrics` 的 `decision` 一栏：能力声明与今天的命中数（一个来源）。"""
        return {
            **self.decisions.snapshot(),
            "supports_shared": self.supports_shared,
            "supports_batch": self.supports_batch,
        }


__all__ = ["DecisionMetrics", "DecisionRunner", "Encoded", "shared_prefix"]
