"""决策 runner 的共性契约：共享前缀判据、两层编排（`answers_for` / `answers_batch`）、能力声明
与回退计数。

三件事分开钉：`shared_prefix` 在什么条件下才算"同一条前缀"、基类拿到五个原语之后怎么分派
（请求内共享 / 跨请求合批 / 逐题回退 / 拒绝）、以及计数器的形状与归属。三条策略自己的接线在
`test_split_runner.py` / `test_pooled_runner.py` / `test_vendor_runner.py`，服务层那一栏在
`test_service_app.py`。
"""

from __future__ import annotations

import pytest

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Answer, DecisionRequest, Question
from sysone.engine.strategies.decision import (
    DecisionMetrics,
    DecisionRunner,
    Encoded,
    shared_prefix,
)
from sysone.errors import ConfigError, RequestTooLongError


def _encoded(ids, *, prefix_len=None, labels=("A", "B"), indexes=(0, 1)) -> Encoded:
    return Encoded(
        ids=tuple(ids), prefix_len=prefix_len, labels=tuple(labels), indexes=tuple(indexes)
    )


# ---------------------------------------------------------------- 声明了边界


def test_a_declared_boundary_is_reused_when_every_question_agrees() -> None:
    """mude 那条路：`common_prefix` 量出同一个边界、边界前的词元也一致。"""
    assert shared_prefix([_encoded((1, 2, 3), prefix_len=2),
                          _encoded((1, 2, 9), prefix_len=2)]) == (1, 2)


def test_a_different_declared_boundary_is_not_a_shared_prefix() -> None:
    """长度不同就不能复用：前缀行的可见集合会跟着边界走。"""
    assert shared_prefix([_encoded((1, 2, 3), prefix_len=2),
                          _encoded((1, 2, 3), prefix_len=1)]) is None


def test_the_tokens_must_match_up_to_the_declared_boundary() -> None:
    assert shared_prefix([_encoded((1, 2, 3), prefix_len=2),
                          _encoded((1, 9, 3), prefix_len=2)]) is None


def test_a_declared_prefix_longer_than_the_prompt_is_refused() -> None:
    """边界落在 prompt 之外是不可能的输入，不能靠切片把它当成"整条都相同"。"""
    assert shared_prefix([_encoded((1, 2), prefix_len=3),
                          _encoded((1, 2, 3), prefix_len=3)]) is None


def test_a_zero_boundary_means_no_prefix() -> None:
    assert shared_prefix([_encoded((1, 2), prefix_len=0),
                          _encoded((1, 5), prefix_len=0)]) is None


def test_one_question_still_reports_its_own_prefix() -> None:
    """这个函数只回答"是不是同一条前缀"；"要不要共享"要求 ≥2 题，由调用方判。"""
    assert shared_prefix([_encoded((1, 2, 3), prefix_len=1)]) == (1,)
    assert shared_prefix([]) is None


# ---------------------------------------------------------------- 量不出边界


def test_lcp_is_the_boundary_when_the_chain_cannot_declare_one() -> None:
    """h2o 那种只说"整条 prompt"的链：因果下任意切点都等价，取公共前缀。"""
    assert shared_prefix([_encoded((1, 2, 3, 4)), _encoded((1, 2, 7))]) == (1, 2)


def test_lcp_keeps_one_token_for_the_tail() -> None:
    """尾巴不能空：一条序列是另一条的前缀时，LCP 要停在离终点一格的位置。"""
    assert shared_prefix([_encoded((1, 2, 3)), _encoded((1, 2, 3, 4))]) == (1, 2)


def test_lcp_needs_a_common_first_token() -> None:
    assert shared_prefix([_encoded((1, 4, 5)), _encoded((2, 4, 5))]) is None


def test_mixing_declared_and_unknown_boundaries_is_not_shared() -> None:
    """一边是"边界在这里"，一边是"不知道"：没有可证的公共切点，宁可退回逐题。"""
    assert shared_prefix([_encoded((1, 2, 3), prefix_len=2), _encoded((1, 2, 3))]) is None


# ---------------------------------------------------------------- 计数


def test_the_counters_start_at_zero_and_are_per_instance() -> None:
    class Runner(DecisionRunner):
        supports_shared = True

    first, second = Runner(), Runner()
    assert first.decision_stats() == {
        **DecisionMetrics().snapshot(),
        "supports_shared": True,
        "supports_batch": False,
    }
    first.decisions.shared_hit += 1
    assert first.decision_stats()["shared_hit"] == 1
    assert second.decision_stats()["shared_hit"] == 0, "计数字段不能是类属性、被两个实例共用"


# ---------------------------------------------------------------- 编排


def _request(questions: int) -> DecisionRequest:
    return DecisionRequest(
        state="state",
        questions=tuple(Question(text=f"q{index}", options=("甲", "乙"))
                        for index in range(questions)),
    )


class FakeRunner(DecisionRunner):
    """五个原语齐全的替身：每条前向各自记账，按选项数给等概率。

    `prefixes` 逐题给前缀（默认 `(7, 8)`），`boundary` 是它声明给 `Encoded` 的边界 —— 两者
    分开是为了能造"声明长度相同、词元不同"这种真会发生的错配。

    `tiled` 表示它还有块对角原语；`fits` 是那条原语对"这一批装不装得下"的回答（装不下时它
    返回 `None`，编排退回逐题）。
    """

    def __init__(self, *, prefixes=None, boundary: int = 2, max_len: int = 64,
                 shared: bool = True, tiled: bool = False, fits: bool = True) -> None:
        self.supports_shared = shared
        self.supports_batch = tiled
        self._max_len = max_len
        self._image_token = None
        self._prefixes = list(prefixes or [])
        self._boundary = boundary
        self._call = 0
        self._fits = fits
        self.serial: list[tuple[tuple[int, ...], int, bool]] = []
        self.shared: list[tuple[tuple[int, ...], list[tuple[int, ...]], tuple[int, ...], bool]] = []
        self.tiled: list[tuple[tuple[tuple[int, ...], ...], bool]] = []

    def encode(self, state, question, *, images=0):
        index = self._call
        self._call += 1
        prefix = self._prefixes[index] if index < len(self._prefixes) else (7, 8)
        labels = tuple(question.options)
        return Encoded(
            ids=tuple(prefix) + (ord(question.text[0]), ord(question.text[-1])),
            prefix_len=self._boundary,
            labels=labels,
            indexes=tuple(range(len(labels))),
        )

    def forward(self, encoded, *, prefix, causal=False, mm=None):
        self.serial.append((encoded.ids, int(prefix), bool(causal)))
        return [1.0 / len(encoded.labels)] * len(encoded.labels)

    def forward_shared(self, prefix, tails, *, options, causal=False, mm=None):
        self.shared.append((tuple(prefix), [tuple(tail) for tail in tails], tuple(options),
                            bool(causal)))
        return [[1.0 / int(count)] * int(count) for count in options]

    def forward_batched(self, encoded, *, causal=False, use_prefix=True):
        self.tiled.append((
            tuple(item.ids for item in encoded),
            bool(causal),
            tuple(0 if not use_prefix or item.prefix_len is None else item.prefix_len
                  for item in encoded),
        ))
        if not self._fits:
            return None
        return [[1.0 / len(item.labels)] * len(item.labels) for item in encoded]

    def answer(self, encoded, raw, *, index, kind):
        values = tuple(float(value) for value in raw)
        return Answer(
            question_index=index,
            kind=kind,
            value=encoded.labels[0],
            probs=values,
            labels=encoded.labels,
            confidence=max(values),
        )


def test_the_orchestration_shares_one_prefix_across_questions() -> None:
    """两题一个 state：前缀只跑一遍，每题只跑自己的尾巴。"""
    runner = FakeRunner()
    answers, tokens, _ = runner.answers_for(_request(2), MaskMode.PREFIX_NONCAUSAL)
    assert len(runner.shared) == 1, "两题共用前缀时只该有一次共享前向"
    prefix, tails, options, causal = runner.shared[0]
    assert prefix == (7, 8) and len(tails) == 2 and options == (2, 2)
    assert causal is False, "`prefix_noncausal` 的掩码仍是双向"
    assert runner.serial == [], "共享路径不该再逐题整趟"
    assert [answer.question_index for answer in answers] == [0, 1]
    assert [answer.value for answer in answers] == ["甲", "甲"]
    assert tokens == sum(len(prefix) + len(tail) for tail in tails), "计费口径是整条 prompt"
    assert runner.decisions.shared_hit == 1 and runner.decisions.shared_miss == 0


def test_a_lone_question_goes_down_the_serial_path() -> None:
    """单题没有"与谁共享"这件事：逐题整趟，而且一个 `shared_*` 都不记。"""
    runner = FakeRunner()
    runner.answers_for(_request(1), MaskMode.PREFIX_NONCAUSAL)
    assert runner.shared == []
    assert len(runner.serial) == 1 and runner.serial[0][1] == 2, "单题也要按前缀整趟跑"
    assert runner.decisions.shared_hit == 0 and runner.decisions.shared_miss == 0


def test_a_different_prefix_per_question_falls_back_and_records_the_miss() -> None:
    """分词把两题的前缀切错开时复用会改数值：退回逐题，并把这笔账记下来。"""
    runner = FakeRunner(prefixes=[(7, 8), (7, 9)])
    runner.answers_for(_request(2), MaskMode.PREFIX_NONCAUSAL)
    assert runner.shared == [] and len(runner.serial) == 2
    assert runner.decisions.shared_miss == 1 and runner.decisions.shared_hit == 0


def test_a_runner_without_the_primitive_falls_back_instead_of_failing() -> None:
    """没声明共享的 runner（h2o 今天就是这样）必须是逐题，而不是 `AttributeError`。"""
    runner = FakeRunner(shared=False)
    runner.answers_for(_request(2), MaskMode.PREFIX_NONCAUSAL)
    assert runner.shared == [] and len(runner.serial) == 2
    assert runner.decisions.shared_miss == 1, "有能力却没做是 miss；没这个能力不算"


def test_noncausal_full_has_no_prefix_to_share_and_sends_zero() -> None:
    """双向前向里前缀行与题面耦合：既不共享，也不记 miss，前缀按 0 传。"""
    runner = FakeRunner()
    runner.answers_for(_request(2), MaskMode.NONCAUSAL_FULL)
    assert runner.shared == [] and len(runner.serial) == 2
    assert all(prefix == 0 for _, prefix, _ in runner.serial)
    assert runner.decisions.shared_miss == 0


def test_an_over_long_prompt_is_refused_before_anything_runs() -> None:
    """超长是 422：编码那一关没过，前向与共享的账都不该动。"""
    runner = FakeRunner(max_len=3)
    with pytest.raises(RequestTooLongError):
        runner.answers_for(_request(2), MaskMode.PREFIX_NONCAUSAL)
    assert runner.serial == [] and runner.shared == []
    assert runner.decisions.snapshot() == DecisionMetrics().snapshot()


# ---------------------------------------------------------------- 跨请求合批


def test_a_batch_with_one_prefix_shares_it_across_requests() -> None:
    """决策负载的主形态：几条请求同一个 state，前缀只跑一遍、所有题一次尾巴相。"""
    runner = FakeRunner()
    out = runner.answers_batch([_request(1), _request(2)], MaskMode.PREFIX_NONCAUSAL)
    assert len(runner.shared) == 1 and len(runner.shared[0][1]) == 3
    assert runner.shared[0][0] == (7, 8)
    assert runner.tiled == [] and runner.serial == [], "共享命中之后不该再有别的前向"
    assert runner.decisions.batch_hit == 1
    assert [len(answers) for answers, _, _ in out] == [1, 2], "答案按请求切回去"
    assert [tokens for _, tokens, _ in out] == [4, 8], "token 也按请求切"


def test_independent_requests_tile_into_one_pass_when_the_shapes_fit() -> None:
    """前缀不同、形状装得下：块对角一条序列一个槽。"""
    runner = FakeRunner(prefixes=[(7, 8), (9, 8)], tiled=True)
    out = runner.answers_batch([_request(1), _request(1)], MaskMode.PREFIX_NONCAUSAL)
    assert runner.shared == [] and len(runner.tiled) == 1
    assert [sequence[0] for sequence in runner.tiled[0][0]] == [7, 9], "两条序列原样并排"
    assert runner.decisions.batch_hit == 1 and runner.decisions.batch_fallback == 0
    assert [answer.value for answers, _, _ in out for answer in answers] == ["甲", "甲"]


def test_a_batch_whose_shapes_do_not_fit_falls_back_to_serial() -> None:
    """装不进一趟：退回逐题，答案与逐条调用**逐位相同**，只记一笔 fallback。"""
    tiled = FakeRunner(prefixes=[(7, 8), (9, 8)], tiled=True, fits=False)
    serial = FakeRunner(prefixes=[(7, 8), (9, 8)])
    requests = [_request(1), _request(1)]
    batched = tiled.answers_batch(requests, MaskMode.PREFIX_NONCAUSAL)
    alone = [serial.answers_for(request, MaskMode.PREFIX_NONCAUSAL)[0] for request in requests]
    assert len(tiled.serial) == 2 and tiled.shared == []
    assert [answers for answers, _, _ in batched] == alone
    assert tiled.decisions.batch_fallback == 1 and tiled.decisions.batch_hit == 0


def test_a_single_sequence_through_the_batch_route_is_a_miss() -> None:
    """引擎把单条也送进这条路：一条题没有可合的东西，但那不叫"有机会没合上"。"""
    runner = FakeRunner()
    runner.answers_batch([_request(1)], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_miss == 1
    assert runner.decisions.batch_hit == 0 and runner.decisions.batch_fallback == 0
    assert len(runner.serial) == 1


def test_one_request_with_many_questions_is_a_batch_hit() -> None:
    """单条请求多题：引擎仍走批这条路，共用前缀就是真的合批。"""
    runner = FakeRunner()
    runner.answers_batch([_request(3)], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_hit == 1 and runner.decisions.batch_miss == 0
    assert len(runner.shared) == 1


def test_noncausal_full_does_not_share_across_requests() -> None:
    """双向掩码下前缀行与题面耦合：不共享（也不记 miss），但形状合适仍可块对角。"""
    runner = FakeRunner(tiled=True)
    runner.answers_batch([_request(1), _request(1)], MaskMode.NONCAUSAL_FULL)
    assert runner.shared == [] and len(runner.tiled) == 1
    assert runner.decisions.batch_hit == 1 and runner.decisions.shared_miss == 0


def test_noncausal_full_sends_zero_prefixes_into_the_tiled_batch() -> None:
    """`noncausal_full` 没有前缀语义：批里每条序列的前缀长度必须传 0。

    带着 `prefix_len` 进去，流水线会按 `prefix_noncausal` 收行 —— 同一个 mode 的批与逐条就
    用上了两种掩码（真 mude 上这批的概率差了 8.2e-2，判词恰好没翻）。
    """
    runner = FakeRunner(prefixes=[(7, 8), (9, 8)], tiled=True)
    runner.answers_batch([_request(1), _request(1)], MaskMode.NONCAUSAL_FULL)
    assert runner.tiled[0][2] == (0, 0), "全双向的批不该带前缀长度"


def test_a_prefix_mode_keeps_each_sequence_own_prefix() -> None:
    """有前缀语义的模式相反：批里每条序列仍按自己的前缀长度收行。"""
    runner = FakeRunner(prefixes=[(7, 8), (9, 8)], tiled=True)
    runner.answers_batch([_request(1), _request(1)], MaskMode.PREFIX_NONCAUSAL)
    assert runner.tiled[0][2] == (2, 2), "prefix_noncausal 的批要按每条自己的边界"


def test_a_batch_with_images_is_refused_before_encoding() -> None:
    """图按请求走塔与像素管线：批里不许有图，而且是编码之前就拒。"""
    runner = FakeRunner()
    requests = [
        _request(1),
        DecisionRequest(state="s",
                       questions=(Question(text="q0", options=("甲", "乙")),),
                       media=(object(),)),
    ]
    with pytest.raises(ConfigError):
        runner.answers_batch(requests, MaskMode.PREFIX_NONCAUSAL)
    assert runner.shared == [] and runner.serial == [] and runner.tiled == []
    assert runner.decisions.snapshot() == DecisionMetrics().snapshot(), "拒收之前不该记任何账"


def test_the_base_runner_reports_a_batch_limit_of_one() -> None:
    """没有合批原语就没有这条路：引擎不攒批、也不把单条送进 `answers_batch`。"""

    class Runner(DecisionRunner):
        supports_shared = True

    assert Runner().batch_limit == 1
