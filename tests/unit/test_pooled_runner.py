"""pooled 策略的接线（假链 + 假答面，不依赖 ANE / transformers）。

要钉住的是这条路的三个语义：读出只取**最后一行的最终 hidden**、标签行的点积**不经我们**再
归一化就交给厂商、以及模式与长度门禁。共享前缀（Stage 3）另有一条：前缀只算一遍、尾巴按题
切在同一个值上、没有原语或没有流水线的链退回逐题。真链与真 shim 的数值对拍在
`.scratch/g6_h2o_answers.py` 与 `.tmp/probe_stage3.py`。
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_request

from sysone.domain.attention import MaskMode
from sysone.domain.decision import DecisionRequest, Question
from sysone.engine.strategies.decision import shared_prefix
from sysone.engine.strategies.pooled import PooledRunner
from sysone.errors import ConfigError, NotReadyError, RequestTooLongError, UnsupportedModeError
from sysone.models.vendors.h2o_serve import Prepared, Serve

from test_h2o_serve import CONFIG, FakeContract, FakeShim, FakeTokenizer

HIDDEN = 4


class FakeChain:
    """只记账：`last_hidden` 返回给定的那一行，`hidden` 若被调用就露馅。"""

    def __init__(self, last: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0), width: int = HIDDEN):
        self.calls: list[tuple[tuple[int, ...], bool, int]] = []
        self.closed = False
        self.whole_readbacks = 0
        self._last = np.asarray(last, dtype=np.float32)
        self._width = width

    def last_hidden(self, tokens, *, prefix=0, causal=False, mm=None):
        self.calls.append((tuple(tokens), bool(causal), int(prefix)))
        return self._last

    def hidden(self, tokens, *, causal=False):
        """读出只要末行；整段读回是 87 ms 对 0.02 ms，这条路不该再被走到。"""
        self.whole_readbacks += 1
        raise AssertionError("pooled 的读出不该整段读回 hidden")

    def close(self):
        self.closed = True


class SharedChain(FakeChain):
    """会共享前缀的链：前缀一遍，每题一条尾巴，各交一份末行 hidden。"""

    can_share_prefix = True

    def __init__(self, last: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)):
        super().__init__(last=last)
        self.shared_calls: list[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], bool]] = []

    def last_hidden_shared(self, prefix, tails, *, causal=False, mm=None):
        self.shared_calls.append((
            tuple(prefix), tuple(tuple(tail) for tail in tails), bool(causal)))
        return [self._last for _ in tails]


class FloatChain(SharedChain):
    """有共享原语、但链上没有零拷贝流水线（`metal=False` 的 numpy 形态）。"""

    can_share_prefix = False


class _TailContract(FakeContract):
    """把题干也渲染进 prompt：多题共享前缀时，前缀之后才分叉。"""

    def question_prompt(self, state_text: str, spec: dict):
        prompt, names, descs, labels = super().question_prompt(state_text, spec)
        return f"{prompt} {spec['instructions']}", names, descs, labels


class _TailShim(FakeShim):
    Contract = _TailContract


def rows_by_label(label_ids):
    """一热行：`label_ids[k]` 取 hidden 的第 k 维 —— 于是 logits 就是 hidden[-1][:n]。"""
    matrix = np.zeros((len(label_ids), HIDDEN), dtype=np.float32)
    for index in range(len(label_ids)):
        matrix[index, index] = 1.0
    return matrix


def runner(*, last=(0.0, 0.0, 0.0, 1.0), max_len: int = 64, chain=None, serve_override=None,
           loaded: bool = True):
    chain = chain if chain is not None else FakeChain(last=last)
    if serve_override is None:
        subject = Serve(CONFIG, _TailShim(), FakeTokenizer())
    else:
        subject = serve_override
    runner = PooledRunner(lambda: chain, subject, rows_by_label, max_len=max_len)
    if loaded:
        runner.load_weights()
    return runner


def two_option_request() -> DecisionRequest:
    """两个选项的题：`rows_by_label` 于是只取 hidden 的前两维，平局与翻转都看得清。"""
    return DecisionRequest(
        state="record",
        questions=(Question(text="item", options=("A", "B"), kind="choice"),),
        request_id="two",
    )


def two_question_request() -> DecisionRequest:
    """一 state 两题：题干不同，所以前缀之后各有一条自己的尾巴。"""
    return DecisionRequest(
        state="record",
        questions=(
            Question(text="alpha", options=("A", "B"), kind="choice"),
            Question(text="beta ray", options=("A", "B"), kind="choice"),
        ),
        request_id="two-q",
    )


# ------------------------------------------------------------------- 答案
def test_answer_reads_the_last_row_through_the_vendor_answer():
    """末位 token 的 hidden × 标签行 → logits → 厂商判答案；整段读回不参与。"""
    subject = runner(last=(0.0, 0.0, 3.0, 1.0))
    request = two_option_request()
    answers, tokens, _ = subject.answers_for(request, MaskMode.CAUSAL)
    answer = answers[0]

    # logits = [0, 0]（前两维）→ 平局，取第一个标签 A
    assert answer.value == "A"
    assert answer.labels == ("A", "B")
    assert answer.probs == pytest.approx((0.5, 0.5), rel=1e-12)
    assert answer.confidence == pytest.approx(0.5, rel=1e-12)
    assert tokens == len(subject._serve.prepare(request.state, spec_for(request)).ids)


def spec_for(request: DecisionRequest) -> dict:
    from sysone.models.vendors.base import question_spec

    return question_spec(request.questions[0])


def test_answer_follows_the_row_the_chain_gives_it():
    """链给的那一行换个方向，答案就要跟着走 —— 这条钉住"点积用的是 `last_hidden`"。"""
    subject = runner(last=(0.0, 5.0, 0.0, 0.0))
    answers, _, _ = subject.answers_for(two_option_request(), MaskMode.NONCAUSAL_FULL)
    # logits = [0, 5] → B 胜出
    assert answers[0].value == "B"


def test_mode_is_forwarded_as_causal_and_prefix_is_refused():
    subject = runner()
    subject.answers_for(make_request(questions=1), MaskMode.CAUSAL)
    assert subject._chain.calls[-1][1] is True
    subject.answers_for(make_request(questions=1), MaskMode.NONCAUSAL_FULL)
    assert subject._chain.calls[-1][1] is False
    with pytest.raises(UnsupportedModeError):
        subject.answers_for(make_request(questions=1), MaskMode.PREFIX_NONCAUSAL)


# ------------------------------------------------------------------- 门禁
def test_prompt_longer_than_the_buffer_is_a_422_not_a_bare_error():
    subject = runner(max_len=8)
    with pytest.raises(RequestTooLongError):
        subject.answers_for(make_request(questions=1), MaskMode.CAUSAL)


def test_unloaded_runner_refuses_to_answer():
    subject = runner(loaded=False)
    with pytest.raises(ConfigError):
        subject.answers_for(make_request(questions=1), MaskMode.CAUSAL)


def test_restarted_warm_up_reports_bad_probabilities():
    """预热时概率不合法要按未就绪报出去，而不是变成第一道真实题的答案。"""

    class BrokenServe:
        def prepare(self, state, spec, *, images=0):
            return Prepared(ids=(1, 2), label_ids=(3, 4), labels=("A", "B"), names=("A", "B"),
                            descs=("a", "b"), kind="choice", temperature=1.0, threshold=None)

        def answer(self, prepared, logits):
            return "A", ("A", "B"), (float("nan"), float("nan")), float("nan")

    subject = runner(serve_override=BrokenServe())
    with pytest.raises(NotReadyError):
        subject.compile_or_warm_up()


def test_shutdown_closes_the_chain():
    subject = runner()
    chain = subject._chain
    subject.shutdown()
    assert chain.closed is True
    assert subject.check_health() is False


def test_answers_are_per_question_with_the_vendor_ordering():
    """链没有共享原语时，一个问题里的多道题各自整链前向一次。"""
    subject = runner()
    answers, _, _ = subject.answers_for(make_request(questions=2), MaskMode.CAUSAL)
    assert [answer.question_index for answer in answers] == [0, 1]
    assert all(answer.kind == "choice" for answer in answers)
    assert len(subject._chain.calls) == 2


# ----------------------------------------------------------------- 共享的账


def test_a_causal_request_with_two_questions_is_counted_as_a_shared_miss():
    """链没有共享前缀原语：因果下"一次 state + 多题"本可以共享，这条 runner 只能退回逐题。

    这个 miss 是 Stage 3 的立项依据 —— 真实评测流量 34 条请求全是 4–5 题、理想共享比是
    3.8–3.9×；不记的话 `/metrics` 里只看到"跑得慢"，看不到"本来能快"。
    """
    subject = runner()
    subject.answers_for(make_request(questions=2), MaskMode.CAUSAL)
    stats = subject.decision_stats()
    assert stats["shared_miss"] == 1 and stats["shared_hit"] == 0
    assert stats["supports_shared"] is False and stats["supports_batch"] is False
    assert subject.batch_limit == 1, "引擎据此不攒批、也不把单条送进批路"


def test_a_shared_chain_runs_the_state_prefix_once():
    """会共享的链：一次请求只跑一遍前缀，每题只跑自己的尾巴，答案与逐题相同。"""
    subject = runner(chain=SharedChain())
    request = two_question_request()
    answers, tokens, _ = subject.answers_for(request, MaskMode.CAUSAL)

    encoded = [subject.encode(request.state, question) for question in request.questions]
    prefix = shared_prefix(encoded)
    tails = [item.ids[len(prefix):] for item in encoded]
    assert subject._chain.shared_calls == [(prefix, tuple(tails), True)]
    assert subject._chain.calls == [], "共享那条路不该再逐题整趟"
    assert tokens == sum(len(item.ids) for item in encoded)
    assert subject.decisions.shared_hit == 1 and subject.decisions.shared_miss == 0
    # `/metrics` 与能力位：这条链现在真的会共享，读的人不该再看到 False。
    assert subject.supports_shared is True
    assert subject.decision_stats()["supports_shared"] is True

    # 同一个 `last` 行喂给两臂：答案必须一致（共享只该省前向，不该改数值）。
    serial = runner(chain=FakeChain())
    reference, _, _ = serial.answers_for(request, MaskMode.CAUSAL)
    assert [(item.value, item.probs) for item in answers] == \
           [(item.value, item.probs) for item in reference]


def test_the_shared_tails_start_after_the_common_prefix():
    """切点与判定是同一个值：尾巴是"整条 prompt 去掉那个前缀"，不是另一套切法。"""
    subject = runner(chain=SharedChain())
    request = two_question_request()
    subject.answers_for(request, MaskMode.CAUSAL)
    prefix, tails, _ = subject._chain.shared_calls[0]
    encoded = [subject.encode(request.state, question) for question in request.questions]
    assert prefix == shared_prefix(encoded)
    assert [item.ids[len(prefix):] for item in encoded] == list(tails)


def test_noncausal_full_has_no_prefix_to_share_so_no_miss():
    """双向前向里前缀行与题面耦合：那不是"漏做共享"，是没这个语义。"""
    subject = runner(chain=SharedChain())
    subject.answers_for(two_question_request(), MaskMode.NONCAUSAL_FULL)
    assert subject._chain.shared_calls == []
    assert len(subject._chain.calls) == 2, "全双向逐题整趟，各带 prefix=0"
    assert subject.decisions.shared_miss == 0


def test_a_lone_question_is_not_a_miss():
    subject = runner(chain=SharedChain())
    subject.answers_for(make_request(questions=1), MaskMode.CAUSAL)
    assert subject._chain.shared_calls == []
    assert subject.decisions.shared_miss == 0 and subject.decisions.shared_hit == 0


def test_a_chain_without_a_prefix_buffer_goes_serial():
    """有共享原语的**名字**、但链没有零拷贝流水线（numpy 形态）：声明能力会把 500 留给它。"""
    subject = runner(chain=FloatChain())
    assert subject.supports_shared is False
    subject.answers_for(two_question_request(), MaskMode.CAUSAL)
    assert subject._chain.shared_calls == []
    assert len(subject._chain.calls) == 2
    assert subject.decisions.shared_miss == 1


# ------------------------------------------------------------- 块对角的账


class BatchedChain(SharedChain):
    """会块对角的链：N 条独立序列一个 pass，各交一份末行 hidden。"""

    def __init__(self, last: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0), *, fits: bool = True):
        super().__init__(last=last)
        self.batched_calls: list[tuple[tuple[tuple[int, ...], ...], tuple[int, ...], bool]] = []
        self._fits = fits

    def last_hidden_batched(self, sequences, *, prefixes=(), causal=False):
        self.batched_calls.append((
            tuple(tuple(sequence) for sequence in sequences),
            tuple(int(prefix) for prefix in prefixes),
            bool(causal),
        ))
        return [self._last for _ in sequences] if self._fits else None


class FloatBatchedChain(BatchedChain):
    """有块对角原语的**名字**、但链没有零拷贝流水线（`metal=False` 的 numpy 形态）。"""

    can_share_prefix = False


def independent_requests() -> tuple[DecisionRequest, DecisionRequest]:
    """两条 state 首词就不同的请求：公共前缀是 0，所以走块对角而不是共享前缀。"""
    return (make_request(questions=1, state="alpha record"),
            make_request(questions=1, state="omega record"))


def test_batch_capability_follows_the_chain_primitive():
    """能力位按链问：有 `last_hidden_batched` 才声明；numpy 形态仍不声明。

    与 `supports_shared` 同一个口径 —— 声明了却调不到是一条 500，退回串行只是一次没有收益的
    前向。
    """
    assert runner().supports_batch is False
    assert runner(chain=BatchedChain()).supports_batch is True
    assert runner(chain=FloatBatchedChain()).supports_batch is False


def test_the_batch_primitive_does_not_turn_batching_on():
    """原语到位 ≠ 默认开：`batch_limit` 仍是 1，引擎据此不攒批、也不把单条送进批路。

    真要对 h2o 开合批，改的是这个上限（且先补"垫列钳位之后连跑 N 遍"的形状矩阵回归），
    不是 `supports_batch`。
    """
    subject = runner(chain=BatchedChain(), max_len=16384)
    assert subject.batch_limit == 1
    assert subject.decision_stats()["supports_batch"] is True


def test_two_independent_requests_share_one_pass():
    """两条前缀不同的请求：走块对角，一条序列一个槽，前缀长度按 0 传。"""
    chain = BatchedChain()
    subject = runner(chain=chain, max_len=16384)
    first, second = independent_requests()
    results = subject.answers_batch((first, second), MaskMode.CAUSAL)

    assert len(results) == 2
    assert len(chain.batched_calls) == 1, "两条请求应该合成一趟"
    sequences, prefixes, causal = chain.batched_calls[0]
    assert len(sequences) == 2
    assert prefixes == (0, 0), "h2o 的答面给不出前缀切点，按 0 传"
    assert causal is True
    assert chain.calls == [], "批那条路不该再逐题整趟"
    assert subject.decisions.batch_hit == 1 and subject.decisions.batch_fallback == 0


def test_a_batch_that_does_not_fit_falls_back_to_serial():
    """形状挤不进一趟时链返回 `None`：退回逐题，答案与逐条调用逐位相同。"""
    subject = runner(chain=BatchedChain(fits=False), max_len=16384)
    first, second = independent_requests()
    batched = subject.answers_batch((first, second), MaskMode.CAUSAL)
    assert subject.decisions.batch_fallback == 1 and subject.decisions.batch_hit == 0

    reference = runner()
    for request, (answers, tokens, _) in zip((first, second), batched):
        expected, expected_tokens, _ = reference.answers_for(request, MaskMode.CAUSAL)
        assert [(item.value, item.probs) for item in answers] == \
               [(item.value, item.probs) for item in expected]
        assert tokens == expected_tokens


def test_a_chain_without_the_batch_primitive_goes_serial():
    """没有原语的链：不去试批，直接逐题 —— 记一次 fallback，而不是假装成功。"""
    subject = runner()
    first, second = independent_requests()
    subject.answers_batch((first, second), MaskMode.CAUSAL)
    assert len(subject._chain.calls) == 2
    assert subject.decisions.batch_fallback == 1 and subject.decisions.batch_hit == 0


def test_a_shared_prefix_wins_over_block_diagonal():
    """两条请求 state 相同：共享前缀那条更便宜，不该按块对角排。"""
    chain = BatchedChain()
    subject = runner(chain=chain, max_len=16384)
    same = make_request(questions=1, state="same record")
    subject.answers_batch((same, same), MaskMode.CAUSAL)
    assert chain.batched_calls == []
    assert chain.shared_calls, "同前缀应该走共享那条"
    assert subject.decisions.batch_hit == 1
