"""pooled 策略的接线（假链 + 假答面，不依赖 ANE / transformers）。

要钉住的是这条路的三个语义：读出只取**最后一行的最终 hidden**、标签行的点积**不经我们**再
归一化就交给厂商、以及模式与长度门禁。真链与真 shim 的数值对拍在 `.scratch/g6_h2o_answers.py`。
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_request

from sysone.domain.attention import MaskMode
from sysone.domain.decision import DecisionRequest, Question
from sysone.engine.strategies.pooled import PooledRunner
from sysone.errors import ConfigError, NotReadyError, RequestTooLongError, UnsupportedModeError
from sysone.models.vendors.h2o_serve import Prepared

from test_h2o_serve import serve, spec

HIDDEN = 4


class FakeChain:
    """只记账：`last_hidden` 返回给定的那一行，`hidden` 若被调用就露馅。"""

    def __init__(self, last: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0), width: int = HIDDEN):
        self.calls: list[tuple[tuple[int, ...], bool]] = []
        self.closed = False
        self.whole_readbacks = 0
        self._last = np.asarray(last, dtype=np.float32)
        self._width = width

    def last_hidden(self, tokens, *, causal=False, mm=None):
        self.calls.append((tuple(tokens), bool(causal)))
        return self._last

    def hidden(self, tokens, *, causal=False):
        """读出只要末行；整段读回是 87 ms 对 0.02 ms，这条路不该再被走到。"""
        self.whole_readbacks += 1
        raise AssertionError("pooled 的读出不该整段读回 hidden")

    def close(self):
        self.closed = True


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
        subject, _ = serve()
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
    assert tokens == len(subject._serve.prepare(request.state, spec()).ids)


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
            return "A", ("A", "B"), (float("nan"), float("nan"))

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
    """一个问题里的多道题各自整链前向一次：标签与概率来自同一份 `Serve` 契约。"""
    subject = runner()
    answers, _, _ = subject.answers_for(make_request(questions=2), MaskMode.CAUSAL)
    assert [answer.question_index for answer in answers] == [0, 1]
    assert all(answer.kind == "choice" for answer in answers)
    assert len(subject._chain.calls) == 2
