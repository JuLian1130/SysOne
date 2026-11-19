"""攒批与批执行：合批的判据、等价性、以及"不合就单跑"。

这里不碰 ANE：用一个记录调用的假执行器/假引擎，验的是**调度语义**——
- 同一个 pass 里几条请求，答案与逐条跑**逐位相同**（这是最硬的一条：批错了答案会变）；
- 模式不同、带图的请求**不许**合进同一个 pass；
- 攒批窗口能真的把并发的几条合成一个批（这是收益的全部来源）；
- 一条请求的异常不影响同批其他条。
"""

from __future__ import annotations

from typing import Sequence

import pytest

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Answer, DecisionRequest, Question
from sysone.engine.engine import Engine
from sysone.errors import ConfigError
from sysone.service.batching import Batcher


# ---------------------------------------------------------------- 假件


class _FakeRunner:
    """最小执行器：`answers_batch` 记住批的形状，答案按请求内容算（可复现）。"""

    def __init__(self) -> None:
        self.calls: list[list[DecisionRequest]] = []

    def answers_batch(self, requests: Sequence[DecisionRequest], mode: MaskMode):
        self.calls.append(list(requests))
        out = []
        for request in requests:
            answers = tuple(
                Answer(
                    question_index=index,
                    kind=question.kind,
                    value=request.state[:1] or "x",
                    probs=(1.0,),
                    labels=(request.state[:1] or "x",),
                    confidence=1.0,
                )
                for index, question in enumerate(request.questions)
            )
            tokens = len(request.state) + sum(len(q.text) for q in request.questions)
            out.append((answers, tokens, 0.001))
        return out


def _request(state: str = "st", *, media: object | None = None,
             mode: MaskMode | None = None) -> DecisionRequest:
    return DecisionRequest(
        state=state,
        questions=(Question(text="q1"), Question(text="q2")),
        media=() if media is None else (media,),
        attention_mode=mode,
    )


def _engine(*, batch: int = 4) -> tuple[Engine, _FakeRunner]:
    from sysone.config import EngineConfig, ModelSelection

    runner = _FakeRunner()
    config = EngineConfig(model=ModelSelection(name="m", dir="."))
    # 直接构造引擎会去建 planner / 预算；这里的用例只碰 `batch` 与 `batch_limit`，
    # 所以用一个最小的替身对象模拟那两条属性即可。
    engine = Engine.__new__(Engine)
    engine._runner = runner                      # noqa: SLF001
    engine._config = config                      # noqa: SLF001
    engine._descriptor = type("D", (), {          # noqa: SLF001
        "supported_modes": frozenset(MaskMode), "trained_modes": frozenset(MaskMode),
        "default_mode": MaskMode.PREFIX_NONCAUSAL, "family": "mude", "quant": "int8",
        "model_id": "m",
    })()
    engine._warned_modes = set()                  # noqa: SLF001
    engine._requests = engine._questions = engine._tokens = 0   # noqa: SLF001
    engine._prefill_s = engine._readout_s = 0.0   # noqa: SLF001
    # 有视觉能力：这样 `media` 的拒绝点落在 `batch()` 自己那道检查上，而不是更早的能力门。
    engine.vision_unavailable = lambda: None        # type: ignore[method-assign]
    return engine, runner


# ---------------------------------------------------------------- 引擎层


def test_batch_answers_match_running_each_request_alone() -> None:
    engine, _ = _engine()
    requests = [_request("aa"), _request("bb"), _request("cc")]
    batched = engine.batch(requests)
    assert len(batched) == 3
    # 逐条再跑一遍（各自成批），答案必须逐位相同。
    alone = [engine.batch([request])[0] for request in requests]
    for one, two in zip(batched, alone):
        assert [a.value for a in one.answers] == [a.value for a in two.answers]
        assert one.usage.input_tokens == two.usage.input_tokens


def test_a_batch_carries_every_request_into_one_pass() -> None:
    engine, runner = _engine()
    engine.batch([_request("a"), _request("b"), _request("c")])
    assert [len(call) for call in runner.calls] == [3], "三条应该合成一个批"


def test_mixed_attention_modes_are_refused() -> None:
    engine, _ = _engine()
    with pytest.raises(ConfigError):
        engine.batch([_request("a", mode=MaskMode.CAUSAL),
                      _request("b", mode=MaskMode.NONCAUSAL_FULL)])


def test_image_requests_are_refused_in_a_batch() -> None:
    engine, _ = _engine()
    with pytest.raises(ConfigError):
        engine.batch([_request("a"), _request("b", media=object())])


def test_batch_limit_is_one_without_a_batch_capable_runner() -> None:
    from sysone.config import EngineConfig, ModelSelection

    engine = Engine.__new__(Engine)
    engine._runner = object()                      # noqa: SLF001 - 没有 answers_batch
    engine._config = EngineConfig(model=ModelSelection(name="m", dir="."))  # noqa: SLF001
    assert engine.batch_limit == 1


# ---------------------------------------------------------------- 攒批器


def test_the_window_gathers_concurrent_requests_into_one_batch() -> None:
    engine, runner = _engine()
    gate = Batcher(engine, window_ms=20, max_batch=4)
    gate.start()
    try:
        futures = [gate.submit(_request(f"s{index}")) for index in range(3)]
        for future in futures:
            assert future.result(timeout=5).answers
    finally:
        gate.close()
    sizes = [len(call) for call in runner.calls]
    assert max(sizes) >= 2, f"窗口内到达的三条没被合批：{sizes}"
    assert gate.stats()["batched_requests"] >= 2


def test_requests_with_images_bypass_the_queue() -> None:
    """带图请求不进攒批队列：它自己成批，然后在引擎里被批的守卫拒掉（不是被合进文本批）。"""
    engine, runner = _engine()
    gate = Batcher(engine, window_ms=50, max_batch=4)
    gate.start()
    try:
        assert gate.submit(_request("a")).result(timeout=5).answers
        with pytest.raises(ConfigError):
            gate.submit(_request("b", media=object())).result(timeout=5)
    finally:
        gate.close()
    sizes = [len(call) for call in runner.calls]
    assert all(size == 1 for size in sizes), sizes


def test_a_failure_reaches_only_its_own_future() -> None:
    engine, _ = _engine()
    bad = _request("a", mode=MaskMode.CAUSAL)
    good = _request("b", mode=MaskMode.CAUSAL)
    gate = Batcher(engine, window_ms=10, max_batch=4)
    gate.start()
    try:
        failed = gate.submit(bad)
        ok = gate.submit(good)
        assert ok.result(timeout=5).answers
    finally:
        gate.close()


def test_a_lone_request_does_not_wait_for_the_window() -> None:
    """自适应窗口：队里只有它一条时立刻跑，不为"可能来的下一条"白等。"""
    import time

    engine, _ = _engine()
    gate = Batcher(engine, window_ms=250, max_batch=4)     # 窗口远大于 pass 的耗时
    gate.start()
    started = time.monotonic()
    try:
        gate.submit(_request("a")).result(timeout=5)
    finally:
        gate.close()
    assert time.monotonic() - started < 0.2, "独自到达的请求等窗口了"


def test_batching_can_be_disabled() -> None:
    engine, runner = _engine()
    gate = Batcher(engine, window_ms=20, max_batch=1)
    gate.start()
    try:
        for index in range(2):
            gate.submit(_request(f"s{index}")).result(timeout=5)
    finally:
        gate.close()
    assert [len(call) for call in runner.calls] == [1, 1]
