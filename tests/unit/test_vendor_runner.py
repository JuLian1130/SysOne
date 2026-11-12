"""vendor 策略的管线测试（用假桥，不依赖 torch）。

真实厂商桥只在装了 torch + int8 检查点时才被 import；这里验证的是 Engine 在
`answers_for` 存在时会把模板、前向与读出整段交给桥，并正确回填 usage/provenance。
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import pytest
from helpers import load_tiny_descriptor, make_request

from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.work import CompilationTimes
from sysone.engine.engine import Engine
from sysone.engine.strategies.vendor import VendorRunner
from sysone.errors import NotReadyError, UnsupportedModeError
from sysone.models.vendors.base import VendorAnswer


class FakeBridge:
    name = "fake"

    def __init__(self, *, trained: Sequence[MaskMode], fail_mode: MaskMode | None = None) -> None:
        self.trained = tuple(trained)
        self.fail_mode = fail_mode
        self.loaded = False
        self.calls: list[tuple[str, dict, MaskMode]] = []
        self.artifact_bytes_value = 1024

    def load(self) -> None:
        self.loaded = True

    def compile_or_warm_up(self) -> CompilationTimes:
        return CompilationTimes(load_s=0.0, compile_s=0.0, warmup_s=0.0)

    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        if mode is self.fail_mode:
            raise NotImplementedError("这种模式厂商不跑")
        self.calls.append((state, question, mode))
        options = tuple(question["criteria"].keys()) or ("A", "B")
        probs = tuple(1.0 / len(options) for _ in options)
        return VendorAnswer(
            value=options[0],
            labels=options,
            probs=probs,
            input_tokens=7,
            readout_s=0.0,
        )

    def artifact_bytes(self) -> int:
        return self.artifact_bytes_value

    def close(self) -> None:
        self.loaded = False


def _config(mode: MaskMode | None = None) -> EngineConfig:
    return EngineConfig(
        model=ModelSelection(name="fake", dir="."),
        scheduler=SchedulerConfig(attention_mode=mode),
        service=ServiceConfig(debug_timings=True),
    )


def _engine(bridge: FakeBridge, config: EngineConfig | None = None) -> Engine:
    from sysone.devices.reference.encoding import CharEncoder

    descriptor = load_tiny_descriptor()
    return Engine(
        descriptor=descriptor,
        runner=VendorRunner(bridge),
        encoder=CharEncoder(vocab=descriptor.vocab),
        config=config or _config(),
    )


def test_vendor_path_answers_every_question() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    engine = _engine(bridge)
    started = engine.start()
    assert started.total_s == 0.0
    assert bridge.loaded

    request = make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=2)
    result = engine.run(request)

    assert len(result.answers) == 2
    assert result.answers[0].question_index == 0
    assert result.answers[1].question_index == 1
    for answer in result.answers:
        assert answer.labels == ("A", "B", "C", "D")
        assert abs(sum(answer.probs) - 1.0) < 1e-9
    assert result.usage.input_tokens == 14
    assert result.usage.prefill_calls == 2
    assert result.provenance.vendor_bridge == "fake"
    assert result.timings is not None
    assert len(bridge.calls) == 2


def test_vendor_mode_gate_uses_descriptor_supported_modes() -> None:
    descriptor = load_tiny_descriptor()
    narrow = dataclasses.replace(
        descriptor,
        supported_modes=frozenset({MaskMode.PREFIX_NONCAUSAL}),
        default_mode=MaskMode.PREFIX_NONCAUSAL,
        trained_modes=frozenset({MaskMode.PREFIX_NONCAUSAL}),
    )
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    from sysone.devices.reference.encoding import CharEncoder

    engine = Engine(
        descriptor=narrow,
        runner=VendorRunner(bridge),
        encoder=CharEncoder(vocab=narrow.vocab),
        config=_config(),
    )
    engine.start()
    with pytest.raises(UnsupportedModeError) as info:
        engine.run(make_request(mode=MaskMode.NONCAUSAL_FULL))
    assert info.value.context["mode"] == MaskMode.NONCAUSAL_FULL.value


def test_vendor_unsupported_mode_from_bridge_becomes_400() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,), fail_mode=MaskMode.PREFIX_NONCAUSAL)
    engine = _engine(bridge)
    engine.start()
    with pytest.raises(UnsupportedModeError) as info:
        engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL))
    assert info.value.status == 400
    assert info.value.context["bridge"] == "fake"


def test_vendor_runner_requires_load_first() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    runner = VendorRunner(bridge)
    with pytest.raises(Exception) as info:
        runner.answers_for(make_request(), MaskMode.CAUSAL)
    assert "尚未装载" in str(info.value)


def test_vendor_runner_artifacts_and_close() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    runner = VendorRunner(bridge)
    assert runner.artifact_bytes() == 1024
    runner.load_weights()
    assert runner.check_health() is True
    runner.shutdown()
    assert bridge.loaded is False


def test_vendor_runner_delegates_warmup() -> None:
    runner = VendorRunner(FakeBridge(trained=(MaskMode.CAUSAL,)))
    times = runner.compile_or_warm_up()
    assert times.total_s == 0.0
    assert times.load_s == 0.0


def test_engine_start_surfaces_bridge_warmup_failure() -> None:
    """预热抛 NotReadyError 时启动就该失败，而不是让第一道题拿到坏概率。"""

    class BrokenWarmup(FakeBridge):
        def compile_or_warm_up(self) -> CompilationTimes:
            raise NotReadyError("预热概率非法", bridge="fake")

    engine = _engine(BrokenWarmup(trained=(MaskMode.CAUSAL,)))
    with pytest.raises(NotReadyError):
        engine.start()


def test_question_spec_carries_options_as_criteria() -> None:
    from sysone.models.vendors.base import question_spec

    question = Question(text="选一个", options=("甲", "乙"))
    spec = question_spec(question)
    assert spec["type"] == "choice"
    assert spec["instructions"] == "选一个"
    assert spec["criteria"] == {"甲": "甲", "乙": "乙"}


def test_wire_answer_shapes() -> None:
    from sysone.models.vendors.base import wire_answer

    value, labels, probs = wire_answer(
        {"type": "choice", "choice": "B", "probabilities": {"A": 0.25, "B": 0.75}}
    )
    assert value == "B"
    assert labels == ("A", "B")
    assert probs == (0.25, 0.75)

    value, labels, probs = wire_answer({"type": "noul", "noul": 0.8})
    assert value == "yes"
    assert labels == ("yes", "no")

    value, labels, probs = wire_answer(
        {"type": "score", "score": 2.5, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}
    )
    assert value == "2.500"
    assert len(labels) == 3


def test_engine_config_is_frozen() -> None:
    config = _config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.service.port = 1  # type: ignore[misc]
