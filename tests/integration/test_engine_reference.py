"""tiny 模型端到端：概率、token 计数、模式门禁、内容限制。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from helpers import make_request, make_tiny
from sysone.config import EngineConfig, ModelSelection, SchedulerConfig
from sysone.devices.reference import CharEncoder, ReferenceRunner
from sysone.domain.attention import MaskMode
from sysone.domain.decision import MediaRef
from sysone.engine.engine import Engine
from sysone.errors import RequestTooLongError, UnsupportedMediaError, UnsupportedModeError


def _engine(tmp_path: Path, **config_overrides) -> tuple[Engine, object]:
    descriptor, store = make_tiny(tmp_path)
    config = EngineConfig(model=ModelSelection(name="tiny", dir=tmp_path))
    if config_overrides:
        config = dataclasses.replace(config, **config_overrides)
    encoder = CharEncoder(descriptor.vocab)
    engine = Engine(descriptor, ReferenceRunner(descriptor, store), encoder, config)
    engine.start()
    return engine, encoder


def _expected_tokens(encoder, request) -> int:
    prompt = encoder.encode(request)
    return len(prompt.prefix_tokens) + sum(len(tail) for tail in prompt.tails)


def test_end_to_end_probs(tmp_path: Path) -> None:
    engine, encoder = _engine(tmp_path)
    request = make_request(questions=2)
    result = engine.run(request)
    assert len(result.answers) == 2
    for index, answer in enumerate(result.answers):
        assert answer.question_index == index
        assert answer.labels == ("A", "B", "C", "D")
        assert answer.value in answer.labels
        assert sum(answer.probs) == pytest.approx(1.0, abs=1e-6)
    assert result.usage.prefill_calls == 2
    assert result.usage.input_tokens == _expected_tokens(encoder, request)
    assert result.provenance.attention_mode is MaskMode.PREFIX_NONCAUSAL
    assert result.provenance.attention_trained is True


def test_modes_change_probabilities(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path)
    causal = engine.run(make_request(mode=MaskMode.CAUSAL)).answers[0]
    prefix = engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL)).answers[0]
    assert causal.probs != prefix.probs
    assert engine.run(make_request(mode=MaskMode.CAUSAL)).provenance.attention_trained is False


def test_unsupported_mode_is_400(tmp_path: Path) -> None:
    descriptor, store = make_tiny(tmp_path)
    narrowed = dataclasses.replace(
        descriptor,
        supported_modes=frozenset({MaskMode.NONCAUSAL_FULL}),
        trained_modes=frozenset({MaskMode.NONCAUSAL_FULL}),
        default_mode=MaskMode.NONCAUSAL_FULL,
    )
    config = EngineConfig(model=ModelSelection(name="tiny", dir=tmp_path))
    engine = Engine(narrowed, ReferenceRunner(narrowed, store), CharEncoder(narrowed.vocab), config)
    engine.start()
    with pytest.raises(UnsupportedModeError) as caught:
        engine.run(make_request(mode=MaskMode.CAUSAL))
    assert caught.value.status == 400


def test_media_is_rejected_with_the_reason_that_it_is_the_path_not_the_request(tmp_path: Path) -> None:
    """reference 执行器没有塔：拒收的理由要说"这条执行路径"，不是"这个模型"。"""
    engine, _ = _engine(tmp_path)
    request = dataclasses.replace(make_request(), media=(MediaRef(kind="image", data=b"x"),))
    with pytest.raises(UnsupportedMediaError) as caught:
        engine.run(request)
    assert caught.value.status == 422
    assert "执行路径" in caught.value.message, caught.value.message


def test_long_request_is_rejected(tmp_path: Path) -> None:
    engine, _ = _engine(tmp_path, scheduler=SchedulerConfig(max_len=2))
    with pytest.raises(RequestTooLongError):
        engine.run(make_request())
