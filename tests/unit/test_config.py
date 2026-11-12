"""聚合配置：校验与产物键。"""

from __future__ import annotations

import dataclasses

import pytest

from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from sysone.domain.attention import MaskMode
from sysone.errors import ConfigError


def _config(**overrides) -> EngineConfig:
    base = EngineConfig(model=ModelSelection(name="tiny", dir="."))
    return dataclasses.replace(base, **overrides)


def test_cache_key_is_stable_and_ignores_service() -> None:
    first = _config().cache_key()
    assert first == _config().cache_key()
    assert _config(service=ServiceConfig(port=9999)).cache_key() == first


def test_cache_key_changes_with_shape_flags() -> None:
    first = _config().cache_key()
    assert _config(scheduler=SchedulerConfig(tile=128)).cache_key() != first
    assert _config(scheduler=SchedulerConfig(attention_mode=MaskMode.CAUSAL)).cache_key() != first


def test_validate_rejects_bad_scheduler() -> None:
    with pytest.raises(ConfigError):
        _config(scheduler=SchedulerConfig(ring=0)).validate()


def test_validate_rejects_empty_model() -> None:
    with pytest.raises(ConfigError):
        EngineConfig(model=ModelSelection(name="", dir=".")).validate()
