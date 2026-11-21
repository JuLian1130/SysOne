"""聚合配置：校验与产物键。"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

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


# ---- 产物落点：默认必须持久（首启 26 s 与 2 s 的差别全在它身上）-------------------------


def test_artifacts_dir_defaults_to_a_persistent_platform_cache() -> None:
    from sysone.config import StorageConfig
    from sysone.config.storage import default_artifacts_dir

    resolved = StorageConfig().resolved_artifacts_dir()
    assert resolved == default_artifacts_dir()
    assert resolved.is_absolute()
    assert resolved != Path(tempfile.gettempdir())
    assert "sysone" in resolved.parts


def test_artifacts_dir_keeps_what_the_caller_gave() -> None:
    from sysone.config import StorageConfig

    assert StorageConfig(artifacts_dir=Path("/var/art")).resolved_artifacts_dir() == Path("/var/art")


def test_weights_dir_defaults_to_a_persistent_platform_cache() -> None:
    from sysone.config import StorageConfig
    from sysone.config.storage import default_weights_dir

    resolved = StorageConfig().resolved_weights_dir()
    assert resolved == default_weights_dir()
    assert resolved.is_absolute()
    assert resolved != Path(tempfile.gettempdir())
    assert "sysone" in resolved.parts


def test_weights_dir_keeps_what_the_caller_gave() -> None:
    from sysone.config import StorageConfig

    assert StorageConfig(weights_dir=Path("/var/weights")).resolved_weights_dir() == Path("/var/weights")
