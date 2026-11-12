"""模型家族注册表：重复注册、未知家族、已注册清单。"""

from __future__ import annotations

import pytest

from sysone.errors import RegistrationError, UnknownModelError
from sysone.models import families  # noqa: F401  触发注册
from sysone.models.registry import family, families as registered, register_family

from helpers import load_tiny_descriptor


def _noop_plan(descriptor, config):
    return None  # 注册表不调用它


def test_three_families_registered() -> None:
    assert {"h2o", "mude", "torchcast"} <= set(registered())


def test_unknown_family_reports_available() -> None:
    with pytest.raises(UnknownModelError) as caught:
        family("nope")
    assert "available" in caught.value.context


def test_double_registration_is_an_error() -> None:
    with pytest.raises(RegistrationError):
        register_family("temp", lambda path: load_tiny_descriptor(), _noop_plan)
        register_family("temp", lambda path: load_tiny_descriptor(), _noop_plan)
