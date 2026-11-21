"""环境变量只登记四项，解析规则固定。"""

from __future__ import annotations

import pytest

from sysone import env
from sysone.errors import ConfigError

_DEFAULTS = {
    "SYS1_LOG_LEVEL": "info",
    "DECISION_API_KEY": "",
    "SYS1_ARTIFACTS_DIR": "",
    "SYS1_WEIGHTS_DIR": "",
}


def test_default_is_used_when_unset() -> None:
    assert env.read({}) == _DEFAULTS


def test_value_is_read() -> None:
    assert env.read(
        {"SYS1_LOG_LEVEL": "debug", "DECISION_API_KEY": "k",
         "SYS1_ARTIFACTS_DIR": "/var/artifacts", "SYS1_WEIGHTS_DIR": "/var/weights"}
    ) == {
        "SYS1_LOG_LEVEL": "debug",
        "DECISION_API_KEY": "k",
        "SYS1_ARTIFACTS_DIR": "/var/artifacts",
        "SYS1_WEIGHTS_DIR": "/var/weights",
    }


def test_empty_string_falls_back_to_default() -> None:
    assert env.read({"SYS1_LOG_LEVEL": ""})["SYS1_LOG_LEVEL"] == "info"


def test_bool_and_int_parsing_rules_still_hold() -> None:
    """解析器仍支持 bool/int，即便 M0 只登记了一个 str 项。"""
    spec_bool = env.EnvSpec("X_BOOL", "bool", False, "")
    assert env._parse(spec_bool, "yes") is True
    assert env._parse(spec_bool, "off") is False
    with pytest.raises(ConfigError):
        env._parse(spec_bool, "maybe")
    spec_int = env.EnvSpec("X_INT", "int", 0, "")
    with pytest.raises(ConfigError):
        env._parse(spec_int, "abc")
