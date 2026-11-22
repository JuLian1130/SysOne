"""`sysone` 根 logger：级别、JSON 形状，以及 CLI 入口有没有把它接上。"""

from __future__ import annotations

import io
import json
import logging

import pytest

from sysone import cli
from sysone.errors import ConfigError
from sysone.logging import configure, get_logger


@pytest.fixture(autouse=True)
def _restore_logger():
    """`configure` 改的是进程级状态：每条用例跑完都要还回去。"""
    root = logging.getLogger("sysone")
    handlers, level, propagate = root.handlers[:], root.level, root.propagate
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    root.propagate = propagate


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_configure_writes_json_at_the_requested_level() -> None:
    stream = io.StringIO()
    configure(level="warning", stream=stream)
    logger = get_logger("engine.engine")
    logger.info("隐藏")
    logger.warning("显示", extra={"data": {"mode": "causal"}})

    records = _records(stream)
    assert len(records) == 1
    assert records[0]["level"] == "warning"
    assert records[0]["logger"] == "sysone.engine.engine"
    assert records[0]["msg"] == "显示"
    assert records[0]["data"] == {"mode": "causal"}


def test_configure_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("SYS1_LOG_LEVEL", "debug")
    configure(stream=io.StringIO())
    assert logging.getLogger("sysone").level == logging.DEBUG


def test_an_unknown_level_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        configure(level="loud", stream=io.StringIO())


def test_main_configures_logging_before_it_runs_a_command(monkeypatch, capsys) -> None:
    """CLI 一进来就接线：不给这一句，`SYS1_LOG_LEVEL` 就算设了也无效。"""
    calls: list[bool] = []
    monkeypatch.setattr(cli, "configure", lambda: calls.append(True))

    assert cli.main(["doctor", "--platform", "reference"]) == 0
    capsys.readouterr()
    assert calls == [True]
