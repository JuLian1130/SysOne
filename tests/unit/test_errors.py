"""错误层次：码表、状态码、上下文字符串。"""

from __future__ import annotations

import pytest

from sysone import errors


@pytest.mark.parametrize(
    ("cls", "code", "status"),
    [
        (errors.UnknownModelError, "unknown_model", 400),
        (errors.UnsupportedModeError, "unsupported_mode", 400),
        (errors.QueueFullError, "queue_full", 429),
        (errors.NotReadyError, "not_ready", 503),
        (errors.RequestTooLongError, "request_too_long", 422),
        (errors.UnsupportedMediaError, "unsupported_media", 422),
    ],
)
def test_code_and_status(cls, code: str, status: int) -> None:
    exc = cls("boom", k=1)
    assert exc.code == code
    assert exc.status == status
    assert "k=1" in str(exc)


def test_base_defaults(tmp_path) -> None:
    exc = errors.SysOneError("base")
    assert exc.code == "internal"
    assert exc.status == 500
    assert exc.context == {}
