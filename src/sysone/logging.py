"""结构化日志：一行一条 JSON。

字段固定为 `ts / level / logger / msg`，调用方用 `logger.info("...", extra={"data": {...}})`
附带结构化上下文；异常自动带 `exc`。
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import IO

from . import env
from .errors import ConfigError


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        data = getattr(record, "data", None)
        if data:
            payload["data"] = data
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(level: str | None = None, stream: IO[str] | None = None) -> None:
    """配置 `sysone` 根 logger；`level=None` 时读环境变量。"""
    resolved = (level or str(env.read()["SYS1_LOG_LEVEL"])).upper()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("sysone")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """取 `sysone.<name>` logger（`name` 已带前缀则原样返回）。"""
    if name.startswith("sysone"):
        return logging.getLogger(name)
    return logging.getLogger(f"sysone.{name}")


def file_logger(path: str | Path, name: str = "sysone.access") -> logging.Logger:
    """新建一个只写文件的 logger（一行一条 JSON），调用方负责关闭它的 handler。

    不动 `sysone` 根 logger：访问日志落在指定文件，进程里其它日志的去向不变。
    """
    logger = logging.Logger(name, level=logging.INFO)
    logger.propagate = False
    try:
        handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        raise ConfigError("访问日志无法打开", path=str(path), error=str(exc)) from exc
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
