"""环境变量：唯一一处读环境的模块（注册表 + 类型校验）。

M0 只有日志级别有真实消费者；其余调参项都走 `EngineConfig`，不在这里登记。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .errors import ConfigError


@dataclass(frozen=True)
class EnvSpec:
    name: str
    kind: str
    default: object
    help: str


REGISTRY: tuple[EnvSpec, ...] = (
    EnvSpec("SYS1_LOG_LEVEL", "str", "info", "日志级别（debug/info/warning/error）"),
)

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})


def read(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """读取并校验已登记的环境变量。"""
    source = os.environ if environ is None else environ
    values: dict[str, object] = {}
    for spec in REGISTRY:
        raw = source.get(spec.name)
        values[spec.name] = spec.default if raw is None or raw == "" else _parse(spec, raw)
    return values


def _parse(spec: EnvSpec, raw: str) -> object:
    if spec.kind == "str":
        return raw
    if spec.kind == "bool":
        lowered = raw.strip().lower()
        if lowered in _BOOL_TRUE:
            return True
        if lowered in _BOOL_FALSE:
            return False
        raise ConfigError("环境变量不是布尔值", name=spec.name, value=raw)
    if spec.kind == "int":
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError("环境变量不是整数", name=spec.name, value=raw) from exc
    return raw
