"""环境变量与本机临时产物落点：唯一一处读环境的模块（注册表 + 类型校验）。

登记的只有真正有消费者的三项：日志级别（`logging`）、服务的 bearer key（`service/app`）、
产物落点（`cli`）。其余调参项都走 `EngineConfig`，不在这里登记。

`scratch_root()` 不是环境变量，但同属"东西从哪来"这个问题，所以也放在这里。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
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
    EnvSpec(
        "DECISION_API_KEY",
        "str",
        "",
        "服务层的 bearer key；设了非空值后 /v1/models 与 /v1/systemone 必须带 Authorization",
    ),
    EnvSpec(
        "SYS1_ARTIFACTS_DIR",
        "str",
        "",
        "编译产物的落点（默认平台缓存目录）；`--artifacts` 优先于它",
    ),
)

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off"})

# 本机临时产物的目录名（仓库根下，`.gitignore` 忽略）：dylib、Metal 的几何头、测试依赖、
# 探针输出。它们都能重算，但不该每次重启都重算一遍。
SCRATCH_DIRNAME = ".tmp"


def scratch_root() -> Path:
    """`<仓库>/.tmp`：本机临时产物的根。

    为什么不放 `/tmp` 或 `$TMPDIR`：本机开机会清空 `/tmp`；`$TMPDIR`（`/var/folders/.../T`）
    躲得过重启，但 macOS 会按访问时间清理里面的条目（2026-11-19 一个测试依赖目录就是这样
    消失的，同一目录里更大、更新近的产物活着）。两类失效都表现为"重启后一切从零重建"。

    锚点是本文件位置（`<仓库>/src/sysone/env.py`）。本仓库只在源码树里运行（没有
    `pyproject.toml`/`setup.py`，靠 `PYTHONPATH=src`），所以这个锚点成立；**真打包安装后它
    会指到 site-packages 的上级目录**，那时该换成平台缓存目录，而不是加一条回退分支。
    """
    return Path(__file__).resolve().parents[2] / SCRATCH_DIRNAME


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
