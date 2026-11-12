"""聚合配置：`EngineConfig` 是引擎唯一的外部输入。

`cache_key()` 同时用于两件事（见 architecture.md §10）：
1. 编译产物键的一部分（产物跨运行复用）；
2. 服务层启动时校验父进程与 worker 的配置一致。
端口、日志级别等无关项刻意不进哈希。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from ..errors import ConfigError
from .model import ModelSelection
from .platform import PlatformConfig
from .scheduler import SchedulerConfig
from .service import ServiceConfig
from .storage import StorageConfig

CACHE_VERSION = "sysone-m0-v1"


@dataclass(frozen=True)
class EngineConfig:
    model: ModelSelection
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def validate(self) -> None:
        self.model.validate()
        self.scheduler.validate()
        self.storage.validate()
        self.platform.validate()
        self.service.validate()

    def cache_key(self) -> str:
        """对产物有影响的字段做 sha256（不含端口、日志级别、超时）。"""
        payload = {
            "version": CACHE_VERSION,
            "model": {
                "name": self.model.name,
                "dir": str(self.model.dir.resolve()),
                "precision": self.model.precision.value,
            },
            "attention_mode": (
                None if self.scheduler.attention_mode is None else self.scheduler.attention_mode.value
            ),
            "resident_w": self.scheduler.resident_w,
            "ring": self.scheduler.ring,
            "tile": self.scheduler.tile,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_config(model: ModelSelection) -> EngineConfig:
    """按 `--model` 构造默认配置并校验；CLI 的唯一入口。"""
    config = EngineConfig(model=model)
    try:
        config.validate()
    except ConfigError:
        raise
    return config
