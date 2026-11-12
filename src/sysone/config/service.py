"""服务层配置（M0 只有 CLI，字段先落定以便 M2 直接用）。"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    workers: int = 1
    api_key: str | None = None
    request_timeout_s: float = 120.0
    debug_timings: bool = False

    def validate(self) -> None:
        if not (0 < self.port < 65536):
            raise ConfigError("端口非法", port=self.port)
        if self.workers < 1:
            raise ConfigError("workers 必须 >= 1", workers=self.workers)
        if self.request_timeout_s <= 0:
            raise ConfigError("request_timeout_s 必须为正", request_timeout_s=self.request_timeout_s)
