"""平台（设备）选择配置。"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError

PLATFORMS = ("auto", "reference", "apple")


@dataclass(frozen=True)
class PlatformConfig:
    """`preferred="auto"` 时按可用平台择优；M0 只有 `reference`。"""

    preferred: str = "auto"
    wired_limit: int | None = None
    threads: int = 0

    def validate(self) -> None:
        if self.preferred not in PLATFORMS:
            raise ConfigError("平台未知", preferred=self.preferred, allowed=PLATFORMS)
        if self.wired_limit is not None and self.wired_limit <= 0:
            raise ConfigError("wired_limit 必须为正", wired_limit=self.wired_limit)
        if self.threads < 0:
            raise ConfigError("threads 不能为负", threads=self.threads)
