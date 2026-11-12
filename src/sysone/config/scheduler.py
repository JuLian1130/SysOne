"""调度与注意力配置。"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.attention import MaskMode
from ..errors import ConfigError


@dataclass(frozen=True)
class SchedulerConfig:
    """调度器与掩码相关的参数。

    - `attention_mode=None`：跟随模型自身的 `default_mode`（mude 建议 prefix_noncausal）。
    - `resident_w` / `ring`：常驻窗口与层环大小（M0 只做校验，M2 才真正分块）。
    - `tile`：掩码分窗的 token 粒度，0 表示整段一个窗口。
    - `queue`：准入等待队列上限（服务层用，M0 CLI 不用）。
    - `max_len`：序列上限，超了直接 422。
    """

    attention_mode: MaskMode | None = None
    resident_w: int = 4
    ring: int = 4
    tile: int = 0
    queue: int = 2
    max_len: int = 16384

    def validate(self) -> None:
        if self.resident_w < 1:
            raise ConfigError("resident_w 必须 >= 1", resident_w=self.resident_w)
        if self.ring < 1:
            raise ConfigError("ring 必须 >= 1", ring=self.ring)
        if self.queue < 0:
            raise ConfigError("queue 必须 >= 0", queue=self.queue)
        if self.tile < 0:
            raise ConfigError("tile 必须 >= 0", tile=self.tile)
        if self.max_len < 1:
            raise ConfigError("max_len 必须 >= 1", max_len=self.max_len)
        if self.tile > self.max_len:
            raise ConfigError("tile 不能大于 max_len", tile=self.tile, max_len=self.max_len)
