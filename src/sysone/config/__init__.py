"""配置对象：构造出来后不可变（frozen dataclass）。

只有 `config` 包读环境变量（`sysone.env`），其余模块只接收 `EngineConfig`。
"""

from .engine import CACHE_VERSION, EngineConfig
from .model import ModelSelection
from .platform import PlatformConfig
from .scheduler import SchedulerConfig
from .service import ServiceConfig
from .storage import StorageConfig

__all__ = [
    "CACHE_VERSION",
    "EngineConfig",
    "ModelSelection",
    "PlatformConfig",
    "SchedulerConfig",
    "ServiceConfig",
    "StorageConfig",
]
