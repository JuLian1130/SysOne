"""Apple 平台实现：`AneRuntime`（dylib 句柄）与 `ApplePlatform`（能力与预算）。"""

from .platform import (
    MEASURED_GEOMETRIES,
    MAX_SEQ,
    MAX_TILE,
    ApplePlatform,
    available,
    measured,
    warn_if_unmeasured,
)
from .runtime import AneRuntime, Surface

__all__ = [
    "AneRuntime",
    "ApplePlatform",
    "MEASURED_GEOMETRIES",
    "MAX_SEQ",
    "MAX_TILE",
    "Surface",
    "available",
    "measured",
    "warn_if_unmeasured",
]
