"""两条执行策略：`vendor`（M1：H2O / Torchcast）与 `split`（M2：mude 的 ANE+GPU 链）。"""

from .split import SplitRunner
from .vendor import VendorRunner

__all__ = ["SplitRunner", "VendorRunner"]
