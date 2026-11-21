"""导入即注册四个家族。

`sysone.models.families` 只在这里被 import：启动路径（CLI / 服务）显式 import 本包，
一次性触发 `register_family`，避免"注册表里有什么取决于谁先 import 了什么"。
"""

from . import d3, h2o, mude, torchcast  # noqa: F401  仅注册副作用

__all__ = ["d3", "h2o", "mude", "torchcast"]
