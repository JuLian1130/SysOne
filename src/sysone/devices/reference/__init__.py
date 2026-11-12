"""参考后端：numpy 实现（`ReferenceRunner` + `ReferencePlatform`）与编码器。"""

from .encoding import CharEncoder
from .numpy import ReferencePlatform, ReferenceRunner

__all__ = ["CharEncoder", "ReferencePlatform", "ReferenceRunner"]
