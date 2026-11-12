"""厂商桥：把权重交给厂商自己的前向与读出（H2O、Torchcast）。

这一层只在选了 vendor 策略时才被 import，因此 torch 依赖不会污染核心面。
"""

from .base import VendorAnswer, VendorBridge, question_spec, wire_answer

__all__ = ["VendorAnswer", "VendorBridge", "question_spec", "wire_answer"]
