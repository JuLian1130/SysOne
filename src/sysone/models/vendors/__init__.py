"""厂商侧适配：题面与答面（不碰 torch），以及把前向交回厂商的桥。

`base` / `h2o_serve` / `torchcast_serve` 只做"文本 → id"与判答案，两条前向（厂商桥与我们的
ANE 链）共用它们；`*_bridge` 才 import torch，只有选了 `vendor` 策略时才装载。
"""

from .base import AnswerSurface, Prepared, VendorAnswer, VendorBridge, question_spec, wire_answer

__all__ = [
    "AnswerSurface",
    "Prepared",
    "VendorAnswer",
    "VendorBridge",
    "question_spec",
    "wire_answer",
]
