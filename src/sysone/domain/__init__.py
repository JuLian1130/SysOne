"""领域类型（domain）：不 import torch、不 import 设备与模型实现。

这一层只放"引擎与硬件无关的表达"：请求、答案、注意力语义、缓冲声明、量化描述。
所有类型都是 frozen dataclass 或 StrEnum，便于跨进程/跨设备传递与快照测试。
"""
