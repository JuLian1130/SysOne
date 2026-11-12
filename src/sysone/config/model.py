"""模型选择：启动时只指定一个模型（M0 不做多模型常驻）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..domain.quant import DType
from ..errors import ConfigError


@dataclass(frozen=True)
class ModelSelection:
    """`--model <family>` + 权重目录 + 计算精度。

    `name` 是家族名（mude / h2o / torchcast），由 `sysone.models.registry` 解析；
    `dir` 是权重目录，`precision` 是计算精度（存储精度由权重形态决定，二者正交）。
    """

    name: str
    dir: Path
    precision: DType = DType.BF16

    def __post_init__(self) -> None:
        # CLI / JSON 传进来常是字符串；统一成 Path，其他地方不再做类型判断。
        object.__setattr__(self, "dir", Path(self.dir))

    def validate(self) -> None:
        if not self.name:
            raise ConfigError("模型家族名不能为空")
        if not str(self.dir):
            raise ConfigError("权重目录不能为空")
