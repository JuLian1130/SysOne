"""存储与编译产物配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageConfig:
    """权重与编译产物的位置。

    `artifacts_budget=0` 表示不额外限制产物占用；>0 时超出即 `BudgetExceededError`。
    """

    weights_dir: Path | None = None
    artifacts_dir: Path | None = None
    artifacts_budget: int = 0

    def validate(self) -> None:
        if self.artifacts_budget < 0:
            from ..errors import ConfigError

            raise ConfigError("artifacts_budget 不能为负", artifacts_budget=self.artifacts_budget)
