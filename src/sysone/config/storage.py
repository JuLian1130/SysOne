"""存储与编译产物配置。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# 变量名跟着消费者走，`sysone.env.REGISTRY` 里再写一份字面量（与 `DECISION_API_KEY` 同一种
# 写法：注册表管解析与默认值，消费者管语义）。
ARTIFACTS_VARIABLE = "SYS1_ARTIFACTS_DIR"
WEIGHTS_VARIABLE = "SYS1_WEIGHTS_DIR"


def default_artifacts_dir() -> Path:
    """编译产物的默认落点：**持久**缓存目录。

    首启与二次启动差一个数量级（26.10 s 对 2.14 s，`plan.md` 的 R6），差别的全部来源是这
    个目录里有没有产物。放在系统临时目录里等于每次重启、每次清缓存都重编一遍，所以默认按
    平台给缓存目录；`SYS1_ARTIFACTS_DIR` 与 `--artifacts` 可覆盖。

    平台目录按 `sys.platform` 取而不是读 `XDG_CACHE_HOME`：读环境的唯一入口是 `env.py`，
    而这里给的是"没配任何东西时的默认值"。
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sysone" / "artifacts"
    return Path.home() / ".cache" / "sysone" / "artifacts"


def default_weights_dir() -> Path:
    """首启自动下载权重的默认落点：平台缓存目录，与产物目录同一套规矩。

    不落仓库下的 `models/`：服务的工作目录不固定，同一台机器上多份 checkout 还会各下
    一份 9 GB；缓存目录与 artifacts 同源，删掉只意味着下次重下。`SYS1_WEIGHTS_DIR` 可覆盖。
    """
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sysone" / "weights"
    return Path.home() / ".cache" / "sysone" / "weights"


@dataclass(frozen=True)
class StorageConfig:
    """权重与编译产物的位置。

    `artifacts_dir=None` 表示用 `default_artifacts_dir()`；`artifacts_budget=0` 表示不额外
    限制产物占用，>0 时超出即 `BudgetExceededError`。
    """

    weights_dir: Path | None = None
    artifacts_dir: Path | None = None
    artifacts_budget: int = 0

    def resolved_artifacts_dir(self) -> Path:
        """产物落点：构造时给的就用，没给换成平台缓存目录。"""
        return default_artifacts_dir() if self.artifacts_dir is None else Path(self.artifacts_dir)

    def resolved_weights_dir(self) -> Path:
        """自动下载的权重落点：构造时给的就用，没给换成平台缓存目录。"""
        return default_weights_dir() if self.weights_dir is None else Path(self.weights_dir)

    def validate(self) -> None:
        if self.artifacts_budget < 0:
            from ..errors import ConfigError

            raise ConfigError("artifacts_budget 不能为负", artifacts_budget=self.artifacts_budget)
