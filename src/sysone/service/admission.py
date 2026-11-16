"""准入：容量 = 正在跑 + 排队；满了立刻 429，排队等太久同样 429。

与上游 `d3_server.py` 的一把锁不同，这里把"在跑"与"在等"分开记账，于是同一时刻最多
`workers` 个请求真的在引擎里跑、最多 `queue` 个在等；再进来的**在读 body 之前**就拒绝
（连接要关：body 还没读，不能复用这条连接）。
"""

from __future__ import annotations

import threading
import time

from ..errors import ConfigError


class Admission:
    """一个引擎前面的小门：`workers` 个执行位 + `queue` 个等待位。"""

    def __init__(
        self,
        *,
        workers: int = 1,
        queue: int = 2,
        retry_after: int = 1,
        timeout: float = 120.0,
    ) -> None:
        if workers < 1:
            raise ConfigError("workers 必须 >= 1", workers=workers)
        if queue < 0:
            raise ConfigError("queue 不能为负", queue=queue)
        self.workers = int(workers)
        self.queue = int(queue)
        self.retry_after = int(retry_after)
        self.timeout = float(timeout)
        self._cond = threading.Condition()
        self._running = 0
        self._waiting = 0
        # 累计计数（`snapshot` 那两个是瞬时值）：`/metrics` 要的是可聚合的这两笔。
        self._entered = 0
        self._rejected = 0

    @property
    def capacity(self) -> int:
        """同时在册的请求上限（在跑 + 在等）。"""
        return self.workers + self.queue

    def enter(self) -> bool:
        """拿一个执行位；容量已满（立刻）或等过 `timeout` 都返回 `False`（即 429）。"""
        with self._cond:
            if self._running + self._waiting >= self.capacity:
                self._rejected += 1
                return False
            self._waiting += 1
            deadline = time.monotonic() + self.timeout
            while self._running >= self.workers:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._waiting -= 1
                    self._rejected += 1
                    return False
                self._cond.wait(remaining)
            self._waiting -= 1
            self._running += 1
            self._entered += 1
            return True

    def leave(self) -> None:
        with self._cond:
            self._running -= 1
            self._cond.notify_all()

    def snapshot(self) -> tuple[int, int]:
        """(在跑, 在等)，给拒绝文案与日志用。"""
        with self._cond:
            return self._running, self._waiting

    def stats(self) -> dict[str, int]:
        """瞬时值加上累计计数，给 `/metrics` 用。

        `entered` / `rejected` 是自启动以来的累计：拒绝**包括**"立刻满"与"排队超时"两种
        （两者都回 429），所以它与 `entered` 一起就能算出被挡掉的比例。
        """
        with self._cond:
            return {
                "active": self._running,
                "waiting": self._waiting,
                "entered": self._entered,
                "rejected": self._rejected,
                "capacity": self.capacity,
            }
