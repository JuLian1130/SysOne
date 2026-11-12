"""调度器：接口 + 串行实现（chunked prefill 等策略在 M4 落地）。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from ..domain.decision import Answer
from ..domain.work import WorkPlan


@dataclass(frozen=True)
class SchedulerStats:
    """调度器计数。"""

    added: int = 0
    finished: int = 0
    pending: int = 0


class Scheduler(Protocol):
    """调度器接口：只依赖计划与答案，不碰硬件。"""

    def add(self, plan: WorkPlan) -> None: ...

    def schedule(self) -> tuple[WorkPlan, ...]: ...

    def finish(self, plan: WorkPlan, answer: Answer) -> None: ...

    def has_unfinished(self) -> bool: ...

    def drain(self) -> list[tuple[WorkPlan, Answer]]: ...

    def stats(self) -> SchedulerStats: ...


class SerialScheduler:
    """一次只调度一份计划的串行实现（M0 的唯一实现）。"""

    def __init__(self) -> None:
        self._pending: deque[WorkPlan] = deque()
        self._done: list[tuple[WorkPlan, Answer]] = []
        self._added = 0
        self._finished = 0

    def add(self, plan: WorkPlan) -> None:
        self._pending.append(plan)
        self._added += 1

    def schedule(self) -> tuple[WorkPlan, ...]:
        if not self._pending:
            return ()
        return (self._pending.popleft(),)

    def finish(self, plan: WorkPlan, answer: Answer) -> None:
        self._done.append((plan, answer))
        self._finished += 1

    def has_unfinished(self) -> bool:
        return bool(self._pending)

    def drain(self) -> list[tuple[WorkPlan, Answer]]:
        done, self._done = self._done, []
        return done

    def stats(self) -> SchedulerStats:
        return SchedulerStats(
            added=self._added, finished=self._finished, pending=len(self._pending)
        )
