"""串行调度器与会话池的行为。"""

from __future__ import annotations

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Answer
from sysone.domain.work import QuestionPlan, ResidencyPlan, WorkPlan
from sysone.engine.planner import build_mask_plan
from sysone.engine.scheduler import SerialScheduler
from sysone.engine.session import SessionPool


def _plan(length: int = 4) -> WorkPlan:
    mask = build_mask_plan(MaskMode.CAUSAL, length, 0, 2)
    return WorkPlan(
        lane="lane",
        tokens=tuple(range(length)),
        prefix_len=0,
        questions=(QuestionPlan(index=0, start=0, end=length, labels=("A", "B"), kind="choice"),),
        mask=mask,
        residency=ResidencyPlan(),
    )


def test_scheduler_serial_order() -> None:
    scheduler = SerialScheduler()
    first, second = _plan(), _plan()
    scheduler.add(first)
    scheduler.add(second)
    assert scheduler.has_unfinished() is True
    scheduled = scheduler.schedule()
    assert scheduled == (first,)
    scheduler.finish(first, Answer(0, "choice", "A", (0.5, 0.5), ("A", "B"), 0.5))
    assert scheduler.has_unfinished() is True
    assert scheduler.schedule() == (second,)
    scheduler.finish(second, Answer(0, "choice", "B", (0.5, 0.5), ("A", "B"), 0.5))
    drained = scheduler.drain()
    assert [answer.value for _, answer in drained] == ["A", "B"]
    assert scheduler.has_unfinished() is False
    assert scheduler.stats().added == 2


def test_session_pool_reuses_and_reaps() -> None:
    pool = SessionPool(ttl_s=1.0)
    first = pool.acquire(_plan(4))
    assert first.refs == 1
    later = pool.acquire(_plan(6))
    assert later is not first
    pool.release(first)
    pool.release(later)
    assert first.released is True
    again = pool.acquire(_plan(4))
    assert again is first and again.refs == 1
    pool.release(again)
    assert pool.reap(now=again.released_at + 2.0) == 2
    assert pool.stats()["sessions"] == 0
