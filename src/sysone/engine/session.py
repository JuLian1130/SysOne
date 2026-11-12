"""会话与缓冲：一块预填充状态的生命周期。

M0 只做形状兼容的复用（同 `seq_len`/`prefix_len` 的空闲会话），真正的
前缀缓存（radix + 引用计数）留到 M2。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..domain.attention import MaskPlan
from ..domain.work import WorkPlan


@dataclass
class Session:
    """一个可执行单元的缓冲身份。"""

    seq_len: int
    prefix_len: int
    mask: MaskPlan
    scope: str = ""
    refs: int = 1
    created_at: float = field(default_factory=time.monotonic)
    released_at: float | None = None
    buffers: dict[str, object] | None = None

    @property
    def released(self) -> bool:
        return self.refs <= 0

    def compatible(self, seq_len: int, prefix_len: int) -> bool:
        return self.seq_len == seq_len and self.prefix_len == prefix_len and self.released


class SessionPool:
    """按形状复用空闲会话，并按 TTL 回收。"""

    def __init__(self, ttl_s: float | None = None) -> None:
        self._ttl = ttl_s
        self._live: list[Session] = []

    def acquire(self, plan: WorkPlan) -> Session:
        for session in self._live:
            if session.compatible(plan.seq_len(), plan.prefix_len):
                session.refs = 1
                session.released_at = None
                return session
        session = Session(
            seq_len=plan.seq_len(),
            prefix_len=plan.prefix_len,
            mask=plan.mask,
            scope=plan.lane,
        )
        self._live.append(session)
        return session

    def release(self, session: Session) -> None:
        session.refs -= 1
        if session.refs <= 0:
            session.refs = 0
            session.released_at = time.monotonic()

    def reap(self, now: float | None = None) -> int:
        if self._ttl is None:
            return 0
        current = time.monotonic() if now is None else now
        kept: list[Session] = []
        reaped = 0
        for session in self._live:
            if session.released and session.released_at is not None:
                if current - session.released_at > self._ttl:
                    session.buffers = None
                    reaped += 1
                    continue
            kept.append(session)
        self._live = kept
        return reaped

    def stats(self) -> dict[str, int]:
        return {"sessions": len(self._live), "live": sum(1 for s in self._live if not s.released)}
