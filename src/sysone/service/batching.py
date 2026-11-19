"""攒批：把几毫秒内到达的几条请求拼成一个 pass，再一起交给引擎。

**为什么在服务层**：`SentenceTransformer` 式的"批"不能由执行器自己攒 —— 执行器是被调用的，
它不知道还有没有请求在路上。只有前门知道：请求从 socket 上来，谁先谁后、能不能等几毫秒。

**怎么攒**：一个后台线程守着队列。第一条进来后等 `window_ms`（默认 5 ms，旧引擎同值），
把这期间到达的、**能合批的**一并取走，最多 `max_batch` 条。

"能合批"是三条硬条件，任一条不满足就单独跑：

| 条件 | 为什么 |
|---|---|
| 没有 `media` | 带图请求要走塔与像素管线，塞进文本批里会把两套调度搅在一起 |
| 同一个 `attention_mode` | 掩码不同，拼进一个 pass 答案就错了（而且看不出来） |
| `max_batch > 1` | 执行器不支持 `answers_batch` 时引擎会把上限压到 1 |

**等多久**：`window_ms` 是**延迟换吞吐**的旋钮，默认 5 ms（旧引擎同值）。窗口是**自适应**
的 —— 只有队里已经有人在等时才开：独自到达的那一条立刻跑，不为"可能来的下一条"白等。
这样单请求的延迟完全不变，而积压时（正是并发发生的时刻）照样能攒批。

**失败面**：一条请求的异常只回到它自己的 future，同批的其他请求不受影响 —— 但引擎的批是
一个 pass，异常通常来自解析/长度校验（那些在进批之前就发生了），真正的批内异常会让整批一起
失败，这是可接受的：同一个 pass 里出的事故没有"部分成功"可言。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Callable, Sequence

from ..domain.decision import DecisionRequest
from ..engine.engine import Engine


class Batcher:
    """把请求攒成小批，串行交给引擎（引擎自己不再并行）。"""

    def __init__(
        self,
        engine: Engine,
        *,
        window_ms: float = 5.0,
        max_batch: int | None = None,
    ) -> None:
        self.engine = engine
        self.window_s = max(0.0, float(window_ms)) / 1000.0
        self.max_batch = max(1, int(max_batch if max_batch is not None else engine.batch_limit))
        self._queue: deque[tuple[DecisionRequest, Future]] = deque()
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None
        # 记账：成型了几个批、其中几条真的合在一起跑了。
        self.batches = 0
        self.requests = 0
        self.batched_requests = 0

    # -- 生命周期 ------------------------------------------------------------

    def start(self) -> None:
        with self._cond:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="sysone-batcher", daemon=True)
            self._thread.start()

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
            self._thread = None

    # -- 提交 ----------------------------------------------------------------

    def submit(self, request: DecisionRequest) -> Future:
        """把请求排进来，返回它的 future（调用方 `.result()` 等答案）。"""
        future: Future = Future()
        if self.max_batch <= 1 or request.media:
            # 单条路径不进队列：带图请求本来就不批，也没必要为它开一个线程往返。
            self._run_now([request], [future])
            return future
        with self._cond:
            self._queue.append((request, future))
            self._cond.notify()
        return future

    # -- 内部 ----------------------------------------------------------------

    def _run_now(self, requests: Sequence[DecisionRequest], futures: Sequence[Future]) -> None:
        try:
            results = self.engine.batch(tuple(requests))
        except BaseException as exc:            # noqa: BLE001 - 原样交给各自的 future
            for future in futures:
                if not future.done():
                    future.set_exception(exc)
            return
        for future, result in zip(futures, results):
            if not future.done():
                future.set_result(result)

    def _compatible(self, head: DecisionRequest, other: DecisionRequest) -> bool:
        """能不能和 `head` 合进一个 pass。"""
        return (not other.media) and other.attention_mode == head.attention_mode

    def _take(self) -> list[tuple[DecisionRequest, Future]]:
        """取一批：等第一条 → 开窗口 → 把这期间能合的取走。"""
        with self._cond:
            while self._running and not self._queue:
                self._cond.wait(0.2)
            if not self._queue:
                return []
            head = self._queue.popleft()
            # **自适应窗口**：队里已经有人等才开窗。独自到达的那一条立刻跑，不为"可能来的
            # 下一条"白等 5 ms —— 单请求的延迟不该为不一定存在的批买单。
            if self.window_s > 0 and self.max_batch > 1 and self._queue:
                deadline = time.monotonic() + self.window_s
                while len(self._queue) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(remaining)
            batch = [head]
            # 与 head 不合的留在队里，下一轮再走 —— 不能因为一条不同 mode 的请求把整批拆散。
            rest: deque[tuple[DecisionRequest, Future]] = deque()
            while self._queue and len(batch) < self.max_batch:
                item = self._queue.popleft()
                if self._compatible(head[0], item[0]):
                    batch.append(item)
                else:
                    rest.append(item)
            self._queue.extendleft(reversed(rest))
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take()
            if not batch:
                with self._cond:
                    if not self._running:
                        return
                continue
            self.batches += 1
            self.requests += len(batch)
            if len(batch) > 1:
                self.batched_requests += len(batch)
            self._run_now([item[0] for item in batch], [item[1] for item in batch])

    def stats(self) -> dict[str, int]:
        return {
            "batches": self.batches,
            "requests": self.requests,
            "batched_requests": self.batched_requests,
            "max_batch": self.max_batch,
            "window_ms": int(self.window_s * 1000),
        }
