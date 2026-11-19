"""在飞的 pass 长度合计闸：两条（或更多）序列一起超过预算时，后来的排队，**单条永远整趟跑**。

**为什么需要**：每个进程的工作区按**最长在飞 pass** 一次性分配，而且是稳态不是泄漏
（票据 07 §7.25e / §7.25g 实测：2k 是 0.72 GiB、16k 是 1.41 GiB，三趟连跑不再增长）。
`--workers 2` 是**两个进程**，各自一份，所以两个 16k 同时跑就是 2.82 GiB —— 这台 16 GB 的机器
扛不住。把"同时在飞的长度之和"压到预算以内，最坏情况是从 2.82 GiB 降到约 2.0 GiB。

**预算是多少、为什么**：`BUDGET_TOKENS = 16384`，正好是 `SchedulerConfig.max_len`（一条满宽
序列）。也就是说：**合计在飞的长度不超过一条满宽序列**。代入实测的拟合
`ws(T) ≈ 0.62 + 0.049·T/1000 GiB`：

| 在飞组合 | 合计 | 工作区估算 |
|---|---:|---:|
| 一条 16k | 16 384 | 0.62 + 1.41 ≈ 2.0 GiB（另一进程只有常驻，跑不了 pass） |
| 8k + 8k | 16 384 | 0.90 + 0.90 = 1.80 GiB |
| 16k + 2k | 18 432 > 预算 | 后来的那条排队 |
| 16k + 16k | 32 768（今天的无闸情形） | 2.82 GiB |

决策题的常态 prompt 是 1–2k，两条同时在飞只占 2–4k，**离预算还很远** —— 所以常态并发不受影响，
被挡住的只有"两条都很长"这一种。

**单条不受阈值影响**：预算按"合计"算，任何一条序列自己都整趟跑完（`max_len` 已经把单条限在
16 384 以内）。一条长序列可能要**等**别人退出，但永远不会被切开、也不会被拒。

**为什么用 flock 而不是 `threading.Semaphore`**：闸要管的是**另一个 worker 进程**，进程内的
信号量看不见它。做法是把预算切成 `QUANTUM_TOKENS` 一格、每格一个锁文件，一条 pass 按
`ceil(length / 4096)` 格拿锁（拿满为止、拿不到就全放回去下一次重试）。锁跟着 fd 走，fd 关闭或
进程退出（含崩溃）时由内核释放，所以不会留下死锁；锁文件放在系统临时目录，它们不承载状态，
重启后消失没有任何影响。

**今天它是惰性的**：`--workers` 已撤、`serve` 只起一个进程，一次只有一个 pass 在飞，
合计永远到不了预算，闸不会真的阻塞任何人。
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 在飞长度的合计预算：一条满宽序列（`SchedulerConfig.max_len`）。
BUDGET_TOKENS = 16384

# 一格预算。格子越小越贴合预算，但拿锁的次数越多；4 格 = 预算 16k 时每格 4k。
QUANTUM_TOKENS = 4096

# 拿不到格子时的重试间隔。
POLL_SECONDS = 0.02


class LongPass:
    """跨进程的在飞长度预算。`hold(length)` 在合计超预算时排队，单条永远放行。"""

    def __init__(self, directory: str | os.PathLike | None = None, *,
                 budget: int = BUDGET_TOKENS, quantum: int = QUANTUM_TOKENS) -> None:
        self.directory = (Path(directory) if directory is not None
                          else Path(tempfile.gettempdir()))
        self.quantum = max(1, int(quantum))
        self.quanta = max(1, int(budget) // self.quantum)
        # 关掉闸（`budget <= 0`）：任何长度都直接放行。
        self.enabled = int(budget) > 0
        # 只为观测：真正拿过格的次数与累计等待秒数。
        self.acquired = 0
        self.waited_s = 0.0

    # -- 预算换算 ------------------------------------------------------------

    def quanta_for(self, length: int) -> int:
        """一条 pass 要占几格；单条最多占满全部格子（再长也不排队等自己）。"""
        if not self.enabled:
            return 0
        need = -(-int(length) // self.quantum)          # 向上取整
        return max(1, min(need, self.quanta))

    def _path(self, index: int) -> Path:
        return self.directory / f"sysone-pass-budget-{index}.lock"

    # -- 拿格 / 还格 ---------------------------------------------------------

    def _try_take(self, need: int) -> list[int] | None:
        """非阻塞地拿 `need` 格；拿不齐就把已拿的放回去（不留半套）。"""
        held: list[int] = []
        for index in range(self.quanta):
            if len(held) >= need:
                break
            fd = os.open(self._path(index), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            held.append(fd)
        if len(held) == need:
            return held
        for fd in held:
            os.close(fd)
        return None

    @contextmanager
    def hold(self, length: int) -> Iterator[None]:
        """占住这条 pass 的预算；合计超了就在这里排队。"""
        need = self.quanta_for(length)
        if need == 0:
            yield
            return
        started = time.monotonic()
        held = self._try_take(need)
        while held is None:
            time.sleep(POLL_SECONDS)
            held = self._try_take(need)
        self.waited_s += time.monotonic() - started
        self.acquired += 1
        try:
            yield
        finally:
            for fd in held:
                os.close(fd)        # 关 fd 即释放；异常与崩溃同样如此
