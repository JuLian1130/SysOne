"""长 pass 的跨进程闸：同一时刻只让一个"超过阈值的 pass"在飞。

**为什么需要**：每个进程的工作区按**最长在飞 pass** 一次性分配，而且这是稳态不是泄漏
（票据 07 §7.25e / §7.25g 实测：2k 是 0.72 GiB、16k 是 1.41 GiB，三趟连跑不再增长）。
`--workers 2` 是**两个进程**，各自一份工作区，所以两个 16k 同时跑就是 2.82 GiB —— 这台 16 GB
的机器扛不住。把长 pass 串起来，最坏情况降到 `1.41（一个长）+ 0.82（一个 4k 内的短）≈ 2.23 GiB`。

**为什么是 flock 而不是 `threading.Semaphore`**：闸必须**跨进程**，同一进程内的信号量管不到
另一个 worker。flock 的锁跟着 fd 走，fd 关闭或进程退出（含崩溃）时由内核释放，所以不会留下
死锁；文件本身不承载任何状态，因此放在系统临时目录里是对的（重启后它消失没有任何影响）。

**阈值 4096 是怎么来的**：不是拍的，是从两次实测插出来的 —— 2k→0.72 GiB、16k→1.41 GiB，
线性外推 4k≈0.82 GiB；取 4096 有两个附带理由：它等于出货桶宽 1024 的 4 倍，也正好是旧引擎
"ANE program 一律按 S ≤ 4096 编译"那条设计规则的界（`docs/ANE/e1a-e1b-results.md` §7）。
决策题的实际 prompt 是 1–2k token，所以**常态请求永远不会被这道闸挡住**。

**今天它是惰性的**：`--workers` 已撤、`serve` 只起一个进程，一次只有一个 pass 在飞，
闸不会真的阻塞任何人。它在这里是为了让 `--workers` 回来时正确性已经就位，代价是每个长 pass
一次 `flock` 系统调用（微秒级）。
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# 超过这个 token 数的 pass 才算"长"。见模块 docstring 的推导。
LONG_PASS_TOKENS = 4096

LOCK_NAME = "sysone-longpass.lock"


class LongPass:
    """长 pass 的闸：`hold(length)` 在 `length > tokens` 时拿独占锁，否则什么都不做。"""

    def __init__(self, path: str | os.PathLike | None = None, *,
                 tokens: int = LONG_PASS_TOKENS) -> None:
        self.path = Path(path) if path is not None else Path(tempfile.gettempdir()) / LOCK_NAME
        self.tokens = int(tokens)
        # 只为观测：过闸次数与累计等待秒数（`tokens <= 0` 时恒为 0）。
        self.acquired = 0
        self.waited_s = 0.0

    def blocked(self, length: int) -> bool:
        """这个长度的 pass 要不要过闸。"""
        return self.tokens > 0 and int(length) > self.tokens

    @contextmanager
    def hold(self, length: int) -> Iterator[None]:
        """`blocked(length)` 为真时串行化；否则直接让行。"""
        if not self.blocked(length):
            yield
            return
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            try:
                # 先非阻塞探一次：没被占（常态）时省掉阻塞路径。
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                started = time.monotonic()
                fcntl.flock(fd, fcntl.LOCK_EX)      # 另一个进程正在跑长 pass
                self.waited_s += time.monotonic() - started
            self.acquired += 1
            yield
        finally:
            os.close(fd)        # 关 fd 即释放 flock；异常与崩溃同样如此
