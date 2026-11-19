"""长 pass 闸的语义：跨 fd 互斥、短 pass 不挡、异常与进程退出都会释放。

`flock` 的锁跟 fd 走，所以同一进程里**两个不同的 fd** 也会互相阻塞 —— 这正是它比
`threading.Semaphore` 更适合这里的原因（闸要管的是另一个 worker 进程，不是另一个线程）。
真跨进程那一段用子进程验，其余用线程：都属于毫秒级、不碰 ANE。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from sysone.devices.apple.longpass import LONG_PASS_TOKENS, LongPass

SRC = Path(__file__).resolve().parents[2] / "src"


def _hold_for(slot: LongPass, length: int, seconds: float, *, name: str,
              events: dict[str, float]) -> threading.Thread:
    def body() -> None:
        # 记**绝对**单调时刻：两个线程各记相对时刻的话，基线不同，比较就没有意义。
        with slot.hold(length):
            events[f"{name}_at"] = time.monotonic()
            time.sleep(seconds)
        events[f"{name}_done"] = time.monotonic()

    thread = threading.Thread(target=body)
    thread.start()
    return thread


def test_the_threshold_is_the_documented_one() -> None:
    """阈值是票里的数：超出桶宽（1024）四倍才算长 pass。"""
    assert LONG_PASS_TOKENS == 4096
    slot = LongPass()
    assert not slot.blocked(LONG_PASS_TOKENS), "等于阈值不算长"
    assert slot.blocked(LONG_PASS_TOKENS + 1)


def test_a_long_pass_waits_for_another_long_pass(tmp_path: Path) -> None:
    slot = LongPass(tmp_path / "gate.lock")
    events: dict[str, float] = {}
    first = _hold_for(slot, 16_384, 0.30, name="first", events=events)
    time.sleep(0.05)                      # 让第一条先进闸
    second = _hold_for(slot, 16_384, 0.0, name="second", events=events)
    first.join()
    second.join()
    assert events["second_at"] >= events["first_done"] - 0.01, "第二条必须等第一条退出"
    assert events["second_at"] - events["first_at"] >= 0.20, "第二条没有真的等"
    assert slot.acquired == 2


def test_a_short_pass_never_waits(tmp_path: Path) -> None:
    port = LongPass(tmp_path / "gate.lock")
    events: dict[str, float] = {}
    holder = _hold_for(port, 16_384, 0.30, name="long", events=events)
    time.sleep(0.05)
    with port.hold(2048):                 # 短 pass 直接过，不等那把锁
        pass
    holder.join()
    assert port.acquired == 1, "只有长 pass 记账"


def test_the_lock_is_released_when_the_pass_raises(tmp_path: Path) -> None:
    slot = LongPass(tmp_path / "gate.lock")
    with pytest.raises(RuntimeError):
        with slot.hold(8192):
            raise RuntimeError("pass 炸了")
    # 还能再拿到 —— 异常没把锁留下
    with slot.hold(8192):
        pass
    assert slot.acquired == 2


def test_another_process_blocks_on_the_same_gate(tmp_path: Path) -> None:
    """真跨进程：子进程拿锁期间，父进程的长 pass 必须等。"""
    gate = tmp_path / "gate.lock"
    child = textwrap.dedent(
        """
        import sys, time
        from pathlib import Path
        sys.path.insert(0, sys.argv[1])
        from sysone.devices.apple.longpass import LongPass
        slot = LongPass(Path(sys.argv[2]))
        with slot.hold(16384):
            print("held", flush=True)
            time.sleep(0.6)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", child, str(SRC), str(gate)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"     # 子进程已进闸
        slot = LongPass(gate)
        started = time.monotonic()
        with slot.hold(16_384):
            waited = time.monotonic() - started
        assert waited >= 0.3, f"父进程只等了 {waited:.2f} s，闸没生效"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_disabling_the_gate_makes_every_length_a_pass(tmp_path: Path) -> None:
    """`tokens <= 0` 是"关掉"—— 部署在内存宽裕的机器上时用这一档。"""
    slot = LongPass(tmp_path / "gate.lock", tokens=0)
    assert not slot.blocked(1 << 20)
    with slot.hold(1 << 20):
        pass
    assert slot.acquired == 0
