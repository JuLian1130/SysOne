"""在飞长度预算的语义：单条永远整趟跑、合计超预算才排队、锁随 fd 释放。

要点（票 §7.25h）：
- 预算按**合计**算，不按单条 —— 单条 16k 拿满 4 格，自己永远能跑；
- 两条的合计超过预算时，后来的那条排队，先来的不会被抢；
- 锁是 `flock`，所以**跨进程**有效，且进程崩溃/被 kill 时由内核释放。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from sysone.devices.apple.longpass import BUDGET_TOKENS, QUANTUM_TOKENS, LongPass

SRC = Path(__file__).resolve().parents[2] / "src"


def _hold(slot: LongPass, length: int, seconds: float, *, name: str,
          events: dict[str, float]) -> threading.Thread:
    """跑一条 pass；记**绝对**单调时刻（各线程记相对时刻基线不同，比较没有意义）。"""

    def body() -> None:
        with slot.hold(length):
            events[f"{name}_at"] = time.monotonic()
            time.sleep(seconds)
        events[f"{name}_done"] = time.monotonic()

    thread = threading.Thread(target=body)
    thread.start()
    return thread


def test_the_budget_is_one_16k_sequence() -> None:
    assert BUDGET_TOKENS == 16_384, "预算固定 16k 当量；--width 更大时单条占满格子独自跑"
    assert QUANTUM_TOKENS * 4 == BUDGET_TOKENS
    slot = LongPass()
    assert slot.quanta_for(1) == 1
    assert slot.quanta_for(QUANTUM_TOKENS) == 1
    assert slot.quanta_for(QUANTUM_TOKENS + 1) == 2
    assert slot.quanta_for(8_192) == 2
    assert slot.quanta_for(BUDGET_TOKENS) == 4
    # 32k 的单条：需要的格子超过预算，按"占满"处理 —— 它照样整趟跑，只是没有别的 pass 能并行。
    assert slot.quanta_for(32_768) == 4


def test_a_single_long_sequence_always_runs_whole(tmp_path: Path) -> None:
    """单条超过一格预算也要整趟跑完 —— 只是它会把全部格子占满。"""
    slot = LongPass(tmp_path)
    with slot.hold(BUDGET_TOKENS):
        assert slot.acquired == 1
    assert slot.quanta_for(BUDGET_TOKENS) == slot.quanta


def test_two_short_passes_run_at_the_same_time(tmp_path: Path) -> None:
    """合计在预算以内：第二条不等。"""
    slot = LongPass(tmp_path)
    events: dict[str, float] = {}
    first = _hold(slot, 2_048, 0.30, name="a", events=events)
    time.sleep(0.05)
    second = _hold(slot, 2_048, 0.30, name="b", events=events)
    first.join()
    second.join()
    assert events["b_at"] - events["a_at"] < 0.15, "两条短的应该并行"


def test_two_long_passes_serialize_when_the_sum_exceeds_the_budget(tmp_path: Path) -> None:
    """8k + 12k = 20k > 16k ⇒ 后来的排队（8k+8k 恰好等于预算，是并行的那一侧）。"""
    slot = LongPass(tmp_path)
    events: dict[str, float] = {}
    first = _hold(slot, 8_192, 0.30, name="a", events=events)
    time.sleep(0.05)
    second = _hold(slot, 12_288, 0.0, name="b", events=events)
    first.join()
    second.join()
    assert events["b_at"] >= events["a_done"] - 0.01, "合计 20k 超过 16k，不该并行"
    assert events["b_at"] - events["a_at"] >= 0.20


def test_the_budget_is_exactly_reachable(tmp_path: Path) -> None:
    """8k + 8k = 16k 恰好等于预算 ⇒ 并行（边界按"不超过"算）。"""
    slot = LongPass(tmp_path)
    events: dict[str, float] = {}
    first = _hold(slot, 8_192, 0.30, name="a", events=events)
    time.sleep(0.05)
    second = _hold(slot, 8_192, 0.30, name="b", events=events)
    first.join()
    second.join()
    assert events["b_at"] - events["a_at"] < 0.15, "合计正好等于预算时应该并行"


def test_a_failed_pass_gives_its_budget_back(tmp_path: Path) -> None:
    slot = LongPass(tmp_path)
    with pytest.raises(RuntimeError):
        with slot.hold(BUDGET_TOKENS):
            raise RuntimeError("pass 炸了")
    with slot.hold(BUDGET_TOKENS):       # 4 格都还回来了
        pass
    assert slot.acquired == 2


def test_another_process_holds_the_budget(tmp_path: Path) -> None:
    """真跨进程：子进程占满预算期间，父进程的 pass 必须排队。"""
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
    proc = subprocess.Popen([sys.executable, "-c", child, str(SRC), str(tmp_path)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"
        slot = LongPass(tmp_path)
        started = time.monotonic()
        with slot.hold(2_048):
            waited = time.monotonic() - started
        assert waited >= 0.3, f"父进程只等了 {waited:.2f} s，预算没生效"
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_disabling_the_budget_lets_everything_through(tmp_path: Path) -> None:
    slot = LongPass(tmp_path, budget=0)
    assert slot.quanta_for(1 << 20) == 0
    with slot.hold(1 << 20):
        pass
    assert slot.acquired == 0
