"""Metal 流水线与 numpy 链的端到端对拍（需要真 mude 权重 + 两个 dylib）。

只在设了 `SYS1_MUDE_WEIGHTS` 时跑：它会各建一条 20 层的链（一次约 40 s，同一个 module 内复用）。

`prefix` 参数覆盖 mude 出货的两种读法：0 是全长可见（`noncausal_full`），非 0 是
`prefix_noncausal`（前缀行只看前缀、尾巴行看全部）——后者在流水线里走两遍 attention dispatch。
"""

import os
from pathlib import Path

import numpy as np
import pytest

MUDE = os.environ.get("SYS1_MUDE_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not MUDE or not Path(MUDE).is_dir(),
    reason="设 SYS1_MUDE_WEIGHTS 指向 mude-bench 权重目录才会跑（约 40 s）",
)

S = 256
TOKENS = [1234, 5678, 9012, 3456, 7890, 123, 456, 789]
TOL = 5e-3
PREFIXES = (0, 4)


def _chains():
    from sysone.devices.apple.chain import MudeChain
    from sysone.devices.apple.metal import available as metal_available
    from sysone.devices.apple.runtime import AneRuntime

    try:
        AneRuntime()
    except Exception as exc:  # noqa: BLE001 - 没有 dylib 就是跳过
        pytest.skip(f"ANE dylib 不可用：{exc}")
    if not metal_available():
        pytest.skip("Metal dylib 不可用")
    return MudeChain


@pytest.fixture(scope="module")
def chains():
    MudeChain = _chains()
    cpu = MudeChain.load(MUDE, s=S)
    try:
        gpu = MudeChain.load(MUDE, s=S, metal=True)
    except BaseException:
        cpu.close()
        raise
    try:
        yield cpu, gpu
    finally:
        gpu.close()
        cpu.close()


@pytest.fixture(scope="module")
def chunked():
    """桶宽只有 S/4 的链：同一个 256 列的缓冲要分四段算，用来钉住"桶宽 / 缓冲宽"这条分界。"""
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S // 4, width=S, metal=True)
    try:
        yield chain
    finally:
        chain.close()


@pytest.mark.parametrize("prefix", PREFIXES)
def test_metal_chain_matches_the_numpy_chain(chains, prefix: int) -> None:
    cpu, gpu = chains
    lcpu = cpu.logits(TOKENS, prefix=prefix)
    lgpu = gpu.logits(TOKENS, prefix=prefix)
    rel = float(np.linalg.norm(lgpu - lcpu) / np.linalg.norm(lcpu))
    assert rel < TOL, f"prefix={prefix} 时 metal 与 numpy 链的 logits 偏差 {rel:.4e} 超过 {TOL}"


def test_metal_chain_keeps_the_last_layer_hidden_in_range(chains) -> None:
    cpu, gpu = chains
    trace: list = []
    cpu.logits(TOKENS, prefix=0, trace=trace)
    gpu.logits(TOKENS, prefix=0)
    last = gpu._pipeline.last_hidden(len(TOKENS))
    ref = np.asarray(trace[-1], np.float32)
    rel = float(np.linalg.norm(last - ref) / np.linalg.norm(ref))
    assert rel < 5e-2, f"末层 hidden 偏差 {rel:.4e}"


def test_chunked_bucket_matches_the_single_bucket_run(chains, chunked) -> None:
    """桶宽 S/4（四段）与桶宽 S（一段）跑同一个满宽 prompt，结果应逐位相同。

    这条用例专门钉住 `GdnArgs.Stile` 的双重身份：它既是列界也是通道步长
    （`shaders/e8_gdn.metal:332-360` 的 `proj[c*Stile + j]`），传成"本段有效列数"会让短
    prompt 全错而满桶恰好正确 —— 见 `docs/sysone/plan.md` 的 M2.2 第 4 片 B。
    """
    _, gpu = chains
    tokens = TOKENS * (S // len(TOKENS))  # 正好填满 256 列的缓冲
    assert len(tokens) == S
    once = gpu.logits(tokens, prefix=0)
    segment = chunked.logits(tokens, prefix=0)
    rel = float(np.linalg.norm(segment - once) / np.linalg.norm(once))
    assert rel < 1e-6, f"四段与单段结果不一致：{rel:.4e}"


@pytest.fixture(scope="module")
def windowed():
    """待命窗口 W=4 的链：只在块边界换常驻，其余与 `gpu` 完全一样。"""
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S, metal=True, resident=4)
    try:
        yield chain
    finally:
        chain.close()


@pytest.mark.parametrize("prefix", PREFIXES)
def test_resident_window_keeps_the_answer(chains, windowed, prefix) -> None:
    """窗口只换常驻、不换数值：答案与 numpy 链在同一容差内。"""
    cpu, _ = chains
    lwin = windowed.logits(TOKENS, prefix=prefix)
    lcpu = cpu.logits(TOKENS, prefix=prefix)
    rel = float(np.linalg.norm(lwin - lcpu) / np.linalg.norm(lcpu))
    assert rel < TOL, f"prefix={prefix} 时窗口链的 logits 偏差 {rel:.4e} 超过 {TOL}"


def test_resident_window_returns_to_standby(windowed) -> None:
    """一轮跑完必须落在待命窗口上：4 层常驻、峰值 2W、换窗不泄漏。

    这条就是服务契约"回到前 W 层预装载"的实测版。`programs` 是 dylib 的**进程级**计数
    （同一个进程里还有别的链），所以这里只钉"一轮换窗前后不变"，绝对层数由 `resident` 钉。
    """
    before = windowed._runtime.programs
    windowed.logits(TOKENS, prefix=0)
    stats = windowed._pipeline._residency.stats()
    assert stats["resident"] == 4
    assert stats["standby"] == [0, 1, 2, 3]
    assert stats["peak_resident"] == 8       # 块 0 的窗口 + 块 1 的预取 = 2W
    assert stats["loads"] >= 20              # 一轮要重装窗口外的层（每轮一次全量）
    assert windowed._runtime.programs == before
    assert windowed._runtime.pending is False


def test_dead_prefix_tiles_are_skipped_without_changing_the_answer(chunked) -> None:
    """S3：最后一层里整段落在公共前缀中的瓦片只做第一相，答案不变、注意力瓦片少一片。

    桶宽 S/4、满宽 prompt 正好给 4 个瓦片，`prefix = S/4` 让第 0 个瓦片整段落在前缀里。

    这条用例拿**第一次** `logits()` 比，不预热。分块链的首趟曾经与后续不同（对 numpy 链
    `relL2 1.47e-1`，pass2 起 `1.31e-3`）：`proj` 的 ANE 写不在任何一条 lane 上，而读同一张
    `proj_out` 的 `gather` 在 lane 0 —— 首趟拷贝冷页慢，写就把上一片的读盖掉了。修法是在
    `proj` 发射前也等到同一槽位上一片 `gather` 的 CB（见 `pipeline._issue`，票据 02）。

    量两件事：**出货形态（双链 + S3 开）与单链全量计算逐位相同**，以及 **S3 真的少发了一片
    注意力**（数 `wait_upto` 的 lane 0 次数，不数的话"答案一致"也可能是因为根本没跳过）。
    票据 01 原先把 `_skip_dead = False`（双链）那臂记成"坏参照"——那一臂其实是该进程的
    第一次运行，票据已据此改写。
    """
    from sysone.devices.apple.metal import LANE_LONG

    tokens = TOKENS * (S // len(TOKENS))
    prefix = S // 4
    pipeline = chunked._pipeline
    assert pipeline is not None
    runtime = pipeline._runtime
    original_nch = pipeline._nch
    original = runtime.wait_upto

    def run(nch: int, skip: bool, counted: dict | None = None) -> np.ndarray:
        pipeline._nch = nch
        pipeline._skip_dead = skip
        if counted is not None:
            def counted_wait(lane: int, count: int) -> float:
                if lane == LANE_LONG:
                    counted["n"] += 1
                return original(lane, count)

            runtime.wait_upto = counted_wait     # type: ignore[method-assign]
        try:
            return chunked.logits(tokens, prefix=prefix)
        finally:
            runtime.wait_upto = original          # type: ignore[method-assign]

    try:
        shipping = run(original_nch, True)
        full_single = run(1, False)
        assert np.array_equal(shipping, full_single), (
            "出货形态（双链 + S3 开）与单链全量计算不一致"
        )
        dead_count, live_count = {"n": 0}, {"n": 0}
        with_dead = run(1, True, dead_count)
        without_dead = run(1, False, live_count)
        assert dead_count["n"] == live_count["n"] - 1, (
            f"死瓦片没有少收注意力瓦片：S3 开 {dead_count['n']} 次、关 {live_count['n']} 次"
        )
        assert np.array_equal(with_dead, without_dead), "S3 改变了答案（跳过的瓦片不该有读者）"
    finally:
        pipeline._nch = original_nch
        pipeline._skip_dead = True
