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


def test_dead_prefix_tiles_are_skipped_without_changing_the_answer(chunked) -> None:
    """S3：最后一层里整段落在公共前缀中的瓦片只做第一相，结果必须逐位不变。

    桶宽 S/4、满宽 prompt 正好给 4 个瓦片，`prefix = S/4` 让第 0 个瓦片整段落在前缀里。
    A/B 用 `_skip_dead`：关掉它只是不省那些瓦片，答案不该有任何差别 —— 被跳过的活没有读者
    （读出在最后一个瓦片里，K/V 出自投影那一步）。另外数一下 `wait_upto` 的次数，确认注意力
    瓦片真的少收一个；不数的话"答案一致"也可能是因为根本没跳过。
    """
    from sysone.devices.apple.metal import LANE_LONG

    tokens = TOKENS * (S // len(TOKENS))
    prefix = S // 4
    pipeline = chunked._pipeline
    assert pipeline is not None
    runtime = pipeline._runtime
    counted = {"n": 0}
    original = runtime.wait_upto

    def counted_wait(lane: int, count: int) -> float:
        if lane == LANE_LONG:
            counted["n"] += 1
        return original(lane, count)

    runtime.wait_upto = counted_wait     # type: ignore[method-assign]
    try:
        counted["n"] = 0
        with_dead = chunked.logits(tokens, prefix=prefix)
        pumped_with_dead = counted["n"]
        pipeline._skip_dead = False
        try:
            counted["n"] = 0
            without_dead = chunked.logits(tokens, prefix=prefix)
            pumped_without_dead = counted["n"]
        finally:
            pipeline._skip_dead = True
    finally:
        runtime.wait_upto = original   # type: ignore[method-assign]

    assert pumped_with_dead == pumped_without_dead - 1, (
        f"死瓦片没有少收注意力瓦片：S3 开 {pumped_with_dead} 次、关 {pumped_without_dead} 次"
    )
    assert np.array_equal(with_dead, without_dead), "S3 改变了答案（跳过的瓦片不该有读者）"
