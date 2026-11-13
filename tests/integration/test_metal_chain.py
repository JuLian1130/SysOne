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
