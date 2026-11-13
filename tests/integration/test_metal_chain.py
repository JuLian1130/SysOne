"""Metal 流水线与 numpy 链的端到端对拍（需要真 mude 权重 + 两个 dylib）。

只在设了 `SYS1_MUDE_WEIGHTS` 时跑：它会各建一条 20 层的链，约 40 s。
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


def test_metal_chain_matches_the_numpy_chain():
    MudeChain = _chains()
    cpu = MudeChain.load(MUDE, s=S)
    try:
        gpu = MudeChain.load(MUDE, s=S, metal=True)
        try:
            length = len(TOKENS)
            lcpu = cpu.logits(TOKENS, prefix=0)
            lgpu = gpu.logits(TOKENS, prefix=0)
            rel = float(np.linalg.norm(lgpu - lcpu) / np.linalg.norm(lcpu))
            assert rel < TOL, f"metal 与 numpy 链的 logits 偏差 {rel:.4e} 超过 {TOL}"
        finally:
            gpu.close()
    finally:
        cpu.close()


def test_metal_chain_keeps_the_last_layer_hidden_in_range():
    MudeChain = _chains()
    cpu = MudeChain.load(MUDE, s=S)
    try:
        gpu = MudeChain.load(MUDE, s=S, metal=True)
        try:
            trace: list = []
            cpu.logits(TOKENS, prefix=0, trace=trace)
            gpu.logits(TOKENS, prefix=0)
            last = gpu._pipeline.last_hidden(len(TOKENS))
            ref = np.asarray(trace[-1], np.float32)
            rel = float(np.linalg.norm(last - ref) / np.linalg.norm(ref))
            assert rel < 5e-2, f"末层 hidden 偏差 {rel:.4e}"
        finally:
            gpu.close()
    finally:
        cpu.close()
