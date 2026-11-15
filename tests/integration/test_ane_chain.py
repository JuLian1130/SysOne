"""20 层链：概率形状/归一、重复调用逐位一致（缺 dylib 或真实权重就跳过）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple.chain import SplitChain
from sysone.models.families import mude

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
WEIGHTS = MUDE_DIR / "model.safetensors"


def _skip_reason() -> str | None:
    """与 `test_ane_layer.py` 同一套跳过条件：非 macOS、缺权重、缺 dylib。"""
    if sys.platform != "darwin":
        return "ANE 只在 macOS 上可用"
    if not WEIGHTS.is_file():
        return f"缺少真实权重：{WEIGHTS}"
    from sysone.devices.apple import available

    if not available():
        return "ANE dylib 不可用（先跑 tools/build_ane.sh）"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_chain_probabilities_and_repeat(tmp_path: Path) -> None:
    """最省的一次前向：形状对、概率归一，而且第二次调用与第一次逐位相同。"""
    chain = SplitChain.load(MUDE_DIR, weights=mude.weight_plan(MUDE_DIR), s=256,
                            scratch_dir=tmp_path / "programs")
    try:
        tokens = [1, 2, 3, 4]
        probabilities = chain.forward(tokens, options=3)
        assert probabilities.shape == (3,)
        assert np.all(probabilities >= 0.0)
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
        assert chain.logits(tokens).shape == (255,)
        # ANE 程序与 numpy 一侧都是确定性的：重复调用必须逐位相同。
        assert np.array_equal(chain.forward(tokens, options=3), probabilities)
    finally:
        chain.close()
