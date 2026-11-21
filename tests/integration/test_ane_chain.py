"""20 层链：概率形状/归一、重复调用逐位一致（缺 dylib 或真实权重就跳过）。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple.chain import SplitChain, artifact_state
from sysone.errors import NotReadyError
from sysone.models.families import mude
from sysone.models.vision.merge import Placement

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", ""))
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
        # 冷产物目录（`tmp_path` 是新的）：编译那一段真的花了时间，且与装载分开记。
        # 这两条是"启动报告"在真链上的落点 —— 少了它们，`compile_s` 又可以是恒 0 的摆设。
        assert chain.compile_s > 0.0
        assert chain.load_s > 0.0
        # `doctor` 的产物一栏与装载共用同一份身份计算：刚编出来的必须都算"可复用"，
        # 否则那份报告就会对着一个热目录说"要重编"。
        state = artifact_state(MUDE_DIR, weights=mude.weight_plan(MUDE_DIR), s=256,
                               artifacts_dir=tmp_path / "programs")
        assert state.hits == state.total > 0
        tokens = [1, 2, 3, 4]
        probabilities = chain.forward(tokens, options=3)
        assert probabilities.shape == (3,)
        assert np.all(probabilities >= 0.0)
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
        assert chain.logits(tokens).shape == (255,)
        # 图像合流（票据 07 §7.4）：`mm` 覆盖的那几行必须**就是**换成那些 token 的效果。
        # 这条断言抓的错很具体 —— 忽略 `mm`（图"进去过"、注意力按占位符自己的 embedding 跑）
        # 或改错行，都会让它当场变红；而它不需要跑塔，所以在这条最便宜的用例里就能钉住。
        spliced = np.asarray(
            chain._store.read_rows(chain._embed_name, np.array([7, 7, 7])), np.float32)
        placement = Placement(spans=((1, 3),), rows=spliced,
                              positions=np.tile(np.arange(len(tokens)), (3, 1)))
        got = chain.forward(tokens, options=3, mm=placement)
        want = chain.forward([tokens[0], 7, 7, 7], options=3)
        assert np.array_equal(got, want), "splice 之后与直接用那些 token 不等价"
        # ANE 程序与 numpy 一侧都是确定性的：重复调用必须逐位相同。
        assert np.array_equal(chain.forward(tokens, options=3), probabilities)
    finally:
        chain.close()


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_quant_modes_keep_separate_artifact_directories(tmp_path: Path) -> None:
    """int8 与 fp16 两档各占一个目录：切档不该顶掉另一档的产物。"""
    plan = mude.weight_plan(MUDE_DIR)
    int8 = artifact_state(MUDE_DIR, weights=plan, s=256, artifacts_dir=tmp_path / "programs")
    fp16 = artifact_state(MUDE_DIR, weights=plan, s=256, artifacts_dir=tmp_path / "programs",
                          quantize=False)
    assert int8.directory.name == "s256"
    assert fp16.directory.name == "s256-fp16"
    assert int8.directory != fp16.directory


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_fp16_chain_loads_and_runs(tmp_path: Path) -> None:
    """fp16 档走的是 `ane_layer_weights` + `build_blob(quantize=False)`，与 int8 不同的那条路。"""
    chain = SplitChain.load(MUDE_DIR, weights=mude.weight_plan(MUDE_DIR), s=256,
                            scratch_dir=tmp_path / "programs", quantize=False)
    try:
        probabilities = chain.forward([1, 2, 3, 4], options=3)
        assert probabilities.shape == (3,)
        assert np.all(probabilities >= 0.0)
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
        assert chain.compile_s > 0.0
    finally:
        chain.close()
