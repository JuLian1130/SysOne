"""M2.1 单层对拍（真 ANE + 真权重）。

拿 mude 的第 0 层 `in_proj_qkv`（2048 → 6144）在 `S=1024` 上跑一次 1×1 conv，和 numpy 用
同一份反量化权重算出的结果比 relL2。判据比 docs 的端到端阈值（2.5e-2）严得多，因为这里
只测一层、且没有量化误差混入（参考就是 ANE 读到的权重）。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple import available
from sysone.devices.apple.compile import AneProjection, projection_mil, write_program
from sysone.devices.apple.runtime import AneRuntime
from sysone.weights.packing import ane_blob
from sysone.weights.store import SafeTensorFile

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
WEIGHTS = MUDE_DIR / "model.safetensors"
TENSOR = "language_model.layers.0.linear_attn.in_proj_qkv.weight"
S = 1024
REL_L2_BUDGET = 1e-2

def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "ANE 只在 macOS 上"
    if not WEIGHTS.is_file():
        return f"缺真实权重：{WEIGHTS}（可用 SYS1_MUDE_DIR 指向 mude-bench）"
    if not available():
        return "ANE dylib 不可用（运行 tools/build_ane.sh）"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_projection_matches_numpy(tmp_path: Path) -> None:
    source = SafeTensorFile(WEIGHTS)
    try:
        weight = np.ascontiguousarray(source.read(TENSOR), dtype=np.float16)
    finally:
        source.close()
    assert weight.shape[1] > 0

    rng = np.random.default_rng(7)
    x = rng.standard_normal((weight.shape[1], S)).astype(np.float16)
    q = ane_blob.quant_per_channel(weight)
    blob, entries = ane_blob.build([("w", weight)], quantize=True)
    mil = projection_mil(hidden=weight.shape[1], cout=weight.shape[0], s=S, entry=entries["w"])
    files = write_program(tmp_path / "proj", mil=mil, blob=blob, s=S, meta={"tensor": TENSOR})
    assert files.reused is False  # 首次落盘

    runtime = AneRuntime()
    projection = AneProjection(
        runtime, files, name="m21_parity", hidden=weight.shape[1], cout=weight.shape[0], s=S
    )
    try:
        first, _ = projection.run(x)
        second, _ = projection.run(x)
    finally:
        projection.close()
        runtime.close()

    reference = q.dequant.astype(np.float32) @ x.astype(np.float32)
    rel = float(np.linalg.norm(first.astype(np.float64) - reference)
                / np.linalg.norm(reference.astype(np.float64)))
    assert rel < REL_L2_BUDGET, f"relL2 {rel:.2e} 超过 {REL_L2_BUDGET}"
    np.testing.assert_array_equal(first, second, "同一输入两次执行必须逐位一致")


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_program_files_are_reused_on_second_write(tmp_path: Path) -> None:
    _, entries = ane_blob.build([("w", np.eye(8, dtype=np.float16))], quantize=False)
    mil = projection_mil(hidden=8, cout=8, s=16, entry=entries["w"])
    first = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert first.reused is False
    second = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert second.reused is True
    third = write_program(tmp_path / "p8", mil=mil, blob=b"y" * 4, s=16)
    assert third.reused is False, "blob 变了就必须重写，不能沿用旧 sidecar"
