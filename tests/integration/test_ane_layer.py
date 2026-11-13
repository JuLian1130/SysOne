"""M2.1 一层三段 ANE 计算的对拍（真 ANE + mude 真权重）。

一层在 ANE 上分三段：投影（`hidden → cout`）、输出投影 + 残差（`2×hidden → hidden`）、
融合 FFN（`2×hidden → hidden`，gate/up 在 ANE 内做 silu×up，down 的 K 维按 `ffn/hidden`
切段后求和）。判据比 docs 的端到端阈值（2.5e-2）严，因为这里只测一段、且参考用的是
**ANE 实际读到的同一份反量化权重** —— 量化误差不混进来，留到 M2.2 的 20 层链。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple import available
from sysone.devices.apple.compile import (
    AneProgram,
    ffn_mil,
    ffn_parts,
    out_mil,
    projection_mil,
    write_program,
)
from sysone.devices.apple.runtime import AneRuntime
from sysone.errors import ConfigError
from sysone.weights.packing import ane_blob
from sysone.weights.store import SafeTensorFile

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
WEIGHTS = MUDE_DIR / "model.safetensors"
LAYER0 = "language_model.layers.0"
PROJ = f"{LAYER0}.linear_attn.in_proj_qkv.weight"
OUT = f"{LAYER0}.linear_attn.out_proj.weight"
GATE = f"{LAYER0}.mlp.gate_proj.weight"
UP = f"{LAYER0}.mlp.up_proj.weight"
DOWN = f"{LAYER0}.mlp.down_proj.weight"
S = 1024

PROJ_BUDGET = 1e-2
OUT_BUDGET = 1e-2
FFN_BUDGET = 5e-3


def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "ANE 只在 macOS 上"
    if not WEIGHTS.is_file():
        return f"缺真实权重：{WEIGHTS}（可用 SYS1_MUDE_DIR 指向 mude-bench）"
    if not available():
        return "ANE dylib 不可用（运行 tools/build_ane.sh）"
    return None


def _load(names: list[str]) -> dict[str, np.ndarray]:
    source = SafeTensorFile(WEIGHTS)
    try:
        return {n: np.ascontiguousarray(source.read(n), dtype=np.float16) for n in names}
    finally:
        source.close()


def _rel_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    a = actual.astype(np.float64)
    b = expected.astype(np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _run(files, *, name: str, runtime: AneRuntime, cin: int, cout: int, x: np.ndarray):
    program = AneProgram(runtime, files, name=name, cin=cin, cout=cout, s=S)
    try:
        first, ms = program.run(x)
        second, _ = program.run(x)
    finally:
        program.close()
    np.testing.assert_array_equal(first, second, "同一输入两次执行必须逐位一致")
    return first, ms


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_projection_matches_numpy(tmp_path: Path) -> None:
    weight = _load([PROJ])[PROJ]
    hidden, cout = weight.shape[1], weight.shape[0]
    x = np.random.default_rng(7).standard_normal((hidden, S)).astype(np.float16)

    blob, entries = ane_blob.build([("w", weight)], quantize=True)
    mil = projection_mil(hidden=hidden, cout=cout, s=S, entry=entries["w"])
    files = write_program(tmp_path / "proj", mil=mil, blob=blob, s=S, meta={"tensor": PROJ})
    assert files.reused is False  # 首次落盘

    runtime = AneRuntime()
    try:
        y, _ = _run(files, name="m21_proj", runtime=runtime, cin=hidden, cout=cout, x=x)
    finally:
        runtime.close()

    reference = ane_blob.quant_per_channel(weight).dequant.astype(np.float32) @ x.astype(np.float32)
    rel = _rel_l2(y, reference)
    assert rel < PROJ_BUDGET, f"relL2 {rel:.2e} 超过 {PROJ_BUDGET}"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_out_projection_matches_numpy(tmp_path: Path) -> None:
    weight = _load([OUT])[OUT]
    hidden = weight.shape[1]
    rng = np.random.default_rng(11)
    v = rng.standard_normal((hidden, S)).astype(np.float16)   # 注意力输出
    r = rng.standard_normal((hidden, S)).astype(np.float16)   # 残差

    blob, entries = ane_blob.build([("o", weight)], quantize=True)
    mil = out_mil(hidden=hidden, s=S, entry=entries["o"])
    files = write_program(tmp_path / "out", mil=mil, blob=blob, s=S, meta={"tensor": OUT})

    runtime = AneRuntime()
    try:
        y, _ = _run(files, name="m21_out", runtime=runtime, cin=2 * hidden, cout=hidden,
                    x=np.concatenate([v, r], axis=0))
    finally:
        runtime.close()

    w = ane_blob.quant_per_channel(weight).dequant.astype(np.float32)
    reference = (w @ v.astype(np.float32) + r.astype(np.float32)).astype(np.float16)
    rel = _rel_l2(y, reference)
    assert rel < OUT_BUDGET, f"relL2 {rel:.2e} 超过 {OUT_BUDGET}"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_ffn_matches_numpy(tmp_path: Path) -> None:
    gate_w = _load([GATE])[GATE]
    up_w = _load([UP])[UP]
    down_w = _load([DOWN])[DOWN]
    hidden = gate_w.shape[1]
    ffn = down_w.shape[1]
    parts = ffn_parts(hidden, ffn)
    assert parts == ffn // hidden
    chunks = [down_w[:, i * hidden:(i + 1) * hidden] for i in range(parts)]

    rng = np.random.default_rng(13)
    xn = rng.standard_normal((hidden, S)).astype(np.float16)  # 归一化后的 hidden
    r = rng.standard_normal((hidden, S)).astype(np.float16)   # 残差

    items = [("gate", gate_w), ("up", up_w)] + [(f"down{i}", c) for i, c in enumerate(chunks)]
    blob, entries = ane_blob.build(items, quantize=True)
    mil = ffn_mil(hidden=hidden, ffn=ffn, s=S, entries=entries)
    files = write_program(tmp_path / "ffn", mil=mil, blob=blob, s=S, meta={"tensors": [UP, GATE]})

    runtime = AneRuntime()
    try:
        y, _ = _run(files, name="m21_ffn", runtime=runtime, cin=2 * hidden, cout=hidden,
                    x=np.concatenate([xn, r], axis=0))
        peak = runtime.peak_footprint_gb()
    finally:
        runtime.close()
    assert peak < 1.0, f"单段 FFN 的进程占用 {peak:.3f} GB 明显超标"

    gate = ane_blob.quant_per_channel(gate_w).dequant.astype(np.float32)
    up = ane_blob.quant_per_channel(up_w).dequant.astype(np.float32)
    down = [ane_blob.quant_per_channel(c).dequant.astype(np.float32) for c in chunks]
    h32 = _silu(gate @ xn.astype(np.float32)) * (up @ xn.astype(np.float32))
    h16 = h32.astype(np.float16)
    accumulated = np.zeros((hidden, S), dtype=np.float32)
    for i, chunk in enumerate(down):
        accumulated += chunk @ h16[i * hidden:(i + 1) * hidden].astype(np.float32)
    reference = (accumulated + r.astype(np.float32)).astype(np.float16)

    rel = _rel_l2(y, reference)
    assert rel < FFN_BUDGET, f"relL2 {rel:.2e} 超过 {FFN_BUDGET}"


def test_ffn_parts_is_three_for_mude_geometry() -> None:
    assert ffn_parts(2048, 6144) == 3


def test_ffn_parts_rejects_4b_geometry() -> None:
    """4B 是 9216/2560=3.6，需要变长切分；M2 只做 mude，这里必须显式失败。"""
    with pytest.raises(ConfigError) as excinfo:
        ffn_parts(2560, 9216)
    assert excinfo.value.context["hidden"] == 2560
    assert excinfo.value.context["ffn"] == 9216


def test_program_files_are_reused_on_second_write(tmp_path: Path) -> None:
    _, entries = ane_blob.build([("w", np.eye(8, dtype=np.float16))], quantize=False)
    mil = projection_mil(hidden=8, cout=8, s=16, entry=entries["w"])
    first = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert first.reused is False
    second = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert second.reused is True
    third = write_program(tmp_path / "p8", mil=mil, blob=b"y" * 4, s=16)
    assert third.reused is False, "blob 变了就必须重写，不能沿用旧 sidecar"
