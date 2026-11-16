"""M2.1 一层三段 ANE 计算的对拍（真 ANE + mude 真权重）。

一层在 ANE 上分三段：投影（`hidden → cout`）、输出投影 + 残差（`core_out + hidden → hidden`）、
融合 FFN（`2×hidden → hidden`，gate/up 在 ANE 内做 silu×up，down 的 K 维按 `ffn_segments()`
切段后求和 —— 每段 `hidden` 列，余数落在最后一段）。`core_out` 是 `o_proj` 的输入行数：2B 恰好等于 `hidden`（所以看起来像
`2×hidden`），4B 是 4096 ≠ 2560。判据比 docs 的端到端阈值（2.5e-2）严，因为这里只测一段、且参考用的是
**ANE 实际读到的同一份反量化权重** —— 量化误差不混进来，留到 M2.2 的 20 层链。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple import available
from sysone.devices.apple.compile import (
    AneLayer,
    AneProgram,
    SurfaceSet,
    ffn_mil,
    ffn_segments,
    layer_mil,
    out_mil,
    projection_mil,
    write_program,
)
from sysone.devices.apple.runtime import AneRuntime
from sysone.domain.attention import AttentionRole
from sysone.devices.apple.blobplan import ane_layer_segments
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
    hidden, core_out = weight.shape[0], weight.shape[1]
    rng = np.random.default_rng(11)
    v = rng.standard_normal((core_out, S)).astype(np.float16)   # 核心输出
    r = rng.standard_normal((hidden, S)).astype(np.float16)     # 残差

    blob, entries = ane_blob.build([("o", weight)], quantize=True)
    mil = out_mil(hidden=hidden, core_out=core_out, s=S, entry=entries["o"])
    files = write_program(tmp_path / "out", mil=mil, blob=blob, s=S, meta={"tensor": OUT})

    runtime = AneRuntime()
    try:
        y, _ = _run(files, name="m21_out", runtime=runtime, cin=core_out + hidden,
                    cout=hidden, x=np.concatenate([v, r], axis=0))
    finally:
        runtime.close()

    w = ane_blob.quant_per_channel(weight).dequant.astype(np.float32)
    reference = (w @ v.astype(np.float32) + r.astype(np.float32)).astype(np.float16)
    rel = _rel_l2(y, reference)
    assert rel < OUT_BUDGET, f"relL2 {rel:.2e} 超过 {OUT_BUDGET}"


@pytest.mark.skipif(not available(), reason="ANE dylib 不可用（先跑 tools/build_ane.sh）")
def test_out_projection_at_the_4b_width_matches_numpy(tmp_path: Path) -> None:
    """4B 宽度的 out program（`[4096 | 2560] → 2560`）在 ANE 上真算一遍。

    权重是**合成的**：这条只钉 **program 形状与切片语义**，不钉那份模型（真权重的整链证据在
    票据 15 §22.2，那条要 5 GiB 级的检查点，不进闸门）。参考口径与 mude 那条一样 —— 用 ANE
    实际读到的那份反量化权重，量化误差不混进来。
    """
    hidden, core_out = 2560, 4096
    rng = np.random.default_rng(29)
    weight = (rng.standard_normal((hidden, core_out)) * 0.05).astype(np.float16)
    v = rng.standard_normal((core_out, S)).astype(np.float16)   # 核心输出（比 hidden 宽）
    r = rng.standard_normal((hidden, S)).astype(np.float16)     # 残差

    blob, entries = ane_blob.build([("o", weight)], quantize=True)
    mil = out_mil(hidden=hidden, core_out=core_out, s=S, entry=entries["o"])
    files = write_program(tmp_path / "out_4b", mil=mil, blob=blob, s=S,
                          meta={"case": "m21-out-4b"})

    runtime = AneRuntime()
    try:
        y, ms = _run(files, name="m21_out_4b", runtime=runtime,
                    cin=core_out + hidden, cout=hidden,
                    x=np.concatenate([v, r], axis=0))
    finally:
        runtime.close()

    w = ane_blob.quant_per_channel(weight).dequant.astype(np.float32)
    reference = (w @ v.astype(np.float32) + r.astype(np.float32)).astype(np.float16)
    rel = _rel_l2(y, reference)
    assert rel < OUT_BUDGET, f"relL2 {rel:.2e} 超过 {OUT_BUDGET}（4B 宽度的 out 投影）"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_the_4b_layer_program_runs_with_an_uneven_ffn_split(tmp_path: Path) -> None:
    """h2o 的整层 program 在 ANE 上真跑一遍：12352 行投影、`[4096|2560]` out、四段 FFN。

    条目表用**生产路径的那一份**（`blobplan.ane_layer_segments`），所以这条同时钉住三件事：
    4B 的 9216/2560 真的切成了 2560/2560/2560/1536、最后一段的窄 conv 被 ANE 接受、
    四段分别算完累加出来的就是 `down @ h`。权重是合成的 —— 这条钉的是形状，真权重的整链
    证据在票据 15 §22.2。
    """
    hidden, ffn, core_out, cout = 2560, 9216, 4096, 12352
    segments = ane_layer_segments(layer=0, layer_type=AttentionRole.LINEAR,
                                 hidden=hidden, ffn=ffn, prefix="language_model")
    down = {name: columns[0][1] for name, columns in segments.items()
            if name.startswith("down")}
    assert [columns.stop - columns.start for columns in down.values()] == [2560, 2560, 2560, 1536]

    rng = np.random.default_rng(31)

    def matrix(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.05).astype(np.float16)

    weights = {
        "proj": matrix(cout, hidden), "o": matrix(hidden, core_out),
        "gate": matrix(ffn, hidden), "up": matrix(ffn, hidden),
        **{name: matrix(hidden, columns.stop - columns.start) for name, columns in down.items()},
    }
    blob, entries = ane_blob.build(weights, quantize=True)
    mil = layer_mil(hidden=hidden, core_out=core_out, cout=cout, ffn=ffn, s=S, entries=entries)
    files = write_program(tmp_path / "layer0_4b", mil=mil, blob=blob, s=S,
                          meta={"case": "m21-layer-4b"})

    x = rng.standard_normal((hidden, S)).astype(np.float16)
    v = rng.standard_normal((core_out, S)).astype(np.float16)
    r = rng.standard_normal((hidden, S)).astype(np.float16)

    runtime = AneRuntime()
    layer = AneLayer(runtime, files, name="m21_layer_4b", hidden=hidden, core_out=core_out,
                     cout=cout, ffn=ffn, s=S)
    try:
        y_proj, _ = layer.project(x)
        y_out, _ = layer.output(v, r)
        y_ffn, _ = layer.feed_forward(x, r)
    finally:
        layer.close()
        runtime.close()

    w = {name: ane_blob.quant_per_channel(weight).dequant.astype(np.float32)
         for name, weight in weights.items()}
    assert _rel_l2(y_proj, w["proj"] @ x.astype(np.float32)) < PROJ_BUDGET
    assert _rel_l2(y_out, w["o"] @ v.astype(np.float32) + r.astype(np.float32)) < OUT_BUDGET
    h = (_silu(w["gate"] @ x.astype(np.float32)) * (w["up"] @ x.astype(np.float32))).astype(np.float16)
    accumulated = np.zeros((hidden, S), dtype=np.float32)
    for name, columns in down.items():
        accumulated += w[name] @ h[columns].astype(np.float32)
    assert _rel_l2(y_ffn, accumulated + r.astype(np.float32)) < FFN_BUDGET


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_ffn_matches_numpy(tmp_path: Path) -> None:
    gate_w = _load([GATE])[GATE]
    up_w = _load([UP])[UP]
    down_w = _load([DOWN])[DOWN]
    hidden = gate_w.shape[1]
    ffn = down_w.shape[1]
    segments = ffn_segments(hidden, ffn)
    chunks = [down_w[:, columns] for columns in segments]

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
    for chunk, columns in zip(down, segments):
        accumulated += chunk @ h16[columns].astype(np.float32)
    reference = (accumulated + r.astype(np.float32)).astype(np.float16)

    rel = _rel_l2(y, reference)
    assert rel < FFN_BUDGET, f"relL2 {rel:.2e} 超过 {FFN_BUDGET}"


def test_ffn_is_cut_into_equal_parts_for_mude_geometry() -> None:
    segments = ffn_segments(2048, 6144)
    assert [columns.stop - columns.start for columns in segments] == [2048, 2048, 2048]


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_split_projection_matches_numpy(tmp_path: Path) -> None:
    """全注意力层（第 3 层）：投影拆成 q 段 + kv 段，两段各写各自的行，`project()` 拼回整块。

    这条用例同时钉住拆投影（票据 06）的两个前提：**切行不改变量化**（int8 是按输出行量的，
    所以两半的 codes 与整块逐字节相同），以及"比声明小的输出张量可以落在更大的面上"
    （kv 段只有 1024 行，落到 `out_in` 的 4096 行里 —— 这是 A1 借面的全部依据）。
    """
    full = 3
    qkv = tuple(f"language_model.layers.{full}.self_attn.{n}_proj.weight" for n in ("q", "k", "v"))
    rest = (
        f"language_model.layers.{full}.self_attn.o_proj.weight",
        f"language_model.layers.{full}.mlp.gate_proj.weight",
        f"language_model.layers.{full}.mlp.up_proj.weight",
        f"language_model.layers.{full}.mlp.down_proj.weight",
    )
    w = _load(list(qkv) + list(rest))
    proj = np.concatenate([w[name] for name in qkv], axis=0)
    hidden = int(proj.shape[1])
    q_rows = int(w[qkv[0]].shape[0])
    kv_rows = int(proj.shape[0]) - q_rows
    assert (q_rows, kv_rows) == (4096, 1024), "第 3 层的 q/kv 行数不是 mude 几何"

    ffn = int(w[rest[3]].shape[1])
    entries = {
        "proj_q": proj[:q_rows], "proj_kv": proj[q_rows:],
        "o": w[rest[0]], "gate": w[rest[1]], "up": w[rest[2]],
    }
    entries.update({f"down{i}": w[rest[3]][:, columns]
                    for i, columns in enumerate(ffn_segments(hidden, ffn))})

    blob, blob_entries = ane_blob.build(entries, quantize=True)
    mil = layer_mil(hidden=hidden, core_out=hidden, cout=q_rows + kv_rows, ffn=ffn, s=S,
                    entries=blob_entries, split=True)
    assert mil.count("func ") == 4
    files = write_program(tmp_path / "layer3_split", mil=mil, blob=blob, s=S)

    rng = np.random.default_rng(23)
    x = rng.standard_normal((hidden, S)).astype(np.float16)
    runtime = AneRuntime()
    layer = AneLayer(runtime, files, name="m22_layer3_split", hidden=hidden,
                     core_out=hidden, cout=q_rows + kv_rows, ffn=ffn, s=S,
                     split_rows=(q_rows, kv_rows))
    try:
        y, ms = layer.project(x)
        # 两段各自读回来：q 段落在 proj_out，kv 段落在 out_in 的前 kv_rows 行
        layer._write("proj_in", hidden, x)
        layer.eval_proc("proj_q")
        got_q = layer._read("proj_out", q_rows)
        layer._write("proj_in", hidden, x)
        layer.eval_proc("proj_kv")
        got_kv = layer._read("out_in", kv_rows)
    finally:
        layer.close()
        runtime.close()

    # 整块量化（与按行切开再量化逐字节相同）就是两段共同的参考
    reference = ane_blob.quant_per_channel(proj).dequant.astype(np.float32) @ x.astype(np.float32)
    assert _rel_l2(y, reference) < PROJ_BUDGET, "拼回的整块与 numpy 参考不一致"
    assert _rel_l2(got_q, reference[:q_rows]) < PROJ_BUDGET, "q 段与整块的前 4096 行不一致"
    assert _rel_l2(got_kv, reference[q_rows:]) < PROJ_BUDGET, "kv 段与整块的后 1024 行不一致"
    assert y.shape == (q_rows + kv_rows, S)
    assert ms > 0


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_fused_layer_program_runs_three_procedures(tmp_path: Path) -> None:
    """整层一个 program、三个 procedure，一次 `load` 拿到三段计算。

    第 0 层是线性注意力层，投影的输出通道是 `8224 = qkv 6144 + z 2048 + a 16 + b 16`
    （全注意力层是 `5120 = q 2048 + k 512 + v 512 + gate 2048`）。这里按真实权重拼出
    那一整块，顺便验证"投影矩阵是行拼接"这件事在真数据上成立。
    """
    names = [
        PROJ,
        f"{LAYER0}.linear_attn.in_proj_z.weight",
        f"{LAYER0}.linear_attn.in_proj_a.weight",
        f"{LAYER0}.linear_attn.in_proj_b.weight",
        OUT, GATE, UP, DOWN,
    ]
    w = _load(names)
    proj = np.concatenate([
        w[PROJ],
        w[f"{LAYER0}.linear_attn.in_proj_z.weight"],
        w[f"{LAYER0}.linear_attn.in_proj_a.weight"],
        w[f"{LAYER0}.linear_attn.in_proj_b.weight"],
    ], axis=0)
    assert proj.shape[0] == 8224
    hidden, cout = proj.shape[1], proj.shape[0]
    ffn = w[DOWN].shape[1]
    chunks = [w[DOWN][:, columns] for columns in ffn_segments(hidden, ffn)]

    items = {"proj": proj, "o": w[OUT], "gate": w[GATE], "up": w[UP]}
    items.update({f"down{i}": c for i, c in enumerate(chunks)})
    blob, entries = ane_blob.build(items, quantize=True)
    mil = layer_mil(hidden=hidden, core_out=hidden, cout=cout, ffn=ffn, s=S, entries=entries)
    assert mil.count("func ") == 3
    assert "func main<ios18>" in mil and "func p1<ios18>" in mil and "func p2<ios18>" in mil

    files = write_program(tmp_path / "layer0", mil=mil, blob=blob, s=S)
    rng = np.random.default_rng(17)
    x = rng.standard_normal((hidden, S)).astype(np.float16)
    v = rng.standard_normal((hidden, S)).astype(np.float16)
    r = rng.standard_normal((hidden, S)).astype(np.float16)

    runtime = AneRuntime()
    layer = AneLayer(runtime, files, name="m22_layer0", hidden=hidden, core_out=hidden,
                     cout=cout, ffn=ffn, s=S)
    try:
        y_proj, ms0 = layer.project(x)
        y_out, ms1 = layer.output(v, r)
        y_ffn, ms2 = layer.feed_forward(x, r)
    finally:
        layer.close()
        runtime.close()

    q_proj = ane_blob.quant_per_channel(proj).dequant.astype(np.float32)
    ref_proj = q_proj @ x.astype(np.float32)
    assert _rel_l2(y_proj, ref_proj) < PROJ_BUDGET

    q_out = ane_blob.quant_per_channel(w[OUT]).dequant.astype(np.float32)
    ref_out = (q_out @ v.astype(np.float32) + r.astype(np.float32)).astype(np.float16)
    assert _rel_l2(y_out, ref_out) < OUT_BUDGET

    q_gate = ane_blob.quant_per_channel(w[GATE]).dequant.astype(np.float32)
    q_up = ane_blob.quant_per_channel(w[UP]).dequant.astype(np.float32)
    q_down = [ane_blob.quant_per_channel(c).dequant.astype(np.float32) for c in chunks]
    h = (_silu(q_gate @ x.astype(np.float32)) * (q_up @ x.astype(np.float32))).astype(np.float16)
    acc = np.zeros((hidden, S), dtype=np.float32)
    for i, chunk in enumerate(q_down):
        acc += chunk @ h[i * hidden:(i + 1) * hidden].astype(np.float32)
    ref_ffn = (acc + r.astype(np.float32)).astype(np.float16)
    assert _rel_l2(y_ffn, ref_ffn) < FFN_BUDGET

    # 三个 procedure 的耗时是分开计的，加总就是整层 ANE 时间
    assert ms0 > 0 and ms1 > 0 and ms2 > 0
    assert y_proj.shape == (cout, S) and y_out.shape == (hidden, S) and y_ffn.shape == (hidden, S)


def test_ffn_is_cut_into_three_parts_and_a_tail_for_4b_geometry() -> None:
    """9216/2560 = 3.6：三段等长的加一段 1536，覆盖整个 K 维。

    段长不齐是 ANE 的输入事实（实测收），所以这里不再是"必须整除"的拒绝，而是钉住切法。
    """
    segments = ffn_segments(2560, 9216)
    assert [columns.stop - columns.start for columns in segments] == [2560, 2560, 2560, 1536]
    assert sum(columns.stop - columns.start for columns in segments) == 9216


def test_program_files_are_reused_on_second_write(tmp_path: Path) -> None:
    _, entries = ane_blob.build([("w", np.eye(8, dtype=np.float16))], quantize=False)
    mil = projection_mil(hidden=8, cout=8, s=16, entry=entries["w"])
    first = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert first.reused is False
    second = write_program(tmp_path / "p8", mil=mil, blob=b"x" * 4, s=16)
    assert second.reused is True
    third = write_program(tmp_path / "p8", mil=mil, blob=b"y" * 4, s=16)
    assert third.reused is False, "blob 变了就必须重写，不能沿用旧 sidecar"


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_pool_slots_and_async_eval(tmp_path: Path) -> None:
    """同一 program 的两套面各绑一个槽并独立执行；`issue_proc`/`reap_proc` 能异步跑。

    这两个能力是第 3 步（双链：两条链同时用同一份权重、各用一套面）的机制验收：槽 1 是为了"一次 eval 在飞
    时用另一套面准备下一段"，异步 eval 是为了"eval 在飞时 CPU 不阻塞"。这里用小几何合成
    权重，只钉机制不改数值。
    """
    # 几何要落在 ANE 肯跑的范围内：通道数太窄（如 hidden=8 / ffn=24）时 ANE 会在
    # `evaluate` 阶段直接报 `Program Inference error`（status=0x1d）。这里取 64/192/64，
    # `ffn_segments(64, 192)` 刚好三段等长，覆盖 down 的三段切片。
    hidden, ffn, cout, s = 64, 192, 64, 64
    rng = np.random.default_rng(3)
    items = {
        "proj": rng.standard_normal((cout, hidden)).astype(np.float16),
        "o": rng.standard_normal((hidden, hidden)).astype(np.float16),
        "gate": rng.standard_normal((ffn, hidden)).astype(np.float16),
        "up": rng.standard_normal((ffn, hidden)).astype(np.float16),
        **{f"down{i}": rng.standard_normal((hidden, columns.stop - columns.start)).astype(np.float16)
           for i, columns in enumerate(ffn_segments(hidden, ffn))},
    }
    blob, entries = ane_blob.build(items, quantize=False)
    mil = layer_mil(hidden=hidden, core_out=hidden, cout=cout, ffn=ffn, s=s,
                    entries=entries)
    files = write_program(tmp_path / "slots", mil=mil, blob=blob, s=s)

    runtime = AneRuntime()
    pool = [SurfaceSet(runtime, hidden=hidden, core_out=hidden, cout=cout, s=s)
            for _ in range(2)]
    layer = AneLayer(runtime, files, name="m22_slots", hidden=hidden, core_out=hidden,
                     cout=cout, ffn=ffn, s=s, pool=pool)
    try:
        x = rng.standard_normal((hidden, s)).astype(np.float16)
        layer._write("proj_in", hidden, x)
        layer.eval_proc("proj")
        slot0 = layer._read("proj_out", cout)

        # 槽 1 绑同一对面（机制验收：多槽走的是 `e8_attach_slot` / `e8_eval_slot`）。
        # 槽 1 = 池里第 1 套面：同一个逻辑位置写同一份输入，走 `e8_eval_slot` 执行。
        pool[1].surfaces["proj_in"].array(np.float16, (1, hidden, 1, s))[...] = \
            x.reshape(1, hidden, 1, s)
        layer.eval_proc("proj", slot=1)
        slot1 = pool[1].surfaces["proj_out"].array(np.float16, (1, cout, 1, s))
        np.testing.assert_array_equal(slot0, slot1.reshape(cout, s),
                                      "同一输入在两个槽位必须逐位一致")

        # 异步：发射后立刻返回，回收时给出毫秒数。
        layer.issue_proc("proj")
        assert runtime.pending is True
        ms = layer.reap_proc()
        assert runtime.pending is False
        assert ms > 0, "回收得到的应该是这次 eval 的毫秒数"
    finally:
        layer.close()
        for surface_set in pool:
            surface_set.free()
        runtime.close()
