"""Metal 核的对拍：q/k 的准备（归一化 + 半维 RoPE + gate 的 sigmoid）与注意力。

两个用例都把参考路径与 GPU 路径的输入对齐到**同一份 fp16 取整后的数值**，所以量出来的差额
就是核自己的算术，不含 ANE 的 fp16 地板（2.07e-2 量级，见 `docs/ANE/README.md` §10.22）。
把 prep 与 attention 分成两段对拍，是为了失败时一眼看得出是哪一段错了。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple import kernels, vision
from sysone.devices.apple.chain import _rope_table
from sysone.devices.apple.metal import MetalRuntime
from sysone.devices.apple.ops import apply_rotary, attention, rms_norm, sigmoid
from sysone.errors import NotReadyError
from sysone.weights.store import SafeTensorFile

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
_WEIGHTS = MUDE_DIR / "model.safetensors"
LAYER = 3
S = 96
TOL = 5e-3  # prep/attention 的算术都在 fp32 里做，5e-3 已经比 fp16 落盘舍入宽两个数量级


def _skip_reason() -> str | None:
    e8 = Path(os.environ.get("SYS1_E8_DIR", "/tmp/e8"))
    if not (e8 / "libe8mtl.dylib").is_file():
        return "没有 libe8mtl.dylib（先跑 tools/build_ane.sh）"
    if not _WEIGHTS.is_file():
        return f"缺少 mude 权重：{_WEIGHTS}"
    return None


pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="只在 macOS 上跑"),
    pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or ""),
]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    fa, fb = a.reshape(-1), b.reshape(-1)
    return float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12))


@pytest.fixture(scope="module")
def store() -> SafeTensorFile:
    return SafeTensorFile(_WEIGHTS)


@pytest.fixture(scope="module")
def runtime() -> MetalRuntime:
    return MetalRuntime()


def _theta() -> float:
    text = json.loads((MUDE_DIR / "config.json").read_text())
    return float(text.get("text_config", text).get("rope_theta", 1e7))


Q_ROWS = kernels.Q_HEADS * kernels.HEAD_DIM * 2  # 4096：每头的 [q(256)|gate(256)]
KV_ROWS = kernels.KV_PROJ * 2  # 1024：k|v 各 512


def _inputs(store: SafeTensorFile, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """真实权重下的投影与参考 prep 结果。

    返回 `(proj_fp16 [5120,s], ref_qg [4096,s], ref_kv [1024,s])`：两个 ref 数组就是 GPU 侧
    两个 buffer 应有的内容（fp32，未取整），方便逐段比较。
    """
    prefix = f"language_model.layers.{LAYER}.self_attn"
    w_qkv = np.concatenate(
        [store.read(f"{prefix}.{name}_proj.weight").astype(np.float32) for name in ("q", "k", "v")],
        axis=0,
    )
    q_norm = store.read(f"{prefix}.q_norm.weight").astype(np.float32) + 1.0  # 零中心存，真值 w+1
    k_norm = store.read(f"{prefix}.k_norm.weight").astype(np.float32) + 1.0
    cos, sin = _rope_table(_theta(), s, kernels.ROTARY_DIM)

    xn = np.random.default_rng(7).normal(size=(kernels.HID, s)).astype(np.float32)
    proj = (w_qkv @ xn).astype(np.float16)
    p = proj.astype(np.float32)

    # ---- 参考 prep（与 chain._full_layer 同一套算子）
    qg = p[:Q_ROWS].T.reshape(s, kernels.Q_HEADS, 2, kernels.HEAD_DIM)
    query = apply_rotary(rms_norm(qg[:, :, 0, :], q_norm), cos, sin, rotary_dim=kernels.ROTARY_DIM)
    gate = sigmoid(qg[:, :, 1, :])
    key = apply_rotary(
        rms_norm(p[Q_ROWS : Q_ROWS + kernels.KV_PROJ].T.reshape(s, kernels.KV_HEADS, kernels.HEAD_DIM),
                 k_norm),
        cos, sin, rotary_dim=kernels.ROTARY_DIM,
    )
    value = p[Q_ROWS + kernels.KV_PROJ :].T.reshape(s, kernels.KV_HEADS, kernels.HEAD_DIM)

    ref_qg = np.zeros((Q_ROWS, s), np.float32)
    ref_kv = np.zeros((KV_ROWS, s), np.float32)
    for h in range(kernels.Q_HEADS):
        base = h * 2 * kernels.HEAD_DIM
        ref_qg[base : base + kernels.HEAD_DIM] = query[:, h, :].T
        ref_qg[base + kernels.HEAD_DIM : base + 2 * kernels.HEAD_DIM] = gate[:, h, :].T
    for h in range(kernels.KV_HEADS):
        base = h * kernels.HEAD_DIM
        ref_kv[base : base + kernels.HEAD_DIM] = key[:, h, :].T
        ref_kv[kernels.KV_PROJ + base : kernels.KV_PROJ + base + kernels.HEAD_DIM] = value[:, h, :].T
    return proj, ref_qg, ref_kv


def test_argument_structs_match_the_c_header() -> None:
    """CPU-only：结构体字段顺序/宽度就是与 `e8_common.h` 的契约。"""
    assert kernels._NRM.size == 36          # A1（票据 06）在尾部加了 `pos0`
    assert kernels._COPY.size == 32
    assert kernels._ATTN.size == 72         # M3 在尾部加了 `caus`（批尾巴是 qseq + 4 个 16-bit 表）
    assert kernels._GDN.size == 32          # 共享前缀在尾部加了 `cols`（真实列数）
    args = kernels.nrm_args(stile=8, s0=2, src_stride=64, eps=1e-6, pos0=96)
    assert kernels._NRM.unpack(args) == (8, 2, 64, pytest.approx(1e-6), 0, 0, 0, 0, 96)
    assert kernels._ATTN.unpack(kernels.attn_args(T=10, stile=8, s0=0, scale=0.5, nk=8, nq=8,
                                                  qgT=16)) == (
        10, 8, 0, 0.5, 8, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    # M3：`caus` 在最后一位（索引 17）——插在中间会让 `qseq` 之后整段错位。
    assert kernels._ATTN.unpack(kernels.attn_args(
        T=10, stile=8, s0=0, scale=0.5, nk=8, nq=8, qseq=7, caus=1))[12:] == (7, 0, 0, 0, 0, 1)
    # 8 个槽位的表：低槽位在 `nkw0` 的低 16 位
    assert kernels._ATTN.unpack(kernels.attn_args(
        T=10, stile=8, s0=0, scale=0.5, nk=8, nq=8, nka=1024, segw=128, qseq=1024,
        nkw_slots=[60, 63, 66, 69, 72, 75, 80, 90]))[12:] == (
        1024, 60 | (63 << 16), 66 | (69 << 16), 72 | (75 << 16), 80 | (90 << 16), 0,
    )
    # GDN 的 `s0` 是"源表面首列"（票据 10 的逐槽扫描），与 `AttnArgs.s0`（qg 环列）、
    # `CopyArgs.s0`（目标列偏移）同名不同义 —— 这里用非零值钉住它在索引 1。
    assert kernels._GDN.unpack(kernels.gdn_args(stile=4, src_s0=7, T=4, first=True,
                                                eps=1e-6))[:4] == (4, 7, 4, 1)


def test_q_and_kv_prep_match_numpy(store: SafeTensorFile, runtime: MetalRuntime) -> None:
    proj, ref_qg, ref_kv = _inputs(store, S)
    cos, sin = _rope_table(_theta(), S, kernels.ROTARY_DIM)
    prefix = f"language_model.layers.{LAYER}.self_attn"
    q_norm = store.read(f"{prefix}.q_norm.weight").astype(np.float32) + 1.0
    k_norm = store.read(f"{prefix}.k_norm.weight").astype(np.float32) + 1.0

    qg_buf = runtime.alloc(Q_ROWS * S * 2)
    kv_buf = runtime.alloc(KV_ROWS * S * 2)
    qn_buf = runtime.alloc(kernels.HEAD_DIM * 4)
    kn_buf = runtime.alloc(kernels.HEAD_DIM * 4)
    cos_buf = runtime.alloc(S * (kernels.ROTARY_DIM // 2) * 4)
    sin_buf = runtime.alloc(S * (kernels.ROTARY_DIM // 2) * 4)

    qg_buf.array(np.float16, (Q_ROWS, S))[:] = proj[:Q_ROWS]
    kv_buf.array(np.float16, (KV_ROWS, S))[:] = proj[Q_ROWS:]
    qn_buf.array(np.float32, (kernels.HEAD_DIM,))[:] = q_norm
    kn_buf.array(np.float32, (kernels.HEAD_DIM,))[:] = k_norm
    cos_buf.array(np.float32, (S, kernels.ROTARY_DIM // 2))[:] = cos
    sin_buf.array(np.float32, (S, kernels.ROTARY_DIM // 2))[:] = sin

    kernels.q_prep(runtime, qg_buf, qn_buf, cos_buf, sin_buf, stile=S, src_stride=S, eps=1e-6)
    kernels.kv_prep(runtime, kv_buf, kn_buf, cos_buf, sin_buf, stile=S, src_stride=S, eps=1e-6)
    runtime.commit()
    runtime.wait()

    got_qg = qg_buf.array(np.float16, (Q_ROWS, S)).astype(np.float32)
    got_kv = kv_buf.array(np.float16, (KV_ROWS, S)).astype(np.float32)
    rel_qg = _rel_l2(got_qg, ref_qg)
    rel_kv = _rel_l2(got_kv, ref_kv)
    # 分块报告：q 半区 / gate 半区 / k 半区 / v 半区，失败时能直接指到是哪一步错了
    parts = {
        "q": (np.concatenate([got_qg[h * 512 : h * 512 + 256] for h in range(kernels.Q_HEADS)], 0),
              np.concatenate([ref_qg[h * 512 : h * 512 + 256] for h in range(kernels.Q_HEADS)], 0)),
        "gate": (np.concatenate([got_qg[h * 512 + 256 : (h + 1) * 512] for h in range(kernels.Q_HEADS)], 0),
                 np.concatenate([ref_qg[h * 512 + 256 : (h + 1) * 512] for h in range(kernels.Q_HEADS)], 0)),
        "k": (np.concatenate([got_kv[h * 256 : (h + 1) * 256] for h in range(kernels.KV_HEADS)], 0),
              np.concatenate([ref_kv[h * 256 : (h + 1) * 256] for h in range(kernels.KV_HEADS)], 0)),
        "v": (np.concatenate([got_kv[512 + h * 256 : 512 + (h + 1) * 256] for h in range(kernels.KV_HEADS)], 0),
              np.concatenate([ref_kv[512 + h * 256 : 512 + (h + 1) * 256] for h in range(kernels.KV_HEADS)], 0)),
    }
    detail = "，".join(f"{name} relL2={_rel_l2(a, b):.3e}" for name, (a, b) in parts.items())
    assert rel_qg < TOL and rel_kv < TOL, (
        f"prep 对不上：qg relL2={rel_qg:.3e} (cos={_cos(got_qg, ref_qg):.6f})，"
        f"kv relL2={rel_kv:.3e} (cos={_cos(got_kv, ref_kv):.6f})｜{detail}"
    )
    for buf in (qg_buf, kv_buf, qn_buf, kn_buf, cos_buf, sin_buf):
        buf.free()


@pytest.mark.parametrize("fast", [False, True])
def test_attention_prefix_two_pass_matches_numpy(
    store: SafeTensorFile, runtime: MetalRuntime, fast: bool
) -> None:
    """`prefix_noncausal`：前缀行只看前缀、尾巴行看全部，靠两遍 dispatch 表达。

    一遍 dispatch 只有一个全局上界 `nk`（`shaders/e8_common.h:117`），所以第一遍只放前缀行
    （`s0=0`、`Stile=prefix`、`nk=prefix`），第二遍只放尾巴行（`s0=prefix`、
    `Stile=tail`、`nk=S`），两遍各自写到 `out` 的 `out_s0` 列上。
    """
    prefix = 32
    _, ref_qg, ref_kv = _inputs(store, S)
    qg16, kv16 = ref_qg.astype(np.float16), ref_kv.astype(np.float16)
    p_q, p_kv = qg16.astype(np.float32), kv16.astype(np.float32)
    query = np.stack([p_q[h * 2 * kernels.HEAD_DIM : h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM].T
                      for h in range(kernels.Q_HEADS)], axis=1)
    gate = np.stack([p_q[h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM : (h + 1) * 2 * kernels.HEAD_DIM].T
                     for h in range(kernels.Q_HEADS)], axis=1)
    key = np.stack([p_kv[h * kernels.HEAD_DIM : (h + 1) * kernels.HEAD_DIM].T
                    for h in range(kernels.KV_HEADS)], axis=1)
    value = np.stack([p_kv[kernels.KV_PROJ + h * kernels.HEAD_DIM : kernels.KV_PROJ + (h + 1) * kernels.HEAD_DIM].T
                      for h in range(kernels.KV_HEADS)], axis=1)
    repeat = kernels.Q_HEADS // kernels.KV_HEADS
    idx = np.arange(S)
    mask = (idx[:, None] >= prefix) | (idx[None, :] < prefix)  # prefix_noncausal
    context = attention(
        query, np.repeat(key, repeat, axis=1), np.repeat(value, repeat, axis=1), mask,
    ).reshape(S, kernels.QK_DIM)
    expected = (context * gate.reshape(S, kernels.QK_DIM)).T

    qg_buf = runtime.alloc(Q_ROWS * S * 2)
    kv_buf = runtime.alloc(KV_ROWS * S * 2)
    out_buf = runtime.alloc(kernels.HID * 2 * S * 2)
    scratch = runtime.alloc(16 if fast else 8 * S * S * 4)
    qg_buf.array(np.float16, (Q_ROWS, S))[:] = qg16
    kv_buf.array(np.float16, (KV_ROWS, S))[:] = kv16
    out_buf.array(np.float16, (kernels.HID * 2, S))[:] = 0

    kernels.attention(
        runtime, qg_buf, kv_buf, out_buf, scratch,
        T=S, stile=prefix, nk=prefix, nq=prefix, s0=0,
        out_stride=S, out_s0=0, fast=fast,
    )
    kernels.attention(
        runtime, qg_buf, kv_buf, out_buf, scratch,
        T=S, stile=S - prefix, nk=S, nq=S - prefix, s0=prefix,
        out_stride=S, out_s0=prefix, fast=fast,
    )
    runtime.commit()
    runtime.wait()

    got = out_buf.array(np.float16, (kernels.HID * 2, S))[: kernels.HID].astype(np.float32)
    rel, cos_sim = _rel_l2(got, expected), _cos(got, expected)
    # 分两段报告：前缀行被尾巴污染只会坏第一段，尾巴行漏看前缀只会坏第二段
    head_rel = _rel_l2(got[:, :prefix], expected[:, :prefix])
    tail_rel = _rel_l2(got[:, prefix:], expected[:, prefix:])
    assert rel < TOL, (
        f"两遍 prefix 注意力对不上：relL2={rel:.3e} cos={cos_sim:.6f}"
        f"｜前缀段 {head_rel:.3e}、尾巴段 {tail_rel:.3e}"
    )
    assert cos_sim > 0.999, f"方向不对：cos={cos_sim:.6f}"
    assert np.all(np.isfinite(got)), "输出里有 NaN/inf"
    for buf in (qg_buf, kv_buf, out_buf, scratch):
        buf.free()


@pytest.mark.parametrize("fast", [False, True])
def test_attention_matches_numpy(store: SafeTensorFile, runtime: MetalRuntime, fast: bool) -> None:
    """喂**已 prep 好的** q/k/v（fp16 取整后再落盘），只量注意力核本身。"""
    _, ref_qg, ref_kv = _inputs(store, S)
    qg16 = ref_qg.astype(np.float16)
    kv16 = ref_kv.astype(np.float16)

    # 参考路径从同一份 fp16 数值重建 q/k/v
    p_q, p_kv = qg16.astype(np.float32), kv16.astype(np.float32)
    query = np.stack([p_q[h * 2 * kernels.HEAD_DIM : h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM].T
                      for h in range(kernels.Q_HEADS)], axis=1)  # [S,8,256]
    gate = np.stack([p_q[h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM : (h + 1) * 2 * kernels.HEAD_DIM].T
                     for h in range(kernels.Q_HEADS)], axis=1)  # 已是 sigmoid
    key = np.stack([p_kv[h * kernels.HEAD_DIM : (h + 1) * kernels.HEAD_DIM].T
                    for h in range(kernels.KV_HEADS)], axis=1)
    value = np.stack([p_kv[kernels.KV_PROJ + h * kernels.HEAD_DIM : kernels.KV_PROJ + (h + 1) * kernels.HEAD_DIM].T
                      for h in range(kernels.KV_HEADS)], axis=1)
    repeat = kernels.Q_HEADS // kernels.KV_HEADS
    context = attention(
        query, np.repeat(key, repeat, axis=1), np.repeat(value, repeat, axis=1),
        np.ones((S, S), dtype=bool),  # nk = 全长 ⇒ 双向；prefix_noncausal 走同一条核
    ).reshape(S, kernels.QK_DIM)
    expected = (context * gate.reshape(S, kernels.QK_DIM)).T  # [HID, S]

    qg_buf = runtime.alloc(Q_ROWS * S * 2)
    kv_buf = runtime.alloc(KV_ROWS * S * 2)
    out_buf = runtime.alloc(kernels.HID * 2 * S * 2)
    scratch = runtime.alloc(16 if fast else 8 * S * S * 4)  # 快核不读，但 buffer 必须绑定
    qg_buf.array(np.float16, (Q_ROWS, S))[:] = qg16
    kv_buf.array(np.float16, (KV_ROWS, S))[:] = kv16

    kernels.attention(
        runtime, qg_buf, kv_buf, out_buf, scratch,
        T=S, stile=S, nk=S, out_stride=S, fast=fast,
    )
    runtime.commit()
    runtime.wait()

    got = out_buf.array(np.float16, (kernels.HID * 2, S))[: kernels.HID].astype(np.float32)
    rel, cos_sim = _rel_l2(got, expected), _cos(got, expected)
    assert rel < TOL, f"注意力差得太多：relL2={rel:.3e} cos={cos_sim:.6f}"
    assert cos_sim > 0.999, f"注意力方向不对：cos={cos_sim:.6f}"
    assert np.all(np.isfinite(got)), "注意力输出里有 NaN/inf"
    for buf in (qg_buf, kv_buf, out_buf, scratch):
        buf.free()


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(("s", "start"), [(96, 0), (96, 16), (256, 16)])
def test_causal_attention_matches_numpy(
    store: SafeTensorFile, runtime: MetalRuntime, fast: bool, s: int, start: int
) -> None:
    """M3：`caus=1` 的核对上 numpy 的三角注意力（`k <= q`），快核与参考核都测。

    `start` 非 0 时 query 段是整条序列的后半截（`s0 = qseq = start`），这正是 split 路径的
    形态：一段 query 从序列第 `start` 列起，行的位置不能从 `s0` 反推，只能由 `qseq` 给。
    `s = 256` 那组把一段切成两个 threadgroup（AT_BM = 128），第二个的词块上界必须由
    `qseq + j0` 算出来 —— 用 `j0` 代替会少算一两个 key 块（前 16 行的对角线）。
    """
    _, ref_qg, ref_kv = _inputs(store, s)
    qg16 = ref_qg.astype(np.float16)
    kv16 = ref_kv.astype(np.float16)

    p_q, p_kv = qg16.astype(np.float32), kv16.astype(np.float32)
    query = np.stack([p_q[h * 2 * kernels.HEAD_DIM : h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM].T
                      for h in range(kernels.Q_HEADS)], axis=1)  # [s,8,256]
    gate = np.stack([p_q[h * 2 * kernels.HEAD_DIM + kernels.HEAD_DIM : (h + 1) * 2 * kernels.HEAD_DIM].T
                     for h in range(kernels.Q_HEADS)], axis=1)
    key = np.stack([p_kv[h * kernels.HEAD_DIM : (h + 1) * kernels.HEAD_DIM].T
                    for h in range(kernels.KV_HEADS)], axis=1)
    value = np.stack([p_kv[kernels.KV_PROJ + h * kernels.HEAD_DIM : kernels.KV_PROJ + (h + 1) * kernels.HEAD_DIM].T
                      for h in range(kernels.KV_HEADS)], axis=1)
    repeat = kernels.Q_HEADS // kernels.KV_HEADS

    rows = np.arange(start, s)[:, None]        # 这一段的 query 行的序列列号
    cols = np.arange(s)[None, :]
    context = attention(                       # k <= q，含对角线
        query[start:], np.repeat(key, repeat, axis=1), np.repeat(value, repeat, axis=1),
        (cols <= rows),
    ).reshape(s - start, kernels.QK_DIM)
    expected = (context * gate[start:].reshape(s - start, kernels.QK_DIM)).T  # [HID, s-start]

    qg_buf = runtime.alloc(Q_ROWS * s * 2)
    kv_buf = runtime.alloc(KV_ROWS * s * 2)
    out_buf = runtime.alloc(kernels.HID * 2 * s * 2)
    scratch = runtime.alloc(64 if fast else 8 * (s - start) * s * 4)
    qg_buf.array(np.float16, (Q_ROWS, s))[:] = qg16
    kv_buf.array(np.float16, (KV_ROWS, s))[:] = kv16

    kernels.attention(
        runtime, qg_buf, kv_buf, out_buf, scratch,
        T=s, stile=s - start, nk=s, nq=s - start, s0=start, qseq=start, out_s0=start,
        out_stride=s, caus=1, fast=fast,
    )
    runtime.commit()
    runtime.wait()

    full = out_buf.array(np.float16, (kernels.HID * 2, s))[: kernels.HID].astype(np.float32)
    got = full[:, start:]
    rel, cos_sim = _rel_l2(got, expected), _cos(got, expected)
    assert rel < TOL, (
        f"causal 注意力对不上 numpy：relL2={rel:.3e} cos={cos_sim:.6f}（fast={fast} start={start}）"
    )
    assert cos_sim > 0.999, f"causal 注意力方向不对：cos={cos_sim:.6f}"
    assert np.all(np.isfinite(got)), "causal 输出里有 NaN/inf"
    for buf in (qg_buf, kv_buf, out_buf, scratch):
        buf.free()


# ---- A1（票据 06）：`qg` 成环之后，列号与 rope 位置是两件事 -----------------------------
RING = S + 32   # 环的列数：要能同时装下"槽位偏移"与"整窗位置"两个概念


def test_q_prep_positions_come_from_pos0_not_the_column(
    store: SafeTensorFile, runtime: MetalRuntime
) -> None:
    """位置 64..95 的 q 摆进槽位 32..63：结果的 rope 位置必须是 64..95（`NormArgs.pos0`）。

    `_inputs` 已经算好位置 0..95 的整窗参考，这里只取它的第 64..95 列 —— 参考列（64）与
    摆放列（32）故意不同：拿列号当位置的实现会错得很明显，stride 写错则会读到槽位外。
    """
    start, slot, width = 64, 32, 32
    proj, ref_qg, _ = _inputs(store, S)
    prefix = f"language_model.layers.{LAYER}.self_attn"
    q_norm = store.read(f"{prefix}.q_norm.weight").astype(np.float32) + 1.0
    cos, sin = _rope_table(_theta(), S, kernels.ROTARY_DIM)

    qg_buf = runtime.alloc(Q_ROWS * RING * 2)
    view = qg_buf.array(np.float16, (Q_ROWS, RING))
    view[...] = 0
    view[:, slot:slot + width] = proj[:Q_ROWS, start:start + width]
    qn_buf = runtime.alloc(kernels.HEAD_DIM * 4)
    qn_buf.array(np.float32, (kernels.HEAD_DIM,))[:] = q_norm
    cos_buf = runtime.alloc(S * (kernels.ROTARY_DIM // 2) * 4)
    sin_buf = runtime.alloc(S * (kernels.ROTARY_DIM // 2) * 4)
    cos_buf.array(np.float32, (S, kernels.ROTARY_DIM // 2))[:] = cos
    sin_buf.array(np.float32, (S, kernels.ROTARY_DIM // 2))[:] = sin

    kernels.q_prep(runtime, qg_buf, qn_buf, cos_buf, sin_buf,
                   stile=width, s0=slot, src_stride=RING, eps=1e-6, pos0=start - slot)
    runtime.commit()
    runtime.wait()

    got = view[:, slot:slot + width].astype(np.float32)
    want = ref_qg[:, start:start + width]
    rel = _rel_l2(got, want)
    assert rel < TOL, f"环槽位里的 prep 对不上：relL2={rel:.3e}"
    for buf in (qg_buf, qn_buf, cos_buf, sin_buf):
        buf.free()


def test_attention_reads_q_from_the_ring_slot(
    store: SafeTensorFile, runtime: MetalRuntime
) -> None:
    """注意力从环的第 `slot` 列读 q/gate（`AttnArgs.qgT`），同时仍按整窗读 kv（`AttnArgs.T`）。"""
    start, slot, width = 64, 32, 32
    _, ref_qg, ref_kv = _inputs(store, S)
    qg16, kv16 = ref_qg.astype(np.float16), ref_kv.astype(np.float16)
    p_q, p_kv = qg16.astype(np.float32), kv16.astype(np.float32)
    d, qh, kvh = kernels.HEAD_DIM, kernels.Q_HEADS, kernels.KV_HEADS
    query = np.stack([p_q[h * 2 * d : h * 2 * d + d, start:start + width].T for h in range(qh)],
                     axis=1)
    gate = np.stack([p_q[h * 2 * d + d : (h + 1) * 2 * d, start:start + width].T
                     for h in range(qh)], axis=1)
    key = np.stack([p_kv[h * d : (h + 1) * d].T for h in range(kvh)], axis=1)
    value = np.stack([p_kv[kernels.KV_PROJ + h * d : kernels.KV_PROJ + (h + 1) * d].T
                      for h in range(kvh)], axis=1)
    repeat = qh // kvh
    context = attention(query, np.repeat(key, repeat, axis=1), np.repeat(value, repeat, axis=1),
                        np.ones((width, S), dtype=bool)).reshape(width, kernels.QK_DIM)
    expected = (context * gate.reshape(width, kernels.QK_DIM)).T

    qg_buf = runtime.alloc(Q_ROWS * RING * 2)
    kv_buf = runtime.alloc(KV_ROWS * S * 2)
    out_buf = runtime.alloc(kernels.HID * 2 * width * 2)
    scratch = runtime.alloc(16)   # 快核不读，但 buffer 必须绑定
    ring = qg_buf.array(np.float16, (Q_ROWS, RING))
    ring[...] = 0
    ring[:, slot:slot + width] = qg16[:, start:start + width]
    kv_buf.array(np.float16, (KV_ROWS, S))[:] = kv16
    out_buf.array(np.float16, (kernels.HID * 2, width))[:] = 0

    kernels.attention(runtime, qg_buf, kv_buf, out_buf, scratch,
                      T=S, stile=width, nk=S, nq=width, s0=slot, qgT=RING,
                      out_stride=width, out_s0=0, fast=True)
    runtime.commit()
    runtime.wait()

    got = out_buf.array(np.float16, (kernels.HID * 2, width))[: kernels.HID].astype(np.float32)
    rel = _rel_l2(got, expected)
    assert rel < TOL, f"环槽位里的注意力对不上：relL2={rel:.3e}"
    for buf in (qg_buf, kv_buf, out_buf, scratch):
        buf.free()


def _vattn_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, heads: int,
                     dim: int) -> np.ndarray:
    """通道主序的 softmax(QKᵀ)·V，逐头、非因果、fp32 中间量（视觉塔的注意力参考）。"""
    scale = dim ** -0.5
    out = np.empty((heads * dim, q.shape[1]), dtype=np.float16)
    for h in range(heads):
        b = h * dim
        qh = q[b:b + dim].astype(np.float32)
        kh = k[b:b + dim].astype(np.float32)
        vh = v[b:b + dim].astype(np.float32)
        sc = (qh.T @ kh) * scale                        # (nq, T)
        sc -= sc.max(axis=1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(axis=1, keepdims=True)
        out[b:b + dim] = (p @ vh.T).T.astype(np.float16)
    return out


@pytest.mark.parametrize(("T", "nq", "s0"), [
    (25, 25, 0),        # 奇数 T、一档都没铺满：尾列写入的守卫（漏了会踩掉下一行第 0 列）
    (96, 96, 0),        # 三档铺满
    (128, 70, 8),       # query 尾（第二格只写 6 列）+ 列偏移
    (300, 300, 0),      # 尾档 300 % 32 = 12
])
def test_vision_attention_matches_numpy(runtime: MetalRuntime, T: int, nq: int,
                                        s0: int) -> None:
    """`k_vattn`（非因果、head_dim 64、64 列 query 一格、键 32 一档）对 numpy。

    视觉的注意力核没有权重，所以这条用例的输入是随机构造的；形状是按会咬人的边界挑的：
    奇数 `T`、尾档不满、query 不满一格、以及非零 `s0`。
    """
    heads, dim = 16, 64
    hidden = heads * dim
    rng = np.random.default_rng(5)
    qkv = (rng.standard_normal((3 * hidden, T)) * 0.6).astype(np.float16)
    sentinel = np.float16(-777.0)
    out = np.full((hidden, T), sentinel, dtype=np.float16)
    qbuf = runtime.alloc(qkv.nbytes)
    obuf = runtime.alloc(out.nbytes)
    try:
        qbuf.array(np.float16, qkv.shape)[:] = qkv
        obuf.array(np.float16, out.shape)[:] = out
        kernels.vattn(runtime, qbuf, obuf, T=T, nq=nq, hidden=hidden, dim=dim,
                      scale=dim ** -0.5, s0=s0)
        runtime.commit()
        runtime.wait()
        got = obuf.array(np.float16, (hidden, T)).copy()
    finally:
        qbuf.free()
        obuf.free()

    # q 只切出这次产出的列；k/v 必须是全部 T 列（核按全 T 个键做 softmax）
    col = slice(s0, s0 + nq)
    want = _vattn_reference(qkv[:hidden, col], qkv[hidden:2 * hidden], qkv[2 * hidden:],
                            heads, dim)
    diff = np.abs(got[:, col].astype(np.float32) - want.astype(np.float32))
    peak = float(np.abs(want.astype(np.float32)).max())
    assert diff.max() / peak < TOL, f"视觉注意力对不上：rel={diff.max() / peak:.3e}"
    assert (got[:, col] != sentinel).all(), "有 query 列没被写过（网格没铺满 nq）"
    outside = np.ones(got.shape, dtype=bool)
    outside[:, col] = False
    assert (got[outside] == sentinel).all(), "写到了本次调用范围之外的列"


@pytest.mark.parametrize(("T", "nq", "s0", "k0", "k1"), [
    (300, 64, 0, 0, 150),        # 前半段当键：尾部与"只看一部分键"的遮罩
    (300, 64, 128, 100, 233),    # 键区间既不在 0 也不在尾部，且不与 query 对齐
    (128, 128, 0, 40, 41),       # 只看一个键：softmax 退化成恒等
])
def test_vision_attention_honours_the_key_range(runtime: MetalRuntime, T: int, nq: int,
                                                s0: int, k0: int, k1: int) -> None:
    """`k0`/`k1` 是这一片 query 能看的键区间 —— 多图打包靠它把一张图的注意力关在自己那张图里。

    判据不是"差不多"，是逐位等于"把键先切出来单独算一遍"：越界的键必须真的被排除（漏掉就
    会像正常结果一样安静地错），而区间里的键必须一个不少。
    """
    heads, dim = 16, 64
    hidden = heads * dim
    rng = np.random.default_rng(6)
    qkv = (rng.standard_normal((3 * hidden, T)) * 0.6).astype(np.float16)
    out = np.full((hidden, T), np.float16(-777.0), dtype=np.float16)
    qbuf = runtime.alloc(qkv.nbytes)
    obuf = runtime.alloc(out.nbytes)
    try:
        qbuf.array(np.float16, qkv.shape)[:] = qkv
        obuf.array(np.float16, out.shape)[:] = out
        kernels.vattn(runtime, qbuf, obuf, T=T, nq=nq, hidden=hidden, dim=dim,
                      scale=dim ** -0.5, s0=s0, k0=k0, k1=k1)
        runtime.commit()
        runtime.wait()
        got = obuf.array(np.float16, (hidden, T)).copy()
    finally:
        qbuf.free()
        obuf.free()
    col = slice(s0, s0 + nq)
    keys = slice(k0, k1)
    want = _vattn_reference(qkv[:hidden, col], qkv[hidden:2 * hidden, keys],
                            qkv[2 * hidden:, keys], heads, dim)
    diff = np.abs(got[:, col].astype(np.float32) - want.astype(np.float32))
    peak = float(np.abs(want.astype(np.float32)).max())
    assert diff.max() / peak < TOL, f"键区间没被遵守：rel={diff.max() / peak:.3e}"


def test_vision_attention_rejects_other_head_dims(runtime: MetalRuntime) -> None:
    """核里 8 档通道是按 head_dim == 64 写死的：别的值要当场拒绝，不能算出垃圾。"""
    qbuf = runtime.alloc(3 * 256 * 4 * 2)
    obuf = runtime.alloc(256 * 4 * 2)
    try:
        with pytest.raises(NotReadyError):
            kernels.vattn(runtime, qbuf, obuf, T=4, nq=4, hidden=256, dim=32, scale=0.1)
    finally:
        qbuf.free()
        obuf.free()


@pytest.mark.parametrize(("T", "dim", "hidden"), [
    (25, 64, 1024),      # 一格都没填满：尾列与 grid-stride 的守卫
    (300, 64, 1024),     # 640×480 的图（30×20 的 patch 网格）
    (513, 64, 1024),     # 跨出第一格（512 列）一列
    (1024, 64, 1024),    # 512² 的真实形状
])
def test_vision_rope_matches_host_bitwise(runtime: MetalRuntime, T: int, dim: int,
                                          hidden: int) -> None:
    """`k_vrope` 与 `vision.py::rope_qk` **逐位**相同。

    判据是逐位、不是 relL2：两边都是「四个乘各舍入一次、一减一加、最后落 f16」，核里关掉了
    fma 合并。只要有一位不同，就说明编译器把乘加折了，或者配对/转置写错了 —— 这两种错都不会
    被 5e-3 的容差拦下来（配对写反时 relL2 只有 0.5 量级，而容差是 5e-3，拦得住；但 f32→f16
    的最后一位不同拦不住，而它恰恰是"逐位对齐主机"这个约定的全部内容）。
    """
    half = dim // 2
    rng = np.random.default_rng(11)
    qkv = (rng.standard_normal((3 * hidden, T)) * 0.6).astype(np.float16)
    rows = np.arange(T) // 32
    cols = np.arange(T) % 32
    cos, sin = vision.rope_tables(rows, cols, head_dim=dim, theta=1e4)
    cos16, sin16 = cos.astype(np.float16), sin.astype(np.float16)

    want = vision.rope_qk(qkv.copy(), cos16, sin16, hidden=hidden, head_dim=dim)

    qbuf = runtime.alloc(qkv.nbytes)
    cbuf = runtime.alloc(half * T * 2)
    sbuf = runtime.alloc(half * T * 2)
    try:
        qbuf.array(np.float16, qkv.shape)[:] = qkv
        cbuf.array(np.float16, (half, T))[...] = cos16[:, :half].T
        sbuf.array(np.float16, (half, T))[...] = sin16[:, :half].T
        kernels.vrope(runtime, qbuf, cbuf, sbuf, T=T, hidden=hidden, dim=dim)
        runtime.commit()
        runtime.wait()
        got = qbuf.array(np.float16, qkv.shape).copy()
    finally:
        for buf in (qbuf, cbuf, sbuf):
            buf.free()

    bad = np.argwhere(got != want)
    assert bad.size == 0, (f"k_vrope 与主机 rope 不是逐位相同：{bad.shape[0]} 处不同，"
                           f"第一处 {bad[0].tolist() if bad.size else '-'}")
    assert (got[2 * hidden:] == qkv[2 * hidden:]).all(), "v 段不该被碰"


def test_vision_rope_rejects_bad_geometry(runtime: MetalRuntime) -> None:
    """核按 `dim/2` 配对、按 `dim` 切头：奇数 dim 或除不尽的通道数要当场拒绝。"""
    qbuf = runtime.alloc(3 * 1024 * 16 * 2)
    cbuf = runtime.alloc(8 * 16 * 2)
    sbuf = runtime.alloc(8 * 16 * 2)
    try:
        for hidden, dim in ((1024, 63), (1000, 64)):
            with pytest.raises(NotReadyError):
                kernels.vrope(runtime, qbuf, cbuf, sbuf, T=16, hidden=hidden, dim=dim)
    finally:
        for buf in (qbuf, cbuf, sbuf):
            buf.free()
