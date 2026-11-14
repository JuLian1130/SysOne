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

from sysone.devices.apple import kernels
from sysone.devices.apple.chain import _rope_table
from sysone.devices.apple.metal import MetalRuntime
from sysone.devices.apple.ops import apply_rotary, attention, rms_norm, sigmoid
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
    assert kernels._ATTN.size == 68         # 批尾巴（票据 10）在尾部加了 qseq + 4 个 16-bit 表
    assert kernels._GDN.size == 32          # 共享前缀在尾部加了 `cols`（真实列数）
    args = kernels.nrm_args(stile=8, s0=2, src_stride=64, eps=1e-6, pos0=96)
    assert kernels._NRM.unpack(args) == (8, 2, 64, pytest.approx(1e-6), 0, 0, 0, 0, 96)
    assert kernels._ATTN.unpack(kernels.attn_args(T=10, stile=8, s0=0, scale=0.5, nk=8, nq=8,
                                                  qgT=16)) == (
        10, 8, 0, 0.5, 8, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0,
    )
    # 8 个槽位的表：低槽位在 `nkw0` 的低 16 位
    assert kernels._ATTN.unpack(kernels.attn_args(
        T=10, stile=8, s0=0, scale=0.5, nk=8, nq=8, nka=1024, segw=128, qseq=1024,
        nkw_slots=[60, 63, 66, 69, 72, 75, 80, 90]))[12:] == (
        1024, 60 | (63 << 16), 66 | (69 << 16), 72 | (75 << 16), 80 | (90 << 16),
    )
    assert kernels._GDN.unpack(kernels.gdn_args(stile=4, s0=0, T=4, first=True, eps=1e-6))[:4] == (
        4, 0, 4, 1,
    )


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
