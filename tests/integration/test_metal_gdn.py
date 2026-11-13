"""GDN 核的对拍：conv1d + silu、delta 递推、门控 RMSNorm 一整段。

与 `test_metal_kernels.py` 同一个思路：参考路径与 GPU 路径吃**同一份 fp16 取整后的投影**，
所以量出来的差额是核自己的算术。参考核（`k_gdn_ref`）与快核（4 个 dispatch）分别对拍，
失败时一眼看得出是"整段错"还是"某一步错"。

第 0 层是 mude 的第一个线性注意力层，权重形状与旧引擎的 8224 行布局一致：
`[in_proj_qkv 6144 | in_proj_z 2048 | in_proj_b 16 | in_proj_a 16]`（`b` 在前、`a` 在后，
这是 kernel 的约定，与本仓库 numpy 链的装配顺序相反）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple import kernels
from sysone.devices.apple.metal import MetalRuntime
from sysone.devices.apple.ops import (
    causal_depthwise_conv,
    delta_rule,
    rms_norm_gated,
    sigmoid,
    softplus,
)
from sysone.weights.store import SafeTensorFile

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
_WEIGHTS = MUDE_DIR / "model.safetensors"
LAYER = 0
S = 96
TOL = 5e-3
EPS = 1e-6


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


@pytest.fixture(scope="module")
def store() -> SafeTensorFile:
    return SafeTensorFile(_WEIGHTS)


@pytest.fixture(scope="module")
def runtime() -> MetalRuntime:
    return MetalRuntime()


def _inputs(store: SafeTensorFile) -> dict[str, np.ndarray]:
    """真实权重下的一次 GDN：投影（fp16）、卷积权重、A_log / dt_bias / 门控 norm 与参考输出。"""
    base = f"language_model.layers.{LAYER}.linear_attn"
    w_proj = np.concatenate(
        [
            store.read(f"{base}.in_proj_qkv.weight"),
            store.read(f"{base}.in_proj_z.weight"),
            store.read(f"{base}.in_proj_b.weight"),
            store.read(f"{base}.in_proj_a.weight"),
        ],
        axis=0,
    ).astype(np.float32)
    assert w_proj.shape == (kernels.COUT_GDN, kernels.HID)

    xn = np.random.default_rng(11).normal(size=(kernels.HID, S)).astype(np.float32)
    proj = (w_proj @ xn).astype(np.float16)          # [8224, S]，与 ANE 的输出同布局

    conv = store.read(f"{base}.conv1d.weight").astype(np.float32).reshape(-1, kernels.CONV_K)
    a_log = store.read(f"{base}.A_log").astype(np.float32)
    dt_bias = store.read(f"{base}.dt_bias").astype(np.float32)
    gnorm = store.read(f"{base}.norm.weight").astype(np.float32)  # 不是零中心存，直接读

    p = proj.astype(np.float32).T                    # [S, 8224]
    heads, dim = kernels.L_HEADS, kernels.L_KD
    conv_rows = 3 * heads * dim                       # q丨k丨v
    mixed = causal_depthwise_conv(p[:, :conv_rows], conv)   # 已过 silu
    query = mixed[:, : heads * dim].reshape(S, heads, dim)
    key = mixed[:, heads * dim : 2 * heads * dim].reshape(S, heads, dim)
    value = mixed[:, 2 * heads * dim :].reshape(S, heads, dim)
    z = p[:, conv_rows : conv_rows + kernels.HID].reshape(S, heads, dim)
    beta = sigmoid(p[:, kernels.COUT_GDN - 32 : kernels.COUT_GDN - 16])
    g = -np.exp(a_log) * softplus(p[:, kernels.COUT_GDN - 16 :] + dt_bias)
    core = delta_rule(query, key, value, g, beta)
    gated = rms_norm_gated(
        core.reshape(S * heads, dim), z.reshape(S * heads, dim), gnorm, eps=EPS
    ).reshape(S, heads * dim)

    return {
        "proj": proj, "conv": conv, "a_log": a_log, "dt_bias": dt_bias,
        "gnorm": gnorm, "gated": gated.T.astype(np.float32),
    }


def _run(t: dict[str, np.ndarray], runtime: MetalRuntime, *, fast: bool) -> np.ndarray:
    """把一次 GDN 跑在 GPU 上，返回 `out` 的前 HID 行（`[HID, S]`，fp16 取整后的 fp32 视图）。"""
    hid = kernels.HID
    proj_buf = runtime.alloc(t["proj"].nbytes)
    proj_buf.array(np.float16, t["proj"].shape)[:] = t["proj"]
    state = runtime.alloc(4 * kernels.L_HEADS * kernels.L_KD * kernels.L_VD)
    tail = runtime.alloc(2 * kernels.CONV_DIM * 3)
    conv_buf = runtime.alloc(2 * t["conv"].size)
    conv_buf.array(np.float16, t["conv"].shape)[:] = t["conv"]
    a_buf = runtime.alloc(4 * t["a_log"].size)
    a_buf.array(np.float32, t["a_log"].shape)[:] = t["a_log"]
    dt_buf = runtime.alloc(4 * t["dt_bias"].size)
    dt_buf.array(np.float32, t["dt_bias"].shape)[:] = t["dt_bias"]
    norm_buf = runtime.alloc(2 * t["gnorm"].size)
    norm_buf.array(np.float16, t["gnorm"].shape)[:] = t["gnorm"]
    out = runtime.alloc(2 * 2 * hid * S)
    scratch = runtime.alloc(2 * 8192 * S)
    try:
        kernels.gdn(
            runtime, proj_buf, state, tail, conv_buf, a_buf, dt_buf, norm_buf, out, scratch,
            stile=S, eps=EPS, first=True, T=S, fast=fast,
        )
        runtime.commit()
        runtime.wait()
        return out.array(np.float16, (2 * hid, S))[:hid].astype(np.float32)
    finally:
        for buf in (proj_buf, state, tail, conv_buf, a_buf, dt_buf, norm_buf, out, scratch):
            buf.free()


@pytest.fixture(scope="module")
def t(store: SafeTensorFile) -> dict[str, np.ndarray]:
    return _inputs(store)


@pytest.mark.parametrize("fast", [False, True], ids=["ref", "fast"])
def test_gdn_matches_numpy(t: dict[str, np.ndarray], runtime: MetalRuntime, fast: bool) -> None:
    out = _run(t, runtime, fast=fast)
    err = _rel_l2(out, t["gated"])
    assert err < TOL, f"fast={fast} relL2={err:.3e}"
