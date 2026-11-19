"""GDN 核在 4B 几何下的对拍：v 头数是 k 头数的两倍，且 `hidden != qk_dim`、`ffn != conv_dim`。

和 `test_metal_gdn.py` 同一个思路（GPU 与参考吃同一份 fp16 投影，差额是核自己的算术），
但**必须跑在子进程里**：`MetalRuntime` 是进程级单例、一个几何一份编译好的库
（`metal.py` 的类 docstring），同一个 pytest 进程里先跑的 `test_metal_gdn.py` 已经把 2B
的宏编进去了，再要 4B 就会撞上 "已经按另一套几何编译过"。

参考路径按 `trunked_gated_delta_rule` 的约定把 q/k 用 `repeat_interleave` 扩张到 v 头数
（`l_v_heads / l_k_heads = 2`），核里那一步由寻址隐含完成，所以这里量的就是"展开对不对"。

投影不来自真权重：核只吃 `[cout_gdn, S]` 的 fp16 面，参考吃同一份，所以形状与寻址是这里
唯一要证明的东西（真实 4B 权重的对拍要等引擎侧能跑 h2o，属于 #1 后续）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
TOL = 5e-3


def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "只在 macOS 上跑"
    from sysone.devices.apple.paths import dylib_dir

    e8 = dylib_dir()
    if not (e8 / "libe8mtl.dylib").is_file():
        return "没有 libe8mtl.dylib（先跑 tools/build_ane.sh）"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


# 子进程脚本：一套 4B 几何 + 随机投影 + numpy 参考，两条核路径各量一次 relL2。
CHILD = r'''
import numpy as np

from sysone.devices.apple import kernels
from sysone.devices.apple.geometry import Geometry
from sysone.devices.apple.metal import MetalRuntime
from sysone.devices.apple.ops import (
    causal_depthwise_conv, delta_rule, rms_norm_gated, sigmoid, softplus,
)

S = 96
EPS = 1e-6

# H2O-Lightning-4B 的线性注意力层：v 头 32 对 k 头 16，hidden 2560 != qk_dim 2048，
# ffn 9216 != conv_dim 8192 —— 这三条正是 2B 上"恰好相等"、会被写死成常量的地方。
GEOM = Geometry(
    hid=2560, ffn=9216, q_heads=16, kv_heads=4, head_dim=256, rotary_dim=64,
    l_k_heads=16, l_k_dim=128, l_v_heads=32, l_v_dim=128, conv_k=4,
)
GEOM.validate()


def reference(proj, conv_w, a_log, dt_bias, gnorm):
    rep = GEOM.l_v_heads // GEOM.l_k_heads            # 2
    p = proj.astype(np.float32).T                     # [S, cout_gdn]
    mixed = causal_depthwise_conv(p[:, : GEOM.conv_dim], conv_w)   # 已过 silu
    q = mixed[:, : GEOM.qk_dim].reshape(S, GEOM.l_k_heads, GEOM.l_k_dim)
    k = mixed[:, GEOM.qk_dim: 2 * GEOM.qk_dim].reshape(S, GEOM.l_k_heads, GEOM.l_k_dim)
    v = mixed[:, 2 * GEOM.qk_dim:].reshape(S, GEOM.l_v_heads, GEOM.l_v_dim)
    # 参考实现的 repeat_interleave(q, 2, dim=heads)；核里靠 h -> h/2 的寻址做到同一件事
    q = np.repeat(q, rep, axis=1)
    k = np.repeat(k, rep, axis=1)
    z = p[:, GEOM.conv_dim: GEOM.conv_dim + GEOM.z_dim].reshape(
        S, GEOM.l_v_heads, GEOM.l_v_dim)
    beta = sigmoid(p[:, GEOM.cout_gdn - 2 * GEOM.l_v_heads: GEOM.cout_gdn - GEOM.l_v_heads])
    g = -np.exp(a_log) * softplus(p[:, GEOM.cout_gdn - GEOM.l_v_heads:] + dt_bias)
    core = delta_rule(q, k, v, g, beta).reshape(S * GEOM.l_v_heads, GEOM.l_v_dim)
    gated = rms_norm_gated(core, z.reshape(S * GEOM.l_v_heads, GEOM.l_v_dim), gnorm, eps=EPS)
    return gated.reshape(S, GEOM.l_v_heads * GEOM.l_v_dim).T.astype(np.float32)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


rng = np.random.default_rng(7)
proj = rng.normal(size=(GEOM.cout_gdn, S)).astype(np.float16)
conv_w = rng.normal(size=(GEOM.conv_dim, GEOM.conv_k)).astype(np.float32)
a_log = rng.normal(size=(GEOM.l_v_heads,)).astype(np.float32)
dt_bias = rng.normal(size=(GEOM.l_v_heads,)).astype(np.float32)
gnorm = rng.normal(size=(GEOM.l_v_dim,)).astype(np.float32)
expected = reference(proj, conv_w, a_log, dt_bias, gnorm)

runtime = MetalRuntime(geometry=GEOM)


def run(fast):
    proj_buf = runtime.alloc(2 * GEOM.cout_gdn * S)
    proj_buf.array(np.float16, (GEOM.cout_gdn, S))[:] = proj
    state = runtime.alloc(4 * GEOM.state_floats)
    tail = runtime.alloc(2 * GEOM.conv_dim * 3)
    conv_buf = runtime.alloc(2 * GEOM.conv_dim * GEOM.conv_k)
    conv_buf.array(np.float16, (GEOM.conv_dim, GEOM.conv_k))[:] = conv_w
    a_buf = runtime.alloc(4 * GEOM.l_v_heads)
    a_buf.array(np.float32, (GEOM.l_v_heads,))[:] = a_log
    dt_buf = runtime.alloc(4 * GEOM.l_v_heads)
    dt_buf.array(np.float32, (GEOM.l_v_heads,))[:] = dt_bias
    norm_buf = runtime.alloc(2 * GEOM.l_v_dim)
    norm_buf.array(np.float16, (GEOM.l_v_dim,))[:] = gnorm
    out = runtime.alloc(2 * 2 * GEOM.hid * S)
    scratch = runtime.alloc(2 * GEOM.gdn_scratch_halves * S)
    try:
        kernels.gdn(runtime, proj_buf, state, tail, conv_buf, a_buf, dt_buf, norm_buf, out,
                    scratch, stile=S, eps=EPS, first=True, T=S, fast=fast, geometry=GEOM)
        runtime.sync_all()
        return out.array(np.float16, (2 * GEOM.hid, S))[: GEOM.value_dim].astype(np.float32)
    finally:
        for buf in (proj_buf, state, tail, conv_buf, a_buf, dt_buf, norm_buf, out, scratch):
            buf.free()


for fast in (False, True):
    print(("fast" if fast else "ref"), "%.6e" % rel_l2(run(fast), expected))
'''


def test_gdn_4b_geometry_matches_numpy() -> None:
    """4B 几何：参考核与快核都要在 `5e-3` 以内（与 2B 的 `test_metal_gdn.py` 同一口径）。"""
    result = subprocess.run(
        [sys.executable, "-c", CHILD],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        timeout=300,
    )
    assert result.returncode == 0, f"子进程失败：\n{result.stderr[-2000:]}"
    lines = [line for line in result.stdout.splitlines() if line.startswith(("ref", "fast"))]
    assert len(lines) == 2, f"子进程输出不完整：{result.stdout!r}\n{result.stderr[-800:]}"
    for line in lines:
        name, value = line.split()
        assert float(value) < TOL, f"4B 几何 {name} relL2={value} 超出 {TOL}"
