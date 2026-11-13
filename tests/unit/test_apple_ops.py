"""numpy 数学核：与参考实现的约定对齐之外，主要守三件事 —— 掩码、卷积的时序、分块的边界。"""

import numpy as np

from sysone.devices.apple import ops


def test_rms_norm_matches_hand_computation():
    x = np.array([[3.0, 4.0]], np.float32)          # rms = sqrt(12.5)
    w = np.ones(2, np.float32)
    out = ops.rms_norm(x, w)
    np.testing.assert_allclose(out, np.array([[3.0, 4.0]]) / np.sqrt(12.5 + ops.EPS), rtol=1e-6)


def test_rms_norm_gated_multiplies_silu_of_gate():
    x = np.ones((1, 2), np.float32)
    gate = np.zeros((1, 2), np.float32)             # silu(0) = 0
    assert np.allclose(ops.rms_norm_gated(x, gate, np.ones(2, np.float32)), 0.0)


def test_causal_conv_only_looks_left():
    """第 t 个输出只看 t-3..t；改最后一个输入不能影响更早的输出。"""
    x = np.arange(8, dtype=np.float32).reshape(8, 1)
    w = np.ones((1, 1, 4), np.float32)
    out = ops.causal_depthwise_conv(x, w, activation=False)
    assert out.shape == (8, 1)
    assert np.allclose(out[:3, 0], [0.0, 1.0, 3.0])   # 左侧补零
    assert np.allclose(out[3:, 0], [6.0, 10.0, 14.0, 18.0, 22.0])
    changed = x.copy()
    changed[-1] = 100.0
    out2 = ops.causal_depthwise_conv(changed, w, activation=False)
    np.testing.assert_allclose(out2[:-1], out[:-1])


def test_attention_respects_boolean_mask():
    s, h, d = 3, 1, 2
    q = np.ones((s, h, d), np.float32)
    k = np.ones((s, h, d), np.float32)
    v = np.arange(s, dtype=np.float32).reshape(s, 1, 1) * np.ones((1, 1, d))
    mask = np.array([[True, False, False],
                     [True, True, False],
                     [True, True, True]])
    out = ops.attention(q, k, v, mask)
    np.testing.assert_allclose(out[0, 0, 0], 0.0)          # 只能看第 0 个 token
    np.testing.assert_allclose(out[1, 0, 0], 0.5)          # 前两个平均
    np.testing.assert_allclose(out[2, 0, 0], 1.0)          # 三个平均


def test_rotary_is_identity_at_angle_zero():
    x = np.random.default_rng(0).standard_normal((4, 2, 8)).astype(np.float32)
    cos = np.ones((4, 4), np.float32)
    sin = np.zeros((4, 4), np.float32)
    np.testing.assert_array_equal(ops.apply_rotary(x, cos, sin, rotary_dim=8), x)


def test_delta_rule_is_linear_in_value():
    """β 只缩放 v、state 只由 v 累加，所以 v 通道是线性的 —— 分块不许引入非线性。"""
    rng = np.random.default_rng(1)
    s, h, d = 70, 2, 8
    q, k, v1, v2 = (rng.standard_normal((s, h, d)).astype(np.float32) for _ in range(4))
    g = -np.abs(rng.standard_normal((s, h)).astype(np.float32)) * 0.1
    beta = rng.random((s, h)).astype(np.float32)
    a = ops.delta_rule(q, k, v1, g, beta)
    b = ops.delta_rule(q, k, v2, g, beta)
    ab = ops.delta_rule(q, k, v1 + v2, g, beta)
    np.testing.assert_allclose(ab, a + b, rtol=1e-5, atol=1e-5)


def test_delta_rule_padding_tail_does_not_disturb_prefix():
    """越过 64 的分块边界：多出来的全零 token 不影响前 s 个输出。"""
    rng = np.random.default_rng(2)
    s, h, d = 100, 2, 8
    q, k, v = (rng.standard_normal((s, h, d)).astype(np.float32) for _ in range(3))
    g = -np.abs(rng.standard_normal((s, h)).astype(np.float32)) * 0.1
    beta = rng.random((s, h)).astype(np.float32)
    out = ops.delta_rule(q, k, v, g, beta)

    p = 28  # 100 + 28 = 128，正好补满两个块
    zeros_hd = np.zeros((p, h, d), np.float32)
    zeros_h = np.zeros((p, h), np.float32)
    qp = np.concatenate([q, zeros_hd])
    kp = np.concatenate([k, zeros_hd])
    vp = np.concatenate([v, zeros_hd])
    gp = np.concatenate([g, zeros_h])
    bp = np.concatenate([beta, zeros_h])
    out_padded = ops.delta_rule(qp, kp, vp, gp, bp)
    np.testing.assert_allclose(out_padded[:s], out, rtol=1e-5, atol=1e-5)


def test_softplus_is_stable_for_large_input():
    x = np.array([[-1e4, 0.0, 1e4]], np.float32)
    out = ops.softplus(x)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[:, 2], 1e4, rtol=1e-6)
