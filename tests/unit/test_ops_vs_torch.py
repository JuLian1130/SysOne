"""把 numpy 数学核钉在 transformers 的参考实现上（只在装了 torch 的环境里跑）。

`delta_rule` 曾经在参考掩码上少保了对角线（`intra` 的 `q_i·k_i` 项整块丢失），
逐层 cosine 掉到 0.85、20 层链答错选项；这条用例就是给那次回归上的锁。
"""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from sysone.devices.apple.ops import causal_depthwise_conv, delta_rule, rms_norm_gated


def _compare(ours: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    a = np.asarray(ours, np.float64).reshape(-1)
    b = np.asarray(ref, np.float64).reshape(-1)
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return rel, cos


@pytest.mark.parametrize("length", [17, 130])
def test_delta_rule_matches_transformers(length: int) -> None:
    """17 只走一个 chunk；130 跨 3 个 chunk，专门守住 chunk 间状态递推。

    跨 chunk 路径曾经完全没被覆盖（`length=17 < CHUNK=64`），层 0 的偏差因此
    一度往投影/GDN 里找。两个长度都必须过。
    """
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule

    rng = np.random.default_rng(7)
    heads, dim = 16, 128
    q = rng.normal(size=(1, length, heads, dim)).astype(np.float32)
    k = rng.normal(size=(1, length, heads, dim)).astype(np.float32)
    v = rng.normal(size=(1, length, heads, dim)).astype(np.float32)
    beta = (1.0 / (1.0 + np.exp(-rng.normal(size=(1, length, heads)))).astype(np.float32))
    g = -np.abs(rng.normal(size=(1, length, heads))).astype(np.float32) * 0.5

    ours = delta_rule(q[0], k[0], v[0], g[0], beta[0])
    ref = torch_chunk_gated_delta_rule(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v),
        torch.from_numpy(g), torch.from_numpy(beta),
        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
    )[0].numpy()[0]
    rel, cos = _compare(ours, ref)
    assert rel < 1e-5, f"delta_rule 偏离参考实现：relL2={rel:.3e} cos={cos:.6f}"


def test_causal_conv_matches_transformers() -> None:
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import causal_conv1d_fn

    rng = np.random.default_rng(11)
    length, channels = 21, 3 * 16 * 128
    x = rng.normal(size=(length, channels)).astype(np.float32)
    w = rng.normal(size=(channels, 1, 4)).astype(np.float32)

    ours = causal_depthwise_conv(x, w)
    ref = causal_conv1d_fn(
        torch.from_numpy(x.T[None]), torch.from_numpy(w.squeeze(1)), None, activation="silu",
    )[0].numpy().T
    rel, cos = _compare(ours, ref)
    assert rel < 1e-5, f"因果卷积偏离参考实现：relL2={rel:.3e} cos={cos:.6f}"


def test_gated_rms_norm_matches_reference() -> None:
    import torch

    rng = np.random.default_rng(13)
    rows, dim = 3 * 16, 128
    core = rng.normal(size=(rows, dim)).astype(np.float32)
    gate = rng.normal(size=(rows, dim)).astype(np.float32)
    weight = rng.normal(size=(dim,)).astype(np.float32)

    ours = rms_norm_gated(core, gate, weight)
    ref = (
        torch.nn.functional.rms_norm(torch.from_numpy(core), (dim,), torch.from_numpy(weight), eps=1e-6)
        * torch.nn.functional.silu(torch.from_numpy(gate))
    ).numpy()
    rel, cos = _compare(ours, ref)
    assert rel < 1e-5, f"门控 RMSNorm 偏离参考实现：relL2={rel:.3e} cos={cos:.6f}"
