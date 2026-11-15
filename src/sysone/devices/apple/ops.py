"""mude 一层的 numpy 数学：M2.2 第 2 片先用它跑通 20 层链，再用 Metal 换掉两个热点。

每个函数只做一件事，形状写在 docstring 里；全部 float32，与 transformers 的参考实现
逐条对齐（`modeling_qwen3_5.py`：`torch_chunk_gated_delta_rule` :301、`Qwen3_5GatedDeltaNet`
:550、`Qwen3_5RMSNormGated` :225、`Qwen3_5Attention` :776）。这里不做性能优化，够快就行。
"""

from __future__ import annotations

import numpy as np

from ...errors import MaskPlanError

EPS = 1e-6
CHUNK = 64


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:
    """`x` 末维归一化后乘 `weight`；参考实现用 fp32 累计方差。"""
    x32 = np.asarray(x, dtype=np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(variance + eps)) * np.asarray(weight, dtype=np.float32)


def rms_norm_gated(x: np.ndarray, gate: np.ndarray, weight: np.ndarray, eps: float = EPS) -> np.ndarray:
    """`RMSNormGated`：归一化乘权重之后再过 `silu(gate)`。"""
    return rms_norm(x, weight, eps) * silu(np.asarray(gate, dtype=np.float32))


def l2norm(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def causal_depthwise_conv(x: np.ndarray, weight: np.ndarray, *, activation: bool = True) -> np.ndarray:
    """`x` 是 `[S, C]`，`weight` 是 `[C, 1, 4]`（无 bias）；核内因果卷积后过 silu。

    参考实现调的是 `causal_conv1d_fn`：第 t 个输出只看 t-3..t（左侧补零）。
    """
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32).reshape(x.shape[1], -1)  # [C, 4]
    k = w.shape[1]
    out = np.zeros_like(x)
    for j in range(k):
        shift = k - 1 - j  # j=0 对应最老的输入
        if shift:
            out[shift:] += x[:-shift] * w[:, j]
        else:
            out += x * w[:, j]
    return silu(out) if activation else out


def apply_rotary(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, *, rotary_dim: int) -> np.ndarray:
    """`x` 是 `[S, heads, head_dim]`；只旋转前 `rotary_dim` 维，按 RoPE 的"前半/后半"配对。

    两个半区必须先 `.copy()`：它们都是 `x` 的视图，先把新值写回半区、再用另一个半区算第二半，
    会让第二半用上**已经旋转过的**第一半（M2.2 第 3 片接 Metal 时就是靠这条对拍抓到的）。
    """
    x = np.asarray(x, dtype=np.float32).copy()
    half = rotary_dim // 2
    x1 = x[..., :half].copy()
    x2 = x[..., half:rotary_dim].copy()
    c = cos[:, None, :]
    s = sin[:, None, :]
    x[..., :half] = x1 * c - x2 * s
    x[..., half:rotary_dim] = x2 * c + x1 * s
    return x


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray | None = None,
              scale: float | None = None) -> np.ndarray:
    """`q/k/v` 是 `[S, heads, dim]`，`mask` 是 `[S,S]` 布尔（True 可见）。"""
    q, k, v = (np.asarray(t, dtype=np.float32) for t in (q, k, v))
    heads = q.shape[1]
    # [S, heads, dim] -> [heads, S, dim]
    qh, kh, vh = (np.transpose(t, (1, 0, 2)) for t in (q, k, v))
    dk = kh.shape[-1]
    s = np.einsum("hsd,htd->hst", qh, kh) * (scale if scale is not None else dk ** -0.5)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[0]
        if m.shape != (qh.shape[1], kh.shape[1]):
            # 不接受隐式截断：掩码长度与 k/v 长度不一致说明调用方契约错
            #（M2.2 首版曾按未裁剪的全序列长度建掩码，由这里兜底掩盖了真问题）。
            raise MaskPlanError("注意力掩码形状与 q/k 长度不一致",
                                mask=tuple(m.shape), queries=qh.shape[1], keys=kh.shape[1])
        s = np.where(m[None, :, :], s, -np.inf)
    a = softmax(s, axis=-1)
    # 全遮的行（理论不会出现）会得到 nan，置 0 免得污染下游
    a = np.where(np.isnan(a), 0.0, a)
    return np.transpose(np.einsum("hst,htd->hsd", a, vh), (1, 0, 2))


def delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    *,
    repeat: int = 1,
) -> np.ndarray:
    """chunked gated delta rule，`[S, heads, dim]` 进、`[S, heads, dim]` 出。

    逐条对齐 `torch_chunk_gated_delta_rule`：q/k 先 l2norm、q 再乘 `dim**-0.5`；按 64 分块；
    `pairwise_decay[i,j] = exp(cum[i] - cum[j])`（i<=j 置 0）；解 `(I + tril(ut, -1)) x = v_beta`
    得 `new_values` 与 `k_cumdecay`；块内 `intra = (q @ kᵀ) * pairwise`；逐块
    `v_new = new_values - k_cumdecay @ state`、`out = q @ state + intra @ v_new`、
    `state = state * chunk_decay + kᵀ @ v_new`。

    `repeat > 1` 是参考实现的 `repeat_interleave(q, repeat, dim=1)`：v 的头数是 k 的
    `repeat` 倍（h2o 是 32 vs 16），q/k 每一头供 `repeat` 个 v 头共用。`g` 与 `beta` 是
    **按 v 头**给的（`A_log` / `dt_bias` / `in_proj_b` / `in_proj_a` 都是 v 头宽），所以只展开
    q 与 k，不碰它们。Metal 核里这一步由寻址隐含完成（`E8G_R`，见 `e8_gdn.metal`）。
    """
    q, k, v = (np.asarray(t, dtype=np.float32) for t in (q, k, v))
    s_len, keys, dim = q.shape
    if q.shape != k.shape or v.ndim != 3:
        raise ValueError(f"delta_rule 的 q/k/v 形状不配对：{q.shape} / {k.shape} / {v.shape}")
    if int(repeat) < 1 or v.shape[1] != keys * int(repeat):
        raise ValueError(
            f"v 头数 {v.shape[1]} 与 k 头数 {keys}×repeat={int(repeat)} 不一致"
        )
    v_heads = v.shape[1]
    q = l2norm(q) * (dim ** -0.5)
    k = l2norm(k)
    if int(repeat) > 1:
        q = np.repeat(q, int(repeat), axis=1)
        k = np.repeat(k, int(repeat), axis=1)
    heads = v_heads
    g = np.asarray(g, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)

    n = (s_len + CHUNK - 1) // CHUNK
    pad = n * CHUNK - s_len
    if pad:
        q = np.concatenate([q, np.zeros((pad, heads, dim), np.float32)], axis=0)
        k = np.concatenate([k, np.zeros((pad, heads, dim), np.float32)], axis=0)
        v = np.concatenate([v, np.zeros((pad, heads, dim), np.float32)], axis=0)
        g = np.concatenate([g, np.zeros((pad, heads), np.float32)], axis=0)
        beta = np.concatenate([beta, np.zeros((pad, heads), np.float32)], axis=0)

    # [n, 64, heads, dim]
    q = q.reshape(n, CHUNK, heads, dim)
    k = k.reshape(n, CHUNK, heads, dim)
    v = v.reshape(n, CHUNK, heads, dim)
    g = g.reshape(n, CHUNK, heads)
    beta = beta.reshape(n, CHUNK, heads)

    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    cum = np.cumsum(g, axis=1)                       # [n, 64, heads]
    decayed_k_beta = k_beta * np.exp(cum)[..., None]
    q_decayed = q * np.exp(cum)[..., None]

    idx = np.arange(CHUNK)
    # 参考实现遮掉的是严格上三角，**对角线保留**（intra 的对角项 q_i·k_i 非零，
    # 少了它整块输出就偏；ut 的对角线被 unitriangular 求解忽略，保留也无害）。
    lower = idx[:, None] >= idx[None, :]
    # 先在掩码掉的那半边填 -inf 再取 exp，否则 i>j 时的正指数会溢出
    pairwise = np.exp(np.where(lower[:, :, None], cum[:, :, None, :] - cum[:, None, :, :], -np.inf))
    pairwise = np.where(lower[:, :, None], pairwise, 0.0)

    ut = np.einsum("nihd,njhd->nijh", k_beta, k) * pairwise       # [n, 64, 64, heads]
    intra = np.einsum("nihd,njhd->nijh", q, k) * pairwise

    # 单位下三角前代：x_i = b_i - sum_{j<i} ut[i,j] * x_j
    def solve(b_like: np.ndarray) -> np.ndarray:
        out = np.zeros_like(b_like)
        for i in range(CHUNK):
            acc = b_like[:, i]
            for j in range(i):
                acc = acc - ut[:, i, j][..., None] * out[:, j]
            out[:, i] = acc
        return out

    new_values = solve(v_beta)                        # [n, 64, heads, dim]
    k_cumdecay = solve(decayed_k_beta)

    key_decayed = k * np.exp(cum[:, -1:, :] - cum)[..., None]    # 供 state 累加
    chunk_decay = np.exp(cum[:, -1])                            # [n, heads]

    out = np.zeros((n, CHUNK, heads, dim), np.float32)
    state = np.zeros((heads, dim, dim), np.float32)              # [heads, dk, dv]
    for i in range(n):
        v_new = new_values[i] - np.einsum("jhd,hdk->jhk", k_cumdecay[i], state)
        out[i] = (np.einsum("ihd,hdk->ihk", q_decayed[i], state)
                  + np.einsum("ijh,jhd->ihd", intra[i], v_new))
        state = (state * chunk_decay[i][:, None, None]
                 + np.einsum("ihd,ihk->hdk", key_decayed[i], v_new))
    out = out.reshape(n * CHUNK, heads, dim)[:s_len]
    return out
