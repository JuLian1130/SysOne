"""torch 侧装载的单元测试（没有 torch 的环境自动跳过）。

这里不碰 transformers：用一个手写的小网络验证 `fill_model` 的四条规则 ——
int8 线性层被替换成 `Int8Linear`、keep 张量就地赋值、视觉塔缺失可以留 meta、
tied 的 lm_head 与 embedding 共享同一个 Parameter。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sysone.models.vendors.torch_support import (  # noqa: E402
    Int8Linear,
    fill_model,
    output_weight,
    torch_dtype,
)


class _FakeEntry:
    def __init__(self, name: str, kind: str, shape: tuple[int, ...]) -> None:
        self.name = name
        self.kind = kind
        self.shape = shape


class _FakeManifest:
    def __init__(self, entries: dict[str, _FakeEntry]) -> None:
        self.entries = entries

    def quantized(self) -> frozenset[str]:
        return frozenset(name for name, entry in self.entries.items() if entry.kind == "int8")


class _FakeStore:
    def __init__(self, entries: dict[str, _FakeEntry], payload: dict[str, np.ndarray]) -> None:
        self.manifest = _FakeManifest(entries)
        self._payload = payload

    def names(self) -> frozenset[str]:
        return frozenset(self._payload)

    def read_raw(self, name: str) -> np.ndarray:
        return self._payload[name]

    def read_parts(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._payload[name + "__q"], self._payload[name + "__s"]


def _tiny_model() -> torch.nn.Module:
    model = torch.nn.Module()
    body = torch.nn.Module()
    body.layers = torch.nn.ModuleList([torch.nn.Module()])
    body.layers[0].q_proj = torch.nn.Linear(4, 4)
    body.embed_tokens = torch.nn.Embedding(8, 4)
    body.norm = torch.nn.LayerNorm(4)
    model.language_model = body
    model.lm_head = torch.nn.Linear(4, 8, bias=False)
    model.visual = torch.nn.Module()
    model.visual.patch_embed = torch.nn.Linear(3, 5)
    return model


def test_int8_linear_matches_float_linear() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((6, 5)).astype(np.float32)
    scale = (np.abs(weight).max(axis=1) / 127.0).astype(np.float32)
    scale[scale == 0.0] = 1.0
    codes = np.clip(np.round(weight / scale[:, None]), -127, 127).astype(np.int8)
    layer = Int8Linear(torch.from_numpy(codes), torch.from_numpy(scale))

    x = torch.from_numpy(rng.standard_normal((2, 5)).astype(np.float32))
    got = layer(x).detach().numpy()
    restored = (codes.astype(np.float32) * scale[:, None])
    want = x.numpy() @ restored.T
    assert np.abs(got - want).max() < 1e-5


def test_fill_model_replaces_int8_linears_and_ties() -> None:
    model = _tiny_model()
    entries = {
        "language_model.layers.0.q_proj.weight": _FakeEntry(
            "language_model.layers.0.q_proj.weight", "int8", (4, 4)
        ),
        "language_model.layers.0.q_proj.bias": _FakeEntry(
            "language_model.layers.0.q_proj.bias", "keep", (4,)
        ),
        "language_model.embed_tokens.weight": _FakeEntry(
            "language_model.embed_tokens.weight", "keep", (8, 4)
        ),
        "language_model.norm.weight": _FakeEntry("language_model.norm.weight", "keep", (4,)),
        "language_model.norm.bias": _FakeEntry("language_model.norm.bias", "keep", (4,)),
        "lm_head.weight": _FakeEntry("lm_head.weight", "keep", (8, 4)),
    }
    payload = {
        "language_model.layers.0.q_proj.weight__q": np.zeros((4, 4), dtype=np.int8),
        "language_model.layers.0.q_proj.weight__s": np.ones((4,), dtype=np.float32),
        "language_model.layers.0.q_proj.bias": np.zeros((4,), dtype=np.float32),
        "language_model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float32),
        "language_model.norm.weight": np.ones((4,), dtype=np.float32),
        "language_model.norm.bias": np.zeros((4,), dtype=np.float32),
        "lm_head.weight": np.zeros((8, 4), dtype=np.float32),
    }
    config = SimpleNamespace(tie_word_embeddings=True)
    filled = fill_model(
        model,
        config,
        _FakeStore(entries, payload),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert isinstance(filled.language_model.layers[0].q_proj, Int8Linear)
    assert filled.language_model.embed_tokens.weight.data.abs().sum() == 0
    # 视觉塔不在 payload 里，保持在 meta 上（纯文本前向不会碰它）
    assert filled.visual.patch_embed.weight.device.type == "meta"
    # tied：lm_head 与 embedding 共用同一个 Parameter
    embed = filled.language_model.embed_tokens._parameters["weight"]
    assert filled.lm_head._parameters["weight"] is embed


def test_output_weight_prefers_tied_embedding() -> None:
    model = _tiny_model()
    assert output_weight(model).shape == (8, 4)


def test_torch_dtype_mapping() -> None:
    assert torch_dtype("bf16") is torch.bfloat16
    with pytest.raises(Exception) as info:
        torch_dtype("nf4")
    assert "未知的权重精度" in str(info.value)
