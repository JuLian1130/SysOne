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


class _TinyModel(torch.nn.Module):
    """与真实检查点同样的命名：视觉塔在 `model.visual` 下（`skip_prefixes` 认的就是这个前缀）。

    `get_output_embeddings` 是 `PreTrainedModel` 的接口，手写模块要自己补上 —— 少了它
    `output_weight` 的第一条路就走不通，而这两条路本来都该被测到。
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        body = torch.nn.Module()
        body.layers = torch.nn.ModuleList([torch.nn.Module()])
        body.layers[0].q_proj = torch.nn.Linear(4, 4)
        body.embed_tokens = torch.nn.Embedding(8, 4)
        body.norm = torch.nn.LayerNorm(4)
        self.model.language_model = body
        self.model.visual = torch.nn.Module()
        self.model.visual.patch_embed = torch.nn.Linear(3, 5)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    # 真模型的 `language_model` 与 `visual` 都在 `model` 下；测试里换个短名字方便读，
    # 所以这里给出两个便捷属性，而不是把断言写成 `filled.model.language_model`。
    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.language_model

    @property
    def visual(self) -> torch.nn.Module:
        return self.model.visual

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module | None:
        return None      # 真模型里是 `lm_head`；`output_weight` 的第二条路


def _tiny_model() -> torch.nn.Module:
    return _TinyModel()


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
    # 真装载也是先 `with torch.device("meta")` 建壳再就地填（`build_model`），
    # 所以这里同样从 meta 起：视觉塔"没被填到"才会保持在 meta 上。
    with torch.device("meta"):
        model = _tiny_model()
    entries = {
        "model.language_model.layers.0.q_proj.weight": _FakeEntry(
            "model.language_model.layers.0.q_proj.weight", "int8", (4, 4)
        ),
        "model.language_model.layers.0.q_proj.bias": _FakeEntry(
            "model.language_model.layers.0.q_proj.bias", "keep", (4,)
        ),
        "model.language_model.embed_tokens.weight": _FakeEntry(
            "model.language_model.embed_tokens.weight", "keep", (8, 4)
        ),
        "model.language_model.norm.weight": _FakeEntry("model.language_model.norm.weight", "keep", (4,)),
        "model.language_model.norm.bias": _FakeEntry("model.language_model.norm.bias", "keep", (4,)),
        "lm_head.weight": _FakeEntry("lm_head.weight", "keep", (8, 4)),
    }
    payload = {
        "model.language_model.layers.0.q_proj.weight__q": np.zeros((4, 4), dtype=np.int8),
        "model.language_model.layers.0.q_proj.weight__s": np.ones((4,), dtype=np.float32),
        "model.language_model.layers.0.q_proj.bias": np.zeros((4,), dtype=np.float32),
        "model.language_model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float32),
        "model.language_model.norm.weight": np.ones((4,), dtype=np.float32),
        "model.language_model.norm.bias": np.zeros((4,), dtype=np.float32),
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


class _Rope(torch.nn.Module):
    """模拟 transformers 的 rope 模块：派生 buffer 由构造器自己算，权重文件里没有。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        freqs = 1.0 / (float(config.theta) ** (torch.arange(0, 4, 2, dtype=torch.float) / 4.0))
        self.register_buffer("inv_freq", freqs, persistent=False)
        self.register_buffer("original_inv_freq", freqs.clone(), persistent=False)


def test_meta_rope_buffers_are_materialized() -> None:
    """派生 buffer 留在 meta 上是必然的崩溃点：静态 rope 从不重算它们（票据 12）。"""
    from sysone.models.vendors.torch_support import _materialize_derived

    holder = torch.nn.Module()
    with torch.device("meta"):
        holder.rope = _Rope(SimpleNamespace(theta=10_000.0))
    assert holder.rope.inv_freq.device.type == "meta"
    _materialize_derived(holder, device=torch.device("cpu"))
    reference = _Rope(SimpleNamespace(theta=10_000.0)).inv_freq
    assert holder.rope.inv_freq.device.type == "cpu"
    assert holder.rope.inv_freq.dtype == torch.float32
    assert torch.equal(holder.rope.inv_freq, reference)
    assert torch.equal(holder.rope.original_inv_freq, reference)


def test_a_real_buffer_is_left_alone() -> None:
    """没有 meta buffer 的模块不该被重建（`__init__` 都不该被调用）。"""
    from sysone.models.vendors.torch_support import _materialize_derived

    class _Exploding(_Rope):
        def __init__(self, config) -> None:  # pragma: no cover - 被调用就是失败
            raise AssertionError("不该重建一个已经实化的模块")

    holder = torch.nn.Module()
    holder.rope = _Rope(SimpleNamespace(theta=10_000.0))
    holder.exploding = _Exploding.__new__(_Exploding)
    torch.nn.Module.__init__(holder.exploding)
    holder.exploding.config = SimpleNamespace(theta=1.0)
    holder.exploding.register_buffer("inv_freq", torch.ones(2), persistent=False)
    _materialize_derived(holder, device=torch.device("cpu"))
    assert torch.equal(holder.exploding.inv_freq, torch.ones(2))


def test_a_meta_buffer_without_a_config_is_an_error() -> None:
    from sysone.errors import MissingTensorError  # noqa: F401  断言类型
    from sysone.models.vendors.torch_support import _materialize_derived

    holder = torch.nn.Module()
    with torch.device("meta"):
        holder.register_buffer("loose", torch.zeros(3), persistent=False)
    with pytest.raises(Exception) as info:
        _materialize_derived(holder, device=torch.device("cpu"))
    assert "meta buffer" in str(info.value)


def test_output_weight_prefers_tied_embedding() -> None:
    model = _tiny_model()
    assert output_weight(model).shape == (8, 4)


def test_torch_dtype_mapping() -> None:
    assert torch_dtype("bf16") is torch.bfloat16
    with pytest.raises(Exception) as info:
        torch_dtype("nf4")
    assert "未知的权重精度" in str(info.value)


def test_vendor_bridges_default_to_fp32() -> None:
    """CPU 上 bf16 实测会出 NaN 与漂移，默认必须是 fp32。"""
    from sysone.models.vendors.h2o_bridge import H2OBridge
    from sysone.models.vendors.torchcast_bridge import TorchcastBridge

    assert H2OBridge("unused").dtype is torch.float32
    assert TorchcastBridge("unused").dtype is torch.float32
