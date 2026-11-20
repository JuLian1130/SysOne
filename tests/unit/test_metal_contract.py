"""两条被 h2o 的 Metal 对拍逼出来的约束（不需要 GPU / ANE 也能钉）。

1. `metal.available()` **不许**构造 `MetalRuntime`。那个运行时是进程级单例，构造一次就把
   当时的几何编进 kernel 库并钉死；只问"能不能用"就把它编成默认（mude）几何，会让**一次
   探测**决定整个进程只能跑 mude —— A1 的 h2o `(2560, 9216)` 对拍就是这么被自己挡住的。
2. `SplitChain.hidden()` 两条路同形：Metal 那条以前只返回末位的一维向量，与 docstring 的
   `[length, hidden]` 不符，逐位置对拍与 `PooledRunner` 的形状自检都会撞上。
"""

from __future__ import annotations

import numpy as np

from sysone.devices.apple import metal as metal_module
from sysone.devices.apple.chain import SplitChain, _rope_table
from sysone.devices.apple.geometry import Geometry
from sysone.devices.apple.ops import rms_norm


def test_available_does_not_compile_a_runtime(monkeypatch) -> None:
    """问一句"能不能用"不该污染进程：`_shared` 必须还是空的。"""
    monkeypatch.setattr(metal_module.MetalRuntime, "_shared", None)
    metal_module.available()
    assert metal_module.MetalRuntime._shared is None


def test_available_reports_false_without_a_dylib(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(metal_module.MetalRuntime, "_shared", None)
    assert metal_module.available(tmp_path) is False
    assert metal_module.MetalRuntime._shared is None


class StubPipeline:
    """只回答"最后一层整段 hidden"的假流水线（`[length, hidden]`，行主序）。"""

    def __init__(self, rows: np.ndarray) -> None:
        self._rows = rows
        self.runs: list[dict] = []
        self.whole_readbacks = 0

    def run(self, rows, *, length, prefix, causal, last_row_only=False):
        self.runs.append({"length": length, "prefix": prefix, "causal": causal,
                          "last_row_only": last_row_only})

    def hidden_rows(self, length: int) -> np.ndarray:
        self.whole_readbacks += 1
        return np.asarray(self._rows[:length], np.float32)

    def last_hidden(self, length: int) -> np.ndarray:
        return np.asarray(self._rows[length - 1], np.float32)


class StubStore:
    """按行给 embedding：行号就是值，方便看出读了哪几行。"""

    def read_rows(self, name, ids):
        return np.repeat(np.asarray(ids, np.float32)[:, None], 4, axis=1)


def make_chain(rows: np.ndarray, *, final_norm: np.ndarray, width: int) -> SplitChain:
    chain = SplitChain.__new__(SplitChain)
    chain._pipeline = StubPipeline(rows)
    chain._store = StubStore()
    chain._embed_name = "language_model.embed_tokens.weight"
    chain._final_norm = final_norm
    chain._geometry = Geometry(hid=4, ffn=8, q_heads=1, kv_heads=1, head_dim=4, rotary_dim=2,
                               l_k_heads=1, l_k_dim=2, l_v_heads=1, l_v_dim=2)
    chain._s = width
    chain._width = width
    chain._cos, chain._sin = _rope_table(10_000.0, width, chain._geometry.rotary_dim)
    chain._rope_theta = 10_000.0
    chain._mrope_section = None
    chain._rope_dirty = 0
    chain._mrope_interleaved = False
    chain._layers = []
    chain._params = []
    chain._roles = []
    return chain


def test_metal_hidden_returns_every_position():
    """Metal 那条路也要给 `[length, hidden]`：逐位置对拍和 `hidden[-1]` 都靠这个形状。"""
    rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    final_norm = np.ones(4, dtype=np.float32)
    chain = make_chain(rows, final_norm=final_norm, width=8)

    out = chain.hidden([11, 22, 33], causal=True)

    assert out.shape == (3, 4)
    # 整段都过了 final norm（与 numpy 那条路同一个函数），不是只归一了末位那一行
    np.testing.assert_allclose(out, rms_norm(rows, final_norm), rtol=1e-6, atol=1e-6)
    assert chain._pipeline.runs == [{"length": 3, "prefix": 0, "causal": True,
                                     "last_row_only": False}]


def test_metal_hidden_applies_the_final_norm_to_every_row():
    rows = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], np.float32)
    final_norm = np.full(4, 2.0, dtype=np.float32)
    chain = make_chain(rows, final_norm=final_norm, width=4)

    out = chain.hidden([7, 8], causal=False)

    np.testing.assert_allclose(out, rms_norm(rows, final_norm), rtol=1e-6, atol=1e-6)


def test_metal_last_hidden_reads_only_the_last_row():
    """`last_hidden()` 与 `hidden()[-1]` 是同一行，但**不做整段读回**。

    读出（`logits()` 与 pooled 的 `PooledRunner`）只要这一行，而整段读回 + 归一在 16k 上是
    87 ms、末行 0.02 ms。钉住的是"别调 `hidden_rows`"，这是这条出口存在的全部理由。
    """
    rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    final_norm = np.full(4, 2.0, dtype=np.float32)
    chain = make_chain(rows, final_norm=final_norm, width=8)

    out = chain.last_hidden([11, 22, 33], causal=True)

    assert out.shape == (4,)
    np.testing.assert_allclose(out, rms_norm(rows, final_norm)[-1], rtol=1e-6, atol=1e-6)
    assert chain._pipeline.whole_readbacks == 0
    assert chain._pipeline.runs == [{"length": 3, "prefix": 0, "causal": True,
                                     "last_row_only": True}]


def test_metal_last_hidden_can_disable_the_skip_for_ab():
    """A/B 用：关掉末行规则时同一条出口仍然只读末行，只是流水线不省死瓦片。"""
    rows = np.arange(12, dtype=np.float32).reshape(3, 4)
    final_norm = np.full(4, 2.0, dtype=np.float32)
    chain = make_chain(rows, final_norm=final_norm, width=8)

    out = chain.last_hidden([11, 22, 33], causal=True, last_row_only=False)

    np.testing.assert_allclose(out, rms_norm(rows, final_norm)[-1], rtol=1e-6, atol=1e-6)
    assert chain._pipeline.runs == [{"length": 3, "prefix": 0, "causal": True,
                                     "last_row_only": False}]
