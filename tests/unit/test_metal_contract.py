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
from sysone.devices.apple.chain import SplitChain
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

    def run(self, hidden, *, length, prefix, causal):
        self.runs.append({"length": length, "prefix": prefix, "causal": causal})

    def hidden_rows(self, length: int) -> np.ndarray:
        return np.asarray(self._rows[:length], np.float32)


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
    assert chain._pipeline.runs == [{"length": 3, "prefix": 0, "causal": True}]


def test_metal_hidden_applies_the_final_norm_to_every_row():
    rows = np.array([[3.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0]], np.float32)
    final_norm = np.full(4, 2.0, dtype=np.float32)
    chain = make_chain(rows, final_norm=final_norm, width=4)

    out = chain.hidden([7, 8], causal=False)

    np.testing.assert_allclose(out, rms_norm(rows, final_norm), rtol=1e-6, atol=1e-6)
