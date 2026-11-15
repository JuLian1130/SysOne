"""h2o 布局的检查点能不能装进 `SplitChain`：张量名带 `model.` 前缀、读出是 pooled。

这一步（票据 15 第 2 步）之前，`chain.py` 把两件事写死在设备层：张量名 `language_model.…`
与"一定有 `readout.safetensors`"。任一不符就装不起来 —— 与几何无关，是纯粹的两处布局假设。

形状用合成的小模型（真 h2o 是 32 层 5.6 GiB，装一遍要分钟级），但**布局是真的**：
`tensor_prefix="model.language_model"`、目录里没有 `readout.safetensors`、descriptor 的
读出计划是厂商的 `pooled`。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import make_mude_checkpoint
from sysone.models.families import h2o

S = 128
TOKENS = [5, 9, 14, 21]


def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "ANE 只在 macOS 上"
    from sysone.devices.apple import available

    if not available():
        return "ANE dylib 不可用（先跑 tools/build_ane.sh）"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory):
    from sysone.devices.apple.chain import SplitChain

    model = make_mude_checkpoint(
        tmp_path_factory.mktemp("h2o_layout") / "model",
        tensor_prefix="model.language_model",
        with_readout=False,
    )
    loaded = SplitChain.load(
        model, weights=h2o.weight_plan(model), s=S,
        scratch_dir=tmp_path_factory.mktemp("programs"),
    )
    try:
        yield model, loaded
    finally:
        loaded.close()


def test_the_h2o_layout_loads_and_runs(chain) -> None:
    """带 `model.` 前缀的权重、没有码表：链要能装、能跑完，并给出最终 hidden。

    装得起来本身就是这条用例的重点 —— 以前 `load` 会在读第一个张量时
    `MissingTensorError`（前缀不对）、或在 `readout.safetensors` 处失败（pooled 没有这张表）。
    """
    _model, loaded = chain
    hidden = loaded.hidden(TOKENS)
    assert hidden.shape == (len(TOKENS), loaded._geometry.hid)
    assert np.isfinite(hidden).all()


def test_a_pooled_readout_says_so_instead_of_inventing_a_table(chain) -> None:
    """没有码表时 `logits()` 必须明确报错，而不是拿一张不存在的表算出一堆像模像样的数。"""
    from sysone.errors import NotReadyError

    _model, loaded = chain
    with pytest.raises(NotReadyError) as excinfo:
        loaded.logits(TOKENS)
    assert excinfo.value.context["readout_kind"] == "pooled"


def test_the_traces_reach_every_layer(chain) -> None:
    """逐层 trace 仍然可用 —— 这是 G6 之前"h2o 算得对不对"唯一的抓手。"""
    _model, loaded = chain
    trace: list[np.ndarray] = []
    loaded.hidden(TOKENS, trace=trace)
    assert len(trace) == len(loaded._roles) + 1, "embedding 一项 + 每层一项"
    assert all(row.shape == (loaded._geometry.hid,) for row in trace)
