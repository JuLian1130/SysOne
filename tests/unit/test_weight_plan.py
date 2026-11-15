"""家族交给设备层的那份纯数据：`WeightPlan`。

`WeightPlan` 是"这份 checkpoint 怎么存"的四项 —— 形状（descriptor）、张量名前缀、norm 的
存法偏移、RoPE 角基。设备层因此不再认识任何具体 checkpoint（`chain.py` 曾经直接
`import models.families.mude` 拿 `CODE_ROWS` / 张量名 / norm 约定）。

张量名前缀这一项有实战意义：mude 的层叫 `language_model.layers.N`，h2o 的叫
`model.language_model.layers.N`，写死在设备层就装不了第二种。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import FIXTURES
from sysone.devices.apple.geometry import Geometry, mude_geometry
from sysone.errors import ConfigError
from sysone.models.families import h2o, mude


def test_mude_plan_is_the_one_the_shipped_chain_uses() -> None:
    plan = mude.weight_plan(FIXTURES / "mude")
    assert plan.prefix == "language_model"
    assert plan.norm_offset == mude.NORM_OFFSET == 1.0
    assert plan.rope_theta == 10_000_000.0
    # 形状从 descriptor 投影出来，与写死的 mude 几何逐项相同（这条在 `test_parse.py` 也钉着，
    # 这里从 `WeightPlan` 这条路再走一遍）。
    assert Geometry.from_descriptor(plan.descriptor) == mude_geometry()


def test_h2o_plan_names_the_outer_model_prefix_and_the_zero_centred_norm() -> None:
    """h2o 的两处差异：多一层 `model.` 前缀、norm 也是零中心存（厂商 `1 + weight`）。"""
    plan = h2o.weight_plan(FIXTURES / "h2o")
    assert plan.prefix == "model.language_model"
    assert plan.norm_offset == h2o.NORM_OFFSET == 1.0
    assert plan.rope_theta == 10_000_000.0
    assert plan.descriptor.readout.kind == "pooled"

    geometry = Geometry.from_descriptor(plan.descriptor)
    assert (geometry.hid, geometry.ffn) == (2560, 9216)
    assert geometry.qk_dim == 2048
    assert geometry.value_dim == 4096          # v 头 32 × 128，不是 hidden
    assert geometry.conv_dim == 8192
    assert geometry.cout_gdn == 12352
    assert geometry.core_out == 4096


def test_the_rope_base_is_read_not_guessed(tmp_path: Path) -> None:
    """缺 `rope_parameters.rope_theta` 就报错：猜一个（比如 1e4）不会崩，只会让所有位置角
    整体缩放 —— 最难发现的那类错。"""
    config = json.loads((FIXTURES / "mude" / "config.json").read_text("utf-8"))
    text = config.get("text_config", config)
    del text["rope_parameters"]["rope_theta"]
    directory = tmp_path / "no-rope-base"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), "utf-8")

    with pytest.raises(ConfigError) as excinfo:
        mude.weight_plan(directory)
    assert "rope_theta" in str(excinfo.value) or excinfo.value.context
