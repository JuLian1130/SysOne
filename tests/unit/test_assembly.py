"""按注册表装配执行器：家族声明的策略决定谁来跑，CLI 不认识家族名。

工厂在这里换成假的 —— 这条测的是"装配"而不是链：形状从哪个函数来、参数有没有传下去、
后端与声明不符时说不说得清。真链的装配在集成测试与 `.scratch` 的探针里跑。
"""

from __future__ import annotations

import pytest
from helpers import make_mude_checkpoint

from sysone.engine.strategies import assembly
from sysone.errors import ConfigError
from sysone.models.registry import family


class FakeBuilder:
    """记账用的执行器工厂：把收到的参数原样存下来，返回一个哨兵对象。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[dict] = []

    def __call__(self, model_dir, *, bucket=None, width=None, resident=None):
        self.calls.append(
            {"dir": str(model_dir), "bucket": bucket, "width": width, "resident": resident}
        )
        return f"runner:{self.label}"


@pytest.fixture()
def builders(monkeypatch):
    split, pooled = FakeBuilder("split"), FakeBuilder("pooled")
    monkeypatch.setitem(assembly._BUILDERS, "split", split)
    monkeypatch.setitem(assembly._BUILDERS, "pooled", pooled)
    return split, pooled


def test_families_declare_how_their_answers_are_read():
    """声明是**家族**给的：mude 固定码表、h2o 标签行、torchcast 走厂商桥。"""
    assert family("mude").strategy == "split"
    assert family("h2o").strategy == "pooled"
    assert family("torchcast").strategy == "vendor"


def test_assemble_builds_the_runner_the_family_declares(tmp_path, builders):
    split, pooled = builders
    root = make_mude_checkpoint(tmp_path)

    assembled = assembly.assemble(root, "mude", backend="split", bucket=1024, width=2048,
                                 resident=4)

    assert assembled.strategy == "split"
    assert assembled.runner == "runner:split"
    # 形状来自家族的 `parse_config`，不是从命令行拼的
    assert assembled.descriptor.model_id == root.name
    assert split.calls == [{"dir": str(root), "bucket": 1024, "width": 2048, "resident": 4}]
    assert pooled.calls == []


def test_a_backend_that_contradicts_the_family_is_refused(tmp_path, builders):
    """`--backend split --model h2o` 不许"按后端跑"：那条路没为 h2o 验证过。"""
    _split, pooled = builders
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(tmp_path, "h2o", backend="split")
    context = caught.value.context
    assert context["family"] == "h2o"
    assert context["strategy"] == "pooled"
    assert context["backend"] == "split"
    assert pooled.calls == []


def test_a_strategy_without_a_builder_says_which_ones_exist(tmp_path, builders):
    """torchcast 声明 `vendor`，今天没有这条执行器 —— 报错要说出已经有哪几条。

    从 CLI 走会先落到这一条（`--backend` 里根本没有 `vendor`），所以报错不能建议一个
    选不出来的后端。
    """
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(tmp_path, "torchcast", backend="split")
    assert caught.value.context["strategy"] == "vendor"
    assert caught.value.context["ready"] == ["pooled", "split"]


def test_an_option_the_strategy_does_not_take_is_refused(tmp_path, builders):
    """`resident` 是 Metal 流水线的待命窗口：pooled 不认它，也不许悄悄丢掉。"""
    _split, pooled = builders
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(tmp_path, "h2o", backend="pooled", resident=4)
    assert caught.value.context["refused"] == ["resident"]
    assert caught.value.context["accepted"] == ["bucket", "width"]
    assert pooled.calls == []


def test_unknown_family_is_the_registry_error(tmp_path, builders):
    from sysone.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        assembly.assemble(tmp_path, "nope", backend="split")
