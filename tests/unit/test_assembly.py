"""按注册表装配执行器：策略决定谁来跑，配方决定这家怎么读，CLI 不认识家族名。

工厂在这里换成假的 —— 这条测的是"装配"而不是链：形状从哪个函数来、配方与 weight_plan
有没有原样转交、缺配方时说不说得清、后端与声明不符时说不说得清。真链的装配在集成测试与
`.scratch` 的探针里跑。
"""

from __future__ import annotations

import pytest
from helpers import FIXTURES, make_mude_checkpoint

from sysone.engine.strategies import assembly
from sysone.errors import ConfigError
from sysone.models.recipe import PooledRecipe, SplitRecipe, VendorRecipe
from sysone.models.registry import Family, family
from sysone.models import registry


class FakeBuilder:
    """记账用的执行器工厂：把收到的参数原样存下来，返回一个哨兵对象。

    `recipe` / `weights` 单独记两份，因为它们是对象身份（"是不是这一家的配方"），与形状
    参数（值）分开断言更直白。
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[dict] = []
        self.recipes: list[object] = []
        self.weights: list[object] = []

    def __call__(self, model_dir, *, recipe=None, weights=None, bucket=None, width=None,
                 resident=None, scratch_dir=None, quantize=None, vision=None,
                 vision_idle=None, vision_bucket=None):
        self.recipes.append(recipe)
        self.weights.append(weights)
        self.calls.append(
            {"dir": str(model_dir), "bucket": bucket, "width": width,
             "resident": resident, "scratch_dir": scratch_dir, "quantize": quantize,
             "vision": vision, "vision_idle": vision_idle, "vision_bucket": vision_bucket}
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


def test_each_family_brings_the_pieces_of_its_own_readout():
    """策略只是共享实现，题面与读出是每家的配方 —— 新模型能接进来靠的就是这一层。

    三个配方类型各不相同（码表 / 答面 / 桥），因为它们各自需要的东西本来就不一样；共同点
    只有一个：都由家族给，不写在策略里。
    """
    assert isinstance(family("mude").recipe, SplitRecipe)
    assert isinstance(family("h2o").recipe, PooledRecipe)
    assert isinstance(family("torchcast").recipe, VendorRecipe)


def test_the_family_recipe_and_weight_plan_reach_the_factory(tmp_path, builders):
    """配方与 weight_plan 都由家族给、装配层原样转交：换一家只改家族模块，工厂不动。"""
    from sysone.models.families import mude

    split, _ = builders
    root = make_mude_checkpoint(tmp_path)
    assembly.assemble(root, "mude", backend="split")

    assert split.recipes == [mude.RECIPE]
    assert split.weights[0] is not None
    assert split.weights[0].prefix == mude.PREFIX
    assert split.weights[0].descriptor.family == "mude"


def test_a_family_without_a_recipe_cannot_borrow_another(tmp_path, builders, monkeypatch):
    """新家族只声明策略、不带配方时必须报错，而且不许落到任何一家的工厂上。

    早先的写法按策略名查工厂：新家族声明 `split` 会静默拿到 mude 的 `MudePrompter` 与 255 行
    码表，算得出形状正常的答案，错得没有声音。这条钉的就是"借不到"。
    """
    monkeypatch.setitem(registry._REGISTRY, "bare-split", Family(
        name="bare-split",
        parse_config=lambda path: family("mude").parse_config(FIXTURES / "mude"),
        build_layer_plan=lambda descriptor, config: None,
        strategy="split",
    ))
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(FIXTURES / "mude", "bare-split", backend="split")
    context = caught.value.context
    assert context["family"] == "bare-split"
    assert context["strategy"] == "split"
    assert "接入配方" in caught.value.message
    assert "recipe=" in context["hint"]
    assert builders[0].calls == []


def test_runners_take_no_family_at_all():
    """三条工厂都不认识家族名：它们的签名里只有配方、weight_plan 与形状参数。

    这是一条"改回去会被拦下"的检查：给工厂再加一个 `family=` 之类的口子，新模型又要靠
    特判才能接。
    """
    import inspect

    from sysone.engine.strategies.pooled import pooled_runner
    from sysone.engine.strategies.split import split_runner
    from sysone.engine.strategies.vendor import vendor_runner

    for factory in (split_runner, pooled_runner, vendor_runner):
        parameters = inspect.signature(factory).parameters
        assert "recipe" in parameters, factory.__name__
        assert "weights" in parameters, factory.__name__
        assert not {"family", "name"} & set(parameters), factory.__name__


def test_assemble_builds_the_runner_the_family_declares(tmp_path, builders):
    split, pooled = builders
    root = make_mude_checkpoint(tmp_path)

    assembled = assembly.assemble(root, "mude", backend="split", bucket=1024, width=2048,
                                 resident=4)

    assert assembled.strategy == "split"
    assert assembled.runner == "runner:split"
    # 形状来自家族的 `parse_config`，不是从命令行拼的
    assert assembled.descriptor.model_id == root.name
    assert split.calls == [{"dir": str(root), "bucket": 1024, "width": 2048,
                            "resident": 4, "scratch_dir": None, "quantize": None,
                            "vision": None, "vision_idle": None, "vision_bucket": None}]
    assert pooled.calls == []


def test_artifacts_dir_reaches_the_executor(tmp_path, builders):
    """产物的落点是调用方说的：装配层把它当执行器的 `scratch_dir` 递下去。"""
    split, _ = builders
    root = make_mude_checkpoint(tmp_path)
    artifacts = tmp_path / "artifacts"

    assembly.assemble(root, "mude", backend="split", artifacts_dir=artifacts)

    assert split.calls[0]["scratch_dir"] == artifacts


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


def test_a_vendor_family_assembles_through_its_bridge(tmp_path, monkeypatch):
    """torchcast 声明 `vendor`：工厂收到权重目录与产物落点，链形状参数一个都不到它手里。

    这里换掉工厂，是为守住"装配"这一层：真正的桥在模块级 import torch，只有真跑
    vendor 路径时才该付这笔导入（`test_import_light.py` 钉着核心面不带 torch）。
    """
    calls: list[dict] = []

    def builder(model_dir, *, recipe=None, weights=None, scratch_dir=None):
        calls.append({"dir": str(model_dir), "scratch_dir": scratch_dir})
        return "runner:vendor"

    monkeypatch.setitem(assembly._BUILDERS, "vendor", builder)
    assembled = assembly.assemble(
        FIXTURES / "torchcast", "torchcast", backend="vendor", artifacts_dir=tmp_path / "art"
    )
    assert assembled.runner == "runner:vendor"
    assert assembled.strategy == "vendor"
    assert assembled.descriptor.model_id == "torchcast"
    assert calls == [{"dir": str(FIXTURES / "torchcast"), "scratch_dir": tmp_path / "art"}]


def test_a_vendor_family_refuses_chain_knobs(tmp_path, monkeypatch):
    """vendor 的前向与缓冲在厂商引擎里：链形状参数一个也不认，给了要报错。"""
    monkeypatch.setitem(assembly._BUILDERS, "vendor", lambda *a, **k: "runner:vendor")
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(FIXTURES / "torchcast", "torchcast", backend="vendor", resident=4)
    assert caught.value.context["refused"] == ["resident"]
    assert caught.value.context["accepted"] == []


def test_both_strategies_take_the_standby_window(tmp_path, builders):
    """`resident` 是 Metal 流水线的待命窗口；两条策略都有 Metal 形态，所以都认它。"""
    _split, pooled = builders
    assembly.assemble(make_mude_checkpoint(tmp_path), "h2o", backend="pooled", resident=4)
    assert pooled.calls[-1]["resident"] == 4


def test_a_backend_that_does_not_match_the_family_names_both_sides(tmp_path, builders):
    """`--backend split --model torchcast` 报错并说出两边，而不是按 split 跑一条没验证过的路。"""
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(FIXTURES / "torchcast", "torchcast", backend="split")
    context = caught.value.context
    assert context["backend"] == "split"
    assert context["strategy"] == "vendor"
    assert context["available"] == ["split", "pooled", "vendor"]
    assert context["hint"].endswith("用 --backend vendor")


def test_an_option_the_strategy_does_not_take_is_refused(tmp_path, builders, monkeypatch):
    """给了不认的参数要报错，不许悄悄丢：否则"我设了窗口"和"它没生效"会同时成立。"""
    _split, pooled = builders
    monkeypatch.setitem(assembly._OPTIONS, "pooled", frozenset({"bucket", "width"}))
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(tmp_path, "h2o", backend="pooled", resident=4)
    assert caught.value.context["refused"] == ["resident"]
    assert caught.value.context["accepted"] == ["bucket", "width"]
    assert pooled.calls == []


def test_unknown_family_is_the_registry_error(tmp_path, builders):
    from sysone.errors import UnknownModelError

    with pytest.raises(UnknownModelError):
        assembly.assemble(tmp_path, "nope", backend="split")


def test_视觉的三个旋钮都到达执行器(tmp_path, builders):
    """`vision` / `vision_idle` / `vision_bucket` 是三条独立的路：少传一个，部署时就会出现
    "我设了它、它没生效"，而三条都不会报错。"""
    split, _pooled = builders
    root = make_mude_checkpoint(tmp_path)
    assembly.assemble(root, "mude", backend="split", bucket=1024, vision=True,
                      vision_idle=30.0, vision_bucket=512)
    call = split.calls[-1]
    assert (call["vision"], call["vision_idle"], call["vision_bucket"]) == (True, 30.0, 512)


def test_quant_mode_is_translated_to_the_executor_switch(tmp_path, builders):
    """装配层收档位名字，执行器只认布尔：`fp16` 必须落到 `quantize=False`。"""
    split, _ = builders
    root = make_mude_checkpoint(tmp_path)

    assembly.assemble(root, "mude", backend="split", quant="fp16")
    assert split.calls[-1]["quantize"] is False

    assembly.assemble(root, "mude", backend="split", quant="int8")
    assert split.calls[-1]["quantize"] is True


def test_an_unknown_quant_mode_is_refused_before_the_executor_is_called(tmp_path, builders):
    _split, _pooled = builders
    with pytest.raises(ConfigError) as caught:
        assembly.assemble(tmp_path, "mude", backend="split", quant="int4")
    assert caught.value.context["available"] == ["int8", "fp16"]
    assert builders[0].calls == []
