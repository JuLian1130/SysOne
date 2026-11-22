"""模型家族注册表：重复注册、未知家族、已注册清单。"""

from __future__ import annotations

import pytest

from sysone.errors import RegistrationError, UnknownModelError
from sysone.models import families  # noqa: F401  触发注册
from sysone.models.registry import family, families as registered, register_family

from helpers import load_tiny_descriptor


def _noop_plan(descriptor, config):
    return None  # 注册表不调用它


def test_four_families_registered() -> None:
    assert {"d3", "h2o", "mude", "torchcast"} <= set(registered())


# 每个家族固定一份发布检查点，档位用厂商的命名：mude 是 2B（Qwen3.5-2B 剪到 20 层），
# 其余都是 4B —— 包括 torchcast（`decision-4b`，实测 5.17B，因为它多一份解绑的 `lm_head`）
# 与 h2o（实测 4.66B）。精确值只进文档的内存账，不进命令行。
PARAMS = {
    "d3": "4B",
    "h2o": "4B",
    "mude": "2B",
    "pplx12": "4B",
    "pplx13": "4B",
    "singvl": "4B",
    "torchcast": "4B",
}


def test_every_family_names_the_parameter_count_of_its_checkpoint() -> None:
    """`--model` 的名字只说"哪一家"，档位在每个注册项的 `params` 里 —— 漏一个就等于入参
    不知道指的是多大的模型。"""
    assert {name: family(name).params for name in registered()} == PARAMS


def test_unknown_family_reports_available() -> None:
    with pytest.raises(UnknownModelError) as caught:
        family("nope")
    assert "available" in caught.value.context


def test_double_registration_is_an_error() -> None:
    with pytest.raises(RegistrationError):
        register_family("temp", lambda path: load_tiny_descriptor(), _noop_plan,
                        strategy="split")
        register_family("temp", lambda path: load_tiny_descriptor(), _noop_plan,
                        strategy="split")


def test_a_family_must_declare_its_readout_strategy() -> None:
    """不声明策略就注册不了：装配时"没人认领这个家族"比早退难查得多。"""
    with pytest.raises(RegistrationError):
        register_family("no-strategy", lambda path: load_tiny_descriptor(), _noop_plan,
                        strategy="")


def test_executor_families_declare_a_weight_plan() -> None:
    """设备层要的那四项从注册表就能拿到：查产物（`sysone doctor`）不必先装配执行器。

    四家都居于设备链上（mude / d3 走码表，h2o 与 torchcast 走池化读出），所以四家都有；
    这个字段仍然可选，`vendor` 策略不建设备链，接进来的家族可以不给。
    """
    for name in ("d3", "h2o", "mude", "torchcast"):
        assert family(name).weight_plan is not None
