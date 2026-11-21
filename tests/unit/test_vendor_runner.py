"""vendor 策略的管线测试（用假桥，不依赖 torch）。

真实厂商桥只在装了 torch + int8 检查点时才被 import；这里验证的是 Engine 在
`answers_for` 存在时会把模板、前向与读出整段交给桥，并正确回填 usage/provenance。
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import pytest
from helpers import load_tiny_descriptor, make_request

from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.work import CompilationTimes
from sysone.engine.engine import Engine
from sysone.engine.strategies.vendor import VendorRunner
from sysone.errors import NotReadyError, UnsupportedModeError
from sysone.models.vendors.base import VendorAnswer


class FakeBridge:
    name = "fake"

    def __init__(self, *, trained: Sequence[MaskMode], fail_mode: MaskMode | None = None) -> None:
        self.trained = tuple(trained)
        self.fail_mode = fail_mode
        self.loaded = False
        self.calls: list[tuple[str, dict, MaskMode]] = []
        self.artifact_bytes_value = 1024

    def load(self) -> None:
        self.loaded = True

    def compile_or_warm_up(self) -> CompilationTimes:
        return CompilationTimes(load_s=0.0, compile_s=0.0, warmup_s=0.0)

    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        if mode is self.fail_mode:
            raise NotImplementedError("这种模式厂商不跑")
        self.calls.append((state, question, mode))
        options = tuple(question["criteria"].keys()) or ("A", "B")
        probs = tuple(1.0 / len(options) for _ in options)
        return VendorAnswer(
            value=options[0],
            labels=options,
            probs=probs,
            input_tokens=7,
            readout_s=0.0,
        )

    def artifact_bytes(self) -> int:
        return self.artifact_bytes_value

    def close(self) -> None:
        self.loaded = False


def _config(mode: MaskMode | None = None) -> EngineConfig:
    return EngineConfig(
        model=ModelSelection(name="fake", dir="."),
        scheduler=SchedulerConfig(attention_mode=mode),
        service=ServiceConfig(debug_timings=True),
    )


def _engine(bridge: FakeBridge, config: EngineConfig | None = None) -> Engine:
    from sysone.devices.reference.encoding import CharEncoder

    descriptor = load_tiny_descriptor()
    return Engine(
        descriptor=descriptor,
        runner=VendorRunner(bridge),
        encoder=CharEncoder(vocab=descriptor.vocab),
        config=config or _config(),
    )


def test_vendor_path_answers_every_question() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    engine = _engine(bridge)
    started = engine.start()
    assert started.total_s == 0.0
    assert bridge.loaded

    request = make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=2)
    result = engine.run(request)

    assert len(result.answers) == 2
    assert result.answers[0].question_index == 0
    assert result.answers[1].question_index == 1
    for answer in result.answers:
        assert answer.labels == ("A", "B", "C", "D")
        assert abs(sum(answer.probs) - 1.0) < 1e-9
    assert result.usage.input_tokens == 14
    assert result.usage.prefill_calls == 2
    assert result.provenance.vendor_bridge == "fake"
    assert result.timings is not None
    assert len(bridge.calls) == 2


def test_vendor_mode_gate_uses_descriptor_supported_modes() -> None:
    descriptor = load_tiny_descriptor()
    narrow = dataclasses.replace(
        descriptor,
        supported_modes=frozenset({MaskMode.PREFIX_NONCAUSAL}),
        default_mode=MaskMode.PREFIX_NONCAUSAL,
        trained_modes=frozenset({MaskMode.PREFIX_NONCAUSAL}),
    )
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    from sysone.devices.reference.encoding import CharEncoder

    engine = Engine(
        descriptor=narrow,
        runner=VendorRunner(bridge),
        encoder=CharEncoder(vocab=narrow.vocab),
        config=_config(),
    )
    engine.start()
    with pytest.raises(UnsupportedModeError) as info:
        engine.run(make_request(mode=MaskMode.NONCAUSAL_FULL))
    assert info.value.context["mode"] == MaskMode.NONCAUSAL_FULL.value


def test_vendor_unsupported_mode_from_bridge_becomes_400() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,), fail_mode=MaskMode.PREFIX_NONCAUSAL)
    engine = _engine(bridge)
    engine.start()
    with pytest.raises(UnsupportedModeError) as info:
        engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL))
    assert info.value.status == 400
    assert info.value.context["bridge"] == "fake"


def test_vendor_runner_requires_load_first() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    runner = VendorRunner(bridge)
    with pytest.raises(Exception) as info:
        runner.answers_for(make_request(), MaskMode.CAUSAL)
    assert "尚未装载" in str(info.value)


def test_vendor_runner_artifacts_and_close() -> None:
    bridge = FakeBridge(trained=(MaskMode.PREFIX_NONCAUSAL,))
    runner = VendorRunner(bridge)
    assert runner.artifact_bytes() == 1024
    runner.load_weights()
    assert runner.check_health() is True
    runner.shutdown()
    assert bridge.loaded is False


def test_vendor_runner_delegates_warmup() -> None:
    runner = VendorRunner(FakeBridge(trained=(MaskMode.CAUSAL,)))
    times = runner.compile_or_warm_up()
    assert times.total_s == 0.0
    assert times.load_s == 0.0


def test_engine_start_surfaces_bridge_warmup_failure() -> None:
    """预热抛 NotReadyError 时启动就该失败，而不是让第一道题拿到坏概率。"""

    class BrokenWarmup(FakeBridge):
        def compile_or_warm_up(self) -> CompilationTimes:
            raise NotReadyError("预热概率非法", bridge="fake")

    engine = _engine(BrokenWarmup(trained=(MaskMode.CAUSAL,)))
    with pytest.raises(NotReadyError):
        engine.start()


def test_question_spec_carries_options_as_criteria() -> None:
    from sysone.models.vendors.base import question_spec

    question = Question(text="选一个", options=("甲", "乙"))
    spec = question_spec(question)
    assert spec["type"] == "choice"
    assert spec["instructions"] == "选一个"
    assert spec["criteria"] == {"甲": "甲", "乙": "乙"}


def test_noul_criteria_reach_the_vendor_as_true_and_false() -> None:
    """noul 的判据键是 `true` / `false`：协议层渲染成两行，这里原样拆回映射。

    厂商两条答面都按这两个键取判据描述（h2o shim 的 `options_of`、torchcast 的
    `render.question_of`），给空映射它们只能用默认描述 —— 那是另一套题面。
    """
    from sysone.models.vendors.base import question_spec

    question = Question(
        text="Does the notice hold?",
        options=("true: it was served in time", "false: it was served late"),
        kind="noul",
    )
    assert question_spec(question)["criteria"] == {
        "true": "it was served in time",
        "false": "it was served late",
    }


def test_a_score_scale_reaches_the_vendor_as_an_ordered_array() -> None:
    """score 的 criteria 是**数组**不是映射：顺序即档位高低，厂商按 Σ i·p_i 出期望值。

    协议层给所有题型都渲染 `"<键>: <描述>"` 行，所以这里要按序拆回档位本身。档位文字里
    自带 `": "` 时只拆第一处（`hard-opus-c-long_policy-05` 的档位就是这种）。
    """
    from sysone.models.vendors.base import question_spec

    question = Question(
        text="How many violations?",
        options=(
            "0: Compliant: nothing over the limit",
            "1: Exactly one violation",
            "2: Two or more",
        ),
        kind="score",
    )
    spec = question_spec(question)
    assert spec["type"] == "score"
    assert spec["criteria"] == [
        "Compliant: nothing over the limit",
        "Exactly one violation",
        "Two or more",
    ]


def test_wire_answer_shapes() -> None:
    from sysone.models.vendors.base import wire_answer

    # 厂商答面带的置信度原样带出去（0.4 既不是最大概率，也不是我们的任何公式）
    value, labels, probs, confidence = wire_answer(
        {
            "type": "choice",
            "choice": "B",
            "probabilities": {"A": 0.25, "B": 0.75},
            "confidence": 0.4,
        }
    )
    assert value == "B"
    assert labels == ("A", "B")
    assert probs == (0.25, 0.75)
    assert confidence == 0.4

    # noul 的答面没有这个字段：按 `served_top` 取较大的那一侧
    value, labels, probs, confidence = wire_answer({"type": "noul", "noul": 0.8})
    assert value == "yes"
    assert labels == ("yes", "no")
    assert confidence == pytest.approx(0.8)

    # score 同理；厂商没给时退回最大概率，而不是我们另算一遍
    value, labels, probs, confidence = wire_answer(
        {"type": "score", "score": 2.5, "probabilities": {"0": 0.1, "1": 0.4, "2": 0.5}}
    )
    assert value == "2.500"
    assert len(labels) == 3
    assert confidence == pytest.approx(0.5)


def test_engine_config_is_frozen() -> None:
    config = _config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.service.port = 1  # type: ignore[misc]


# ---- 票据 12：`attention_mode` 是逐请求的，双向开关是进程级的 --------------------------


def test_the_h2o_bridge_can_leave_bidirectional_mode() -> None:
    """双向开关要能关：否则一条 `noncausal_full` 之后所有 `causal` 请求都实际跑双向。

    jeff 只给 `enable()`；这里用假 backbone 核对"置回 `is_causal` + 清模式标记"这一步，
    它正是 jeff 自己的 `causal()` 上下文管理器做的事。
    """
    pytest.importorskip("torch")            # 桥在模块级 import torch
    from sysone.models.vendors.h2o_bridge import H2OBridge

    class _Attention:
        def __init__(self) -> None:
            self.is_causal = False

    class _Layer:
        def __init__(self) -> None:
            self.self_attn = _Attention()

    class _Backbone:
        def __init__(self) -> None:
            self.layers = [_Layer(), _Layer()]
            self._jev_bidirectional = "full"

    bridge = object.__new__(H2OBridge)          # 不装载权重：只测状态机
    bridge._backbone = _Backbone()
    bridge._bidir = True

    bridge._apply_mode(MaskMode.CAUSAL)
    assert [layer.self_attn.is_causal for layer in bridge._backbone.layers] == [True, True]
    assert bridge._backbone._jev_bidirectional is None
    assert bridge._bidir is False

    bridge._leave_bidirectional()               # 幂等：没开的时候什么都不做
    assert bridge._bidir is False


def test_option_descriptions_survive_the_trip_to_a_vendor() -> None:
    """协议把选项拼成 `"<键>: <描述>"`；厂商的 criteria 要的是**两个字段**（票据 12）。

    不拆的话厂商看到的是「键: 键」—— 判据描述整个丢掉，真实题集上 h2o 有 7/105、torchcast 有
    6/105 条判词因此不同。
    """
    from sysone.models.vendors.base import question_spec

    composed = Question(
        text="Which option?",
        options=("approve: pay 45 days after the survey", "deny: sensitive goods"),
    )
    spec = question_spec(composed)
    assert spec["criteria"] == {
        "approve": "pay 45 days after the survey",
        "deny": "sensitive goods",
    }
    # 没有描述时两侧相同（旧行为），noul 没有选项
    bare = Question(text="Which option?", options=("admit", "deny_score"))
    assert question_spec(bare)["criteria"] == {"admit": "admit", "deny_score": "deny_score"}
    assert question_spec(Question(text="Does it hold?", kind="noul"))["criteria"] == {}


def test_the_bridge_declares_no_shared_or_batch_capability() -> None:
    """厂商桥不透明：两条能力位保持 False，也**不记** miss —— 那些数字会永远不动。"""
    bridge = FakeBridge(trained=MaskMode)
    runner = VendorRunner(bridge)
    runner.load_weights()
    stats = runner.decision_stats()
    assert stats["supports_shared"] is False and stats["supports_batch"] is False
    assert stats["shared_hit"] == stats["shared_miss"] == 0
    assert stats["batch_hit"] == stats["batch_miss"] == stats["batch_fallback"] == 0
    assert runner.batch_limit == 1, "厂商桥串行：引擎据此不建攒批器"
