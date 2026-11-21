"""vendor 策略：把一道请求整体交给厂商自带的模板、前向与读出。

与 split 策略的区别只在归属：split 由我们用 MaskPlan 驱动自己的核，vendor
则把 prompt 渲染与读出全交给厂商代码，我们只提供权重、注意力模式和预算。
"""

from __future__ import annotations

import time
from pathlib import Path

from ...domain.attention import MaskMode
from ...domain.decision import Answer, DecisionRequest
from ...domain.work import CompilationTimes
from ...errors import ConfigError, UnsupportedModeError
from ...models.vendors.base import VendorBridge, question_spec
from .decision import DecisionRunner


class VendorRunner(DecisionRunner):
    """实现 ModelRunner 的厂商路径：不排 MaskPlan，一题一次厂商前向。

    厂商桥不透明：模板、前向、读出都在厂商代码里，我们连"多题是不是同一个 state 前缀"都
    看不见。所以两条能力位保持 False、永远串行，也**不记**共享/合批的 miss —— 那些数字会
    永远不动，读的人会以为有机会没做。
    """

    def __init__(self, bridge: VendorBridge) -> None:
        self._bridge = bridge
        self._loaded = False

    @property
    def bridge_name(self) -> str:
        return self._bridge.name

    def init_device(self) -> None:
        return None

    def load_weights(self) -> None:
        self._bridge.load()
        self._loaded = True

    def artifact_bytes(self) -> int:
        return int(self._bridge.artifact_bytes())

    def compile_or_warm_up(self) -> CompilationTimes:
        return self._bridge.compile_or_warm_up()

    def check_health(self) -> bool:
        return self._loaded

    def execute(self, plan: object, session: object) -> object:
        raise ConfigError("vendor 策略按请求整批执行，不走单题 execute")

    def answers_for(self, request: DecisionRequest, mode: MaskMode) -> tuple[tuple[Answer, ...], int, float]:
        """一次请求内的所有题：返回 (答案元组, 输入 token 数, 读出耗时)。"""
        if not self._loaded:
            raise ConfigError("厂商桥尚未装载权重")
        answers: list[Answer] = []
        tokens = 0
        readout_s = 0.0
        for index, question in enumerate(request.questions):
            try:
                result = self._bridge.decide(request.state, question_spec(question), mode=mode)
            except NotImplementedError as exc:
                raise UnsupportedModeError(
                    "该厂商桥暂不支持所选注意力模式",
                    mode=mode.value,
                    bridge=self._bridge.name,
                    reason=str(exc),
                ) from exc
            answers.append(
                Answer(
                    question_index=index,
                    kind=question.kind,
                    value=result.value,
                    probs=tuple(result.probs),
                    labels=tuple(result.labels),
                    confidence=result.confidence,
                )
            )
            tokens += result.input_tokens
            readout_s += result.readout_s
        return tuple(answers), tokens, readout_s

    def shutdown(self) -> None:
        self._loaded = False
        self._bridge.close()


def torchcast_runner(model_dir: str | Path, *, scratch_dir: str | Path | None = None) -> VendorRunner:
    """装一条 torchcast 的 vendor 路径：装载、题面与读出全交给厂商包。

    桥在模块级 import `torch`，所以这里才把它带进来 —— 核心面不依赖 torch（
    `tests/unit/test_import_light.py` 盯着这条线），只有真要走 vendor 路径时才付这笔导入。

    `scratch_dir` 只是 `assemble` 的统一签名：这条路径不编 ANE program、不落产物，
    `artifact_bytes()` 恒为 0 —— 前向在厂商引擎里，用不上我们的产物缓存。

    没装 `vendor` 附加依赖时报一条配置错误而不是裸 `ImportError`：`pip install -e ./src`
    默认不带 torch，而这条路必须要有它。
    """
    try:
        from ...models.vendors.torchcast_bridge import TorchcastBridge
    except ImportError as exc:
        raise ConfigError(
            "vendor 路径需要 torch 与 transformers",
            backend="vendor",
            reason=str(exc),
            hint="pip install -e './src[vendor]'",
        ) from exc

    return VendorRunner(TorchcastBridge(model_dir))
