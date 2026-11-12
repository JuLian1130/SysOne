"""执行器接口：设备侧只暴露"装权重、编译、跑一份计划"的生命周期。"""

from __future__ import annotations

from typing import Protocol

from ..domain.decision import RunnerOutput
from ..domain.work import CompilationTimes, WorkPlan
from .session import Session


class ModelRunner(Protocol):
    """一个模型实例的执行器。"""

    def init_device(self) -> None: ...

    def load_weights(self) -> None: ...

    def compile_or_warm_up(self) -> CompilationTimes: ...

    def execute(self, plan: WorkPlan, session: Session) -> RunnerOutput: ...

    def check_health(self) -> bool: ...

    def shutdown(self) -> None: ...
