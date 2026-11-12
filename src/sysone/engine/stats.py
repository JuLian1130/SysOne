"""运行统计：请求级、阶段级、启动级三层。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestStats:
    requests: int = 0
    questions: int = 0
    tokens: int = 0


@dataclass(frozen=True)
class StageStats:
    prefill_s: float = 0.0
    readout_s: float = 0.0


@dataclass(frozen=True)
class StartupStats:
    load_s: float = 0.0
    compile_s: float = 0.0
    warmup_s: float = 0.0


@dataclass(frozen=True)
class RunStats:
    request: RequestStats
    stage: StageStats
    startup: StartupStats
