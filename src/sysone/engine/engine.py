"""引擎：请求门禁、计划、调度、执行、结果汇总。"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..config import EngineConfig
from ..domain.attention import MaskMode
from ..domain.decision import (
    Answer,
    DecisionRequest,
    DecisionResult,
    Provenance,
    Timings,
    Usage,
)
from ..domain.work import CompilationTimes
from ..errors import ConfigError, UnsupportedMediaError, UnsupportedModeError
from ..models.descriptor import ModelDescriptor
from .planner import Planner
from .runner import ModelRunner
from .scheduler import Scheduler, SerialScheduler
from .session import SessionPool
from .stats import RequestStats, RunStats, StageStats, StartupStats


@dataclass(frozen=True)
class StepResult:
    answers: tuple[Answer, ...]
    has_unfinished: bool


class Engine:
    """一个模型 + 一个执行器 + 一个调度器的最小闭环。"""

    def __init__(
        self,
        descriptor: ModelDescriptor,
        runner: ModelRunner,
        encoder: object,
        config: EngineConfig,
        *,
        scheduler: Scheduler | None = None,
        pool: SessionPool | None = None,
    ) -> None:
        self._descriptor = descriptor
        self._runner = runner
        self._config = config
        self._planner = Planner(descriptor, encoder, config)
        self._scheduler: Scheduler = scheduler or SerialScheduler()
        self._pool = pool or SessionPool()
        self._startup = CompilationTimes(0.0, 0.0, 0.0)
        self._requests = 0
        self._questions = 0
        self._tokens = 0
        self._prefill_s = 0.0
        self._readout_s = 0.0

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    @property
    def config(self) -> EngineConfig:
        return self._config

    def start(self) -> CompilationTimes:
        """装载权重并编译预热；返回可展示的启动耗时。"""
        self._runner.init_device()
        self._runner.load_weights()
        times = self._runner.compile_or_warm_up()
        if not self._runner.check_health():
            raise ConfigError("模型自检未通过", model=self._descriptor.model_id)
        self._startup = times
        return times

    def run(self, request: DecisionRequest) -> DecisionResult:
        started = time.perf_counter()
        if request.media:
            raise UnsupportedMediaError("M0 不支持多模态输入", count=len(request.media))
        mode = request.attention_mode or self._descriptor.default_mode
        if mode not in self._descriptor.supported_modes:
            raise UnsupportedModeError(
                "该模型不支持所选注意力模式",
                mode=mode.value,
                supported=sorted(item.value for item in self._descriptor.supported_modes),
            )
        plans = self._planner.plans(request, mode)
        prepared = time.perf_counter()

        for plan in plans:
            self._scheduler.add(plan)
        readout_before = self._readout_s
        prefill_started = time.perf_counter()
        while self._scheduler.has_unfinished():
            self.step()
        prefill_s = time.perf_counter() - prefill_started
        readout_s = self._readout_s - readout_before

        done = self._scheduler.drain()
        if len(done) != len(plans):
            raise ConfigError("调度结果数量与计划数量不一致", plans=len(plans), done=len(done))
        answers = tuple(answer for _, answer in sorted(done, key=lambda item: item[1].question_index))

        prefix_len = plans[0].prefix_len
        tails = sum(plan.seq_len() - plan.prefix_len for plan in plans)
        usage = Usage(input_tokens=prefix_len + tails, prefill_calls=len(plans))
        provenance = Provenance(
            family=self._descriptor.family,
            artifact_id=self._config.cache_key(),
            quant=self._descriptor.quant,
            attention_mode=mode,
            attention_trained=mode in self._descriptor.trained_modes,
        )
        timings = None
        if self._config.service.debug_timings:
            total = time.perf_counter() - started
            timings = Timings(
                prepare=prepared - started,
                prefill=prefill_s,
                readout=readout_s,
                total=total,
            )

        self._requests += 1
        self._questions += len(plans)
        self._tokens += usage.input_tokens
        self._prefill_s += prefill_s
        return DecisionResult(
            answers=answers,
            usage=usage,
            provenance=provenance,
            timings=timings,
        )

    def step(self) -> StepResult:
        plans = self._scheduler.schedule()
        answers: list[Answer] = []
        for plan in plans:
            session = self._pool.acquire(plan)
            try:
                output = self._runner.execute(plan, session)
            finally:
                self._pool.release(session)
            self._scheduler.finish(plan, output.answer)
            self._readout_s += output.readout_s
            answers.append(output.answer)
        return StepResult(answers=tuple(answers), has_unfinished=self._scheduler.has_unfinished())

    def stats(self) -> RunStats:
        return RunStats(
            request=RequestStats(
                requests=self._requests, questions=self._questions, tokens=self._tokens
            ),
            stage=StageStats(prefill_s=self._prefill_s, readout_s=self._readout_s),
            startup=StartupStats(
                load_s=self._startup.load_s,
                compile_s=self._startup.compile_s,
                warmup_s=self._startup.warmup_s,
            ),
        )

    def shutdown(self) -> None:
        self._runner.shutdown()
