"""引擎：请求门禁、计划、调度、执行、结果汇总。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ..config import EngineConfig
from ..domain.attention import MaskMode
from ..domain.buffers import MemoryPlan
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
from ..models.vision.contract import image_contract as _text_contract
from .memory import plan_memory
from .planner import Planner
from .runner import ModelRunner
from .scheduler import Scheduler, SerialScheduler
from .session import SessionPool
from .stats import RequestStats, RunStats, StageStats, StartupStats

_LOGGER = logging.getLogger(__name__)


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
        self._memory: MemoryPlan | None = None
        self._requests = 0
        self._questions = 0
        self._tokens = 0
        self._prefill_s = 0.0
        self._readout_s = 0.0
        # 未训练掩码的告警按 mode 去重：一条请求打一条会把服务日志淹掉，而这个状态是模型的
        # 属性、不是请求的属性（`Provenance.attention_trained` 逐请求带，日志只报一次）。
        self._warned_modes: set[MaskMode] = set()

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    @property
    def config(self) -> EngineConfig:
        return self._config

    def start(self) -> CompilationTimes:
        """装载权重、做内存预算、编译预热；返回可展示的启动耗时。"""
        self._runner.init_device()
        self._runner.load_weights()
        artifacts = 0
        artifact_bytes = getattr(self._runner, "artifact_bytes", None)
        if callable(artifact_bytes):
            artifacts = int(artifact_bytes())
        self._memory = plan_memory(self._descriptor, self._config, artifacts_bytes=artifacts)
        times = self._runner.compile_or_warm_up()
        if not self._runner.check_health():
            raise ConfigError("模型自检未通过", model=self._descriptor.model_id)
        self._startup = times
        return times

    @property
    def batch_limit(self) -> int:
        """一个 pass 最多装几条请求：执行器报的**形状**上限，再被 `--batch` 压一次。

        读执行器的属性而不是 `getattr("answers_batch")`：那条路现在在 `DecisionRunner` 基类里
        人人都有，鸭子类型不再说明任何事。执行器报 1（pooled / vendor）就不攒批。
        """
        limit = min(int(getattr(self._runner, "batch_limit", 1)),
                    int(self._config.scheduler.batch))
        return max(1, limit)

    def _mode_of(self, request: DecisionRequest) -> MaskMode:
        """校验请求能不能跑，并定下这次用哪个掩码。"""
        if request.media:
            reason = self.vision_unavailable()
            if reason is not None:
                # 说清**哪一种**不行（模型没有塔 / 这次启动关了 / 链是 numpy 形态）：一件
                # "不支持多模态"把三件不同的事压成一句，客户端不知道是换参还是换模型。
                raise UnsupportedMediaError(reason, count=len(request.media))
        # 三级回落：请求 → 配置（`--attention-mode`，`SchedulerConfig` 的 docstring 写明
        # 非 None 就是"用这个"）→ 模型自己的默认。跳过中间一级会让 CLI 的开关只在
        # `cache_key` 里留痕、在请求上不留痕。
        mode = (request.attention_mode or self._config.scheduler.attention_mode
                or self._descriptor.default_mode)
        if mode not in self._descriptor.supported_modes:
            raise UnsupportedModeError(
                "该模型不支持所选注意力模式",
                mode=mode.value,
                supported=sorted(item.value for item in self._descriptor.supported_modes),
            )
        return mode

    def batch(self, requests: Sequence[DecisionRequest]) -> list[DecisionResult]:
        """多条请求合进一个 pass —— 只要它们都没有图、且落在同一个掩码上。

        调用方（服务层的攒批器）负责"同 mode、无图"这两条；这里再核一遍是因为**静默跑错**
        比**报错**贵得多：模式不同的两条拼进一个 pass，答案是错的但看不出来。

        每条请求的 `timings.prefill` / `readout` 拿到的是**整批**的墙时（它确实等了这么久），
        单元计数仍按各自的实际 token 数记。
        """
        if not requests:
            return []
        modes = {self._mode_of(request) for request in requests}
        if len(modes) > 1:
            raise ConfigError("一个批里的注意力模式必须相同", modes=sorted(m.value for m in modes))
        mode = modes.pop()
        if any(request.media for request in requests):
            raise ConfigError("批里不能有带图的请求", images=sum(1 for r in requests if r.media))
        started = time.perf_counter()
        prepared = time.perf_counter()
        ran = self._runner.answers_batch(tuple(requests), mode)
        prefill_s = time.perf_counter() - prepared
        vendor_name = getattr(self._runner, "bridge_name", None)
        return [
            self._result(request, mode, answers, tokens, prefill_s, prefill_s,
                         started, prepared, vendor_name, batched=len(requests))
            for request, (answers, tokens, _) in zip(requests, ran)
        ]

    def run(self, request: DecisionRequest) -> DecisionResult:
        if self.batch_limit > 1 and not request.media:
            # 单条也走批那条路：免得"批里 1 条"和"直接跑 1 条"是两条代码路。
            return self.batch([request])[0]
        started = time.perf_counter()
        mode = self._mode_of(request)
        vendor = getattr(self._runner, "answers_for", None)
        prepared = time.perf_counter()
        vendor_name: str | None = None
        if callable(vendor):
            # vendor 策略：模板、前向与读出都在厂商代码里，不走 MaskPlan。
            prefill_started = time.perf_counter()
            answers, tokens, readout_s = vendor(request, mode)
            prefill_s = time.perf_counter() - prefill_started
            usage = Usage(input_tokens=tokens, prefill_calls=len(answers))
            vendor_name = getattr(self._runner, "bridge_name", None)
        else:
            plans = self._planner.plans(request, mode)
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

        return self._result(request, mode, answers, usage.input_tokens, prefill_s, readout_s,
                           started, prepared, vendor_name, batched=0)

    def _result(
        self, request: DecisionRequest, mode: MaskMode, answers: Sequence[Answer],
        tokens: int, prefill_s: float, readout_s: float, started: float, prepared: float,
        vendor_name: str | None, *, batched: int,
    ) -> DecisionResult:
        """一次（或一批里的一条）请求的收尾：告警、provenance、计数、`DecisionResult`。

        `batched` 只是记账用的旁证（0 = 不是批路径），不改变结果。
        """
        usage = Usage(input_tokens=tokens, prefill_calls=len(answers))
        trained = mode in self._descriptor.trained_modes
        if not trained and mode not in self._warned_modes:
            self._warned_modes.add(mode)
            _LOGGER.warning(
                "attention_mode %s 不在模型的训练集合里（trained=%s）：结果只能当实验读数，"
                "不能当精度承诺",
                mode.value,
                sorted(item.value for item in self._descriptor.trained_modes),
            )
        provenance = Provenance(
            family=self._descriptor.family,
            artifact_id=self._config.cache_key(),
            quant=self._descriptor.quant,
            attention_mode=mode,
            attention_trained=trained,
            vendor_bridge=vendor_name,
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
        self._questions += len(answers)
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

    # ------------------------------------------------------------------ 视觉

    def image_contract(self) -> dict[str, object]:
        """这次装配收不收图、收的话四条预算是多少。**执行器是唯一的来源**。

        没有视觉能力的执行器（reference / 未装配的替身）连 `image_contract` 都没有，那就不
        收图，理由照实说 —— 这里补一个 `supported: False` 的表，而不是让前门去猜 `None`
        是什么意思（那会让"能力"在上游是 `None`、在下游是 `"no vision tower"`，两句话对不上）。
        """
        contract = getattr(self._runner, "image_contract", None)
        if callable(contract):
            return dict(contract())
        return _text_contract(unavailable="这条执行路径没有视觉塔，收不了图")

    def vision_unavailable(self) -> str | None:
        """收不了图的原因；`None` 表示收得了。与 `image_contract()` 同一份数据。"""
        return self.image_contract().get("unavailable")

    def vision_stats(self) -> dict[str, object]:
        """`/metrics` 的 `vision` 一栏；执行器不认识视觉时给一个"不可用"的空壳。"""
        stats = getattr(self._runner, "vision_stats", None)
        if not callable(stats):
            return {"available": False}
        return dict(stats())

    def decision_stats(self) -> dict[str, object] | None:
        """`/metrics` 的 `decision` 一栏：执行器认不认识共享/合批，以及今天各命中多少。

        不认这条契约的执行器（reference / 替身）回 `None`：`/metrics` 里就是空，而不是替它
        伪造一排 0 —— 那些 0 读起来像"测过了，没有机会"。
        """
        stats = getattr(self._runner, "decision_stats", None)
        if not callable(stats):
            return None
        return dict(stats())

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
