"""计划器：请求 → `WorkPlan`，并把掩码分解做成可校验的 `MaskPlan`。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..config import EngineConfig
from ..domain.attention import AttnWindow, MaskMode, MaskPlan
from ..domain.decision import DecisionRequest, EncodedPrompt, PromptEncoder
from ..domain.work import QuestionPlan, ResidencyPlan, WorkPlan
from ..errors import ConfigError, MaskPlanError, RequestTooLongError
from ..models.descriptor import ModelDescriptor


def build_mask_plan(
    mode: MaskMode,
    seq_len: int,
    prefix_len: int,
    tile: int,
    *,
    split_prefix: bool = True,
) -> MaskPlan:
    """按模式把一条序列切成若干 `AttnWindow`。

    - `causal`：每窗 `nk = q_end`（内核在窗内再做三角）；
    - `noncausal_full`：每窗 `nk = seq_len`；
    - `prefix_noncausal`：prefix 窗读到 `prefix`，tail 窗读到 `seq_len`；边界处切一刀，
      保证每窗只有一个语义（`split_prefix=False` 时才出现跨边界的 `nk_boundary`）。
    """
    if seq_len <= 0:
        raise ConfigError("seq_len 必须为正", seq_len=seq_len)
    if not 0 <= prefix_len <= seq_len:
        raise ConfigError("prefix_len 必须在 [0, seq_len]", prefix_len=prefix_len, seq_len=seq_len)

    step = tile if tile > 0 else seq_len
    edges = set(range(0, seq_len + 1, step)) | {0, seq_len}
    if 0 < prefix_len < seq_len and split_prefix:
        edges.add(prefix_len)
    boundaries = sorted(edges)

    windows: list[AttnWindow] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        nk_boundary: int | None = None
        k_boundary = 0
        if mode is MaskMode.CAUSAL:
            nk = min(seq_len, end)
        elif mode is MaskMode.NONCAUSAL_FULL:
            nk = seq_len
        else:
            if end <= prefix_len:
                nk = prefix_len
            elif start >= prefix_len:
                nk = seq_len
            else:
                nk = seq_len
                nk_boundary = prefix_len
                k_boundary = prefix_len - start
        windows.append(
            AttnWindow(
                q_start=start,
                q_len=end - start,
                nk=nk,
                nk_boundary=nk_boundary,
                k_boundary=k_boundary,
            )
        )
    return MaskPlan(mode=mode, seq_len=seq_len, prefix_len=prefix_len, tile=tile, windows=tuple(windows))


class Planner:
    """把一次请求展开成"共享前缀 + 每题 tail"的若干计划。"""

    def __init__(self, descriptor: ModelDescriptor, encoder: PromptEncoder, config: EngineConfig) -> None:
        self._descriptor = descriptor
        self._encoder = encoder
        self._config = config

    @property
    def descriptor(self) -> ModelDescriptor:
        return self._descriptor

    def plans(self, request: DecisionRequest, mode: MaskMode) -> tuple[WorkPlan, ...]:
        if not request.questions:
            raise ConfigError("请求没有问题", request_id=request.request_id)
        encoded = self._encoder.encode(request)
        self._check_encoded(request, encoded)

        prefix_len = len(encoded.prefix_tokens)
        residency = ResidencyPlan(
            resident_w=self._config.scheduler.resident_w,
            ring=self._config.scheduler.ring,
            tile=self._config.scheduler.tile,
        )
        plans: list[WorkPlan] = []
        for index, (tail, labels, kind) in enumerate(
            zip(encoded.tails, encoded.labels, encoded.kinds)
        ):
            if not tail:
                raise ConfigError("编码器给问题产出空 tail", question=index)
            tokens = encoded.prefix_tokens + tail
            if len(tokens) > self._config.scheduler.max_len:
                raise RequestTooLongError(
                    "序列超过上限",
                    length=len(tokens),
                    max_len=self._config.scheduler.max_len,
                )
            mask = build_mask_plan(
                mode, len(tokens), prefix_len, self._config.scheduler.tile
            )
            violations = mask.check()
            if violations:
                raise MaskPlanError("掩码计划不满足不变量", mode=mode.value, violations=violations)
            plans.append(
                WorkPlan(
                    lane=self._lane(request, index),
                    tokens=tokens,
                    prefix_len=prefix_len,
                    questions=(
                        QuestionPlan(
                            index=index,
                            start=prefix_len,
                            end=len(tokens),
                            labels=labels,
                            kind=kind,
                        ),
                    ),
                    mask=mask,
                    residency=residency,
                )
            )
        return tuple(plans)

    def _check_encoded(self, request: DecisionRequest, encoded: EncodedPrompt) -> None:
        count = len(request.questions)
        if not (len(encoded.tails) == len(encoded.labels) == len(encoded.kinds) == count):
            raise ConfigError(
                "编码器返回的条数与问题数不一致",
                questions=count,
                tails=len(encoded.tails),
                labels=len(encoded.labels),
                kinds=len(encoded.kinds),
            )

    def _lane(self, request: DecisionRequest, index: int) -> str:
        base = request.request_id or "req"
        return f"{base}#{index}"
