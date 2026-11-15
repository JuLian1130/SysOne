"""工作单元：一次请求被展开成若干 `WorkPlan`（共享前缀 + 各自 tail）。

`StageOp` 是算子名单（不按模型或设备分叉）；`ResidencyPlan` 与 `QuestionPlan` 描述
这一份工作要占多少层环、每个问题读哪一行。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .attention import MaskPlan
from ..errors import ConfigError


def ffn_segments(hidden: int, ffn: int) -> tuple[slice, ...]:
    """FFN 的 down 投影把 K 维切成哪几段：每段 `hidden` 列，余数落在最后一段。

    ANE 的 conv 一次吃下整个 `ffn` 偏大，旧引擎按 `hidden` 一段一段切 —— mude 是
    6144/2048，正好 3 段等长。4B 是 9216/2560 = 3.6，等长切不出来：前三段各 2560、
    最后一段 1536。段长不齐 ANE 照收，实测与等长切分同量级（S=4096 整层 FFN 41.5 ms，
    见票据 15 §15），所以这里不再要求整除。

    返回值是**列区间**（顺序覆盖 `[0, ffn)`）：装配侧用它切 `down_proj`，编译侧用
    `start/stop` 生成 `slice_by_size` 与对应宽度的 conv，两边共用一套规则。
    """
    if hidden <= 0 or ffn <= 0:
        raise ConfigError("FFN 切分需要正的 hidden/ffn", hidden=hidden, ffn=ffn)
    return tuple(slice(begin, min(begin + hidden, ffn)) for begin in range(0, ffn, hidden))


class StageOp(StrEnum):
    """引擎的算子名单（阶段级，不做融合细节）。

    `norm_to` 与 `norm_resid` 与旧实现的 MIL 阶段一一对应：
    `norm_to` 产出 `[rmsnorm(x) | x]`，`norm_resid` 产出 `[rmsnorm(y) | y]`。
    """

    NORM_TO = "norm_to"
    IN_PROJ = "in_proj"
    ATTENTION = "attention"
    GDN = "gdn"
    OUT_PROJ = "out_proj"
    NORM_RESID = "norm_resid"
    FFN = "ffn"
    READOUT = "readout"
    COPY = "copy"


@dataclass(frozen=True)
class ResidencyPlan:
    """常驻窗口与层环大小（M0 只做记录，M2 起决定实际分组）。"""

    resident_w: int = 4
    ring: int = 4
    tile: int = 0


@dataclass(frozen=True)
class QuestionPlan:
    """一个问题的读出行与候选标签。`end` 是最后一个 token 之后的位置。"""

    index: int
    start: int
    end: int
    labels: tuple[str, ...]
    kind: str = "choice"


@dataclass(frozen=True)
class WorkPlan:
    """一次前向要算的序列、掩码与读出行。

    `tokens` 是该 plan 的完整序列（共享前缀 + 本题 tail）；`prefix_len` 是前缀长度。
    """

    lane: str
    tokens: tuple[int, ...]
    prefix_len: int
    questions: tuple[QuestionPlan, ...]
    mask: MaskPlan
    residency: ResidencyPlan

    def seq_len(self) -> int:
        return len(self.tokens)

    def __post_init__(self) -> None:
        from ..errors import MaskPlanError

        if len(self.tokens) != self.mask.seq_len:
            raise MaskPlanError(
                "WorkPlan 与 MaskPlan 长度不一致",
                tokens=len(self.tokens),
                mask_seq_len=self.mask.seq_len,
            )
        if self.prefix_len != self.mask.prefix_len:
            raise MaskPlanError(
                "WorkPlan 与 MaskPlan 前缀长度不一致",
                prefix_len=self.prefix_len,
                mask_prefix= self.mask.prefix_len,
            )


@dataclass(frozen=True)
class CompilationTimes:
    """启动三阶段的耗时（秒），服务层据此判定就绪。"""

    load_s: float
    compile_s: float
    warmup_s: float

    @property
    def total_s(self) -> float:
        return self.load_s + self.compile_s + self.warmup_s
