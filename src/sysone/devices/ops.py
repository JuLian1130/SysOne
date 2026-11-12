"""数值容差：集中在 `ops` 里，测试与 `doctor` 共用同一套阈值。

来源是 docs 的验收表（bf16 对拍 1e-3、int8 0.05、hidden relL2 2.5e-2、top-1 0.95）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerances:
    """一组容差阈值。"""

    prob_abs: float = 1e-3
    int8_prob_abs: float = 0.05
    hidden_rel_l2: float = 2.5e-2
    top1_agreement: float = 0.95


TOLERANCES = Tolerances()
