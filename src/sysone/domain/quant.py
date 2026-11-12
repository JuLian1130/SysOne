"""量化描述：一份 `QuantSpec` 描述全部形态，两种打包器各自实现。

旧实现有两套互不相干的 int8（ANE 的 per-channel blob + sidecar，和 4B 的 per-row
lazy dequant checkpoint）。这里只保留"方案 / 标度 / 容器"三元组，任何权重形态都能
精确写成一个 `QuantSpec`；打包与解包由 `sysone.weights.packing` 分别实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..errors import ConfigError


class DType(StrEnum):
    """计算或存储精度。`INT8` 只作为存储形态出现（计算仍走 fp16/bf16）。"""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"
    INT8 = "int8"


class Scheme(StrEnum):
    """量化方案：无 / 按输出行 / 按输出通道。"""

    NONE = "none"
    PER_ROW_SYM_INT8 = "per_row_sym_int8"
    PER_CHANNEL_SYM_INT8 = "per_channel_sym_int8"


class Scales(StrEnum):
    """标度精度。"""

    NONE = "none"
    FP16 = "fp16"
    FP32 = "fp32"


class Storage(StrEnum):
    """容器形态：普通张量，或与标度混装的 blob。"""

    TENSOR = "tensor"
    MIXED_BLOB = "mixed_blob"


_ALLOWED = frozenset(
    {
        (Scheme.NONE, Scales.NONE, Storage.TENSOR),
        (Scheme.PER_ROW_SYM_INT8, Scales.FP32, Storage.TENSOR),
        (Scheme.PER_CHANNEL_SYM_INT8, Scales.FP16, Storage.MIXED_BLOB),
    }
)


@dataclass(frozen=True)
class QuantSpec:
    """一份量化描述，例如 `per_row_sym_int8/fp32/tensor`。

    允许组合刻意收窄为三组（见 `_ALLOWED`）：
    - `none/none/tensor`：未量化（bf16 源权重）；
    - `per_row_sym_int8/fp32/tensor`：4B checkpoint 形态（每输出行一个 fp32 标度）；
    - `per_channel_sym_int8/fp16/mixed_blob`：ANE 编译产物形态（码与 fp16 标度同 blob）。
    """

    scheme: Scheme = Scheme.NONE
    scales: Scales = Scales.NONE
    storage: Storage = Storage.TENSOR

    def validate(self) -> None:
        if (self.scheme, self.scales, self.storage) not in _ALLOWED:
            raise ConfigError(
                "量化组合不受支持",
                scheme=self.scheme.value,
                scales=self.scales.value,
                storage=self.storage.value,
            )

    @property
    def is_quantized(self) -> bool:
        return self.scheme is not Scheme.NONE

    def __str__(self) -> str:
        return f"{self.scheme.value}/{self.scales.value}/{self.storage.value}"
