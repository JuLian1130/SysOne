"""设备抽象：`Platform`（能力与实现选择）与 `KernelBackend`（单算子执行体）。

两者职责不同、生命周期不同：
- `Platform` 是纯数据 + 纯函数（可 import、可测、无硬件句柄）；
- `KernelBackend` 持有 stream/pool/编译句柄，`run()` 真正执行；
- 编译产物放 `CompiledSet`，与后端分离，便于缓存与跨实例复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from ..domain.attention import AttentionRole, MaskMode
from ..domain.quant import DType, QuantSpec, Scheme
from ..domain.work import StageOp

if TYPE_CHECKING:  # 仅类型引用，避免 devices → models 的运行时依赖
    from ..models.descriptor import ModelDescriptor


@dataclass(frozen=True)
class Shape:
    """一次算子调用的形状参数（不含具体张量）。"""

    s: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Capabilities:
    """平台能做什么（加载期与启动期据此 fail-fast）。"""

    dtypes: frozenset[DType]
    quant: frozenset[Scheme]
    modes: frozenset[MaskMode]
    max_seq: int
    max_tile: int
    zero_copy: bool = False


@dataclass(frozen=True)
class MemoryInfo:
    total: int
    current: int
    usable: int
    wired_limit: int | None = None


@dataclass(frozen=True)
class Placement:
    """缓冲区该放哪（如 `ane` / `gpu` / `cpu`）。"""

    name: str

    def __str__(self) -> str:
        return self.name


@runtime_checkable
class CompiledSet(Protocol):
    """一次编译的产物集合（按 S 桶与算子聚合）。"""

    op: StageOp
    s_bucket: int

    def release(self) -> None: ...


@runtime_checkable
class KernelBackend(Protocol):
    """单个算子的执行体。"""

    op: StageOp

    def supports(
        self, *, role: AttentionRole, mode: MaskMode, dtype: DType, quant: QuantSpec
    ) -> bool: ...

    def prepare(self, shape: Shape, views: object) -> CompiledSet: ...

    def run(self, compiled: CompiledSet, mask: object, stage: object) -> None: ...

    def verify(self) -> None:
        """自检（M1/M2 起用于启动期一致性检查）。"""
        ...


@runtime_checkable
class Platform(Protocol):
    """设备平台：能力查询、预算、实现类选择与编译。"""

    name: str

    def capabilities(self) -> Capabilities: ...

    def verify_model(self, descriptor: "ModelDescriptor") -> None: ...

    def verify_quant(self, quant: QuantSpec) -> None: ...

    def memory(self) -> MemoryInfo: ...

    def default_placement(self, descriptor: "ModelDescriptor") -> Placement: ...

    def backend(self, op: StageOp, shape: Shape) -> KernelBackend | None: ...

    def compile(self, programs: Sequence[object]) -> CompiledSet: ...

    def sync(self) -> None: ...
