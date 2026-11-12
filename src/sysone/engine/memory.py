"""`BufferSpec` → `MemoryPlan` 与启动期预算拒绝。

缓冲区大小从声明推导，不手算 `wired_gb`；预算只在显式配置时生效，超了就在启动期
抛 `BudgetExceededError`（消息里带上每一项，方便判断是产物太大还是窗口太长）。
"""

from __future__ import annotations

from ..config.engine import EngineConfig
from ..domain.attention import AttentionRole
from ..domain.buffers import BufferName, BufferSpec, MemoryPlan
from ..domain.quant import DType
from ..errors import BudgetExceededError
from ..models.descriptor import ModelDescriptor

_ITEM_BYTES = {DType.BF16: 2, DType.FP16: 2, DType.FP32: 4, DType.INT8: 1}


def buffer_specs(descriptor: ModelDescriptor, config: EngineConfig) -> tuple[BufferSpec, ...]:
    """声明本轮工作负载需要的缓冲区。

    以 bf16 全精度为准，条目含义见 `architecture.md` §6：

    - `h`：`w_h` 与两个激活缓冲（`io_a` / `io_b`）共三个 hidden 宽度；
    - `kv`：只有全注意力层有 K/V，`kv_heads * head_dim * 2` 且只算 FULL 层；
    - `ring`：预取环，与 `--W` 无关，固定 `ring * hidden`；
    - `scratch`：一个 tile 的 fp32 中间值，按 `ffn` 宽度；
    - `gdn_state`：GDN 状态依赖线性层几何（旧实现是 16×128×128 个 fp32），
      描述符目前不带这组字段，随 M2 的 Apple provider 补上。
    """
    item = _ITEM_BYTES[DType.BF16]
    full_layers = sum(1 for role in descriptor.layer_types if role == AttentionRole.FULL)
    kv_per_token = descriptor.kv_heads * descriptor.head_dim * 2 * item * max(full_layers, 1)
    return (
        BufferSpec(BufferName.H, DType.BF16, per_token=3 * descriptor.hidden * item),
        BufferSpec(BufferName.KV, DType.BF16, per_token=kv_per_token),
        BufferSpec(BufferName.RING, DType.BF16, per_layer=int(config.scheduler.ring) * descriptor.hidden * item),
        BufferSpec(BufferName.SCRATCH, DType.FP32, per_layer=descriptor.ffn * _ITEM_BYTES[DType.FP32]),
    )


def plan_memory(
    descriptor: ModelDescriptor, config: EngineConfig, *, artifacts_bytes: int = 0
) -> MemoryPlan:
    """算出内存计划并在超预算时抛错。"""
    seq_len = int(config.scheduler.max_len)
    buffers = {
        spec.name: spec.bytes_for(seq_len=seq_len, layers=1)
        for spec in buffer_specs(descriptor, config)
    }
    plan = MemoryPlan(buffers, int(artifacts_bytes), int(config.storage.artifacts_budget))
    if not plan.fits():
        raise BudgetExceededError(
            "内存或产物超出预算",
            total=plan.total,
            budget=plan.budget,
            breakdown=plan.report(),
        )
    return plan
