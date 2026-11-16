"""Apple 平台（M2.0 骨架）：能力声明、内存预算、加载期校验。

M2.0 只把"平台是什么"讲清楚，并把 ANE 运行时接上；真正按算子选后端要等
M2.1（ANE 投影/FFN）与 M2.2（Metal 注意力/GDN）落 `backend()`，所以这里现在返回
`None`——`selector.select` 会因此报 `NoKernelError`，不会静默回退到 CPU。

几何白名单：`MEASURED_GEOMETRIES` 只收**端到端数值对拍过**的 `(hidden, ffn)` —— ANE 收下
`program` 的形状不证明数值对，所以形状探针不算证据。今天的两个（mude 见
`docs/ANE/README.md` §10 系列，h2o 见票据 15 §20/§21/§24）都是真权重跑通、对过厂商或
numpy 的。

**这张表今天是"可见性检查"，不是门禁**（票据 15 §35）：出货路径（`split` / `pooled` 两个
执行器工厂）在装链前调 `warn_if_unmeasured()`，不在表里就**告警一行**然后照常装 ——
ANE 对形状很宽容，未测几何会编译成功、跑出数字，所以真正要防的不是崩溃，而是"没人知道你
在用一个没验过的形状"。`sysone doctor --model … --weights …` 给同一个判据的查询形态。

不拦的理由是证据：今天没有出货路径会走到表外的几何（换一个 config 不同的检查点才会），
而把它挂成 `Engine.start()` 里的 `raise` 会拒掉测试夹具的小几何（M0 语义裁判那组测试），
代价大于收益。

表里只判 `(hidden, ffn)`，**不判层数与头数**，也**不判量化形态**：检查点存的是
`per_row_sym_int8`（`stored_quant` 报的就是它），而 ANE 吃的是装载时派生的
`per_channel_sym_int8` blob —— 拿 `verify_quant(descriptor.quant)` 去验会当场把 h2o 拒掉
（票据 15 §35.2）。加一项到表里的条件仍然是"端到端数值对拍过"，形状探针不算数。
"""

from __future__ import annotations

import ctypes
import logging
import sys
from pathlib import Path

from ...domain.attention import AttentionRole, MaskMode
from ...domain.quant import DType, QuantSpec, Scheme
from ...domain.work import StageOp
from ...errors import ConfigError, UnsupportedPrecisionError
from ...models.descriptor import ModelDescriptor
from ..base import Capabilities, MemoryInfo, Placement, Shape
from .runtime import AneRuntime

# 实测白名单：**端到端**跑通过（numpy 逐位对拍 + 真权重）的 (hidden, ffn)。加一项要拿证据
# —— 形状探针只证明 ANE 收下这些 program，不证明数值对。
#
# (2560, 9216) 是 h2o：真权重单层对拍（票据 15 §20，全部结构性中间量 = 3.6e-04 的投影地板）、
# 整链对厂商 6.3e-3（§21）、numpy 链对 Metal 流水线 1.3e-3（§24，两种读法、64/128 token）。
MEASURED_GEOMETRIES: frozenset[tuple[int, int]] = frozenset({(2048, 6144), (2560, 9216)})
# 一个 ANE program 的原生 S 上限约 4096（16k 在旧引擎里是 4 × 4096 分块跑的）。
MAX_TILE = 4096
# 平台愿意服务的最大 token 数（16k = 4 × 4096 分块）。它是**能力声明**，不是加载期拒绝的
# 理由：模型自报的 `max_position` 是上限，我们只会用得更少（mude 自报 262144，出货跑 16k）。
MAX_SEQ = 16384

_LOGGER = logging.getLogger(__name__)


def measured(descriptor: ModelDescriptor) -> bool:
    """这个几何有没有**端到端数值对拍**的证据（判据只有 `(hidden, ffn)`）。

    单点判据供两处共用：装链前的告警（`warn_if_unmeasured`）与 `sysone doctor` 的查询，
    这样"哪套形状算验过"只有一处定义。
    """
    return (descriptor.hidden, descriptor.ffn) in MEASURED_GEOMETRIES


def warn_if_unmeasured(descriptor: ModelDescriptor, *, strategy: str) -> None:
    """装链前提示"这套几何没有做过数值对拍"；**不阻断**，理由见模块顶部说明。

    出货路径上唯一会走到表外几何的方式是拿一个 config 不同的检查点，而那种情况下最糟的
    结果不是崩溃：ANE 会编译、会跑，给出看起来合理的数字。所以这里出声，让日志里有据可查。
    """
    if measured(descriptor):
        return
    _LOGGER.warning(
        "几何 hidden=%d ffn=%d 不在 Apple 的实测白名单里（strategy=%s）：ANE 会照常编译并"
        "跑出数字，但这套形状没有做过数值对拍（%s），结果只能当实验读数",
        descriptor.hidden,
        descriptor.ffn,
        strategy,
        descriptor.model_id,
    )

_LIBC = "/usr/lib/libSystem.B.dylib"


def _physical_bytes() -> int:
    """`hw.memsize`。拿不到就返回 0：预算由 `usable` 决定，`total` 只用于展示。"""
    try:
        libc = ctypes.CDLL(_LIBC)
    except OSError:
        return 0
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    libc.sysctlbyname.restype = ctypes.c_int
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(ctypes.c_uint64))
    rc = libc.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0)
    return int(value.value) if rc == 0 else 0


class ApplePlatform:
    """Apple Silicon：ANE 做投影/FFN，GPU 做注意力/GDN，两者零拷贝交接。"""

    name = "apple"

    def __init__(
        self, *, e8_dir: str | Path | None = None, runtime: AneRuntime | None = None
    ) -> None:
        self.e8_dir = None if e8_dir is None else Path(e8_dir)
        self._runtime = runtime

    @property
    def runtime(self) -> AneRuntime:
        if self._runtime is None:
            self._runtime = AneRuntime(self.e8_dir)
        return self._runtime

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # ANE 的算术是 fp16（硬件没有 fp32 路径），所以平台对外只承诺 fp16；
            # GPU 侧的 fp32 归约是实现细节，不上浮成模型精度。
            dtypes=frozenset({DType.FP16}),
            quant=frozenset({Scheme.NONE, Scheme.PER_CHANNEL_SYM_INT8}),
            modes=frozenset(MaskMode),
            max_seq=MAX_SEQ,
            max_tile=MAX_TILE,
            zero_copy=True,
        )

    def verify_model(self, descriptor: ModelDescriptor) -> None:
        if not any(role is AttentionRole.FULL for role in descriptor.layer_types):
            raise ConfigError("模型没有全注意力层", model=descriptor.model_id)
        if not measured(descriptor):
            raise ConfigError(
                "模型几何不在 Apple 的实测白名单里",
                model=descriptor.model_id,
                hidden=descriptor.hidden,
                ffn=descriptor.ffn,
                measured=[{"hidden": hidden, "ffn": ffn}
                          for hidden, ffn in sorted(MEASURED_GEOMETRIES)],
                hint="白名单只收端到端跑通过的几何（见 platform.py 顶部注释）；"
                     "ANE 收下 program 形状还不算数",
            )

    def verify_quant(self, quant: QuantSpec) -> None:
        quant.validate()
        if not quant.is_quantized:
            return
        # 组合合法性由 `QuantSpec.validate()` 收窄到三组，所以这里只剩"这一种量化
        # 是不是 ANE 能吃的"这一个问题。
        if quant.scheme is not Scheme.PER_CHANNEL_SYM_INT8:
            raise UnsupportedPrecisionError(
                "ANE 只接受 per-channel int8（码与 fp16 标度同 blob）",
                quant=str(quant),
                hint="per-row int8 是检查点（及其 GPU 路径）的存法，装载时会派生成 per-channel blob",
            )

    def memory(self) -> MemoryInfo:
        runtime = self.runtime
        usable = int(runtime.avail_gb() * 1024**3)
        return MemoryInfo(total=_physical_bytes(), current=0, usable=usable, wired_limit=None)

    def default_placement(self, descriptor: ModelDescriptor) -> Placement:
        # 投影与 FFN 落 ANE：它们是纯线性、无数据依赖，正好对上 ANE 的强项。
        return Placement("ane")

    def backend(self, op: StageOp, shape: Shape) -> None:
        """M2.0 尚未有算子实现；M2.1/M2.2 逐个填。返回 None 让 selector 报 NoKernel。"""
        return None

    def compile(self, programs: object) -> None:
        raise ConfigError(
            "整层 program 的批量编译随 M2.2 的 pipeline 落地；M2.1 只用 devices/apple/compile.py 的单层 AneProjection",
            op=getattr(programs, "op", None),
        )

    def sync(self) -> None:
        """ANE 的 `eval` 本身就是同步等待；GPU 侧的同步在 M2.2 的 Metal 后端里做。"""
        return None

    def probe(self) -> dict[str, object]:
        """一次可打印的能力探测：M2.0 的验收就是这几行数字。"""
        runtime = self.runtime
        info = self.memory()
        caps = self.capabilities()
        return {
            "platform": self.name,
            "dylib": runtime.path,
            "programs": runtime.programs,
            "avail_gb": runtime.avail_gb(),
            # 常驻看本进程 footprint；`wired_gb` 是整机 wire_count，不能当我们的占用。
            "footprint_gb": round(runtime.footprint_gb(), 3),
            "peak_footprint_gb": round(runtime.peak_footprint_gb(), 3),
            "total_gb": round(info.total / 1024**3, 3) if info.total else 0.0,
            "dtypes": sorted(dtype.value for dtype in caps.dtypes),
            "quant": sorted(scheme.value for scheme in caps.quant),
        }


def available(e8_dir: str | Path | None = None) -> bool:
    """本机能不能跑 Apple 平台（macOS + 可加载的 dylib）。"""
    if sys.platform != "darwin":
        return False
    try:
        AneRuntime(e8_dir)
    except Exception:
        return False
    return True
