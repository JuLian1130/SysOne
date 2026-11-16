"""Apple 平台（M2.0 骨架）：能力声明、内存预算、加载期校验。

M2.0 只把"平台是什么"讲清楚，并把 ANE 运行时接上；真正按算子选后端要等
M2.1（ANE 投影/FFN）与 M2.2（Metal 注意力/GDN）落 `backend()`，所以这里现在返回
`None`——`selector.select` 会因此报 `NoKernelError`，不会静默回退到 CPU。

几何白名单：旧引擎只在 mude 的 20 层 / hidden 2048 / ffn 6144 上**端到端**跑通过
（docs/ANE/README.md 的 §10 系列）。4B 的 32 层 / 2560 / 9216 是 M4 的拆层实验：ANE 已经
收下它的 program 形状（票据 15 §11），但数值还没有真实权重对拍，现在跑它只会得到
"看起来像结果"的错误数字 —— 所以加载期放行的是**实测白名单**里的几何，不是某个尺寸上限。
"""

from __future__ import annotations

import ctypes
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
        if (descriptor.hidden, descriptor.ffn) not in MEASURED_GEOMETRIES:
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
