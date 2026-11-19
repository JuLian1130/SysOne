"""Apple ANE 运行时：加载 `libe8ane.dylib`，把 C ABI 绑成 ctypes 签名。

这一层只管"加载 + 探测 + 最小调用面"。编译一个 program、拼 int8 blob、按 S 桶缓存，
都在 M2.1 的 `compile.py`，不放这里。

dylib 由 `tools/build_ane.sh` 从 `docs/ANE/anebench/e8_ane.m` 重建；产物在
`SYS1_E8_DIR`（默认 `/tmp/e8`），那是易失目录，所以"找不到就报 NotReady"而不是静默降级。
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ...errors import NotReadyError

DYLIB_NAME = "libe8ane.dylib"

# M2.0 用到的符号。少一个就说明 dylib 是旧的或编译坏了，直接报错，别在调用点炸。
_REQUIRED = (
    "e8_init",
    "e8_last_error",
    "e8_load",
    "e8_load_warm",
    "e8_eval",
    "e8_eval_slot",
    "e8_unload",
    "e8_unload_all",
    "e8_count",
    "e8_surface_new",
    "e8_surface_base",
    "e8_surface_bytes",
    "e8_surface_free",
    "e8_attach",
    "e8_attach_slot",
    "e8_attach_n",
    "e8_avail_gb",
    "e8_wired_gb",
    "e8_footprint_gb",
    "e8_peak_footprint_gb",
)


def _dylib_path(explicit: str | os.PathLike | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SYS1_E8_DIR", "/tmp/e8")) / DYLIB_NAME


class Surface:
    """一个 IOSurface 的 Python 视图：零拷贝，`array()` 拿到的是同一块内存。"""

    def __init__(self, runtime: "AneRuntime", handle: int) -> None:
        self._runtime = runtime
        self.handle = int(handle)

    @property
    def nbytes(self) -> int:
        return int(self._runtime.lib.e8_surface_bytes(self.handle))

    @property
    def base(self) -> int:
        pointer = self._runtime.lib.e8_surface_base(self.handle)
        return 0 if pointer is None else int(pointer)

    def array(self, dtype: np.dtype | type = np.float16, shape: tuple[int, ...] | None = None):
        """把 surface 的首地址包成 numpy 视图（不复制）。形状不合时 reshape 报错。"""
        if self.base == 0:
            raise NotReadyError("surface 尚未映射到地址空间", handle=self.handle)
        raw = (ctypes.c_ubyte * self.nbytes).from_address(self.base)
        view = np.frombuffer(raw, dtype=np.dtype(dtype))
        return view if shape is None else view.reshape(shape)

    def free(self) -> None:
        self._runtime.lib.e8_surface_free(self.handle)
        self.handle = 0


class AneRuntime:
    """`libe8ane.dylib` 的强类型句柄。构造即初始化 ANE（`e8_init` 返回非零就抛）。"""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        if sys.platform != "darwin":
            raise NotReadyError("ANE 运行时只在 macOS 上可用", platform=sys.platform)
        dylib = _dylib_path(path)
        if not dylib.is_file():
            raise NotReadyError(
                "找不到 ANE dylib",
                path=str(dylib),
                hint="运行 tools/build_ane.sh 重建",
            )
        try:
            self.lib = ctypes.CDLL(str(dylib))
        except OSError as exc:
            raise NotReadyError("加载 ANE dylib 失败", path=str(dylib), reason=str(exc)) from exc
        missing = [name for name in _REQUIRED if not hasattr(self.lib, name)]
        if missing:
            raise NotReadyError("dylib 缺少符号", missing=missing, path=str(dylib))
        self.path = str(dylib)
        self._bind()
        rc = self.lib.e8_init()
        if rc:
            raise NotReadyError("e8_init 失败", rc=int(rc), reason=self.error())
        # 异步 eval 的工作线程：`e8_eval` 是阻塞 ctypes 调用，但 ctypes 在调用期间释放 GIL，
        # 所以一个普通工作线程就够。**只有一个 eval 在飞**：两个并发发射者在这台机器上慢
        # 1.68×（同一条固件队列），所以这里刻意用 `max_workers=1`（`docs/ANE/anebench/e8_a3.py:168-179`）。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ane")
        self._pending: "Future | None" = None

    def _bind(self) -> None:
        lib = self.lib
        lib.e8_init.restype = ctypes.c_int
        lib.e8_last_error.restype = ctypes.c_char_p
        lib.e8_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.e8_load.restype = ctypes.c_int
        lib.e8_load_warm.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.e8_load_warm.restype = ctypes.c_int
        # 免内容哈希重装的快路（票据 09）。**可选**：老 dylib 没有这两个符号时保持
        # `load_warm` 的行为，而不是让整台设备起不来。
        self._descriptor_hashes = getattr(lib, "e8_descriptor_hashes", None)
        if self._descriptor_hashes is not None:
            self._descriptor_hashes.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
            ]
            self._descriptor_hashes.restype = ctypes.c_int
        self._load_fast = getattr(lib, "e8_load_fast", None)
        if self._load_fast is not None:
            self._load_fast.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_double),
            ]
            self._load_fast.restype = ctypes.c_int
        lib.e8_surface_new.argtypes = [ctypes.c_uint64]
        lib.e8_surface_new.restype = ctypes.c_uint64
        lib.e8_surface_base.argtypes = [ctypes.c_uint64]
        lib.e8_surface_base.restype = ctypes.c_void_p
        lib.e8_surface_bytes.argtypes = [ctypes.c_uint64]
        lib.e8_surface_bytes.restype = ctypes.c_uint64
        lib.e8_surface_free.argtypes = [ctypes.c_uint64]
        lib.e8_eval.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.e8_eval.restype = ctypes.c_double
        lib.e8_attach.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64]
        lib.e8_attach.restype = ctypes.c_int
        lib.e8_attach_slot.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.e8_attach_slot.restype = ctypes.c_int
        lib.e8_attach_n.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.e8_attach_n.restype = ctypes.c_int
        lib.e8_eval_slot.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.e8_eval_slot.restype = ctypes.c_double
        lib.e8_unload.argtypes = [ctypes.c_char_p]
        lib.e8_count.restype = ctypes.c_int
        lib.e8_avail_gb.restype = ctypes.c_double
        lib.e8_wired_gb.restype = ctypes.c_double
        lib.e8_footprint_gb.restype = ctypes.c_double
        lib.e8_peak_footprint_gb.restype = ctypes.c_double

    def error(self) -> str:
        raw = self.lib.e8_last_error()
        return "" if raw is None else raw.decode("utf-8", "replace")

    def _check(self, rc: int, what: str) -> None:
        if rc:
            raise NotReadyError(f"{what} 失败", rc=int(rc), reason=self.error())

    # ---------------------------------------------------------------- ANE 调用面

    @property
    def programs(self) -> int:
        """dylib 当前持有的 program 数（卸载后归零，用于长稳断言）。"""
        return int(self.lib.e8_count())

    def avail_gb(self) -> float:
        return float(self.lib.e8_avail_gb())

    def wired_gb(self) -> float:
        """**整机**的非可分页内存（`vm_statistics64.wire_count`），不是本进程的 ANE 占用。

        M2.0 的 doctor 曾把它当"我们的常驻"读成 2.4 GB，其实 0 个 program 时也是这个数。
        内存验收看 `footprint_gb()`。
        """
        return float(self.lib.e8_wired_gb())

    def footprint_gb(self) -> float:
        """本进程的 `phys_footprint`（活动监视器里的"内存"，含 IOSurface）——4 GB 预算看它。"""
        return float(self.lib.e8_footprint_gb())

    def peak_footprint_gb(self) -> float:
        """`phys_footprint` 的历史峰值；M2 的长稳/峰值验收用这个。"""
        return float(self.lib.e8_peak_footprint_gb())

    def surface(self, nbytes: int) -> Surface:
        handle = self.lib.e8_surface_new(int(nbytes))
        if not handle:
            raise NotReadyError("IOSurface 分配失败", nbytes=int(nbytes), reason=self.error())
        return Surface(self, handle)

    def load(self, name: str, mil: str, blob_path: str | os.PathLike) -> None:
        """编译并装载一个 program；`mil` 是 MIL 文本，`blob` 是它引用的权重 blob。"""
        self._check(
            self.lib.e8_load(name.encode(), mil.encode(), str(blob_path).encode()),
            f"load({name})",
        )

    def load_warm(self, name: str, mil: str, blob_path: str | os.PathLike) -> None:
        """编译缓存感知的装载：aned 已有该 (MIL, 权重) 的降级结果时只做 loadWithQoS。"""
        self._check(
            self.lib.e8_load_warm(name.encode(), mil.encode(), str(blob_path).encode()),
            f"load_warm({name})",
        )

    def descriptor_hashes(self, mil: str, blob_path: str | os.PathLike) -> tuple[str, str, str]:
        """aned 认这一份 (MIL, 权重) 的三段身份哈希 —— `load_fast` 的前半。

        段一次 ~25 ms（票据 09 实测 24.8 ms），所以只在一份产物**第一次**装上之后算一次、
        记进 sidecar；之后的重装都拿它走快路。dylib 没有这条入口、rc≠0、或给出空哈希，都抛
        `NotReadyError`：调用方把"记不下来"当正常情形，装载照旧走 `load_warm`。
        """
        fn = getattr(self, "_descriptor_hashes", None)
        if fn is None:
            raise NotReadyError("dylib 没有 e8_descriptor_hashes")
        ms = ctypes.c_double(0.0)
        buffers = [ctypes.create_string_buffer(256) for _ in range(3)]
        rc = fn(mil.encode(), str(blob_path).encode(),
                buffers[0], 256, buffers[1], 256, buffers[2], 256, ctypes.byref(ms))
        if rc != 0:
            raise NotReadyError("descriptor_hashes 失败", rc=int(rc), reason=self.error())
        texts = tuple(buffer.value.decode() for buffer in buffers)
        if len(texts) != 3 or not all(texts):
            raise NotReadyError("descriptor_hashes 给了空哈希", rc=int(rc))
        return texts                     # type: ignore[return-value]

    def load_fast(self, name: str, mil: str, blob_path: str | os.PathLike,
                  hashes: tuple[str, str, str]) -> bool:
        """用现成身份哈希装载，返回是否装上（rc≠0 = aned 那边没有这条 lowering）。

        不抛异常：缓存被驱逐、dylib 没有这条入口都只是"快路不可用"，由调用方回落 `load_warm`。
        段 0.9–1 ms（票据 09 实测），省掉的是那一次 ~28 ms 的权重内容哈希。
        """
        fn = getattr(self, "_load_fast", None)
        if fn is None:
            return False
        ms = ctypes.c_double(0.0)
        rc = fn(name.encode(), mil.encode(), str(blob_path).encode(),
                hashes[0].encode(), hashes[1].encode(), hashes[2].encode(), ctypes.byref(ms))
        return rc == 0

    def eval(self, name: str, procedure: int = 0) -> float:
        """同步执行一次，返回 dylib 记录的毫秒数（M2.0 只用它做对拍，不做吞吐）。"""
        return float(self.lib.e8_eval(name.encode(), int(procedure)))

    def eval_slot(self, name: str, procedure: int = 0, slot: int = 0) -> float:
        """在指定槽位上同步执行一次；槽位 0 与 `eval()` 等价，其余槽位要先用 `attach_slot` 绑过。"""
        return float(self.lib.e8_eval_slot(name.encode(), int(procedure), int(slot)))

    def attach(self, name: str, procedure: int, surf_in: Surface, surf_out: Surface) -> None:
        """把一对 surface 绑到一个 (program, procedure) 上；绑定后 `eval` 才能跑。"""
        self._check(
            self.lib.e8_attach(
                name.encode(), int(procedure), int(surf_in.handle), int(surf_out.handle)
            ),
            f"attach({name}, {procedure})",
        )

    def attach_slot(
        self, name: str, procedure: int, slot: int, surf_in: Surface, surf_out: Surface
    ) -> None:
        """把另一对 surface 绑到同一个 (program, procedure) 的第 `slot` 个槽位。

        多槽是为了"一次 eval 在飞时准备下一段"：同一段代码在不同槽位上跑，输入输出各是
        独立的 IOSurface，所以 GPU 可以往槽 1 写、ANE 从槽 0 读。槽 0 由 `attach()` 绑，
        这里是槽 1 起的补充。
        """
        self._check(
            self.lib.e8_attach_slot(
                name.encode(), int(procedure), int(slot),
                int(surf_in.handle), int(surf_out.handle),
            ),
            f"attach_slot({name}, {procedure}, slot={slot})",
        )

    def attach_multi(
        self, name: str, procedure: int, surfaces: Sequence[Surface], surf_out: Surface,
        *, slot: int = 0,
    ) -> None:
        """把**多张**输入面 + 一张输出面绑到同一个 (program, procedure, slot) 上。

        两条实测约束（票据 07 §7.19 / §7.23）：面的顺序按**输入名的字典序**接，不按声明序；而
        **只有 procedure 0 的多输入会被接线** —— 1 号及以后的 procedure 声明两张输入面时，
        eval 一律 `Program Inference error 0x2`。所以双输入的那一半必须排在最前面。
        """
        count = len(surfaces)
        if count == 0:
            raise NotReadyError("attach_multi 至少要一张输入面", program=name)
        ins = (ctypes.c_uint64 * count)(*[int(item.handle) for item in surfaces])
        indices = (ctypes.c_int * count)(*range(count))
        self._check(
            self.lib.e8_attach_n(
                name.encode(), int(procedure), int(slot), ins, indices, count,
                int(surf_out.handle),
            ),
            f"attach_multi({name}, {procedure}, slot={slot})",
        )

    # ------------------------------------------------------------ 异步执行

    @property
    def pending(self) -> bool:
        """是否还有一次已发射、未回收的 eval 在飞。"""
        return self._pending is not None

    def issue(self, name: str, procedure: int = 0, *, slot: int = 0) -> None:
        """异步发射一次 eval，立刻返回；用 `reap()` 回收。

        这是旧引擎 `ISSUE_EVAL`（`docs/ANE/anebench/e8_a3.py:168-179`）的语义：
        **不是并发 eval**（两个并发发射者慢 1.68×，同一条固件队列），而是"eval 在飞时
        CPU 不阻塞"，于是这段时间里可以往 GPU 命令缓冲里继续编码互不依赖的 kernel。
        """
        if self._pending is not None:
            raise NotReadyError("上一次 eval 还没回收", name=name, procedure=procedure, slot=slot)
        self._pending = self._executor.submit(self.eval_slot, name, int(procedure), int(slot))

    def reap(self) -> float:
        """回收一次在飞的 eval，返回它的毫秒数；没有在飞的就返回 0。"""
        pending = self._pending
        self._pending = None
        if pending is None:
            return 0.0
        return float(pending.result())

    def unload(self, name: str) -> None:
        self.lib.e8_unload(name.encode())

    def close(self) -> None:
        self.reap()
        self._executor.shutdown(wait=True)
        self.lib.e8_unload_all()
