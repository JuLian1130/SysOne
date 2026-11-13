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
    "e8_unload",
    "e8_unload_all",
    "e8_count",
    "e8_surface_new",
    "e8_surface_base",
    "e8_surface_bytes",
    "e8_surface_free",
    "e8_avail_gb",
    "e8_wired_gb",
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

    def _bind(self) -> None:
        lib = self.lib
        lib.e8_init.restype = ctypes.c_int
        lib.e8_last_error.restype = ctypes.c_char_p
        lib.e8_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.e8_load.restype = ctypes.c_int
        lib.e8_load_warm.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.e8_load_warm.restype = ctypes.c_int
        lib.e8_surface_new.argtypes = [ctypes.c_uint64]
        lib.e8_surface_new.restype = ctypes.c_uint64
        lib.e8_surface_base.argtypes = [ctypes.c_uint64]
        lib.e8_surface_base.restype = ctypes.c_void_p
        lib.e8_surface_bytes.argtypes = [ctypes.c_uint64]
        lib.e8_surface_bytes.restype = ctypes.c_uint64
        lib.e8_surface_free.argtypes = [ctypes.c_uint64]
        lib.e8_eval.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.e8_eval.restype = ctypes.c_double
        lib.e8_unload.argtypes = [ctypes.c_char_p]
        lib.e8_count.restype = ctypes.c_int
        lib.e8_avail_gb.restype = ctypes.c_double
        lib.e8_wired_gb.restype = ctypes.c_double

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
        """进程的 ANE wired 内存（M2 的内存验收看这个，不看 RSS）。"""
        return float(self.lib.e8_wired_gb())

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

    def eval(self, name: str, procedure: int = 0) -> float:
        """同步执行一次，返回 dylib 记录的毫秒数（M2.0 只用它做对拍，不做吞吐）。"""
        return float(self.lib.e8_eval(name.encode(), int(procedure)))

    def unload(self, name: str) -> None:
        self.lib.e8_unload(name.encode())

    def close(self) -> None:
        self.lib.e8_unload_all()
