"""Apple GPU（Metal）运行时：加载 `libe8mtl.dylib`，把 C ABI 绑成 ctypes 签名。

GPU 与 ANE 的差别就在 kernel 的生命周期：ANE 的 program 是编译产物（`compile.py` 管），
Metal 的 kernel 是启动时现场编译的。本机只有 Command Line Tools、没有 `xcrun metal`，
所以 `e8m_init_src` 把 `e8_common.h` + 若干 `.metal` 源码拼成一份源码，交给 Metal 框架里
的运行时编译器；第一个文件必须是公共头（`e8_mtl.m:148-152` 写明约定）。

这一层只做三件事：绑定符号、把错误串变成异常、把一块内存抽象成 `MetalBuffer`。
kernel 的参数结构体与启动网格在 `kernels.py`，接进模型链在 `chain.py`。

dylib 由 `tools/build_ane.sh` 重建，产物在 `SYS1_E8_DIR`（默认 `/tmp/e8`）——易失目录，
所以"找不到就报 `NotReadyError`"，不静默降级。
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from ...errors import ConfigError, NotReadyError, SysOneError
from .geometry import Geometry, mude_geometry

MTL_DYLIB_NAME = "libe8mtl.dylib"

# 编译顺序即拼接顺序：公共头必须在最前，后面的 .metal 里 `#include "e8_common.h"` 会被丢掉
# （运行时编译没有 include 路径）。
#
# 这四份源码是 `docs/ANE/anebench` 的副本，vendored 到包内（`shaders/`）：kernel 需要跟着
# 代码走，而 `docs/` 是独立的本地仓库、不进主仓库也不进安装包。唯一的功能改动写在
# `shaders/e8_attn.metal` 顶部：原实现的分片搬运用了 lambda，本机 MSL 编译器不支持，已展开成
# `at_stage()`，循环体逐字保留。
SOURCES = ("e8_common.h", "e8_kernels.metal", "e8_attn.metal", "e8_gdn.metal")

# 复用到的符号。少一个就说明 dylib 是旧的或编译坏了，在 init 处一次报清楚。
_REQUIRED = (
    "e8m_init_src",
    "e8m_last_error",
    "e8m_wrap",
    "e8m_unwrap",
    "e8m_alloc",
    "e8m_base",
    "e8m_bytes",
    "e8m_free",
    "e8m_dispatch_l",
    "e8m_commit_l",
    "e8m_wait_l",
    "e8m_sync_l",
    "e8m_gpu_ms",
)


def _dylib_path(explicit: str | os.PathLike | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return Path(os.environ.get("SYS1_E8_DIR", "/tmp/e8")) / MTL_DYLIB_NAME


def _sources_dir(explicit: str | os.PathLike | None) -> Path:
    """kernel 源码目录：默认是包内的 `shaders/`（从旧引擎 vendored 过来的那份）。

    `SYS1_MTL_SRC` 可覆盖（调 kernel 时指向实验副本很方便）。
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("SYS1_MTL_SRC")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "shaders"


def _write_geometry_header(geometry: Geometry) -> Path:
    """把几何宏写成一个临时头文件，作为编译时插在最前面的那段源码（architecture §13 R1）。

    内容只由几何决定，所以按宏的哈希命名：同几何复用同一份文件，不同几何各一份，互不覆盖。
    """
    digest = hashlib.sha256(
        "|".join(f"{name}={value}" for name, value in sorted(geometry.defines().items())).encode()
    ).hexdigest()[:16]
    directory = Path(tempfile.gettempdir()) / "sysone-metal"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"e8_geometry_{digest}.h"
    text = geometry.header()
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


class MetalBuffer:
    """一块 Metal buffer 的 Python 视图。

    `wrap()` 出来的是 IOSurface 的零拷贝包装（同样一块内存 ANE 也会写），`alloc()` 出来的
    是独立分配的 buffer。两种都允许 `array()` 直接读写——Metal 侧用 shared storage，
    所以 host 看得见。
    """

    def __init__(self, runtime: "MetalRuntime", handle: int, nbytes: int, wrapped: bool,
                 surface=None) -> None:
        self._runtime = runtime
        self.handle = int(handle)
        self.nbytes = int(nbytes)
        self.wrapped = wrapped
        self.surface = surface

    @property
    def base(self) -> int:
        pointer = self._runtime.lib.e8m_base(self.handle)
        return 0 if pointer is None else int(pointer)

    def array(self, dtype: np.dtype | type = np.float16, shape: tuple[int, ...] | None = None):
        """把 buffer 首地址包成 numpy 视图（不复制）。shape 不合时 reshape 报错。"""
        if self.base == 0:
            raise NotReadyError("Metal buffer 尚未映射到地址空间", handle=self.handle)
        dtype = np.dtype(dtype)
        view = np.frombuffer((ctypes.c_ubyte * self.nbytes).from_address(self.base), dtype=dtype)
        if shape is not None:
            view = view.reshape(shape)
        return view

    def free(self) -> None:
        if not self.handle:
            return
        if self.wrapped:
            # 包装句柄在 dylib 里按 IOSurface 指针做缓存，只能解绑、不能 free，
            # 否则下一次分配复用同一地址时会拿到一条已释放的句柄。
            if self.surface is not None:
                self._runtime.unwind(self.surface)
        else:
            self._runtime.lib.e8m_free(self.handle)
        self.handle = 0

    def __enter__(self) -> "MetalBuffer":
        return self

    def __exit__(self, *_exc) -> None:
        self.free()

    def __repr__(self) -> str:
        return f"<MetalBuffer #{self.handle} {self.nbytes} B wrapped={self.wrapped}>"


class MetalRuntime:
    """设备 + 命令队列 + 编译好的 kernel 库。一个进程一份，由 ApplePlatform 持有。

    **必须共享**：`e8m_init_src` 会重建 dylib 内部的命令队列与 pipeline 状态，第二次调用之后
    之前那次发出的 dispatch 就再也到不了 GPU（实测第二个运行时建出的链输出全零）。所以这里
    按进程缓存，后来的构造返回同一个对象（`tests/integration/test_metal_chain.py` 的两个用例
    会各建一条链，正是踩这个坑才发现的）。
    """

    _shared: "MetalRuntime | None" = None

    def __new__(
        cls,
        path: str | os.PathLike | None = None,
        sources_dir: str | os.PathLike | None = None,
        *,
        verbose: bool = False,
        geometry: "Geometry | None" = None,
    ) -> "MetalRuntime":
        if cls._shared is not None:
            return cls._shared
        self = super().__new__(cls)
        cls._shared = self
        return self

    def __init__(
        self,
        path: str | os.PathLike | None = None,
        sources_dir: str | os.PathLike | None = None,
        *,
        verbose: bool = False,
        geometry: "Geometry | None" = None,
    ) -> None:
        if getattr(self, "_ready", False):
            # 进程级单例：后来的构造只校验几何一致，不再重新编译（见类 docstring）。
            # 首次构造没给几何时，编进去的是 `shaders/` 的默认值，也就等价于 mude 的几何。
            compiled = mude_geometry() if self._geometry is None else self._geometry
            if geometry is not None and geometry != compiled:
                raise ConfigError("同进程的 Metal 运行时已经按另一套几何编译过",
                                  compiled=compiled, requested=geometry)
            return
        if sys.platform != "darwin":
            raise NotReadyError("Metal 运行时只在 macOS 上可用", platform=sys.platform)
        dylib = _dylib_path(path)
        if not dylib.is_file():
            MetalRuntime._shared = None
            raise NotReadyError(
                "找不到 libe8mtl.dylib",
                path=str(dylib),
                hint="运行 tools/build_ane.sh 重建（/tmp/e8 是易失目录）",
            )
        self.path = dylib
        self._geometry = geometry
        self.lib = ctypes.CDLL(str(dylib))
        try:
            self._bind()
            self._compile(_sources_dir(sources_dir), verbose=verbose, geometry=geometry)
        except SysOneError:
            MetalRuntime._shared = None  # 初始化失败就别缓存，下次还能重试
            raise
        self._ready = True

    # -- 绑定 ----------------------------------------------------------------
    def _bind(self) -> None:
        lib = self.lib
        for name in _REQUIRED:
            if not hasattr(lib, name):
                raise NotReadyError("libe8mtl.dylib 缺少符号", symbol=name, path=str(self.path))
        lib.e8m_init_src.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_int]
        lib.e8m_init_src.restype = ctypes.c_int
        lib.e8m_last_error.restype = ctypes.c_char_p
        lib.e8m_wrap.argtypes = [ctypes.c_uint64]
        lib.e8m_wrap.restype = ctypes.c_uint64
        lib.e8m_unwrap.argtypes = [ctypes.c_uint64]
        lib.e8m_unwrap.restype = ctypes.c_int
        lib.e8m_alloc.argtypes = [ctypes.c_uint64]
        lib.e8m_alloc.restype = ctypes.c_uint64
        lib.e8m_base.argtypes = [ctypes.c_uint64]
        lib.e8m_base.restype = ctypes.c_void_p
        lib.e8m_bytes.argtypes = [ctypes.c_uint64]
        lib.e8m_bytes.restype = ctypes.c_uint64
        lib.e8m_free.argtypes = [ctypes.c_uint64]
        lib.e8m_dispatch_l.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        lib.e8m_dispatch_l.restype = ctypes.c_int
        lib.e8m_commit_l.argtypes = [ctypes.c_int]
        lib.e8m_commit_l.restype = ctypes.c_int
        lib.e8m_wait_l.argtypes = [ctypes.c_int]
        lib.e8m_wait_l.restype = ctypes.c_double
        lib.e8m_sync_l.argtypes = [ctypes.c_int]
        lib.e8m_sync_l.restype = ctypes.c_double
        lib.e8m_gpu_ms.restype = ctypes.c_double

    def _compile(self, sources: Path, *, verbose: bool, geometry: "Geometry | None" = None) -> None:
        missing = [name for name in SOURCES if not (sources / name).is_file()]
        if missing:
            raise NotReadyError(
                "找不到 Metal kernel 源码",
                directory=str(sources),
                missing=missing,
                hint="设置 SYS1_MTL_SRC 指向含 e8_common.h 与 *.metal 的目录",
            )
        # 几何头必须是**第一段源码**：`e8m_init_src` 按顺序拼接，并把 `.metal` 里的
        # `#include "e8_common.h"` 删掉，所以写在它前面的 `#define E8_GEOMETRY_SET` 会先于
        # `e8_common.h` 的默认宏块生效（那个块被 `#ifndef` 包着）。
        files = [sources / name for name in SOURCES]
        if geometry is not None:
            header = _write_geometry_header(geometry)
            files.insert(0, header)
        encoded = [str(path).encode() for path in files]
        paths = (ctypes.c_char_p * len(encoded))(*encoded)
        rc = self.lib.e8m_init_src(paths, len(encoded), 1 if verbose else 0)
        if rc != 0:
            raise NotReadyError("Metal kernel 编译失败", reason=self.error(), rc=rc,
                                geometry=None if geometry is None else dict(geometry.defines()))

    # -- 内存 ----------------------------------------------------------------
    def wrap(self, surface) -> MetalBuffer:
        """把一个 ANE 的 IOSurface 包成 Metal buffer（零拷贝；同一块内存）。"""
        handle = int(self.lib.e8m_wrap(int(surface.handle)))
        if handle == 0:
            raise NotReadyError("包装 IOSurface 失败", reason=self.error(), surface=int(surface.handle))
        return MetalBuffer(self, handle, int(self.lib.e8m_bytes(handle)), wrapped=True,
                           surface=surface)

    def unwind(self, surface) -> None:
        """surface 释放前必须调用：包装缓存按 IOSurface 指针键，地址会被下一次分配复用。"""
        self.lib.e8m_unwrap(int(surface.handle))

    def alloc(self, nbytes: int) -> MetalBuffer:
        if nbytes <= 0:
            raise NotReadyError("Metal buffer 大小必须为正", bytes=nbytes)
        handle = int(self.lib.e8m_alloc(int(nbytes)))
        if handle == 0:
            raise NotReadyError("分配 Metal buffer 失败", reason=self.error(), bytes=nbytes)
        return MetalBuffer(self, handle, nbytes, wrapped=False)

    # -- 执行 ----------------------------------------------------------------
    def dispatch(
        self,
        kernel: str,
        buffers: list[MetalBuffer],
        *,
        args: bytes = b"",
        grid: tuple[int, int, int] = (1, 1, 1),
        tg: tuple[int, int, int] = (64, 1, 1),
        lane: int = 0,
    ) -> None:
        handles = (ctypes.c_uint64 * len(buffers))(*[int(b.handle) for b in buffers])
        rc = self.lib.e8m_dispatch_l(
            int(lane),
            kernel.encode(),
            handles,
            len(buffers),
            args,
            len(args),
            int(grid[0]),
            int(grid[1]),
            int(grid[2]),
            int(tg[0]),
            int(tg[1]),
            int(tg[2]),
        )
        if rc != 0:
            raise NotReadyError(
                f"Metal dispatch 失败（{kernel}）", reason=self.error(), kernel=kernel, rc=rc
            )

    def commit(self, lane: int = 0) -> None:
        if self.lib.e8m_commit_l(int(lane)) != 0:
            raise NotReadyError("提交命令缓冲失败", reason=self.error(), lane=lane)

    def wait(self, lane: int = 0) -> float:
        return float(self.lib.e8m_wait_l(int(lane)))

    def sync(self, lane: int = 0) -> float:
        return float(self.lib.e8m_sync_l(int(lane)))

    def gpu_ms(self) -> float:
        return float(self.lib.e8m_gpu_ms())

    def error(self) -> str:
        raw = self.lib.e8m_last_error()
        return "" if raw is None else raw.decode(errors="replace")

    def close(self) -> None:
        # dylib 的生命周期跟进程一样，而且重复初始化会清掉命令队列（见类 docstring），
        # 所以这里什么都不做：运行时按进程共享，调用方随时可以再拿到同一个对象。
        return None


def available(e8_dir: str | os.PathLike | None = None) -> bool:
    """能否在这台机器上跑 Metal 计算（非 macOS 或缺 dylib 都算不能）。"""
    if sys.platform != "darwin":
        return False
    try:
        runtime = MetalRuntime(path=None if e8_dir is None else Path(e8_dir) / MTL_DYLIB_NAME)
    except NotReadyError:
        return False
    runtime.close()
    return True
