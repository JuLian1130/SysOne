"""macOS 解码：图片字节 → `(H, W, 3)` uint8，走 ImageIO（`libe8image.dylib`）。

**为什么要有这一层**：这个仓库的 venv 里没有 pillow、没有 torch、没有 OpenCV（查过），而
纯 Python 解 JPEG 不是该出货的东西。ImageIO 是系统解码器 —— 它认 `Qwen2VLImageProcessor`
接受的每一种格式，而且就是预览与访达用的那一个，所以"这张图在 Mac 上打得开"和"服务读得了"
是同一个问题。

两条与参照处理器一致、且**故意**如此的约定（源码 `tools/native/e8_image.m` 的头注释）：

1. **不应用 EXIF 方向**。`CGImageSourceCreateImageAtIndex` 忽略方向标签，Pillow 的
   `Image.open(...).convert("RGB")` 也一样，于是两边默认就对得上。在这里应用旋转，恰好会在
   最要紧的一类文件（手机照片）上与参照分叉。
2. **丢 alpha，不合成**。`kCGImageAlphaNoneSkipLast` 让目的缓冲没有 alpha 通道，源里的 alpha
   被丢弃而不是与背景混合 —— 与 Pillow 的 `convert("RGB")` 相同。合成到白底是另一个函数。

多帧文件（动图 GIF、HEIC burst）取首帧：处理器是"一个占位符一张图"，挑"最好的一帧"是参照
没有的策略。

**有损格式上与 Pillow 的差是解码器策略，不是算术**：无损格式（PNG/TIFF）实测**逐位相同**；
JPEG 4:4:4 只差反 DCT 的舍入（≤3 个 8 位步长、均值 0.21）；而**色度子采样的 4:2:0**（相机的
默认）差到 82 个步长、均值 1.6 —— 那是色度上采样滤波器的选择不同（Pillow 用三角形滤波，
CoreGraphics 另一套），只在色度有高频的地方显形。这一层的方差大于重采样，所以"视觉够不够准"
的基线是 1e-2 而不是 1e-3（票据 07 §2.3 #3.2）。

dylib 由 `tools/build_ane.sh` 从本仓库的 `tools/native/e8_image.m` 编译到 `SYS1_E8_DIR`
（默认 `<仓库>/.tmp/e8`），所以"找不到就报 `NotReadyError`"而不是静默降级 ——
与 `runtime.py` 同一个形态。**导入这个模块不需要编译**：库在第一次 `decode` 时才加载。

这里的错误分两种，别混：**dylib 不在**是 `NotReadyError`（503，运维该去重建），**字节不是
图**是 `UnsupportedMediaError`（422，客户端该换一张）。
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from ...errors import NotReadyError, UnsupportedMediaError
from .paths import dylib_dir

DYLIB_NAME = "libe8image.dylib"
_REQUIRED = ("e8_img_decode", "e8_img_free", "e8_img_size")
_ERR_LEN = 256

_lib: ctypes.CDLL | None = None
_lib_path: str | None = None


def _dylib_path(explicit: str | os.PathLike | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    return dylib_dir() / DYLIB_NAME


def _load(path: str | os.PathLike | None) -> ctypes.CDLL:
    """加载并绑定 C ABI；进程内缓存（同一份 dylib 只 `dlopen` 一次）。"""
    global _lib, _lib_path
    if _lib is not None and (path is None or str(_dylib_path(path)) == _lib_path):
        return _lib
    if sys.platform != "darwin":
        raise NotReadyError("图片解码只在 macOS 上可用", platform=sys.platform)
    dylib = _dylib_path(path)
    if not dylib.is_file():
        raise NotReadyError("找不到图片解码 dylib", path=str(dylib),
                            hint="运行 tools/build_ane.sh 重建（源码在 tools/native/）")
    try:
        lib = ctypes.CDLL(str(dylib))
    except OSError as exc:
        raise NotReadyError("加载图片解码 dylib 失败", path=str(dylib), reason=str(exc)) from exc
    missing = [name for name in _REQUIRED if not hasattr(lib, name)]
    if missing:
        raise NotReadyError("dylib 缺少符号", missing=missing, path=str(dylib))
    lib.e8_img_decode.restype = ctypes.c_int
    lib.e8_img_decode.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    lib.e8_img_free.restype = None
    lib.e8_img_free.argtypes = [ctypes.c_void_p]
    lib.e8_img_size.restype = ctypes.c_int
    lib.e8_img_size.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_char_p, ctypes.c_size_t,
    ]
    _lib, _lib_path = lib, str(dylib)
    return lib


def available(e8_dir: str | os.PathLike | None = None) -> bool:
    """`decode` 现在能不能跑（非 macOS 或缺 dylib 都算不能）。**不加载库、无副作用。**

    前门（P4）用它决定要不要把视觉报成可用 —— 与 `metal.available()` 同一个理由：探测不该
    顺手把东西编进进程。
    """
    if sys.platform != "darwin":
        return False
    path = Path(e8_dir) / DYLIB_NAME if e8_dir is not None else _dylib_path(None)
    if not path.is_file():
        return False
    try:
        ctypes.CDLL(str(path))
    except OSError:
        return False
    return True


def size(data: bytes, *, path: str | os.PathLike | None = None) -> tuple[int, int]:
    """`bytes` → `(宽, 高)`，**不解码**：只读容器头。

    16 MP 那条拒收要用它：真解码再判，等于先把"这张图不该被解"的那块内存申请下来（30000²
    的 PNG 压缩后几百 KB、解开是 3.6 GB）。读到的是编码尺寸、不是 EXIF 旋转之后的尺寸 ——
    与 PIL 的 `Image.size` 同口径，所以两边的 16 MP 判据落在同一批文件上。
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise UnsupportedMediaError("解码只接受字节", type=type(data).__name__)
    buf = bytes(data)
    lib = _load(path)
    width, height = ctypes.c_int(), ctypes.c_int()
    err = ctypes.create_string_buffer(_ERR_LEN)
    rc = lib.e8_img_size(buf, len(buf), ctypes.byref(width), ctypes.byref(height), err, _ERR_LEN)
    if rc != 0:
        reason = err.value.decode("utf-8", "replace")
        raise UnsupportedMediaError("认不出这张图", reason=reason, bytes=len(buf), rc=int(rc))
    return width.value, height.value


def decode(data: bytes, *, path: str | os.PathLike | None = None) -> np.ndarray:
    """`bytes` → `(H, W, 3)` uint8 RGB（sRGB，丢 alpha、不转 EXIF，见模块头）。

    `path` 只给测试与探针用；出货路径读 `SYS1_E8_DIR`。
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise UnsupportedMediaError("解码只接受字节", type=type(data).__name__)
    buf = bytes(data)
    lib = _load(path)
    ptr = ctypes.POINTER(ctypes.c_uint8)()
    width, height = ctypes.c_int(), ctypes.c_int()
    err = ctypes.create_string_buffer(_ERR_LEN)
    rc = lib.e8_img_decode(buf, len(buf), ctypes.byref(ptr), ctypes.byref(width),
                           ctypes.byref(height), err, _ERR_LEN)
    if rc != 0:
        reason = err.value.decode("utf-8", "replace")
        raise UnsupportedMediaError("读不出这张图", reason=reason, bytes=len(buf), rc=int(rc))
    try:
        # 解码器交回的是 RGBA8（第 4 个字节没用），这里丢掉它 —— 每格式省一个转换分支的代价
        # 是 C 侧多占三分之一内存，正文那一行就是全部的解释。
        flat = np.ctypeslib.as_array(ptr, shape=(height.value * width.value * 4,))
        return np.ascontiguousarray(flat.reshape(height.value, width.value, 4)[:, :, :3])
    finally:
        lib.e8_img_free(ctypes.cast(ptr, ctypes.c_void_p))
