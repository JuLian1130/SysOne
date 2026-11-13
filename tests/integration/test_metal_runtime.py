"""Metal 运行时的冒烟测试：编译 kernel 源码、分配 buffer、派发 `k_copy`、核对数据。

只在 macOS 上跑（其它平台 `MetalRuntime` 直接 `NotReadyError`）。dylib 不存在就 skip——
它由 `tools/build_ane.sh` 生成，落在易失的 `SYS1_E8_DIR`。
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sysone.devices.apple.metal import MTL_DYLIB_NAME, MetalRuntime, _dylib_path
from sysone.errors import NotReadyError


def _runtime() -> MetalRuntime:
    if not _dylib_path(None).is_file():
        pytest.skip(f"缺少 {MTL_DYLIB_NAME}（先跑 tools/build_ane.sh）")
    try:
        return MetalRuntime()
    except NotReadyError as exc:  # 非 darwin 或缺源码
        pytest.skip(str(exc))


def _copy_args(*, nchan, src_c0, dst_c0, stile, s0, src_stride, dst_stride, src_s0) -> bytes:
    """`struct CopyArgs`（e8_common.h:97）——8 个 uint32，共 32 字节。"""
    return struct.pack(
        "<IIIIIIII", nchan, src_c0, dst_c0, stile, s0, src_stride, dst_stride, src_s0
    )


def test_runtime_compiles_the_kernel_sources():
    runtime = _runtime()
    assert runtime.lib is not None
    assert runtime.error() == "" or "warn" in runtime.error().lower()
    runtime.close()


def test_copy_moves_one_channel_band():
    runtime = _runtime()
    src = runtime.alloc(4 * 8 * 2)  # [4 通道, 8 列] fp16
    dst = runtime.alloc(8 * 8 * 2)  # 目标 8 通道，只写前 4 条
    values = np.arange(32, dtype=np.float16).reshape(4, 8)
    src.array(np.float16, (4, 8))[:] = values
    dst.array(np.float16, (8, 8))[:] = np.float16(-1.0)

    runtime.dispatch(
        "k_copy",
        [src, dst],
        args=_copy_args(
            nchan=4, src_c0=0, dst_c0=2, stile=8, s0=0, src_stride=8, dst_stride=8, src_s0=0
        ),
        grid=((8 + 63) // 64, (4 + 7) // 8, 1),
        tg=(64, 8, 1),
    )
    runtime.commit(0)
    runtime.wait(0)

    got = dst.array(np.float16, (8, 8))
    np.testing.assert_array_equal(got[2:6], values)
    np.testing.assert_array_equal(got[0:2], np.float16(-1.0))
    np.testing.assert_array_equal(got[6:], np.float16(-1.0))
    src.free()
    dst.free()
    runtime.close()


def test_copy_stops_at_the_column_limit():
    """`Stile` 是硬边界：越过它的列必须保持原值（grid 会多派 64 的整倍数）。"""
    runtime = _runtime()
    src = runtime.alloc(2 * 70 * 2)
    dst = runtime.alloc(2 * 70 * 2)
    src.array(np.float16, (2, 70))[:] = np.float16(7.0)
    dst.array(np.float16, (2, 70))[:] = np.float16(-1.0)

    runtime.dispatch(
        "k_copy",
        [src, dst],
        args=_copy_args(
            nchan=2, src_c0=0, dst_c0=0, stile=5, s0=0, src_stride=70, dst_stride=70, src_s0=0
        ),
        grid=((5 + 63) // 64, 1, 1),
        tg=(64, 8, 1),
    )
    runtime.commit(0)
    runtime.wait(0)

    got = dst.array(np.float16, (2, 70))
    np.testing.assert_array_equal(got[:, :5], np.float16(7.0))
    np.testing.assert_array_equal(got[:, 5:], np.float16(-1.0))
    src.free()
    dst.free()
    runtime.close()


def test_wraps_an_ane_surface_without_copying():
    """`wrap()` 出来的 buffer 与 ANE surface 是同一块内存：写进去能被 surface 读到。"""
    from sysone.devices.apple.runtime import AneRuntime

    try:
        ane = AneRuntime()
    except NotReadyError as exc:
        pytest.skip(str(exc))
    surface = ane.surface(4096)
    runtime = _runtime()
    buffer = runtime.wrap(surface)
    buffer.array(np.float16, (2048,))[:] = np.float16(3.5)
    np.testing.assert_array_equal(surface.array(np.float16, (2048,)), np.float16(3.5))
    runtime.unwind(surface)
    buffer.free()
    surface.free()
    ane.close()
    runtime.close()


def test_runtime_is_shared_per_process():
    """同一个进程只能初始化一次：第二次 `e8m_init_src` 会重建命令队列，之后的 dispatch 静默失效。

    这条是在 `test_metal_chain.py` 的两个用例里暴露的（第一个过、第二个输出全零），所以把它
    钉在这里，免得以后又把 `close()` 写成"把引用放开、下次重建"。
    """
    first = _runtime()
    second = MetalRuntime()
    assert first is second
    assert second.lib is not None
    second.close()
    assert MetalRuntime() is first, "close() 之后仍应复用同一份运行时"


def test_lanes_are_independent_command_buffers():
    """两条 lane 各有命令缓冲：提交并等待 lane 0 不会把 lane 1 上还没提交的活带出来。

    `e8_mtl.m:51-72` 的约定是 lane 0 放长 stage（attention / prep）、lane 1 放小核
    （norm / copy / gdn）。`sync_all()` 才是"两条都排空"。
    """
    runtime = _runtime()
    src = runtime.alloc(2 * 8 * 2)
    dst0 = runtime.alloc(2 * 8 * 2)
    dst1 = runtime.alloc(2 * 8 * 2)
    src.array(np.float16, (2, 8))[:] = np.float16(5.0)
    dst0.array(np.float16, (2, 8))[:] = np.float16(-1.0)
    dst1.array(np.float16, (2, 8))[:] = np.float16(-1.0)
    args = _copy_args(
        nchan=2, src_c0=0, dst_c0=0, stile=8, s0=0, src_stride=8, dst_stride=8, src_s0=0
    )
    grid, tg = (1, 1, 1), (64, 8, 1)

    # lane 1：只编码、不提交。
    runtime.dispatch("k_copy", [src, dst1], args=args, grid=grid, tg=tg, lane=1)
    # lane 0：编码、提交、等待 —— 不该顺手把 lane 1 的活也执行掉。
    runtime.dispatch("k_copy", [src, dst0], args=args, grid=grid, tg=tg, lane=0)
    runtime.commit(0)
    runtime.wait(0)
    np.testing.assert_array_equal(dst0.array(np.float16, (2, 8)), np.float16(5.0))
    np.testing.assert_array_equal(dst1.array(np.float16, (2, 8)), np.float16(-1.0))

    # 提交 lane 1 之后它才跑，`sync_all()` 把两条一起排空。
    runtime.commit(1)
    runtime.wait(1)
    np.testing.assert_array_equal(dst1.array(np.float16, (2, 8)), np.float16(5.0))
    runtime.sync_all()
    assert runtime.busy_ms(0) >= 0.0 and runtime.busy_ms(1) >= 0.0
    src.free()
    dst0.free()
    dst1.free()
    runtime.close()
