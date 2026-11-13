"""ANE 运行时的装载与 surface 零拷贝（只在 macOS 且有 dylib 时跑）。"""

import sys

import numpy as np
import pytest

from sysone.devices.apple import AneRuntime, available
from sysone.errors import NotReadyError

needs_apple = pytest.mark.skipif(not available(), reason="没有可用的 ANE dylib（先跑 tools/build_ane.sh）")


@needs_apple
def test_runtime_loads_and_reports_capacity():
    runtime = AneRuntime()
    try:
        assert runtime.programs == 0
        assert runtime.avail_gb() > 0
        assert runtime.error() == ""
    finally:
        runtime.close()


@needs_apple
def test_surface_is_a_zero_copy_view():
    runtime = AneRuntime()
    try:
        surface = runtime.surface(4096)
        assert surface.nbytes >= 4096
        view = surface.array(np.uint8)
        view[:4096] = 7
        assert view[0] == 7 and view[4095] == 7
        surface.free()
    finally:
        runtime.close()


def test_missing_dylib_says_how_to_rebuild(tmp_path):
    with pytest.raises(NotReadyError) as excinfo:
        AneRuntime(tmp_path / "libe8ane.dylib")
    assert excinfo.value.context["hint"] == "运行 tools/build_ane.sh 重建"


@pytest.mark.skipif(sys.platform == "darwin", reason="只在非 macOS 上有意义")
def test_non_darwin_is_not_ready():
    with pytest.raises(NotReadyError) as excinfo:
        AneRuntime()
    assert excinfo.value.context["platform"] != "darwin"
