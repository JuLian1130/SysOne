"""视觉塔：装得起来、跑得出形状、重复调用逐位一致（缺 dylib 或真权重就跳过）。

数值本身不在这里钉：与厂商 `Qwen3_5VisionModel` 的对拍要 torch + 一个 fp32 参考模型，那是
门（`.scratch/p2c_tower_gate.py`）的事。这里钉的是**不依赖参考实现**就能判错的三件事：

* 26 个 program 装得起来，`encode` 的出口是 `(L/4, out_hidden)` fp16；
* 同一张图两次调用逐位相同（塔内部有面子复用、host 残差、GPU 缓冲，全是可变状态）；
* 同一个产物目录再建一座塔时 26 份产物全部命中（`compile 0` 的判据）。

一张 64×64 的图（`L=16`、桶宽 256）不触发分趟，所以这条用例快；分趟与单趟的逐位相同在门里
（那里要同时开两个不同 `s` 的塔，一个就要 0.6 GB 产物目录）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.devices.apple.metal import MetalRuntime
from sysone.devices.apple.runtime import AneRuntime
from sysone.devices.apple.vision import VisionTower
from sysone.models.families import mude
from sysone.weights.checkpoint import open_checkpoint

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
WEIGHTS = MUDE_DIR / "model.safetensors"


def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "ANE/Metal 只在 macOS 上可用"
    if not WEIGHTS.is_file():
        return f"缺少真实权重：{WEIGHTS}"
    from sysone.devices.apple import available

    if not available():
        return "ANE dylib 不可用（先跑 tools/build_ane.sh）"
    return None


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
def test_tower_shape_repeat_and_reuse(tmp_path: Path) -> None:
    """一张 64×64 的图：出口形状对、两次逐位相同、再建一座塔 26 份产物全命中。"""
    store = open_checkpoint(MUDE_DIR)
    runtime, metal = AneRuntime(), MetalRuntime()
    descriptor = mude.parse_config(MUDE_DIR)
    tower = VisionTower(store, descriptor=descriptor, s=256, runtime=runtime, metal=metal,
                        artifacts_dir=tmp_path)
    try:
        assert tower.reuse == (0, 26), "冷产物目录却连一份都没现编"
        image = np.random.default_rng(11).uniform(-1.0, 1.0, (64, 64, 3)).astype(np.float32)
        first = tower.encode(image)
        # 64×64 的图 = 4×4 个 patch = 16 个 token，2×2 合并后是 4 个
        assert first.shape == (4, 2048)
        assert first.dtype == np.float16
        second = tower.encode(image)
        assert np.array_equal(first, second), "同一张图两次调用的结果不是逐位相同"
        # 一次前向的 eval 次数是固定的：patch 1 + 24×2 + merger 2 = 51（分趟时按趟乘）。
        # 24 层每层两次，不是三次：`proj → 残差 → norm2 → FFN → 残差` 在同一段里（票据 07 §7.10）。
        assert tower.stats["evals"] == 102, tower.stats

        # 跨瓦片：288² 是 324 个 token，在 s=256 下分成 256 + 68 两片，而**每个瓦片有自己的
        # 面与槽位**（`attach_slot`，票据 07 §7.20）。片数一变，绑错面不会报错、只会让数值
        # 安静地偏掉，所以这条钉的是"两片都走了自己那条路"：出口行数是 324/4 = 81。
        wide = np.random.default_rng(12).uniform(-1.0, 1.0, (288, 288, 3)).astype(np.float32)
        one = tower.encode(wide)
        assert one.shape == (81, 2048), one.shape
        assert np.array_equal(one, tower.encode(wide)), "跨瓦片的两次调用不是逐位相同"
    finally:
        tower.close()

    again = VisionTower(store, descriptor=descriptor, s=256, runtime=runtime, metal=metal,
                        artifacts_dir=tmp_path)
    try:
        assert again.reuse == (26, 26), "产物目录里已经有 26 份产物，却还是要重编"
        same = again.encode(np.random.default_rng(11).uniform(
            -1.0, 1.0, (64, 64, 3)).astype(np.float32))
        assert np.array_equal(same, first), "重新装载后的结果与第一次逐位不同"
    finally:
        again.close()
        runtime.close()
        store.close()
