"""`VisionSlot`：懒加载、空闲卸载、以及"图片字节 → `Media`"那一段接线。

三条都不是风格问题，每一条都有实测依据（票据 07 §2.5）：

* **懒加载**：注册了但没常驻 = 0 内存。启动时装塔等于选了常驻，0 常驻的契约当场作废。
* **真卸载**（不是挂起）：挂起只还 20–56 MB，627.6 MB 里剩下的拿不回来。
* **`media()` 在解码之前拒 16 MP**：真解码再判等于先把这张图不该占的内存申请下来
  （30000² 的 PNG 压缩后几百 KB、解开 3.6 GB）。

塔与 ANE 都是假的（这个文件不该需要设备）：`_FakeTower` 只回答 `stats`/`encode`/`close`。
真塔的那条在 `.scratch/p4_gate.py`（0 常驻、TTL 回基线、真图进模型）。
"""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

from collections.abc import Sequence

import numpy as np
import pytest

from sysone.devices.apple.image import available
from sysone.devices.apple.vision import VisionSlot
from sysone.domain.decision import MediaRef
from sysone.errors import UnsupportedMediaError
from sysone.models.descriptor import VisionSpec
from sysone.models.vision import preprocess

PATCH, MERGE = 16, 2


def _spec() -> VisionSpec:
    return VisionSpec(model_type="qwen3_5_vision", depth=1, hidden=64, patch_size=PATCH,
                      spatial_merge=MERGE, intermediate_size=128, num_heads=2,
                      out_hidden_size=64, temporal_patch_size=2,
                      num_position_embeddings=4)


class _FakeTower:
    """塔的形状：只记调用次数与开关，`encode` 交出一张按行数确定的表。"""

    def __init__(self) -> None:
        self.stats = {"evals": 0, "ane_ms": 0.0, "gpu_ms": 0.0, "host_ms": 0.0}
        self.closed = 0
        self.images = 0
        self.bytes = 1024
        self.reuse = (26, 26)

    def encode(self, img: np.ndarray) -> np.ndarray:
        return self.encode_many([img])[0]

    def encode_many(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        # 打包之后一批图是**一次**调用：eval 数按批算，但每张图的行数各按自己的网格。
        assert images
        out = []
        for img in images:
            assert img.ndim == 3 and img.shape[2] == 3
            self.images += 1
            rows = (img.shape[0] // PATCH) * (img.shape[1] // PATCH) // (MERGE * MERGE)
            out.append(np.full((rows, 64), self.images, np.float16))
        self.stats["evals"] += 3
        self.stats["ane_ms"] += 0.5
        self.stats["gpu_ms"] += 0.25
        return out

    def close(self) -> None:
        self.closed += 1


def _slot(*, idle: float = 60.0):
    built: list[_FakeTower] = []

    def build() -> _FakeTower:
        tower = _FakeTower()
        built.append(tower)
        return tower

    return VisionSlot(build, spec=_spec(), idle=idle), built


def _png() -> tuple[bytes, tuple[int, int]]:
    """一张真的 PNG 与它的 `(高, 宽)`。

    只有这一张（`tests/fixtures/vision_pixels.json` 里与 Pillow 逐位对过的那张）：7 列 5 行。
    所以断言里的几何从**解出来的形状**算，不从文件名猜 —— 宽高写反过一次，正好被这里抓住。
    """
    data = base64.b64decode(_fixture()["png"]["bytes"])
    return data, (5, 7)


def _fixture() -> dict:
    import json

    path = Path(__file__).resolve().parents[1] / "fixtures" / "vision_pixels.json"
    return json.loads(path.read_text(encoding="utf-8"))["decode"]


def test_the_tower_is_not_built_until_the_first_image() -> None:
    slot, built = _slot()
    assert built == [] and slot.stats()["loaded"] is False
    assert slot.stats()["bytes"] == 0

    data, (height, width) = _png()
    media = slot.media([MediaRef(kind="image", data=data)])
    assert len(built) == 1, "第一次带图请求才建塔（不预热）"
    assert slot.stats()["loaded"] is True and slot.stats()["loads"] == 1
    # 小图被 smart_resize 抬到 min_pixels 以上、对齐到 32：网格按**合并之后**记
    oh, ow = preprocess.smart_resize(height, width, factor=PATCH * MERGE)
    assert media.grids == ((oh // PATCH // MERGE, ow // PATCH // MERGE),)
    assert media.counts == (media.grids[0][0] * media.grids[0][1],)
    assert media.rows.shape == (media.counts[0], 64)
    assert media.rows.dtype == np.float16


@pytest.mark.skipif(not available(), reason="没有 libe8image.dylib（先跑 tools/build_ane.sh）")
def test_media_refuses_a_bad_image_with_the_index_of_the_one_that_is_bad() -> None:
    slot, _ = _slot()
    with pytest.raises(UnsupportedMediaError) as caught:
        slot.media([MediaRef(kind="image", data=_png()[0]),
                    MediaRef(kind="image", data=b"\x89PNG\r\n\x1a\n not really")])
    assert "images[1]:" in caught.value.message, caught.value.message
    assert not caught.value.message.startswith("images[0]:")


@pytest.mark.skipif(not available(), reason="没有 libe8image.dylib（先跑 tools/build_ane.sh）")
def test_media_refuses_an_image_with_no_bytes() -> None:
    slot, _ = _slot()
    with pytest.raises(UnsupportedMediaError, match=r"images\[0\]"):
        slot.media([MediaRef(kind="image")])


def test_idle_zero_unloads_at_the_end_of_every_request() -> None:
    slot, built = _slot(idle=0)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    slot.finish()
    assert slot.stats()["loaded"] is False, "idle = 0 的契约是每请求后即卸"
    assert built[0].closed == 1
    # 下一次请求重新装一座（不是复用挂起的那个）
    slot.media([MediaRef(kind="image", data=_png()[0])])
    assert len(built) == 2 and slot.stats()["unloads"] == 1


def test_the_idle_timer_unloads_and_a_new_request_reloads() -> None:
    slot, built = _slot(idle=0.05)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    assert slot.stats()["loaded"] is True
    deadline = time.monotonic() + 5.0
    while slot.stats()["loaded"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert slot.stats()["loaded"] is False, "空闲窗口过了却没卸"
    assert slot.stats()["unloads"] == 1
    slot.media([MediaRef(kind="image", data=_png()[0])])
    assert len(built) == 2


def test_the_timer_restarts_on_use_so_a_busy_tower_is_not_unloaded() -> None:
    """连着来两张图，第一张排的那次卸载不该把第二张正在用的塔拆掉。"""
    slot, built = _slot(idle=0.15)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    time.sleep(0.10)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    time.sleep(0.10)                     # 距第一张 0.20 s（> idle），距第二张 0.10 s
    assert slot.stats()["loaded"] is True, "计时器没有按最近一次使用重排"
    assert built[0].closed == 0


def test_stats_accumulate_across_a_reload() -> None:
    """累计量在卸载之后还在：`/metrics` 不该因为一次卸载就把历史抹掉。"""
    slot, _ = _slot(idle=0)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    slot.finish()
    slot.media([MediaRef(kind="image", data=_png()[0])])
    stats = slot.stats()
    assert stats["evals"] == 6 and stats["ane_ms"] == pytest.approx(1.0)
    assert stats["loads"] == 2 and stats["unloads"] == 1


def test_close_is_idempotent_and_cancels_the_timer() -> None:
    slot, built = _slot(idle=5.0)
    slot.media([MediaRef(kind="image", data=_png()[0])])
    slot.close()
    slot.close()
    assert built[0].closed == 1
    assert slot.stats()["loaded"] is False
    assert threading.active_count() < 100      # 计时器线程没有泄漏堆积


def test_a_negative_idle_window_is_a_configuration_error() -> None:
    with pytest.raises(ValueError, match="不能为负"):
        VisionSlot(lambda: _FakeTower(), spec=_spec(), idle=-1.0)
