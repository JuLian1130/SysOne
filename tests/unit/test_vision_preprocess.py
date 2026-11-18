"""像素管线：几何、双三次、仿射，以及解码（解码要 dylib，缺了就跳过）。

四条判据，各自钉不同的错法：

1. **`smart_resize` 与 HF 的公式全等**（412 个尺寸）。这一条钉的是一份**独立转写**的公式，
   不是"我自己算得跟我自己一样"：旧引擎有两份拷贝，`min_pixels` 那一支一份 `ceil` 一份
   `floor`，只在"缩放后边长不是 32 的倍数"时分叉 —— 而它那四张 fixture 全是倍数，所以一直绿。
2. **双三次与 PIL 逐位相同**（`tests/fixtures/vision_pixels.json`，金标准由真实 pillow 生成）。
   逐位、不是容差：三处语义（外圈 tap ×`a`、两次之间的 8 位量化、`floor(x+0.5)`）漏任何一处
   都是"平均值看着挺好、细节差 20–30 个步长"，容差检查抓不住。**顺序**也是这支 fixture 钉的：
   PIL 先做哪一趟是动态决定的（`61×59 → 32×32` 两种顺序差 4 个步长）。
3. **权重带 == 独立稠密参考**（几千个几何，`tobytes()` 相同）。这一条钉的是"为了快而只缓存
   非零带"没有改算术 —— 参考实现逐行建稠密阵，与带那一版不共用代码。
4. **`prepare` 的出口形状与仿射**：`(H,W,3)` fp32、`grid` 与 `smart_resize` 一致、
   `p·2/255 − 1`（mean == std 时 rescale+normalize 塌成这一条）。

全量几何（4 000+）在 `.scratch/p1_gate.py`；这里取确定性抽样，保持单元测试在秒级。
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sysone.errors import UnsupportedMediaError
from sysone.models.descriptor import VisionSpec
from sysone.models.vision.preprocess import (
    MEAN,
    STD,
    _horizontal_first,
    bicubic_weights,
    prepare,
    resize_bicubic,
    smart_resize,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "vision_pixels.json"
PATCH, MERGE = 16, 2
FACTOR = PATCH * MERGE
MIN_PIXELS, MAX_PIXELS = 65_536, 1_280 * 1_280


def hf_smart_resize(height: int, width: int) -> tuple[int, int]:
    """`Qwen2VLImageProcessor.smart_resize` 的独立转写（对着 transformers 的源码抄的）。

    刻意不复用产品代码：这条判据的意义就是"两份独立实现给出同一个数"。
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def sizes() -> list[tuple[int, int]]:
    """412 个尺寸：400 个随机 + 12 个刁钻的（与旧引擎门 A 的集合同构）。"""
    rng = np.random.default_rng(11)
    out = [(int(rng.integers(16, 4000)), int(rng.integers(16, 4000))) for _ in range(400)]
    out += [(512, 512), (768, 768), (1024, 1024), (1280, 1280), (2560, 2560),
            (100, 100), (101, 203), (77, 301), (133, 189), (59, 61), (33, 33), (31, 999)]
    return out


# ---------------------------------------------------------------- 1. 几何


def test_smart_resize_matches_hf_over_412_sizes() -> None:
    sweep = sizes()
    assert len(sweep) == 412
    wrong = []
    for h, w in sweep:
        try:
            want = hf_smart_resize(h, w)
        except ValueError:
            with pytest.raises(ValueError):
                smart_resize(h, w, factor=FACTOR)
            continue
        got = smart_resize(h, w, factor=FACTOR)
        if got != want:
            wrong.append(((h, w), got, want))
    assert not wrong, f"与 HF 的公式不一致：{wrong[:5]}"


def test_smart_resize_budget_and_alignment() -> None:
    """成品边长都是 32 的倍数，且总像素落在预算内（除非边长比把它压到 factor 那一档）。"""
    for h, w in sizes():
        oh, ow = smart_resize(h, w, factor=FACTOR)
        assert oh % FACTOR == 0 and ow % FACTOR == 0, (h, w, oh, ow)
        if max(h, w) / min(h, w) <= 200:
            # 上限：缩完不超过预算；下限：只在没超过 factor 时保证
            assert oh * ow <= MAX_PIXELS or min(oh, ow) <= FACTOR, (h, w, oh, ow)


def test_smart_resize_rejects_a_ratio_of_200() -> None:
    """HF 在边长比 > 200 时抛，我们照抄 —— 那是**拒收**，不是兜底到某个默认尺寸。"""
    assert smart_resize(10, 2000, factor=FACTOR)          # 恰好 200：收
    with pytest.raises(ValueError):
        smart_resize(10, 2001, factor=FACTOR)


# ---------------------------------------------------------------- 2. 双三次（金标准）


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_resize_matches_pil_bit_for_bit() -> None:
    doc = _fixture()
    assert len(doc["resize"]) >= 5
    for case in doc["resize"]:
        h, w = case["in"]
        oh, ow = case["out"]
        src = np.random.default_rng(case["seed"]).integers(0, 256, (h, w, 3), dtype=np.uint8)
        want = np.frombuffer(base64.b64decode(case["want"]), np.uint8).reshape(oh, ow, 3)
        got = resize_bicubic(src.astype(np.float64), oh, ow)
        assert np.array_equal(got, want.astype(np.float64)), (
            f"{h}x{w} -> {oh}x{ow} 与 PIL 不逐位相同：max|Δ| "
            f"{np.abs(got - want).max()}")


def test_resize_pass_order_follows_pil() -> None:
    """先做哪一趟是 PIL 动态决定的；这条规则本身要钉住，因为它只在缩小方向不对称时显形。"""
    assert _horizontal_first(61, 59, 32, 32)          # 高缩 29、宽缩 27 → 先宽
    assert _horizontal_first(7, 101, 9, 64)           # 高在放大（缩 0 行）→ 先宽
    assert _horizontal_first(1080, 1920, 960, 1696)   # 两轴都缩得差不多 → 先宽
    assert not _horizontal_first(1080, 640, 384, 640)  # 高缩 696 > 宽缩 0 → 先高
    # 只缩一轴时顺序无所谓，但函数必须仍然给出一个稳定的答案
    assert _horizontal_first(100, 61, 100, 32)
    assert not _horizontal_first(100, 61, 64, 61)


# ---------------------------------------------------------------- 3. 权重带 == 稠密参考


def dense_reference(n_out: int, n_in: int, scale: float) -> np.ndarray:
    """独立稠密参考：逐行、逐 tap，与 `_bicubic_band` 不共用任何代码路径。

    最后一步是 PIL 的定点化（`normalize_coeffs_8bpc`）。它必须在这里出现，因为**产品就是
    带定点化的**：金标准那条用例逐位钉着"定点化 + 累加 + 落盘"整条链，这一条只钉"带 == 稠密"。
    """
    a = -0.5
    fs = max(scale, 1.0)
    support = 2.0 * fs
    out = np.zeros((n_out, n_in), np.float64)
    for i in range(n_out):
        c = (i + 0.5) * scale
        lo = max(int(math.floor(c - support + 0.5)), 0)
        hi = min(int(math.floor(c + support + 0.5)), n_in)
        row = np.zeros(n_in, np.float64)
        for x in range(lo, hi):
            t = abs((x - c + 0.5) / fs)
            if t < 1.0:
                row[x] = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0
            elif t < 2.0:
                row[x] = (((t - 5.0) * t + 8.0) * t - 4.0) * a
        # 只对支撑内的 tap 求和，与 PIL 的 `for (x = 0; x < xmax; x++) ww += w;` 一致 ——
        # 把整行的零也加进来会改变求和顺序，末位就不同（`-0.0` 与 `+0.0` 就是这样分出来的）。
        total = row[lo:hi].sum()
        if total:
            row = row / total
        # 末尾的 `+ 0.0` 与产品侧一致：把 `-0.0` 归一成 `+0.0`（算数值一样，逐字节不一样）
        out[i] = (np.trunc(row * (1 << 22) + np.where(row < 0.0, -0.5, 0.5)) + 0.0)
    return out


def _geometries(limit: int) -> list[tuple[int, int, float]]:
    """`smart_resize` 能产生的互不相同的 `(n_out, n_in)` 几何（确定性，与门同一套枚举）。

    稠密参考一份最大 41 MB、平均几 MB，所以单元测试取一个**有上限的抽样**（小的全取、大的
    按尺寸均匀抽），全量 3 000+ 在 `.scratch/p1_gate.py`（那里的抽样与这里同一套枚举）。抽样覆盖边角（taps 被裁到边界、
    放大、极端缩放），这才是这条判据的价值所在，不是重复大阵的开销。
    """
    geos: set[tuple[int, int]] = set()
    for h, w in sizes():
        try:
            oh, ow = smart_resize(h, w, factor=FACTOR)
        except ValueError:
            continue
        geos.add((oh, h))
        geos.add((ow, w))
    small = sorted(g for g in geos if g[1] <= 512)
    big = sorted(g for g in geos if g[1] > 512)
    room = max(limit - len(small), 0)
    picked = small + (big[::max(1, len(big) // room)][:room] if room else [])
    picked.sort()
    return [(n_out, n_in, n_in / n_out) for n_out, n_in in picked[:limit]]


def test_band_matches_dense_reference_bit_for_bit() -> None:
    geos = _geometries(limit=120)
    for n_out, n_in, scale in geos:
        band = bicubic_weights(n_out, n_in, scale)
        dense = dense_reference(n_out, n_in, scale)
        assert band.shape == (n_out, n_in)
        assert band.tobytes() == dense.tobytes(), f"几何 {n_out}×{n_in}（scale {scale}）不同"


def test_band_cache_returns_the_same_matrix() -> None:
    """带是记忆化的：命中与未命中必须逐位相同（旧引擎的这一条抓过一次散回稠密阵的错）。"""
    first = bicubic_weights(384, 640, 640 / 384)
    second = bicubic_weights(384, 640, 640 / 384)
    assert first is second or first.tobytes() == second.tobytes()
    # 归一化在定点上留下 ±1 的残差，但不可能更多 —— 否则就是散回稠密阵时丢/重了格子。
    # （不能用"每行至少 4 个非零项"：边界行的支撑被裁掉，定点化之后只剩 2–3 个。）
    rows = first.sum(axis=1)
    assert np.abs(rows - float(1 << 22)).max() <= 2.0, np.abs(rows - (1 << 22)).max()


# ---------------------------------------------------------------- 4. prepare


def _spec() -> VisionSpec:
    return VisionSpec(model_type="qwen3_5_vision", depth=1, hidden=64, patch_size=PATCH,
                      spatial_merge=MERGE, intermediate_size=128, num_heads=2,
                      out_hidden_size=64, temporal_patch_size=2,
                      num_position_embeddings=4)


def test_prepare_hands_over_the_resized_normalised_image() -> None:
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (300, 200, 3), dtype=np.uint8)
    img, grid = prepare(rgb, spec=_spec())
    oh, ow = smart_resize(300, 200, factor=FACTOR)
    assert img.shape == (oh, ow, 3) and img.dtype == np.float32
    assert grid.tolist() == [[1, oh // PATCH, ow // PATCH]]
    assert img.min() >= -1.0 and img.max() <= 1.0

    # 仿射：mean == std 时 rescale + normalize 塌成 p·2/255 − 1（一步，不是三步）
    raw = resize_bicubic(rgb.astype(np.float64), oh, ow)
    assert np.array_equal(img, (raw * (1.0 / (255.0 * STD)) - (MEAN / STD)).astype(np.float32))


def test_prepare_is_the_only_place_that_chooses_the_geometry() -> None:
    """同一张图两次 `prepare` 逐位相同，且换一张不同尺寸的图会走不同的几何。"""
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    first, grid_a = prepare(a, spec=_spec())
    second, grid_b = prepare(a, spec=_spec())
    assert np.array_equal(first, second) and np.array_equal(grid_a, grid_b)
    _, grid_c = prepare(rng.integers(0, 256, (64, 192, 3), dtype=np.uint8), spec=_spec())
    assert grid_c[0, 2] > grid_a[0, 2]      # 更宽 → 更多列


# ---------------------------------------------------------------- 解码（要 dylib）


def _decode_fixture(name: str):
    from sysone.devices.apple.image import available

    if not available():
        pytest.skip("没有 libe8image.dylib（先跑 tools/build_ane.sh）")
    doc = _fixture()["decode"]
    if name not in doc:
        pytest.skip(f"fixture 里没有 {name} 的金标准")
    case = doc[name]
    want = np.frombuffer(base64.b64decode(case["want"]), np.uint8).reshape(case["shape"])
    return base64.b64decode(case["bytes"]), want, case["max_lsb"]


def test_decode_png_is_bit_identical_to_pillow() -> None:
    """无损格式应当逐位相同：两个解码器对 PNG 没有自由度。（旧引擎：2 430 000 像素逐位相同。）"""
    from sysone.devices.apple.image import decode

    data, want, _ = _decode_fixture("png")
    assert np.array_equal(decode(data), want)


def test_decode_jpeg_differs_from_pillow_by_a_bounded_amount() -> None:
    """有损格式**不该**逐位相同：反 DCT 的舍入是解码器自己的。差距要有上限，不能是"任意"。"""
    from sysone.devices.apple.image import decode

    data, want, max_lsb = _decode_fixture("jpeg")
    got = decode(data)
    diff = int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max())
    assert diff <= max_lsb, f"与 Pillow 的解码差 {diff} 个 8 位步长，上限是 {max_lsb}"
    assert diff > 0, "JPEG 竟然逐位相同了？先确认 fixture 与 dylib 是同一份输入"


def test_decode_rejects_garbage(tmp_path: Path) -> None:
    from sysone.devices.apple.image import available, decode

    if not available():
        pytest.skip("没有 libe8image.dylib")
    with pytest.raises(UnsupportedMediaError):
        decode(b"\x89PNG\r\n\x1a\n not really a png")
    with pytest.raises(UnsupportedMediaError):
        decode(b"")
    with pytest.raises(UnsupportedMediaError):
        decode("不是字节")  # type: ignore[arg-type]
