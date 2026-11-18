"""像素管线：图片几何（`smart_resize`）与可分离双三次，**逐位等价 Pillow**。

**参照物是 `Qwen2VLImageProcessorPil`**，也就是 `preprocessor_config.json` 里
`image_processor_type` 那个 `Fast` 类的 PIL 后端（两者像素算术相同，只是后端不同；`Fast` 要
torchvision，这个 venv 没有）。出口 `patchify(prepare(rgb))` 实测与真处理器的 `pixel_values`
**逐位相同**（6 个尺寸，见 `.scratch/p1_gate.py`）。

四件事是**故意**这么写的，每一件漏了都有实测代价（旧引擎 §4.22 与这次的门）：

1. **可分离 + 中间 8 位量化**：两次一维卷积，中间结果按 `floor(x + 0.5)` + clip 落成 uint8。
   PIL 的中间结果就是 8 位图，所以双三次的振铃在两次之间被夹掉；换成浮点一路到底，在高对比
   细节处差到 **30 个 8 位步长**，而平均误差只有 0.3 —— 正是那种活得过"看均值"检查的 bug。
2. **趟序是动态的**（见 `_horizontal_first`）：高缩得比宽多两倍以上时先做竖直那一趟。
   因为第 1 条，**顺序一换结果就换**（实测 `7×101 → 9×64` 两种顺序差 13 个步长）。
3. **`floor(x + 0.5)` 不是 `np.round`**：后者是四舍六入五取偶，在恰好 .5 的平局上差一格。
4. **外圈 tap 乘 `a`**：PIL 的 `bicubic_filter` 在 `1 <= |t| < 2` 那一支是"多项式 × a"，不是
   裸多项式。漏了它，单调斜坡仍然对得上（0.4 步长），普通细节偏 **20+ 步长**。

还有两件是为了**逐位**而不是为了"更准"：系数照抄 PIL 的 int32 定点网格（`_PRECISION`），
累加留在 float64 里以便整数语义成立。代价是 1.8–2.2× 的墙钟（见 `resize_bicubic` 的注释）。

几何不做成两个函数（旧引擎有 `e8_mm.smart_resize` 与 `p_vision_image.smart_resize` 两份，
`min_pixels` 分支一份 `ceil` 一份 `floor`，只在非 32 倍数的边长上分叉，于是 4 张 fixture
全是倍数、一路绿灯，直到把两份互相比才发现 412 个尺寸里错 16 个）。这里只有一份。
"""

from __future__ import annotations

import math

import numpy as np

from ..descriptor import VisionSpec

# `preprocessor_config.json` 的 `size.shortest_edge`（两个出货 checkpoint 相同，256²）。
MIN_PIXELS = 65_536
# 引擎自定的上界 **1280²**：checkpoint 的 `size.longest_edge` 写的是 16 777 216（4096²），
# 那会让塔的激活在 1280² 之上继续涨（票据 07 §2.5 的内存账按 1280² 算）。所以这是**我们的**
# 上限、不是模型的，`smart_resize` 允许调用方覆盖它（票据 07 §7.6 的退路就是把 1024² 缩到 512²）。
MAX_PIXELS = 1_280 * 1_280
MEAN, STD = 0.5, 0.5          # mean == std 才能把 rescale 与 normalize 塌成一条仿射

# PIL 的系数定点位数：`src/libImaging/Resample.c` 的 `PRECISION_BITS = 32 - 8 - 2`
# （8 位结果 + 2 位给溢出/负面积）。系数因此是 int32 网格上的整数，落盘是
# `clip((2^21 + Σ k·p) >> 22)`。照抄之后才谈得上与 PIL 逐位相同。
_PRECISION = 22

_BAND: dict[tuple[int, int, float], tuple[np.ndarray, np.ndarray]] = {}
_BAND_MAX = 32                # 同时保留的几何数；一条带是几十 KB，不是稠密阵


def smart_resize(
    height: int, width: int, *, factor: int,
    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """`Qwen2VLImageProcessor.smart_resize`：两边都对齐到 `factor`，总像素落在预算内。

    `factor` 没有默认值：它是 `patch_size × merge_size`，调用方从描述符推出来（见 `prepare`）。
    旧引擎的两份拷贝就是这么分叉的 —— 各自写死 32，改一处忘一处，谁都不报错。

    `round` 不是 `floor`（HF 原文就是 `round`），`min_pixels` 那一支是 `ceil`
    （旧引擎写成 `int(x) // factor * factor`，就是那 16 个尺寸的来源）。边长比超过 200 时
    与 HF 一样抛 `ValueError`——那是**拒收**，不是兜底。
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"图片的边长比必须小于 200，收到 {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _bicubic_band(n_out: int, n_in: int, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """权重阵的**非零带**：`(cols, k)`，都是 `(n_out, <=4·ceil(scale)+4)`。

    `cols` 用 -1 补齐，所以散回稠密阵是精确的：真实一行里的列互不相同，没有格子被写两次。

    `k` **已经是 PIL 会用的系数**：归一化之后按 `normalize_coeffs_8bpc` 定点化到 int32 网格
    （精度 `2^-22`），所以它是整数值的 float64，不是实数权重。这一步不是"更精确"，是**照抄**：
    PIL 的双三次算在这个网格上，所以照抄才谈得上逐位相同（见 `_apply`）。

    全程 float64 —— PIL 的 `precompute_coeffs` 用 double 算权重。旧引擎用 float32 算过一版，
    残差 1–2 个 8 位步长（54 个几何里 46 个差 1、4 个差 2）。

    行体一行没改，只是把稠密阵那一层拿掉 —— 唯一被去掉的成本是每行的 Python 开销
    （最多 1600 轮、每轮十来次微小 numpy 调用，9–18 ms/轴）。
    """
    a = -0.5
    fs = max(scale, 1.0)
    support = 2.0 * fs
    inv = 1.0 / fs
    taps = np.arange(n_in, dtype=np.float64)
    centre = (np.arange(n_out) + 0.5) * float(scale)
    lo_all = np.maximum(np.floor(centre - support + 0.5), 0.0).astype(np.int64)
    hi_all = np.minimum(np.floor(centre + support + 0.5), float(n_in)).astype(np.int64)
    cols = np.full((n_out, int((hi_all - lo_all).max())), -1, np.int64)
    k = np.zeros(cols.shape, np.float64)
    for i in range(n_out):
        lo, hi = int(lo_all[i]), int(hi_all[i])
        # `float(...)` 不是装饰：`centre[i]` 是 np.float64，留着它会让整行以 numpy 标量参与
        # 广播（旧引擎在 NEP 50 上踩过：float32 的 taps 被提升成 float64，值偏 3 ULP）。
        c = float(centre[i])
        t = np.abs((taps[lo:hi] - c + 0.5) * inv)
        # PIL 的 `bicubic_filter`：外圈那一支是多项式**乘 a**。漏了它，单调斜坡仍对到 0.4 个
        # 步长，而普通细节偏 20+ 个步长（§4.22 的坑之一）。
        row = np.where(t < 1.0, ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0,
                       np.where(t < 2.0, (((t - 5.0) * t + 8.0) * t - 4.0) * a, 0.0))
        s = row.sum()
        row = row / s if s else row
        # PIL 的落盘：正数 `(int)(0.5 + v·2^22)`、负数 `(int)(-0.5 + v·2^22)` —— round half
        # away from zero，再截断。末尾的 `+ 0.0` 把 `-0.0` 归一成 `+0.0`：外圈的权重可以小到
        # `-1e-12`，定点化之后是"零"，但符号位会留在缓存里，让逐字节比较凭空出现差别。
        k[i, :hi - lo] = (np.trunc(row * (1 << _PRECISION)
                                   + np.where(row < 0.0, -0.5, 0.5)) + 0.0)
        cols[i, :hi - lo] = np.arange(lo, hi)
    return cols, k


def bicubic_weights(n_out: int, n_in: int, scale: float) -> np.ndarray:
    """PIL 的双三次（a = -0.5，缩小抗锯齿）写成 `(n_out, n_in)` 系数阵（整数值 float64）。

    `Image.resize(BICUBIC)` 在缩小时把滤波器按 `scale` 拉开、并对每个目的像素归一化，所以
    朴素 4-tap 立方核并不等于它。

    带是记忆化的：阵只有 6–8 个非零列、却有 896–2600 列，而每张图要建两次（两轴）——
    建它占 `prepare` 的 19–46%，而照片的几何会重复。缓存的是**带**不是稠密阵
    （1600×2600 一份 float64 就是 33 MB），所以稳态内存不随每次调用涨。
    """
    key = (n_out, n_in, scale)
    band = _BAND.get(key)
    if band is None:
        band = _BAND[key] = _bicubic_band(n_out, n_in, scale)
        if len(_BAND) > _BAND_MAX:
            del _BAND[next(iter(_BAND))]
    cols, k = band
    out = np.zeros((n_out, n_in), np.float64)
    rows, at = np.nonzero(cols >= 0)
    out[rows, cols[rows, at]] = k[rows, at]
    return out


def _apply(img: np.ndarray, k: np.ndarray, horizontal: bool) -> np.ndarray:
    """一趟：系数阵乘过去，再按 PIL 的定点落盘（`ss = 2^21 + Σ; out = (ss >> 22)` 夹到 0..255）。

    累加留在 float64 里是**为了实现整数语义**，不是为了"更准"：系数是 int32 网格上的整数、
    像素是整数，所以乘积与和都精确（`|Σ| < 2^40 « 2^53`），`>> 22` 于是等价于
    `floor((ss)·2^-22)`，与 C 里的算术右移一致。
    """
    out = (np.einsum("ow,hwc->hoc", k, img, optimize=True) if horizontal
           else np.einsum("oh,hwc->owc", k, img, optimize=True))
    return np.clip(np.floor((out + (1 << (_PRECISION - 1))) * (2.0 ** -_PRECISION)),
                   0.0, 255.0)


def _horizontal_first(h: int, w: int, out_h: int, out_w: int) -> bool:
    """哪一趟先做 —— **PIL 是动态决定的**，不是固定顺序。

    `ImagingResampleInner`（Pillow 13 的 `src/libImaging/Resample.c:866`）：

        horizontal_first = !((H - out_h) > 0 && (H - out_h) > (W - out_w) * 2)

    也就是"高被缩得比宽多两倍以上时先做竖直那一趟"，那是为了少算几行。数值上这一条**不是**
    无关紧要的：中间结果是 8 位图（两趟之间被量化），顺序一换结果就不同 —— 实测
    `61×59 → 32×32` 两种顺序差 4 个 8 位步长、`7×101 → 9×64` 差 13 个。旧引擎固定"先宽后高"，
    它那四张 fixture 恰好都落在"先宽"那一支，所以一直没暴露。
    """
    shrink_h, shrink_w = h - out_h, w - out_w
    return not (shrink_h > 0 and shrink_h > shrink_w * 2)


def resize_bicubic(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """`(H, W, 3)` 的 0–255 范围 → `(out_h, out_w, 3)`，**逐位等于** `Image.resize(BICUBIC)`。

    两次之间的量化不是优化细节，是算法本身：PIL 的中间结果就是 8 位图，双三次的振铃在那里
    被夹掉，永远到不了第二趟。顺序见 `_horizontal_first`。

    逐位相同的代价是 float64（权重与累加两处）：实测 1080p 的 `prepare` 从 46 ms 到 100 ms
    （2.1×）。这个交换是**有意的** —— 逐位相同之后，"我们的像素与参照不同"就只剩解码器一个
    来源（票据 07 §2.3 #3.2 要的正是这个分界），而 100 ms 正是旧引擎在照片上量的 87–115 ms
    那一档。想换回速度只改两处 dtype，残差会回到 1–2 个 8 位步长。
    """
    h, w = img.shape[:2]
    if (h, w) == (out_h, out_w):
        return img
    if _horizontal_first(h, w, out_h, out_w):
        if w != out_w:
            img = _apply(img, bicubic_weights(out_w, w, w / out_w), True)
        if h != out_h:
            img = _apply(img, bicubic_weights(out_h, h, h / out_h), False)
    else:
        if h != out_h:
            img = _apply(img, bicubic_weights(out_h, h, h / out_h), False)
        if w != out_w:
            img = _apply(img, bicubic_weights(out_w, w, w / out_w), True)
    return img


def prepare(rgb: np.ndarray, *, spec: VisionSpec) -> tuple[np.ndarray, np.ndarray]:
    """`(H, W, 3)` uint8 → `(img (H',W',3) fp32 归一图, grid (1, gh, gw))`。

    这是像素管线的出口，形状不是自由的：塔的 `VisionTower.encode` 吃的就是这张图（它自己
    patchify）。交出展平的 `(L, 1536)` 也能跑，但那样出货路径上就有第二份 `patchify`，而且
    丢了 `gh, gw` —— 塔要靠它查 pos_embed 与轴向 rope。

    三步，按处理器的顺序：`smart_resize` 的几何、一次双三次、一条仿射。mean = std = 0.5 让
    rescale 与 normalize 塌成 `p·(1/(255·std)) − mean/std`，所以是一趟不是三趟（票据 07 §2.3 #3.3）。
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    patch, merge = int(spec.patch_size), int(spec.spatial_merge)
    oh, ow = smart_resize(height, width, factor=patch * merge)
    # 直接进 float64：重采样要在整数网格上做（见 `_apply`），而 uint8 → float32 → float64
    # 是白多走一趟。
    img = resize_bicubic(np.asarray(rgb, dtype=np.float64), oh, ow)
    img = img * (1.0 / (255.0 * STD)) - (MEAN / STD)
    grid = np.asarray([[1, oh // patch, ow // patch]], np.int64)
    return np.ascontiguousarray(img, dtype=np.float32), grid
