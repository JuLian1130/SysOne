"""图片输入的四条预算，与从它们派生出来的能力契约。

**为什么四条不是两条**（上游把这件事搞错过一次，§4.24）：`min_pixels` / `max_pixels` 是
**缩放**预算 —— 超了会被缩，不会被拒；真正拒收的是**源**那一对：8 MB 的编码字节与 16 MP 的
源像素。把 `max_pixels` 当拒绝线写过一次，验收就跑去测了一条不存在的规则。

四个数都有同一个去处：`smart_resize` 读前两条（`preprocess.prepare` 直接调），
`VisionSlot.media` 在**解码之前**读后两条（`image.size` 只读容器头，不解像素），
契约把它们原样发布给客户端（`/v1/models` 的 `image_max_pixels`、`doctor` 的 `vision`）。

放开上界只要有证据：`max_pixels` 是**我们的**引擎上限、不是 checkpoint 的（它写的是 16 777 216，
4096²，那会让塔的激活在 1280² 之上继续涨），理由与数字都在 `preprocess.MAX_PIXELS`。
"""

from __future__ import annotations

from typing import Any

from .preprocess import MAX_PIXELS, MIN_PIXELS

# 一张图的编码字节上限（data URL 里那段 base64 解出来的字节数）。
MAX_BYTES = 8_000_000
# 一张图的**源**像素上限，在解码之前按容器头判（30000² 的 PNG 压缩后几百 KB、解开 3.6 GB）。
MAX_SOURCE_PIXELS = 16_000_000

# 两个模型都是这套塔、这套像素管线，所以这些话对两边都成立。
PLACEMENT = "before the text of the user turn, one placeholder per image, request order"
PATCH_EMBEDDING = "matrix product (equal to the Conv3d with kernel = stride)"


def image_contract(*, unavailable: str | None = None) -> dict[str, Any]:
    """`/v1/models` 与 `doctor` 发布的那个字典；`unavailable` 是"为什么收不了图"。

    键对着上游 `D3.image_contract()`：`supported` / `min_pixels` / `max_pixels` / `placement` /
    `patch_embedding`，加上我们这边的两个预算与两处**真实差异**（不是装饰）：

    * `max_bytes` / `max_source_pixels` —— 拒收线，上游把它们放在 `d3_runtime` 的常量里、
      没有发布；发布出来客户端才能自己先拦。
    * `decoder` —— 我们走 ImageIO 而不是 Pillow。无损格式逐位相同，JPEG 的差在色度上采样
      （4:2:0 最多 82 个 8 位步长），这是**唯一**一处像素与参照不同，写进契约比藏在注释里好。
    * `image_processor` —— 等价物是 `preprocess.prepare`（与 `Qwen2VLImageProcessorPil`
      逐位相同，见 `.scratch/p1_gate.json`），不是某个 HF 类名。
    """
    contract: dict[str, Any] = {
        "supported": unavailable is None,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_bytes": MAX_BYTES,
        "max_source_pixels": MAX_SOURCE_PIXELS,
        "placement": PLACEMENT,
        "patch_embedding": PATCH_EMBEDDING,
        "image_processor": "models.vision.preprocess.prepare (bit-exact to Qwen2VLImageProcessorPil)",
        "decoder": "ImageIO (libe8image): 8-bit sRGB, no EXIF rotation, alpha dropped",
    }
    if unavailable is not None:
        contract["unavailable"] = unavailable
    return contract


__all__ = [
    "MAX_BYTES",
    "MAX_SOURCE_PIXELS",
    "MIN_PIXELS",
    "MAX_PIXELS",
    "PLACEMENT",
    "PATCH_EMBEDDING",
    "image_contract",
]
