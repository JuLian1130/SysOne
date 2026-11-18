"""视觉塔的权重：两个出货 checkpoint 只差一个根前缀，塔的形状与数值约定逐项相同。

mude 的单文件里塔叫 `visual.*`；h2o 的 row-int8 目录里叫 `model.visual.*`（那些塔张量在
`int8_manifest.json` 里是 `keep`/bf16，只有文本的线性层被按行量化）。所以这里**从容器里
发现根**：

- 不看 `vision_config.model_type` —— h2o 的 `vision_config` 写的是 `qwen3_5` 而不是
  `qwen3_5_vision`，按它挑读法会在同一个塔上分叉出两条路径；
- 也不从 `WeightPlan.prefix`（`language_model` / `model.language_model`）做起名手术 ——
  那是文本层的名字，塔挂在它外面。

容器差异（bf16 单文件 / row-int8 目录）留在 `open_checkpoint`：`read()` 一律给 fp32，
所以这下面不认识任何容器。
"""

from __future__ import annotations

import numpy as np

from ..errors import MissingTensorError
from .checkpoint import WeightStore

# 塔根的名字锚点：它在两个 checkpoint 里都只有一个，拿它反推根前缀。
_ROOT_ANCHOR = "visual.patch_embed.proj.weight"

# `key` 是 MIL/blob 里的条目名；值是塔根之后的名字。24 个 block 用同一套（形状逐项相同）。
BLOCK_TENSORS = {
    "n1_g": "norm1.weight",
    "n1_b": "norm1.bias",
    "n2_g": "norm2.weight",
    "n2_b": "norm2.bias",
    "qkv": "attn.qkv.weight",
    "qkv_b": "attn.qkv.bias",
    "proj": "attn.proj.weight",
    "proj_b": "attn.proj.bias",
    "fc1": "mlp.linear_fc1.weight",
    "fc1_b": "mlp.linear_fc1.bias",
    "fc2": "mlp.linear_fc2.weight",
    "fc2_b": "mlp.linear_fc2.bias",
}

PATCH_TENSORS = {
    "pe": "patch_embed.proj.weight",
    "pe_b": "patch_embed.proj.bias",
}

MERGER_TENSORS = {
    "mn_g": "merger.norm.weight",
    "mn_b": "merger.norm.bias",
    "m1": "merger.linear_fc1.weight",
    "m1_b": "merger.linear_fc1.bias",
    "m2": "merger.linear_fc2.weight",
    "m2_b": "merger.linear_fc2.bias",
}

POS_EMBED = "pos_embed.weight"


def tower_root(store: WeightStore) -> str:
    """塔张量的根前缀（`""` 或 `"model."`）：按锚点找，找不到就报错不猜。

    锚点必须唯一：两个以上同名后缀说明容器里有两份塔，那时候"根"不是一个能推出来的东西。
    """
    matches = [name for name in store.names() if name == _ROOT_ANCHOR or
               name.endswith("." + _ROOT_ANCHOR)]
    if len(matches) != 1:
        raise MissingTensorError(
            "找不到唯一的塔根", anchor=_ROOT_ANCHOR, matches=sorted(matches),
        )
    anchor = matches[0]
    return anchor[: -len(_ROOT_ANCHOR)]


def block_entries(store: WeightStore, root: str, index: int) -> dict[str, np.ndarray]:
    """第 `index` 个 block 的 12 个条目（4 个权重 + 4 个 bias + 2×2 个 LayerNorm）。"""
    return _read(store, root, f"blocks.{index}.", BLOCK_TENSORS)


def patch_entries(store: WeightStore, root: str) -> dict[str, np.ndarray]:
    return _read(store, root, "", PATCH_TENSORS)


def merger_entries(store: WeightStore, root: str) -> dict[str, np.ndarray]:
    return _read(store, root, "", MERGER_TENSORS)


def pos_embed(store: WeightStore, root: str) -> np.ndarray:
    """pos_embed 表 `[num_position_embeddings, hidden]`（两个模型都是 `(2304, 1024)`）。

    它不编进 program：塔在 host 上按 `align_corners=True` 插值到每张图的 `(gh, gw)` 网格
    （票据 07 §2.5）。
    """
    return _read(store, root, "", {"pe": POS_EMBED})["pe"]


def _read(store: WeightStore, root: str, scope: str,
          names: dict[str, str]) -> dict[str, np.ndarray]:
    """按 `key -> 后缀` 读一组张量，权重摊成 `[cout, cin]`，一维的（bias / gamma）原样。"""
    out: dict[str, np.ndarray] = {}
    for key, suffix in names.items():
        name = f"{root}visual.{scope}{suffix}"
        tensor = store.read(name)
        out[key] = _flatten(name, tensor)
    return out


def _flatten(name: str, tensor: np.ndarray) -> np.ndarray:
    """把 `nn.Conv3d` 的 5 维权重摊成 1×1 conv 的 `[cout, cin]`。

    patch 投影在 checkpoint 里是 `(1024, 3, 2, 16, 16)`：一个 16×16 的 patch 跨 2 帧、
    3 个通道，正好是 1×1 conv 的 `cin = 3×2×16×16 = 1536`。摊平就是那件事，不是近似。
    """
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 0:
        raise MissingTensorError("视觉塔的张量维度不对", tensor=name, shape=list(tensor.shape))
    return tensor.reshape(int(tensor.shape[0]), -1)
