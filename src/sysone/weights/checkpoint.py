"""检查点目录的两种容器形态：单文件张量，或 row-int8 分片目录。

mude 发布的是 bf16 的 `model.safetensors`；4B 的 h2o 发布的是 row-int8（`__q` 码 + `__s`
fp32 标度 + `int8_manifest.json`）。两者 dtype 与标度精度都不同，但读出来是同一件事：
`read(name)` 给 float32、`read_rows` 给若干行。装载侧只该看见这件事，容器差异留在这里 ——
否则每个读权重的地方都要再长一遍"这份 checkpoint 是哪种形态"的分支。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..errors import MissingTensorError
from .packing.row_int8 import MANIFEST_NAME, RowInt8Store
from .store import SafeTensorFile

# 单文件形态的权重文件名（row-int8 是按 manifest 取数的分片，名单在 manifest 里）。
WEIGHTS_FILE = "model.safetensors"

# 装载侧能拿到的两种 weight store：接口一致（`read` / `read_rows` / `names` /
# `fingerprint` / `close`），只有容器的本事不同。
WeightStore = SafeTensorFile | RowInt8Store


def open_checkpoint(model_dir: str | Path) -> WeightStore:
    """按目录里的容器形态打开权重：有 `int8_manifest.json` 就是 row-int8，否则是单文件张量。

    判据只有 manifest 一个：它是 row-int8 打包时自己写的、也是 `RowInt8Store` 的唯一索引，
    所以"有没有它"与"能不能按 row-int8 读"是同一件事，不需要再猜文件后缀或 dtype。
    """
    root = Path(model_dir)
    if (root / MANIFEST_NAME).is_file():
        return RowInt8Store(root)
    return SafeTensorFile(root / WEIGHTS_FILE)


def int8_parts(store: WeightStore, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """`name` 在容器里的 int8 形态 `(码, fp32 标度)`；存的就是原样权重时返回 `None`。

    row-int8 容器不是整份量化：`MIN_QUANT_ELEMS` 以下的张量（norm、卷积、`in_proj_b/a`
    这类小矩阵）按 16 bit 存，它们没有码可拿，交给调用方现量化。
    """
    if not isinstance(store, RowInt8Store):
        return None
    entry = store.manifest.entries.get(name)
    if entry is None:
        raise MissingTensorError("row-int8 检查点缺少张量", tensor=name, dir=str(store.dir))
    if entry.kind != "int8":
        return None
    return store.read_parts(name)
