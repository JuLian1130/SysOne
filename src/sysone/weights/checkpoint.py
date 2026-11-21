"""检查点目录的两种容器形态：单文件张量，或 row-int8 分片目录。

两份发布权重都是单文件 bf16 `model.safetensors`；row-int8（`__q` 码 + `__s` 标度 +
`int8_manifest.json`）是可选输入 —— 本地打过包就复用它的码，否则引擎在编译期现量化。
两者 dtype 与标度精度都不同，但读出来是同一件事：`read(name)` 给 float32、`read_rows`
给若干行。装载侧只该看见这件事，容器差异留在这里 —— 否则每个读权重的地方都要再长
一遍"这份 checkpoint 是哪种形态"的分支。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..domain.quant import QuantSpec, Scales, Scheme, Storage
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


def stored_quant(model_dir: str | Path) -> QuantSpec:
    """这个目录**存**的是哪种量化形态（`Provenance.quant` 与 `inspect` 报的就是它）。

    判据与 `open_checkpoint` 同一条，不另立一份真相：有 `int8_manifest.json` 就是 row-int8
    （每输出行一个 fp32 标度、码与标度在同一个张量文件里），否则是原样的 16 bit 张量。

    **这不是 ANE 产物那种量化**（`per_channel_sym_int8/fp16/mixed_blob`）：那是编译出来的
    blob，检查点里没有它。所以一张按行审过的 int8 检查点，报的必须是它自己这一种 ——
    不填的话 `QuantSpec()` 的默认值是"未量化"，回包与 `inspect` 都在说假话。
    """
    root = Path(model_dir)
    if (root / MANIFEST_NAME).is_file():
        return QuantSpec(Scheme.PER_ROW_SYM_INT8, Scales.FP32, Storage.TENSOR)
    return QuantSpec()


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
