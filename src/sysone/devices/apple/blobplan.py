"""一层的 ANE blob 由哪些张量怎么拼出来 —— 这是**核的约定**，不是模型的。

条目名（`proj` / `o` / `gate` / `up` / `down{i}`）、行拼接的顺序、列切的位置，全都由
`layer_mil` 生成的那三个 procedure 与 Metal 核的读写位置定死：`k_gdn` 从 `cout_gdn-32`
起读 `b`、从 `cout_gdn-16` 起读 `a`（`e8_common.h:29`），`ffn` 的 `down{i}` 按 `ffn_segments`
一段一段累加。换个模型只要形状对得上就复用同一份规则 —— Qwen3.5 的两个出货模型（mude 2B、
h2o 4B）就是同一份。所以它住在设备层：家族只提供"张量叫什么前缀、norm 偏移多少"这类数据
（`models.layer_plan.WeightPlan`），不提供"怎么拼"。

两份物化函数（`ane_layer_weights` / `ane_layer_quantized`）都从 `ane_layer_segments` 出发，
所以行拼顺序与列切只在 `ane_layer_segments` 里写一次。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...domain.attention import AttentionRole
from ...domain.work import ffn_segments
from ...weights.packing import ane_blob
from .geometry import Geometry

# 一个 blob 条目由哪些张量按行拼成：`(张量名, 列切片)`，`slice(None)` 表示整块。
Segments = dict[str, tuple[tuple[str, slice], ...]]
_ALL = slice(None)


def ane_layer_segments(
    *, layer: int, layer_type: AttentionRole, hidden: int, ffn: int, prefix: str
) -> Segments:
    """一层要装进 blob 的条目表（键与 `layer_mil` 的 `entries` 同名）。

    投影是**整块行拼接**，这是旧引擎 `COUT` 的来历，也是 20 层链要复用的同一事实：
    - 全注意力层 5120 = `q_proj` 4096（每头 `[q(head_dim) | gate(head_dim)]`，与厂商
      `chunk(view(..., head_dim*2), 2, -1)` 交错方式一致）+ `k_proj` 512 + `v_proj` 512；
    - 线性注意力层 8224 = `in_proj_qkv` 6144（q丨k丨v 各 2048）+ `in_proj_z` 2048 +
      `in_proj_b` 16 + `in_proj_a` 16。

    线性层的 `b` 在 `a` 之前，是 Metal 核的约定（`k_gdn` 从 `COUT_GDN-32` 起读 `b`、从
    `COUT_GDN-16` 起读 `a`，见 `docs/ANE/anebench/e8_common.h:29`）；numpy 链按同一顺序读。

    FFN 的 `down_proj` 是 `[hidden, ffn]`，按 `ffn_segments()` 切**输入**维（mude 三段等长、
    4B 最后一段短一截），所以 `down{i}` 切的列区间由那份规则给；`out` 是 `out_proj` /
    `o_proj`（`[hidden, core_out]`，残差在 program 里加）。

    行拼与列切都不动 int8 的标度（量化按输出行），所以这张表对"现量化"和"容器自带码"
    两条路是同一份 —— 两个物化函数都从它出发。
    """
    base = f"{prefix}.layers.{layer}"
    if layer_type is AttentionRole.FULL:
        attn = f"{base}.self_attn"
        proj = ((f"{attn}.q_proj.weight", _ALL), (f"{attn}.k_proj.weight", _ALL),
                (f"{attn}.v_proj.weight", _ALL))
        out = (f"{attn}.o_proj.weight", _ALL)
    else:
        attn = f"{base}.linear_attn"
        proj = tuple(
            (f"{attn}.{name}.weight", _ALL)
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
        )
        out = (f"{attn}.out_proj.weight", _ALL)

    mlp = f"{base}.mlp"
    segments: Segments = {
        "proj": proj,
        "o": (out,),
        "gate": ((f"{mlp}.gate_proj.weight", _ALL),),
        "up": ((f"{mlp}.up_proj.weight", _ALL),),
    }
    down = f"{mlp}.down_proj.weight"
    for index, columns in enumerate(ffn_segments(hidden, ffn)):
        segments[f"down{index}"] = ((down, columns),)
    return segments


def ane_layer_weights(
    read: Callable[[str], np.ndarray],
    *,
    layer: int,
    layer_type: AttentionRole,
    hidden: int,
    ffn: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    """一层的 ANE 权重（`layer_mil` 的 `entries`），来源是原样权重。"""
    entries: dict[str, np.ndarray] = {}
    for name, segments in ane_layer_segments(
        layer=layer, layer_type=layer_type, hidden=hidden, ffn=ffn, prefix=prefix
    ).items():
        entries[name] = np.concatenate([read(tensor)[:, columns] for tensor, columns in segments], axis=0)
    return entries


def ane_layer_quantized(
    read: Callable[[str], np.ndarray],
    parts: Callable[[str], tuple[np.ndarray, np.ndarray] | None],
    *,
    layer: int,
    layer_type: AttentionRole,
    hidden: int,
    ffn: int,
    prefix: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """一层的 ANE 权重，形态是 `(int8 码, 标度)`：容器自带码就直接用，否则现量化。

    `parts(tensor)` 给 `(码, fp32 标度)` 或 `None`（这份容器里它就是原样权重）。容器自带码
    时必须用它 —— 那是检查点对同一批数值已经做过的量化，重算一遍会叠加第二次误差
    （`ane_blob.build_quantized` 的注释里有实测数字）。

    码按行接起来、标度同理：行拼与列切都在"输出行"以外，不动每行的标度。
    """
    entries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, segments in ane_layer_segments(
        layer=layer, layer_type=layer_type, hidden=hidden, ffn=ffn, prefix=prefix
    ).items():
        codes: list[np.ndarray] = []
        scales: list[np.ndarray] = []
        for tensor, columns in segments:
            existing = parts(tensor)
            if existing is None:
                quantized = ane_blob.quant_per_channel(
                    np.asarray(read(tensor)[:, columns], np.float16)
                )
                codes.append(quantized.codes)
                scales.append(quantized.scale)
            else:
                part_codes, part_scale = existing
                codes.append(part_codes[:, columns])
                scales.append(np.asarray(part_scale, np.float16))
        entries[name] = (np.concatenate(codes, axis=0), np.concatenate(scales, axis=0))
    return entries
