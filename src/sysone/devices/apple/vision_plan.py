"""视觉塔的三个 MIL 构造器：patch / block / merger。

从旧引擎的 `docs/ANE/anebench/p_vision_mil.py`（`build_patch` / `build_block` /
`build_merger`）搬来 —— 那三个构造器在 §4.8 对 fp32 参考验到 fp16 `rel 4e-4…4e-3`。入口换成
我们的 `write_program` + `ane_blob.build(quantize=False)`，conv/slice 直接用 `compile.py` 里
那对生成器（旧引擎也这么做："probe 的结果不能怪到一份与出货路径漂开的再实现上"）。

两个只在旧引擎里踩过的坑，这里原样保留：

* **procedure 的面槽位按函数名的字典序排，不按 MIL 文本里的顺序。** 所以三段叫
  `main` / `p1` / `p2`（字典序与顺序一致），不叫 `qkv` / `out` / `ffn`（那样 `ffn` 会排在最
  前，`qkv` 会拿到 2048 宽的输入面，数值 rel 5.8）。
* **LayerNorm 的 gamma/beta 必须是 1 维 `[c]`。** conv 权重用的 `[1, c, 1, 1]` 形状在这里会
  `InvalidMILProgram`。

`gelu` 的 `mode` 是**字符串**枚举且没有可用默认值：裸 `gelu(x)` 直接 `InvalidMILProgram`，
`mode = int32(1)` 也不行。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import NotReadyError
from ...weights.packing.ane_blob import BlobEntry
from .compile import _conv, _func, _program, _slice


# procedure 名 -> 序号：`main` < `p1` 的字典序正好是 0/1（见模块 docstring）。
PROCS = {"main": 0, "p1": 1}


@dataclass(frozen=True)
class ProgramPlan:
    """一份视觉 program：MIL 文本 + 每个 procedure 的面尺寸。

    `io` 是 `procedure -> (cin, sin, cout, sout)`，**四元组**而不是"输入输出同宽的三元组"：
    merger 的 4:1 合并在同一趟里换列数（`s → s//4`），按同宽分配会把输入面开成 4 倍。
    """

    mil: str
    io: dict[str, tuple[int, int, int, int]]


def patch_io(*, cin: int, hidden: int, s: int) -> dict[str, tuple[int, int, int, int]]:
    """patch 投影的面尺寸。与 `patch_plan` 分开是因为**缓存命中时也要它** —— 产物命中就不该
    再读权重、装 blob，但面还是要按同一份布局开。"""
    return {"main": (cin, s, hidden, s)}


def block_io(*, hidden: int, ffn: int, s: int) -> dict[str, tuple[int, int, int, int]]:
    """一个 block 两个 procedure 的面尺寸（见 `block_plan` 的说明）。"""
    return {"main": (hidden, s, 3 * hidden, s), "p1": (2 * hidden, s, hidden, s)}


def merger_io(*, hidden: int, ffn: int, out_hidden: int, s: int
              ) -> dict[str, tuple[int, int, int, int]]:
    """merger 两趟的面尺寸：`main` 在趟内把列数从 `s` 掉到 `s//4`。"""
    q = s // 4
    return {"main": (hidden, s, ffn, q), "p1": (ffn, q, out_hidden, q)}


def patch_plan(*, hidden: int, s: int, entries: dict[str, np.ndarray],
               blob: dict[str, BlobEntry]) -> ProgramPlan:
    """patch 投影：`[1, 3×2×16×16, 1, s]` → `[1, hidden, 1, s]`。

    输入通道数是 `in_channels × temporal × patch²`＝1536，**从权重形状取**而不是写死 ——
    写死就又是一份"塔的形状"的真相。摊平见 `weights/vision.py::_flatten`。
    """
    patch = _columns(entries["pe"])
    body = _conv("pe", cin=patch, cout=hidden, s=s, entry=blob["pe"], src="x", out="pt")
    body += _bias("pe", c=hidden, s=s, src="pt", out="y", entry=blob["pe_b"])
    mil = _program([_func("main", func_input=f"tensor<fp16, [1, {patch}, 1, {s}]> x",
                          body=body)])
    return ProgramPlan(mil=mil, io=patch_io(cin=patch, hidden=hidden, s=s))


def block_plan(*, hidden: int, ffn: int, s: int, entries: dict[str, np.ndarray],
               blob: dict[str, BlobEntry]) -> ProgramPlan:
    """一个视觉 block 的线性半边（两段，注意力在 GPU 上）：

    - `main`：`norm1` → `qkv`(+bias)，出口是 `[3·hidden]`（q/k/v 还没切）；
    - `p1`：入口是 `[attn_ctx | residual]`（两段不同，所以要分开切片）→ `proj`(+bias) → 加残差
      → `norm2` → `fc1`(+bias) → gelu → `fc2`(+bias) → 再加一次残差。第二段的残差就是 `norm2`
      读的那块（`h + mlp(norm2(h))`），所以只需要一次加法，不需要第二张切片。

    这两段原来各是一次 eval（`p1`/`p2`），中间过一张 `hidden × s` 的面。合成一段省掉的是每次
    tile 一次 eval 加一次面的往返，代价是同一趟里两段矩阵乘互相争用（旧引擎 §4.19 量到
    +0.30 ms/eval，而 1024² 整图省下 96 次 eval，见票据 07 §7.10）。

    视觉的每个 `nn.Linear` 都带 bias（文本层不带，所以 `_conv` 那里是对的、这里必须补）。
    """
    body0 = _layernorm("n1", c=hidden, s=s, src="x", out="xn",
                       gamma=blob["n1_g"], beta=blob["n1_b"])
    body0 += _conv("q", cin=hidden, cout=3 * hidden, s=s, entry=blob["qkv"], src="xn", out="qf")
    body0 += _bias("q", c=3 * hidden, s=s, src="qf", out="y", entry=blob["qkv_b"])

    body1 = _slice("sv", begin=0, size=hidden, s=s, src="x", out="v")
    body1 += _slice("sr", begin=hidden, size=hidden, s=s, src="x", out="r")
    body1 += _conv("o", cin=hidden, cout=hidden, s=s, entry=blob["proj"], src="v", out="ov")
    body1 += _bias("o", c=hidden, s=s, src="ov", out="ovb", entry=blob["proj_b"])
    body1.append(f'        tensor<fp16, [1, {hidden}, 1, {s}]> h = add(x = ovb, y = r)'
                 f'[name = string("h")];')
    body1 += _layernorm("n2", c=hidden, s=s, src="h", out="xn",
                        gamma=blob["n2_g"], beta=blob["n2_b"])
    body1 += _conv("u", cin=hidden, cout=ffn, s=s, entry=blob["fc1"], src="xn", out="gt")
    body1 += _bias("u", c=ffn, s=s, src="gt", out="gtb", entry=blob["fc1_b"])
    body1 += _gelu(c=ffn, s=s, src="gtb", out="ga")
    body1 += _conv("d", cin=ffn, cout=hidden, s=s, entry=blob["fc2"], src="ga", out="yb")
    body1 += _bias("d", c=hidden, s=s, src="yb", out="ybb", entry=blob["fc2_b"])
    body1.append(f'        tensor<fp16, [1, {hidden}, 1, {s}]> y = add(x = ybb, y = h)'
                 f'[name = string("y")];')

    mil = _program([
        _func("main", func_input=f"tensor<fp16, [1, {hidden}, 1, {s}]> x", body=body0),
        _func("p1", func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x", body=body1),
    ])
    return ProgramPlan(mil=mil, io=block_io(hidden=hidden, ffn=ffn, s=s))


def merger_plan(*, hidden: int, ffn: int, out_hidden: int, s: int,
                entries: dict[str, np.ndarray], blob: dict[str, BlobEntry]) -> ProgramPlan:
    """merger：`norm(hidden)` → 2×2 空间合并到 `4·hidden`（列数 `s → s//4`）→ fc1 → gelu → fc2。

    两个 procedure：`main` 把 norm + 合并 + fc1 + gelu 做在同一趟里（列数在趟内从 `s` 掉到
    `s//4`，所以 `io` 必须写四元组），`p1` 只做 fc2。
    """
    if s % 4:
        raise NotReadyError("merger 的 2×2 合并要求桶宽能整除 4", s=s)
    q = s // 4
    body0 = _layernorm("mn", c=hidden, s=s, src="x", out="xn",
                       gamma=blob["mn_g"], beta=blob["mn_b"])
    body0 += _merge2x2(c=hidden, s=s, src="xn", out="mm")
    body0 += _conv("m", cin=ffn, cout=ffn, s=q, entry=blob["m1"], src="mm", out="mt")
    body0 += _bias("m", c=ffn, s=q, src="mt", out="mtb", entry=blob["m1_b"])
    body0 += _gelu(c=ffn, s=q, src="mtb", out="y")

    body1 = _conv("n", cin=ffn, cout=out_hidden, s=q, entry=blob["m2"], src="x", out="nt")
    body1 += _bias("n", c=out_hidden, s=q, src="nt", out="y", entry=blob["m2_b"])

    mil = _program([
        _func("main", func_input=f"tensor<fp16, [1, {hidden}, 1, {s}]> x", body=body0),
        _func("p1", func_input=f"tensor<fp16, [1, {ffn}, 1, {q}]> x", body=body1),
    ])
    return ProgramPlan(mil=mil, io=merger_io(hidden=hidden, ffn=ffn, out_hidden=out_hidden, s=s))


# ---------------------------------------------------------------- MIL 片段


def _bias(tag: str, *, c: int, s: int, src: str, out: str, entry: BlobEntry) -> list[str]:
    """`src + b`：b 是 `[1, c, 1, 1]` 的 BLOBFILE 常量，一次广播加就够。

    文本层无 bias（`attention_bias: false`，MLP 也一样），所以 `_conv` 不发 bias 是对的；
    视觉塔的每个 `nn.Linear` 都带 bias。
    """
    path = 'string("@model_path/weights.bin")'
    return [
        f'        tensor<fp16, [1, {c}, 1, 1]> {tag}_bb = const()[name = string("{tag}_bb"), '
        f'val = tensor<fp16, [1, {c}, 1, 1]>(BLOBFILE(path = {path}, '
        f"offset = uint64({entry.offset})))];",
        f'        tensor<fp16, [1, {c}, 1, {s}]> {out} = add(x = {src}, y = {tag}_bb)'
        f'[name = string("{out}")];',
    ]


def _layernorm(tag: str, *, c: int, s: int, src: str, out: str,
               gamma: BlobEntry, beta: BlobEntry, eps: float = 1e-6) -> list[str]:
    """沿通道轴做 LayerNorm；`axes = [1]`，gamma/beta 是 1 维 `[c]`（见模块 docstring）。"""
    path = 'string("@model_path/weights.bin")'
    return [
        f'        tensor<int32, [1]> {tag}_ax = const()[name = string("{tag}_ax"), '
        f"val = tensor<int32, [1]>([1])];",
        f'        fp16 {tag}_eps = const()[name = string("{tag}_eps"), val = fp16({eps})];',
        f'        tensor<fp16, [{c}]> {tag}_g = const()[name = string("{tag}_g"), '
        f'val = tensor<fp16, [{c}]>(BLOBFILE(path = {path}, '
        f"offset = uint64({gamma.offset})))];",
        f'        tensor<fp16, [{c}]> {tag}_b = const()[name = string("{tag}_b"), '
        f'val = tensor<fp16, [{c}]>(BLOBFILE(path = {path}, '
        f"offset = uint64({beta.offset})))];",
        f'        tensor<fp16, [1, {c}, 1, {s}]> {out} = layer_norm(x = {src}, '
        f'axes = {tag}_ax, epsilon = {tag}_eps, gamma = {tag}_g, beta = {tag}_b)'
        f'[name = string("{out}")];',
    ]


def _gelu(*, c: int, s: int, src: str, out: str) -> list[str]:
    """`gelu_pytorch_tanh`：MIL 的 `gelu` 带 `mode = string("TANH_APPROXIMATION")`。"""
    return [
        f'        tensor<fp16, [1, {c}, 1, {s}]> {out} = gelu(x = {src}, '
        f'mode = string("TANH_APPROXIMATION"))[name = string("{out}")];'
    ]


def _merge2x2(*, c: int, s: int, src: str, out: str) -> list[str]:
    """`[1, c, 1, s]` → `[1, 4c, 1, s/4]`：上游是 `x.view(-1, 4·hidden)`，所以合并后的通道带
    `j` 装的是序列位置 `4q + j` 那个 token —— 一个 2×2 窗口里连续的四个 token。

    这是上游自己的顺序，三步做完：`reshape` 把"合并类"单独立一轴、`transpose` 把它挪到通道轴
    前面、再 `reshape` 回来。一次 `gather(axis=3)` 也能做，但这个编译器不认；
    `transpose` 也不能用 `axes =`，属性名是 `perm`。
    """
    q = s // 4
    return [
        f'        tensor<int32, [4]> mg_s1 = const()[name = string("mg_s1"), '
        f"val = tensor<int32, [4]>([1, {c}, {q}, 4])];",
        f'        tensor<fp16, [1, {c}, {q}, 4]> mg_a = reshape(x = {src}, '
        f'shape = mg_s1)[name = string("mg_a")];',
        '        tensor<int32, [4]> mg_pm = const()[name = string("mg_pm"), '
        'val = tensor<int32, [4]>([0, 3, 1, 2])];',
        f'        tensor<fp16, [1, 4, {c}, {q}]> mg_b = transpose(x = mg_a, '
        f'perm = mg_pm)[name = string("mg_b")];',
        f'        tensor<int32, [4]> mg_s2 = const()[name = string("mg_s2"), '
        f"val = tensor<int32, [4]>([1, {4 * c}, 1, {q}])];",
        f'        tensor<fp16, [1, {4 * c}, 1, {q}]> {out} = reshape(x = mg_b, '
        f'shape = mg_s2)[name = string("{out}")];',
    ]


def _columns(weight: np.ndarray) -> int:
    """摊平后的权重列数 = 这个 1×1 conv 的 `cin`。"""
    if weight.ndim != 2:
        raise NotReadyError("视觉塔的权重没有摊成二维", shape=list(weight.shape))
    return int(weight.shape[1])
