"""视觉塔的 MIL 构造器：procedure 序号、面尺寸与 MIL 自己的声明必须对得上。

这里断言的都是**会让数值静默错掉**的那些约定，不是字符串的字面：

- procedure 的面槽位按函数名字典序排（不是 MIL 文本顺序）⇒ 三段必须叫 `main`/`p1`/`p2`；
- 每个 procedure 的输入宽度在 MIL 里声明、在 `ProgramPlan.io` 里再声明一次 ⇒ 两处必须一致，
  对不上时 ANE 会拿错宽的面来算（旧引擎那次 rel 5.8 就是这么来的）；
- merger 的 2×2 合并让列数在趟内 4:1 ⇒ `io` 必须写四元组，而且桶宽要能整除 4。
"""

from __future__ import annotations

import re
import struct

import numpy as np
import pytest

from sysone.devices.apple import kernels, vision_plan as vp
from sysone.errors import NotReadyError
from sysone.weights.packing.ane_blob import build as build_blob

HIDDEN, FFN, OUT, S = 1024, 4096, 2048, 256


def _entries(**shapes) -> dict[str, np.ndarray]:
    """按条目名造一批同形状的零权重；名字必须与 `weights/vision.py` 的键一致。"""
    return {key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()}


def _blob(entries):
    return build_blob(entries, quantize=False)


def _func_inputs(mil: str) -> dict[str, tuple[int, int]]:
    pattern = r"func (\w+)<ios18>\(tensor<fp16, \[1, (\d+), 1, (\d+)\]> x\)"
    return {name: (int(a), int(b)) for name, a, b in re.findall(pattern, mil)}


def test_procedure_names_sort_in_execution_order() -> None:
    """dylib 按**名字的字典序**排面槽位，所以序号必须等于字典序排名。

    叫 `qkv`/`out`/`ffn` 会让 `ffn` 抢到 0 号面（旧引擎实测 rel 5.8），所以这里钉住排名。
    """
    names = sorted(vp.PROCS, key=vp.PROCS.get)
    assert names == sorted(vp.PROCS), "PROCS 的序号不是按名字的字典序排的"
    assert vp.PROCS == {"main": 0, "p1": 1}


def test_patch_program_declares_what_the_plan_allocates() -> None:
    entries = _entries(pe=(HIDDEN, 3 * 2 * 16 * 16), pe_b=(HIDDEN,))
    blob, table = _blob(entries)
    plan = vp.patch_plan(hidden=HIDDEN, s=S, entries=entries, blob=table)
    assert plan.io == {"main": (1536, S, HIDDEN, S)}
    assert _func_inputs(plan.mil) == {"main": (1536, S)}


def test_block_program_declares_what_the_plan_allocates() -> None:
    entries = _entries(
        n1_g=(HIDDEN,), n1_b=(HIDDEN,), n2_g=(HIDDEN,), n2_b=(HIDDEN,),
        qkv=(3 * HIDDEN, HIDDEN), qkv_b=(3 * HIDDEN,),
        proj=(HIDDEN, HIDDEN), proj_b=(HIDDEN,),
        fc1=(FFN, HIDDEN), fc1_b=(FFN,), fc2=(HIDDEN, FFN), fc2_b=(HIDDEN,),
    )
    blob, table = _blob(entries)
    plan = vp.block_plan(hidden=HIDDEN, ffn=FFN, s=S, entries=entries, blob=table)
    # p1 的入口是 `[attn_ctx | residual]`：两段各 HIDDEN 行、两段不同所以不能共用一张面；
    # proj 之后的 norm2+FFN 与它在同一段里（`h` 是这一段的残差），所以只有两段。
    assert plan.io == {"main": (HIDDEN, S, 3 * HIDDEN, S),
                       "p1": (2 * HIDDEN, S, HIDDEN, S)}
    assert _func_inputs(plan.mil) == {"main": (HIDDEN, S), "p1": (2 * HIDDEN, S)}
    # 出口宽度也要与声明一致：`y` 是每个 func 的返回值。
    for cout in (3 * HIDDEN, HIDDEN):
        assert f"tensor<fp16, [1, {cout}, 1, {S}]> y = " in plan.mil
    # 两段各自只加一次残差：p1 里 `h = proj_out + r`、`y = fc2_out + h`。
    assert 'add(x = ovb, y = r)' in plan.mil
    assert 'add(x = ybb, y = h)' in plan.mil


def test_merger_changes_width_inside_one_procedure() -> None:
    """norm+合并+fc1 在同一趟里把列数从 S 掉到 S/4 —— 用"同宽三元组"会把输入面开成 4 倍。"""
    entries = _entries(
        mn_g=(HIDDEN,), mn_b=(HIDDEN,),
        m1=(FFN, FFN), m1_b=(FFN,), m2=(OUT, FFN), m2_b=(OUT,),
    )
    blob, table = _blob(entries)
    plan = vp.merger_plan(hidden=HIDDEN, ffn=FFN, out_hidden=OUT, s=S,
                          entries=entries, blob=table)
    q = S // 4
    assert plan.io == {"main": (HIDDEN, S, FFN, q), "p1": (FFN, q, OUT, q)}
    assert _func_inputs(plan.mil) == {"main": (HIDDEN, S), "p1": (FFN, q)}


def test_merger_refuses_a_bucket_that_cannot_split_into_four() -> None:
    entries = _entries(
        mn_g=(HIDDEN,), mn_b=(HIDDEN,),
        m1=(FFN, FFN), m1_b=(FFN,), m2=(OUT, FFN), m2_b=(OUT,),
    )
    blob, table = _blob(entries)
    with pytest.raises(NotReadyError) as excinfo:
        vp.merger_plan(hidden=HIDDEN, ffn=FFN, out_hidden=OUT, s=250,
                       entries=entries, blob=table)
    assert excinfo.value.context["s"] == 250


def test_gelu_carries_the_string_mode_and_layernorm_one_dimensional_tables() -> None:
    """两个只在旧引擎里踩过的坑：`gelu` 的 mode 是字符串枚举（裸 `gelu(x)` 编不过），
    LayerNorm 的 gamma/beta 必须是 1 维 `[c]`（`[1, c, 1, 1]` 会被拒）。"""
    entries = _entries(
        n1_g=(HIDDEN,), n1_b=(HIDDEN,), n2_g=(HIDDEN,), n2_b=(HIDDEN,),
        qkv=(3 * HIDDEN, HIDDEN), qkv_b=(3 * HIDDEN,),
        proj=(HIDDEN, HIDDEN), proj_b=(HIDDEN,),
        fc1=(FFN, HIDDEN), fc1_b=(FFN,), fc2=(HIDDEN, FFN), fc2_b=(HIDDEN,),
    )
    blob, table = _blob(entries)
    mil = vp.block_plan(hidden=HIDDEN, ffn=FFN, s=S, entries=entries, blob=table).mil
    assert 'gelu(x = gtb, mode = string("TANH_APPROXIMATION"))' in mil
    assert f"tensor<fp16, [{HIDDEN}]> n1_g = const()" in mil
    assert f"tensor<fp16, [{HIDDEN}]> n1_b = const()" in mil
    assert f"tensor<fp16, [1, {HIDDEN}, 1, 1]> n1_g" not in mil


def test_io_helpers_are_the_plans_own_numbers() -> None:
    """`*_io()` 是三个构造器与**缓存命中路径**共用的那一份：命中时不重读权重、不重装 blob，
    但面还得按同一份布局开 —— 两处各写一遍就会在命中与未命中之间漂开。
    """
    entries = _entries(
        n1_g=(HIDDEN,), n1_b=(HIDDEN,), n2_g=(HIDDEN,), n2_b=(HIDDEN,),
        qkv=(3 * HIDDEN, HIDDEN), qkv_b=(3 * HIDDEN,),
        proj=(HIDDEN, HIDDEN), proj_b=(HIDDEN,),
        fc1=(FFN, HIDDEN), fc1_b=(FFN,), fc2=(HIDDEN, FFN), fc2_b=(HIDDEN,),
    )
    blob, table = _blob(entries)
    assert vp.block_plan(hidden=HIDDEN, ffn=FFN, s=S, entries=entries,
                         blob=table).io == vp.block_io(hidden=HIDDEN, ffn=FFN, s=S)
    patch = _entries(pe=(HIDDEN, 3 * 2 * 16 * 16), pe_b=(HIDDEN,))
    patch_blob, patch_table = _blob(patch)
    assert vp.patch_plan(hidden=HIDDEN, s=S, entries=patch,
                         blob=patch_table).io == vp.patch_io(cin=1536, hidden=HIDDEN, s=S)
    merger = _entries(mn_g=(HIDDEN,), mn_b=(HIDDEN,), m1=(FFN, FFN), m1_b=(FFN,),
                      m2=(OUT, FFN), m2_b=(OUT,))
    merger_blob, merger_table = _blob(merger)
    assert vp.merger_plan(hidden=HIDDEN, ffn=FFN, out_hidden=OUT, s=S, entries=merger,
                          blob=merger_table).io == vp.merger_io(hidden=HIDDEN, ffn=FFN,
                                                                out_hidden=OUT, s=S)


def test_vattn_args_layout_is_the_msl_struct() -> None:
    """`k_vattn` 的参数是 24 B：5 个 u32 + 1 个 f32，字段顺序见 `shaders/e8_vattn.metal`。

    dylib 把整块结构体写进 `E8_ARGS_INDEX`，所以格式串就是 ABI 本身：字段顺序错一位，核读到
    的 `T` 会是 `nq`，形状合法而结果错。
    """
    args = kernels.vattn_args(T=4096, nq=4096, s0=0, hidden=1024, dim=64, scale=0.125)
    assert len(args) == 24
    assert struct.unpack("<IIIIIf", args) == (4096, 4096, 0, 1024, 64, 0.125)
