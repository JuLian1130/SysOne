"""视觉塔的 MIL 构造器：procedure 序号、面尺寸与 MIL 自己的声明必须对得上。

这里断言的都是**会让数值静默错掉**的那些约定，不是字符串的字面：

- procedure 的面槽位按函数名字典序排（不是 MIL 文本顺序）⇒ block 的两段叫 `a_p1`/`z_main`
  （双输入那段必须排 0 号：只有 0 号 procedure 的多输入会被接线，票据 07 §7.23）；
- 每个 procedure 的输入宽度在 MIL 里声明、在 `ProgramPlan.io` 里再声明一次 ⇒ 两处必须一致，
  对不上时 ANE 会拿错宽的面来算（旧引擎那次 rel 5.8 就是这么来的）；
- merger 的 2×2 合并让列数在趟内 4:1 ⇒ 输入与输出的列数必须分开写，而且桶宽要能整除 4。
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


def _func_inputs(mil: str) -> dict[str, tuple[tuple[int, ...], int]]:
    """`func 名 -> (每张输入面的行数, 列数)`。多张输入面用逗号隔开，所以按 `func` 整块切。"""
    out: dict[str, tuple[tuple[int, ...], int]] = {}
    for body in re.findall(r"func (\w+)<ios18>\((.*?)\) \{", mil):
        name, args = body
        shapes = re.findall(r"tensor<fp16, \[1, (\d+), 1, (\d+)\]> \w+", args)
        out[name] = (tuple(int(a) for a, _ in shapes), int(shapes[0][1]))
    return out


def test_procedure_names_sort_in_execution_order() -> None:
    """dylib 按**名字的字典序**排面槽位，所以序号必须等于字典序排名。

    叫 `qkv`/`out`/`ffn` 会让 `ffn` 抢到 0 号面（旧引擎实测 rel 5.8），所以这里钉住排名。
    """
    for name, index in vp.PROCS.items():
        assert name, index
    # 每个 program 里用到的段名各自按字典序排名，必须与 PROCS 报的序号一致。
    for group in ({"main"}, {"main", "p1"}, {vp.P1, vp.MAIN}):
        for index, name in enumerate(sorted(group)):
            assert vp.PROCS[name] == index, f"{name} 的序号与字典序排名不符"
    # 双输入的那一段必须是最小名（0 号）：只有 0 号 procedure 的多输入会被接线。
    assert min(vp.P1, vp.MAIN) == vp.P1
    assert vp.PROCS[vp.P1] == 0 and vp.PROCS[vp.MAIN] == 1


def test_patch_program_declares_what_the_plan_allocates() -> None:
    entries = _entries(pe=(HIDDEN, 3 * 2 * 16 * 16), pe_b=(HIDDEN,))
    blob, table = _blob(entries)
    plan = vp.patch_plan(hidden=HIDDEN, s=S, entries=entries, blob=table)
    assert plan.io == {"main": ((1536,), HIDDEN, S)}
    assert _func_inputs(plan.mil) == {"main": ((1536,), S)}


def test_block_program_declares_what_the_plan_allocates() -> None:
    entries = _entries(
        n1_g=(HIDDEN,), n1_b=(HIDDEN,), n2_g=(HIDDEN,), n2_b=(HIDDEN,),
        qkv=(3 * HIDDEN, HIDDEN), qkv_b=(3 * HIDDEN,),
        proj=(HIDDEN, HIDDEN), proj_b=(HIDDEN,),
        fc1=(FFN, HIDDEN), fc1_b=(FFN,), fc2=(HIDDEN, FFN), fc2_b=(HIDDEN,),
    )
    blob, table = _blob(entries)
    plan = vp.block_plan(hidden=HIDDEN, ffn=FFN, s=S, entries=entries, blob=table)
    # p1 的入口是**两张**面（上下文、残差）：各 HIDDEN 行，都按 hidden 开；
    # proj 之后的 norm2+FFN 与它在同一段里（`h` 是这一段的残差），所以只有两段。
    assert plan.io == {vp.MAIN: ((HIDDEN,), 3 * HIDDEN, S),
                       vp.P1: ((HIDDEN, HIDDEN), HIDDEN, S)}
    assert _func_inputs(plan.mil) == {vp.MAIN: ((HIDDEN,), S), vp.P1: ((HIDDEN, HIDDEN), S)}
    # 出口宽度也要与声明一致：`y` 是每个 func 的返回值。
    for cout in (3 * HIDDEN, HIDDEN):
        assert f"tensor<fp16, [1, {cout}, 1, {S}]> y = " in plan.mil
    # 两段各自只加一次残差：p1 里 `h = proj_out + r`、`y = fc2_out + h`。
    assert 'add(x = ovb, y = res)' in plan.mil
    assert 'add(x = ybb, y = h)' in plan.mil


def test_merger_changes_width_inside_one_procedure() -> None:
    """norm+合并+fc1 在同一趟里把列数从 S 掉到 S/4 —— 输入与输出的列数写成一个就会开错面。"""
    entries = _entries(
        mn_g=(HIDDEN,), mn_b=(HIDDEN,),
        m1=(FFN, FFN), m1_b=(FFN,), m2=(OUT, FFN), m2_b=(OUT,),
    )
    blob, table = _blob(entries)
    plan = vp.merger_plan(hidden=HIDDEN, ffn=FFN, out_hidden=OUT, s=S,
                          entries=entries, blob=table)
    q = S // 4
    assert plan.io == {"main": ((HIDDEN,), FFN, q), "p1": ((FFN,), OUT, q)}
    assert _func_inputs(plan.mil) == {"main": ((HIDDEN,), S), "p1": ((FFN,), q)}


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
    """`k_vattn` 的参数是 32 B：7 个 u32 + 1 个 f32，字段顺序见 `shaders/e8_vattn.metal`。

    dylib 把整块结构体写进 `E8_ARGS_INDEX`，所以格式串就是 ABI 本身：字段顺序错一位，核读到
    的 `T` 会是 `nq`，形状合法而结果错。`oT`/`os0` 是**输出面自己的**行跨距与首列（塔按瓦片
    写独立的面，qkv 仍是整图），所以两个布局要分开传。
    """
    args = kernels.vattn_args(T=4096, nq=1024, s0=2048, hidden=1024, dim=64, scale=0.125,
                              out_cols=1024, out_offset=0)
    assert len(args) == 32
    assert struct.unpack("<IIIIIfII", args) == (4096, 1024, 2048, 1024, 64, 0.125, 1024, 0)
    # 不给 out_offset 时沿用它原来的意思：输出与 qkv 同一份整图布局
    same = kernels.vattn_args(T=4096, nq=1024, s0=2048, hidden=1024, dim=64, scale=0.125)
    assert struct.unpack("<IIIIIfII", same)[6:] == (4096, 2048)
