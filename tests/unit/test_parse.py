"""读三份真实 config，检查几何与能力声明。"""

from __future__ import annotations

import dataclasses
import json
import shutil

import pytest

from helpers import FIXTURES, make_code_checkpoint
from sysone.devices.apple.geometry import Geometry, mude_geometry
from sysone.domain.attention import AttentionRole, MaskMode
from sysone.domain.quant import Scheme
from sysone.errors import ConfigError
from sysone.models.families import d3, h2o, mude, qwen35, torchcast


def test_mude_geometry() -> None:
    descriptor = mude.parse_config(FIXTURES / "mude")
    assert descriptor.family == "mude"
    assert descriptor.num_layers == 20
    assert descriptor.hidden == 2048
    assert descriptor.ffn == 6144
    assert (descriptor.heads, descriptor.kv_heads, descriptor.head_dim) == (8, 2, 256)
    full = [index for index, role in enumerate(descriptor.layer_types) if role is AttentionRole.FULL]
    assert full == [3, 7, 11, 15, 19]
    assert descriptor.tied_embeddings is True
    assert descriptor.vision is not None and descriptor.vision.model_type == "qwen3_5_vision"


def test_mude_readout_and_modes() -> None:
    descriptor = mude.parse_config(FIXTURES / "mude")
    assert descriptor.readout.kind == "code_rows"
    assert descriptor.readout.rows == 255
    assert descriptor.readout.file == "readout.safetensors"
    assert descriptor.readout.temperature == pytest.approx(1.2645396590233329)
    assert descriptor.default_mode is MaskMode.PREFIX_NONCAUSAL
    assert descriptor.supported_modes == frozenset(MaskMode)
    assert descriptor.trained_modes == frozenset({MaskMode.PREFIX_NONCAUSAL})


def test_vision_geometry_is_read_whole() -> None:
    """视觉塔的几何要一次读全：编 program 时少一项，装塔就会分到空缓冲。

    两个出货模型的塔逐项相同，**只有 merger 的输出宽度不同**（2048 对 2560）。塔宽与文本
    `hidden` 是两份契约：这里两个模型碰巧也相等，所以"塔宽 = 文本 hidden"这条捷径在两个
    模型上都看着对，第三个模型就会错。
    """
    mude_spec = mude.parse_config(FIXTURES / "mude").vision
    h2o_spec = h2o.parse_config(FIXTURES / "h2o").vision
    assert mude_spec is not None and h2o_spec is not None
    for spec in (mude_spec, h2o_spec):
        assert spec.depth == 24
        assert spec.hidden == 1024
        assert spec.intermediate_size == 4096
        assert (spec.num_heads, spec.patch_size, spec.spatial_merge) == (16, 16, 2)
        assert spec.temporal_patch_size == 2
        assert spec.num_position_embeddings == 2304  # 48×48 的 pos_embed 表
    assert mude_spec.out_hidden_size == 2048
    assert h2o_spec.out_hidden_size == 2560
    # h2o 的 `vision_config` 没有 `rope_parameters`：这两项落到厂商 `Qwen3_5VisionConfig`
    # 的默认值，而不是 0。取 0 会让所有位置角退化成同一个。
    for spec in (mude_spec, h2o_spec):
        assert spec.rope_theta == pytest.approx(10000.0)
        assert spec.rope_type == "axial"
    # `model_type` 只记录、不分支：两个 config 写的不是同一个值，塔却是同一个。
    assert mude_spec.model_type == "qwen3_5_vision"
    assert h2o_spec.model_type == "qwen3_5"


def test_placeholder_ids_and_mrope_sections_are_read() -> None:
    """合流要的三样：占位 token 的 id、M-RoPE 的三段与是否交错。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    assert (
        descriptor.vision_start_token_id,
        descriptor.vision_end_token_id,
        descriptor.image_token_id,
        descriptor.video_token_id,
    ) == (248053, 248054, 248056, 248057)
    assert descriptor.mrope_section == (11, 11, 10)
    assert descriptor.mrope_interleaved is True
    # 三段之和 = head_dim × partial_rotary_factor / 2 = 256 × 0.25 / 2
    assert sum(descriptor.mrope_section) == 32


def test_a_vision_config_missing_a_shape_key_is_an_error() -> None:
    """缺一项就报错：猜 0 会把错误推迟到装塔时，那时已经看不出是 config 少了一项。"""
    config = json.loads((FIXTURES / "h2o" / "config.json").read_text(encoding="utf-8"))
    del config["vision_config"]["temporal_patch_size"]
    with pytest.raises(ConfigError) as excinfo:
        qwen35.vision_spec(config)
    assert excinfo.value.context["missing"] == ["temporal_patch_size"]


def test_a_tower_without_placeholder_ids_cannot_validate() -> None:
    """带塔却说不出占位 token 的 id：合流时只能靠猜。与线性层几何同一套态度。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    with pytest.raises(ConfigError) as excinfo:
        dataclasses.replace(descriptor, image_token_id=None).validate()
    assert "image_token_id" in str(excinfo.value.context["missing"])


def test_mrope_sections_must_add_up_to_half_the_rotary_axis() -> None:
    """三轴分段与旋转维度对不上时会把行轴的频率灌进列轴 —— 错得很安静，所以在这里挡。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    with pytest.raises(ConfigError) as excinfo:
        dataclasses.replace(descriptor, mrope_section=(11, 11, 11)).validate()
    assert excinfo.value.context["rotary_half"] == 32


def test_a_checkpoint_reports_the_quant_form_it_is_stored_in(tmp_path) -> None:
    """`Provenance.quant` / `inspect` 报的是**目录里存**的形态，不是 config 里的字段。

    不填的话 `QuantSpec()` 的默认值是"未量化"，一张按行审过的 int8 检查点会被报成 bf16
    （票据 15 §26 记的这个口径问题）。判据与 `open_checkpoint` 同一条：有没有 manifest。
    """
    shutil.copy(FIXTURES / "h2o" / "config.json", tmp_path / "config.json")
    assert h2o.parse_config(tmp_path).quant.scheme is Scheme.NONE
    (tmp_path / "int8_manifest.json").write_text('{"tensors": {}}', encoding="utf-8")
    assert h2o.parse_config(tmp_path).quant.scheme is Scheme.PER_ROW_SYM_INT8


def test_h2o_geometry() -> None:
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.num_layers == 32
    assert descriptor.hidden == 2560
    assert descriptor.kv_heads == 4
    assert descriptor.mtp_layers == 1
    assert descriptor.tied_embeddings is True
    assert descriptor.readout.kind == "pooled"
    # 两种读法都支持，但默认是因果：只有因果能分块 prefill（长 prompt 的 KV 不整段常驻）。
    assert descriptor.default_mode is MaskMode.CAUSAL
    assert descriptor.supported_modes == frozenset({MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL})
    assert descriptor.trained_modes == frozenset({MaskMode.CAUSAL, MaskMode.NONCAUSAL_FULL})


def test_torchcast_untied() -> None:
    descriptor = torchcast.parse_config(FIXTURES / "torchcast")
    assert descriptor.tied_embeddings is False
    assert descriptor.readout.kind == "letter_rows"
    assert descriptor.readout.weight == "lm_head.weight"
    assert descriptor.supported_modes == frozenset({MaskMode.NONCAUSAL_FULL})
    # 预热与默认掩码只有一种：它没训练过因果读法，`noncausal_full` 也没有前缀可复用。
    assert descriptor.default_mode is MaskMode.NONCAUSAL_FULL


def test_torchcast_reports_the_quantisation_it_is_stored_in(tmp_path) -> None:
    """与 h2o 同一条判据：目录里有 `int8_manifest.json` 就报 row-int8。

    不报的话 `Provenance` 与 `inspect` 会声称"未量化"，而机器上跑的正是那份 int8 权重。
    """
    shutil.copy(FIXTURES / "torchcast" / "config.json", tmp_path / "config.json")
    assert torchcast.parse_config(tmp_path).quant.scheme is Scheme.NONE
    (tmp_path / "int8_manifest.json").write_text('{"tensors": {}}', encoding="utf-8")
    assert torchcast.parse_config(tmp_path).quant.scheme is Scheme.PER_ROW_SYM_INT8


def test_heads_need_not_cover_hidden() -> None:
    """4B 的 `heads * head_dim = 4096 != hidden 2560`，校验不该要求相等。"""
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.heads * descriptor.head_dim != descriptor.hidden
    descriptor.validate()  # 不抛异常


def test_h2o_linear_geometry_keeps_v_apart_from_k() -> None:
    """4B 的 GDN v 头数是 k 的两倍（32 对 16），descriptor 必须分开记。

    这条不变量是 4B 上引擎的门槛：`value_dim = 4096 ≠ hidden 2560`，任何"v 就是 k 的
    副本"的写法都会把 `in_proj_z`、递归状态与 `cout_gdn` 的宽度算错。
    """
    descriptor = h2o.parse_config(FIXTURES / "h2o")
    assert descriptor.linear_key_heads == 16
    assert descriptor.linear_value_heads == 32  # v 头数是 k 的两倍，2B 上两者相等
    assert (descriptor.linear_key_dim, descriptor.linear_value_dim) == (128, 128)
    # qk_dim 2048、value_dim 4096，**都不等于 hidden 2560**（2B 恰好三者重合，4B 不是）
    assert descriptor.linear_key_heads * descriptor.linear_key_dim == 2048
    assert descriptor.linear_value_heads * descriptor.linear_value_dim == 4096
    assert descriptor.conv_kernel == 4
    assert descriptor.partial_rotary_factor == pytest.approx(0.25)


def test_core_out_is_the_out_projection_input_width() -> None:
    """`core_out` = `o_proj` 的输入行数：2B 恰好等于 hidden，4B 是 4096 ≠ 2560。

    两种层共用**同一套** out program，所以它必须同时等于注意力上下文（`q_heads × head_dim`）
    与 GDN 的 `value_dim`；`Geometry.validate()` 在两者不等时报错。
    """
    mude_geom = Geometry.from_descriptor(mude.parse_config(FIXTURES / "mude"))
    h2o_geom = Geometry.from_descriptor(h2o.parse_config(FIXTURES / "h2o"))
    assert mude_geom.core_out == 2048 == mude_geom.hid
    assert h2o_geom.core_out == 4096 and h2o_geom.hid == 2560
    for geom in (mude_geom, h2o_geom):
        assert geom.core_out == geom.value_dim == geom.q_heads * geom.head_dim


def test_geometry_refuses_two_layer_kinds_with_different_core_widths() -> None:
    """注意力上下文与 GDN 的 value_dim 不等时不能猜：那需要按层类型各编一套 out program。"""
    geometry = Geometry(hid=2560, ffn=9216, q_heads=16, kv_heads=4, head_dim=256,
                        rotary_dim=64, l_k_heads=16, l_k_dim=128, l_v_heads=32,
                        l_v_dim=64, conv_k=4)          # value_dim = 2048
    with pytest.raises(ConfigError) as excinfo:
        geometry.validate()
    assert excinfo.value.context["core_out"] == 4096
    assert excinfo.value.context["value_dim"] == 2048


def test_geometry_projection_matches_mude_hardcoding() -> None:
    """`Geometry.from_descriptor` 投影出的几何要与 `mude_geometry()` 逐项相同。

    这是设备层不再 `import models.families.mude`、改由 descriptor 供货的前提：投影算出的
    每个 `#define` 都得是 mude 今天看到的那个值，否则"纯重构"就不成立。
    """
    assert Geometry.from_descriptor(mude.parse_config(FIXTURES / "mude")) == mude_geometry()


def test_gdn_scratch_row_keeps_the_2b_pitch_and_widens_for_4b() -> None:
    """`k_gdn` 的 scratch 行距：2B 保持 8192 的既有刻度，4B 因为 v 段翻倍而加宽。

    `e8_gdn.metal` 的 `E8G_S_ROW` 用同一套公式算这个数，宿主按它分配 buffer；两边算不
    一致就是越界写。运行时守卫是 `test_metal_gdn*.py` 的对拍，这里钉住数字本身。
    """
    assert mude_geometry().gdn_scratch_halves == 8192
    four_b = Geometry.from_descriptor(h2o.parse_config(FIXTURES / "h2o"))
    assert four_b.gdn_scratch_halves == 9216
    need = 2 * four_b.qk_dim + four_b.value_dim + 2 * four_b.l_v_heads
    assert four_b.gdn_scratch_halves >= need


def test_missing_linear_geometry_is_a_config_error() -> None:
    """有线性层却缺几何时解析期就报错，不留到设备层分配缓冲那一步。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    broken = dataclasses.replace(descriptor, linear_value_heads=None)
    with pytest.raises(ConfigError):
        broken.validate()


def test_linear_geometry_is_not_required_without_linear_layers() -> None:
    """只有全注意力层的模型不需要这四项，`validate()` 不该因为它们缺席而报错。"""
    descriptor = mude.parse_config(FIXTURES / "mude")
    all_full = dataclasses.replace(
        descriptor,
        layer_types=tuple(AttentionRole.FULL for _ in descriptor.layer_types),
        linear_key_heads=None,
        linear_key_dim=None,
        linear_value_heads=None,
        linear_value_dim=None,
    )
    all_full.validate()  # 不抛异常


def test_missing_config_is_config_error(tmp_path) -> None:
    with pytest.raises(ConfigError):
        mude.parse_config(tmp_path)


def test_d3_reads_the_mude_wire_format_on_the_4b_body(tmp_path) -> None:
    """d3-mini 是 d3 线格式的 4B 检查点：255 行码表、`language_model` 前缀、只有双向。

    与 mude 的差别只有几何与掩码声明：读出方案、`norm` 存法（`w + 1`）都与那两份 4B 家族
    相同，所以它共用同一条 `SplitChain`，不需要新的设备路径。
    """
    for module, name in ((d3, "d3"),):
        root = make_code_checkpoint(tmp_path / name)
        descriptor = module.parse_config(root)
        assert descriptor.family == name
        assert (descriptor.num_layers, descriptor.hidden, descriptor.ffn) == (32, 2560, 9216)
        assert descriptor.readout.kind == "code_rows"
        assert (descriptor.readout.weight, descriptor.readout.file) == (
            "weight", "readout.safetensors")
        assert descriptor.readout.rows == 255
        assert descriptor.readout.temperature == pytest.approx(1.1)
        # 只训练过整段双向：因果臂是实验读数，不进能力声明（与 torchcast 同一条判据）
        assert descriptor.supported_modes == frozenset({MaskMode.NONCAUSAL_FULL})
        assert descriptor.trained_modes == frozenset({MaskMode.NONCAUSAL_FULL})
        assert descriptor.default_mode is MaskMode.NONCAUSAL_FULL
        plan = module.weight_plan(root)
        assert plan.prefix == "language_model"
        assert plan.norm_offset == 1.0
        assert plan.descriptor.family == name


def test_a_decision_config_without_the_d3_format_id_is_refused(tmp_path) -> None:
    """d3 线格式的码表函数要认 `format_id`：另一套决策检查点（pplx-decider-flash）同样有 255 个
    码，但 `decision_config.json` 里没有它，读法也不同。缺这一项时必须当场拒绝，而不是把别人的
    码表当成 mude 的用。"""
    root = make_code_checkpoint(tmp_path)
    decision = json.loads((root / "decision_config.json").read_text(encoding="utf-8"))
    del decision["format_id"]
    (root / "decision_config.json").write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        mude.read_codes(root)
    assert caught.value.context["format_id"] is None
