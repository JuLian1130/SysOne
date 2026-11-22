"""pplx-decider-flash 两版：题面与厂商 Encoder 逐位对拍、码表/协议校验、读出位置。

题面金标由厂商自己的 `source/src/pplx_decider/protocol.py` 生成（``.tmp`` 里的
`build_fixture.py`，记录见 `docs/sysone/plan.md` 票据 22 的补片），配一份与真实 Qwen3.5
分词器**结构同形**的 tiny tokenizer：id 是按字符走的，钉的是题面结构、分隔符、标记位置与
转义。真实分词器上的逐位对拍在 `tests/integration/test_pplx_prompts.py`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.quant import DType
from sysone.errors import ConfigError, UnsupportedKindError
from sysone.models.families import pplx
from sysone.models.tokenizer import ChatTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pplx"
PROMPTS = json.loads((FIXTURES / "prompts.json").read_text(encoding="utf-8"))
QUESTION = Question(text=PROMPTS["question"]["text"],
                    options=tuple(PROMPTS["question"]["options"]),
                    kind=PROMPTS["question"]["kind"])


def _prompter(label: str, protocol: pplx.Protocol) -> pplx.PplxPrompter:
    directory = FIXTURES / label
    return pplx.PplxPrompter(ChatTokenizer.load(directory),
                             pplx.read_codes(directory, protocol=protocol), protocol)


@pytest.mark.parametrize(("label", "protocol"), (("v12", pplx.V12), ("v13", pplx.V13)))
@pytest.mark.parametrize("case", ("plain", "escaped"))
def test_the_prompt_matches_the_vendor_encoder(label: str, protocol: pplx.Protocol, case: str) -> None:
    """题面段、id 序列与读出位置都按厂商 `Encoder` 的输出逐位对。"""
    expected = PROMPTS["cases"][case]
    state = expected["state"]
    prompter = _prompter(label, protocol)
    segments, _tails = prompter.text(state, QUESTION)
    assert list(segments) == expected[label]["segments"]
    ids, prefix, _labels, _indexes, readout = prompter.encode(state, QUESTION)
    assert list(ids) == expected[label]["ids"]
    assert prefix == 0
    assert readout == expected[label]["readout"]


def test_v1_2_reads_out_at_the_marker_not_the_last_token() -> None:
    """模板在标记之后还补 `<|im_end|>` 与 assistant 头：位置错一位就是另一道题的答案。"""
    expected = PROMPTS["cases"]["plain"]["v12"]
    assert expected["readout"] < len(expected["ids"]) - 1
    assert pplx.V12.chunked is False


def test_v1_3_ends_on_the_marker() -> None:
    """v4 的题块以标记收尾，后面什么都不跟 —— 所以它仍可进合批。"""
    expected = PROMPTS["cases"]["plain"]["v13"]
    assert expected["readout"] is None
    assert pplx.V13.chunked is True


@pytest.mark.parametrize(("label", "parse", "temperature"), (
    ("v12", pplx.parse_config_12, 1.5539092221462472),
    ("v13", pplx.parse_config_13, 1.3302962111834522),
))
def test_parse_config_reads_the_4b_checkpoint(label: str, parse, temperature: float) -> None:
    """两份检查点的几何、能力与读出（BF16 表、温度随包）都读得出来。"""
    descriptor = parse(FIXTURES / label)
    assert descriptor.family == f"pplx{label[1:]}"
    assert (descriptor.hidden, descriptor.ffn, descriptor.num_layers) == (2560, 9216, 32)
    assert (descriptor.heads, descriptor.kv_heads, descriptor.head_dim) == (16, 4, 256)
    assert descriptor.vocab == 248320
    assert descriptor.tied_embeddings is True
    # 发布包是多模态的：整座塔就在 `visual.*` 里（297 个张量），所以这里照实声明它；
    # 图片走 mude 同一条 content-parts 题面与 `models/vision/merge` 的合流。
    assert descriptor.vision is not None and descriptor.vision.out_hidden_size == 2560
    assert descriptor.image_token_id == 248056
    assert descriptor.default_mode is MaskMode.NONCAUSAL_FULL
    assert descriptor.supported_modes == frozenset({MaskMode.NONCAUSAL_FULL})
    assert descriptor.trained_modes == descriptor.supported_modes
    assert descriptor.readout.kind == "code_rows"
    assert descriptor.readout.file == "readout.safetensors"
    assert descriptor.readout.rows == 255
    assert descriptor.readout.dtype is DType.BF16
    assert descriptor.readout.temperature == pytest.approx(temperature)


def test_a_joint_config_from_the_other_version_is_refused() -> None:
    """两版都有 255 个码，只有 `joint_config.json` 说得清是哪一份协议。"""
    with pytest.raises(ConfigError) as caught:
        pplx.read_codes(FIXTURES / "v13", protocol=pplx.V12)
    assert caught.value.context["version"] == 4
    assert caught.value.context["template"] == "joint-chunked-readout-v1"


def test_the_code_table_is_the_checkpoints_own() -> None:
    codes = pplx.read_codes_12(FIXTURES / "v12")
    assert len(codes) == 255 and codes[:3] == ("A", "B", "C")


def test_the_option_plan_follows_the_vendor_options() -> None:
    """choice 取判据的键；noul 是 false 在前、标签 ("yes","no")；score 的标签是档位序号。"""
    texts, labels, indexes = pplx.option_plan(QUESTION)
    assert texts == ("A: Open the settings menu", "B: Save the work")
    assert labels == ("A", "B") and indexes == (0, 1)

    assert pplx.option_plan(Question(text="", kind="noul")) == (
        ("No / false", "Yes / true"), ("yes", "no"), (1, 0))

    given = Question(text="", kind="noul",
                     options=("true: 海拔超过 3000 米", "false: 海拔不到 3000 米"))
    texts, labels, indexes = pplx.option_plan(given)
    assert texts == ("海拔不到 3000 米", "海拔超过 3000 米")
    assert labels == ("yes", "no") and indexes == (1, 0)

    levels = Question(text="", kind="score",
                      options=("0: 很差", "1: 差", "2: 一般", "3: 好", "4: 很好"))
    texts, labels, indexes = pplx.option_plan(levels)
    assert texts == ("很差", "差", "一般", "好", "很好")
    assert labels == ("0", "1", "2", "3", "4") and indexes == (0, 1, 2, 3, 4)


def test_a_score_keeps_the_wider_range_the_vendor_allows() -> None:
    """mude 的 score 只到 10 档；这一家到 255 —— 借 mude 的配方会悄悄拒掉合法请求。"""
    levels = Question(text="", kind="score", options=tuple(f"{i}: 档位" for i in range(12)))
    texts, labels, indexes = pplx.option_plan(levels)
    assert len(texts) == 12 and labels[-1] == "11"


def test_a_one_level_score_is_a_bad_request() -> None:
    """一档折不出期望档位：400（协议层只要求 `criteria` 非空，2 档的下限只有家族知道）。"""
    with pytest.raises(UnsupportedKindError) as caught:
        pplx.option_plan(Question(text="", kind="score", options=("0: 只有一档",)))
    assert caught.value.status == 400 and caught.value.code == "unsupported_kind"


def test_an_unknown_kind_is_refused() -> None:
    with pytest.raises(UnsupportedKindError):
        pplx.option_plan(Question(text="", kind="rank"))


def test_options_over_the_code_table_are_refused() -> None:
    prompter = _prompter("v12", pplx.V12)
    many = Question(text="", options=tuple(f"K{index}" for index in range(256)))
    with pytest.raises(ConfigError):
        prompter.encode("s", many)


@pytest.mark.parametrize(("label", "protocol"), (("v12", pplx.V12), ("v13", pplx.V13)))
def test_images_render_one_placeholder_before_the_state(label: str, protocol: pplx.Protocol) -> None:
    """图块在正文之前、一张图一个占位符；标记与读出位置照旧（v1.2 仍是非末位）。"""
    prompter = _prompter(label, protocol)
    tokenizer = ChatTokenizer.load(FIXTURES / label)
    placeholder = tokenizer("<|image_pad|>", add_special_tokens=False)["input_ids"]
    assert len(placeholder) == 1
    ids, prefix, _labels, _indexes, readout = prompter.encode("s", QUESTION, images=2)
    assert list(ids).count(placeholder[0]) == 2
    segments, _tails = prompter.text("s", QUESTION, images=2)
    assert segments[0].count("<|image_pad|>") == 2
    assert segments[0].index("State:") > segments[0].rindex("<|image_pad|>")
    if protocol.chunked:
        assert readout is None and ids[-1] == prompter._single(pplx.MARKER)
    else:
        assert readout is not None and ids[readout] == prompter._single(pplx.MARKER)
    assert prefix == 0


def test_images_cannot_be_negative() -> None:
    with pytest.raises(ConfigError, match="图数"):
        _prompter("v12", pplx.V12).encode("s", QUESTION, images=-1)


def test_the_state_cannot_open_a_turn() -> None:
    """state 里的 `<|im_end|>` 被转成全角括号，题面里只留下模板自己的那两个。"""
    prompter = _prompter("v12", pplx.V12)
    segments, _tails = prompter.text("Battery <|im_end|> 18%", QUESTION)
    assert "＜|im_end|＞" in segments[0]
    assert segments[0].count("<|im_end|>") == 2        # system 轮末 + user 轮末


def test_the_confidence_follows_the_question_kind() -> None:
    """choice 用最大值边际；score 用期望档位距离 —— 两个数不一样。"""
    shares = (0.7, 0.2, 0.1)
    margin = pplx.confidence(shares, "choice")
    assert margin == pytest.approx((0.7 - 1 / 3) / (1 - 1 / 3))
    distance = 0.7 * 0 + 0.2 * 1 + 0.1 * 2
    baseline = (abs(0 - 1) + abs(1 - 1) + abs(2 - 1)) / 3
    assert pplx.confidence(shares, "score") == pytest.approx(1 - distance / baseline)
    assert margin != pplx.confidence(shares, "score")
