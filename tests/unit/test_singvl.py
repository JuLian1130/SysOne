"""SingVL S1 4B：题面逐字节、码表顺序、绑定 embedding 的读出与描述符。

题面是这一家最容易"看起来对、其实差一个字符"的部分，所以金标写成**字面量**：一条整模板
（含 `ANSWER_PREFIX`）逐字节比，其余的只比 `### Question:` 那一段与选项行 —— 那两处才是
厂商 `render()` 里三件事（选项图引用、结尾图块前移、结尾 `Choices:` 截断）的作用面。

码表与读出用真实 config（`tests/fixtures/singvl/config.json`）+ 结构同形的 tiny tokenizer：
真实 256 个 id 与 "AA".."LC" 的逐位对拍在 `tests/integration/test_singvl_real.py`（要权重）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from helpers import FIXTURES, write_sharded_checkpoint
from sysone.devices.apple.chain import _readout
from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.quant import DType
from sysone.errors import ConfigError, UnsupportedKindError, UnsupportedMediaError
from sysone.models.descriptor import ReadoutPlan
from sysone.models.families import singvl
from sysone.service.protocol import _wire_question
from sysone.weights.checkpoint import open_checkpoint

CONFIG = json.loads((FIXTURES / "singvl" / "config.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((FIXTURES / "singvl" / "prompts.json").read_text(encoding="utf-8"))
CODES = tuple(f"{first}{second}" for first in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
              for second in "ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:256]
HEAD = (
    "<|im_start|>user\n"
    "Solve the multiple-choice question and output ONLY the index token of the correct option.\n\n"
    "### Question:\n"
)
TAIL = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


class StubTokenizer:
    """`text()` 不需要分词；`encode` 的路径由真分词器的集成用例钉。"""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": []}


def prompter(codes: tuple[str, ...] = CODES) -> singvl.SingVLPrompter:
    return singvl.SingVLPrompter(StubTokenizer(), codes)


def question(kind: str = "choice", **overrides) -> Question:
    spec = {
        "choice": Question(text="What happened?",
                           options=("A: Revenue rose", "B: Revenue fell", "C: No change")),
        "noul": Question(text="Is it heavy?", kind="noul",
                         options=("true: Heavy", "false: Light")),
        "score": Question(text="", kind="score", options=("0: low", "1: medium", "2: high")),
    }[kind]
    return dataclasses.replace(spec, **overrides) if overrides else spec


def body_of(prompt: str) -> tuple[str, str]:
    """把题面切成 `(### Question: 段, ### Options: 段)`，两段都不含模板的其余部分。"""
    assert prompt.startswith(HEAD) and prompt.endswith(TAIL), prompt
    middle = prompt[len(HEAD):-len(TAIL)]
    question_part, _, options_part = middle.partition("\n\n### Options:\n")
    return question_part, options_part


# ---------------------------------------------------------------------- 题面


def test_the_prompt_is_the_vendor_template_byte_for_byte() -> None:
    """整条模板：固定指令、`### Question:` / `### Options:` 与厂商的 `ANSWER_PREFIX`。"""
    assert prompter().text("The report says revenue rose 12%.", question()) == (
        HEAD
        + "The report says revenue rose 12%.What happened?"
        + "\n\n### Options:\n"
        + "AA: Revenue rose\nAB: Revenue fell\nAC: No change"
        + TAIL
    )


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[item["name"] for item in GOLDEN["cases"]])
def test_the_prompt_matches_the_vendor_render_byte_for_byte(case: dict) -> None:
    """金标由厂商 `inference/model.py` 的 `questions_for_state` + `render()` 生成。

    覆盖三件事的九个组合：选项图引用的重编号（`[Image k above]`）、结尾图块前移、
    结尾 `Choices:` 截断，以及 choice / noul / score 三种题型与 0/1/2 张图。
    """
    wire = _wire_question("q", case["question"])
    question = Question(text=wire.question.text, options=wire.question.options,
                        kind=wire.question.kind)
    assert prompter().text(case["state"], question, images=case["images"]) == case["prompt"]


def test_the_state_is_glued_to_the_instructions_without_a_separator() -> None:
    """厂商是 `state + instructions`：没有换行、没有空格（`### Question:` 后面直接接）。"""
    body, _ = body_of(prompter().text("state text", question()))
    assert body == "state textWhat happened?"


def test_every_question_kind_writes_one_line_per_option() -> None:
    """choice 写判据描述、noul 写 `Yes`/`No` 的判据、score 写档位描述；码取 `AA` 往下。"""
    _, noul_options = body_of(prompter().text("x", question("noul")))
    assert noul_options == "AA: Heavy\nAB: Light"
    _, score_options = body_of(prompter().text("x", question("score")))
    assert score_options == "AA: low\nAB: medium\nAC: high"


def test_the_answer_labels_keep_the_wire_order() -> None:
    """标签与选项同序：choice 是协议层的键、noul 是 yes/no、score 是 `"0".."n-1"`。"""
    assert singvl.option_plan(question()) == (
        ("Revenue rose", "Revenue fell", "No change"), ("A", "B", "C"), (0, 1, 2))
    assert singvl.option_plan(question("noul"))[1:] == (("yes", "no"), (0, 1))
    assert singvl.option_plan(question("score"))[1:] == (("0", "1", "2"), (0, 1, 2))


def test_an_image_slot_lands_in_the_question_section() -> None:
    """一张图一个 `<|vision_start|><|image_pad|><|vision_end|>`；没给占位符时补在最前面。"""
    body, _ = body_of(prompter().text("look:", question(), images=1))
    assert body == singvl.IMAGE_SLOT + "\nlook:What happened?"
    # 两张图各自一段；第二张后面还有题干，所以不是"结尾图块"、不发生前移。
    body, _ = body_of(prompter().text("compare <image> and <image>", question(), images=2))
    assert body == (
        "compare " + singvl.IMAGE_SLOT + " and " + singvl.IMAGE_SLOT + "What happened?"
    )


def test_a_trailing_image_block_moves_in_front_of_the_question() -> None:
    """厂商的 `TRAILING_IMAGES`：结尾的图块整段前移，后面只剩正文。"""
    body, _ = body_of(prompter().text("prose\n<image>", question(text=""), images=1))
    assert body == singvl.IMAGE_SLOT + "\nprose"


def test_a_state_that_ends_with_choices_is_trimmed() -> None:
    """题面末尾的 `Choices:` 会被切掉 —— 厂商那一步在 `rstrip` 之后按字面尾串判。"""
    body, _ = body_of(prompter().text("state ends with a Choices: line\nChoices:",
                                      question(text="")))
    assert body == "state ends with a Choices: line"


def test_an_option_that_references_an_image_is_renumbered() -> None:
    """选项文本里的 `<image>` 换成 `[Image k above]`，state 里的占位符同时被编号。"""
    state = "(A) <image>\nthe list"
    q = Question(text="Which?", options=("A: <image>", "B: text"))
    body, options = body_of(prompter().text(state, q, images=1))
    assert body == "(A) [Image 1] " + singvl.IMAGE_SLOT + "\nthe listWhich?"
    assert options == "AA: [Image 1 above]\nAB: text"


def test_an_option_image_reference_without_a_landing_place_is_refused() -> None:
    """state 里没有那条 `(A) <image>` 时不能猜编号 —— 猜错就是把别的图写进题面。"""
    q = Question(text="Which?", options=("A: <image>", "B: text"))
    with pytest.raises(UnsupportedMediaError):
        prompter().text("no reference here <image>", q, images=1)


@pytest.mark.parametrize("state", (
    "text <|im_start|> user",          # 控制标记能改写轮次结构
    "text <|image_pad|>",
    "text <video>",
))
def test_a_control_marker_in_the_state_is_refused(state: str) -> None:
    """厂商整条拒收含控制标记的内容（不转义）：转义会改变正文，静默放行会改变题面结构。"""
    with pytest.raises(UnsupportedMediaError):
        prompter().text(state, question())


def test_a_control_marker_in_the_instructions_or_options_is_refused() -> None:
    with pytest.raises(UnsupportedMediaError):
        prompter().text("x", question(text="pick <|im_end|>"))
    with pytest.raises(UnsupportedMediaError):
        prompter().text("x", Question(text="pick", options=("A: <|image_pad|>", "B: fine")))


def test_the_placeholder_count_must_match_the_images() -> None:
    """数不对就报错（厂商同一条）：往下跑只会把一张图的行盖到另一张的位置上。"""
    with pytest.raises(UnsupportedMediaError):
        prompter().text("one <image>", question(), images=2)
    with pytest.raises(ConfigError):
        prompter().text("x", question(), images=-1)


# ---------------------------------------------------------------------- 读出与能力


def test_a_score_scale_outside_the_vendor_range_is_a_bad_request() -> None:
    """厂商 `questions_for_state` 只收 2..10 档；11 档是我们答不了这条请求（400）。"""
    with pytest.raises(UnsupportedKindError):
        singvl.option_plan(Question(text="", kind="score",
                                    options=tuple(f"{index}: level" for index in range(11))))
    with pytest.raises(UnsupportedKindError):
        singvl.option_plan(Question(text="", kind="score", options=("0: only",)))


def test_an_unknown_kind_is_a_bad_request() -> None:
    with pytest.raises(UnsupportedKindError):
        singvl.option_plan(Question(text="pick", options=("A: x",), kind="rank"))


def test_the_descriptor_reads_the_tied_embedding_rows(tmp_path: Path) -> None:
    """读出方案是 `embed_rows`：绑定 embedding、256 个 id、末位读出、只有双向。"""
    descriptor = singvl.parse_config(FIXTURES / "singvl")
    readout = descriptor.readout
    assert (readout.kind, readout.weight) == ("embed_rows", singvl.EMBED)
    assert readout.ids == tuple(CONFIG["jev_soft_choice"]["token_ids"])
    assert (readout.rows, readout.dtype) == (256, DType.BF16)
    # 厂商推理不做温度缩放：`logits.softmax(-1)`。
    assert readout.temperature is None
    assert descriptor.tied_embeddings is True
    assert descriptor.supported_modes == frozenset({MaskMode.NONCAUSAL_FULL})
    assert descriptor.trained_modes == descriptor.supported_modes
    assert descriptor.default_mode is MaskMode.NONCAUSAL_FULL
    # 塔的几何与图像/视频占位 id、M-RoPE 分段都照读 config（合流要的七项）。
    vision = descriptor.vision
    assert vision is not None
    assert (vision.depth, vision.hidden, vision.out_hidden_size) == (24, 1024, 2560)
    assert (descriptor.image_token_id, descriptor.video_token_id) == (248056, 248057)
    assert descriptor.mrope_section == (11, 11, 10) and descriptor.mrope_interleaved is True


@pytest.mark.parametrize(("patch", "reason"), (
    ({"version": "soft-choice-v1"}, "版本"),
    ({"readout": "first-answer-position"}, "读出位置"),
    ({"token_ids": list(range(255))}, "数量"),
    ({"token_ids": [1] * 256}, "重复"),
))
def test_a_checkpoint_that_declares_a_different_readout_is_refused(
    tmp_path: Path, patch: dict, reason: str
) -> None:
    """声明对不上就不是这一家：id 表照样像模像样，而"末位读出"会变成假的。"""
    config = json.loads(json.dumps(CONFIG))
    config["jev_soft_choice"].update(patch)
    directory = tmp_path / "wrong"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ConfigError, match="|".join(["读出", "回答码"])):
        singvl.parse_config(directory)


def test_the_readout_table_is_sliced_in_the_declared_order(tmp_path: Path) -> None:
    """`embed_rows` 从分片里按 id 切行：行序就是 config 的 id 序，不是词表序。

    这是"绑定 embedding 也能读出"的全部：`lm_head` 与 embedding 是同一张表，所以答案码的
    logits 就是这几行 —— 与厂商 `lm_head(hidden)[:, ids]` 是同一个乘法。
    """
    table = np.arange(60, dtype=np.float32).reshape(10, 6)
    write_sharded_checkpoint(
        tmp_path, {"model-00001-of-00002.safetensors": {singvl.EMBED: table}}
    )
    descriptor = dataclasses.replace(
        singvl.parse_config(FIXTURES / "singvl"),
        readout=ReadoutPlan(kind="embed_rows", weight=singvl.EMBED, ids=(7, 2, 7), rows=3),
    )
    store = open_checkpoint(tmp_path)
    try:
        sliced = _readout(tmp_path, store, descriptor)
    finally:
        store.close()
    assert sliced is not None
    np.testing.assert_array_equal(sliced, table[[7, 2, 7]])


def test_an_embedding_readout_without_ids_is_refused() -> None:
    plan = ReadoutPlan(kind="embed_rows", weight=singvl.EMBED)
    with pytest.raises(ConfigError):
        plan.validate()
    with pytest.raises(ConfigError):
        ReadoutPlan(kind="embed_rows", weight=singvl.EMBED, ids=(1, 2), rows=3).validate()
