"""torchcast 的答面：题面与判答逐字节等于厂商那次渲染 —— 金标由厂商模块生成一次。

`torchcast_decision/render.py` 与 `model.py` 是随 checkpoint 发布的厂商文件（Apache-2.0），推理
路径**不 import** 它们（`model.py` 顶层 import torch）；
`tests/fixtures/torchcast_vendor_prompts.json` 是把它们跑一次存下来的结果，生成脚本见
`.scratch/torchcast_vendor_golden.py`。

每个题面用例带三样东西：state、请求里那道题原样、厂商 `render.user_text()` 的输出；这里用出货
路径（`question()` → `user_text()`）重算并逐字节比。答面同理：把一组概率喂给厂商的
`TorchcastDecision._answer()`，与 `Serve._answer()` 的结果比。

这两条以前会是自指的（金标用我们自己的渲染器生成），换 checkpoint 或厂商改题面时不会红。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sysone.errors import (
    ConfigError,
    RequestTooLongError,
    UnsupportedKindError,
    UnsupportedMediaError,
)
from sysone.models.vendors import torchcast_serve
from sysone.models.vendors.torchcast_serve import (
    BUDGET,
    LETTERS,
    Serve,
    chat_text,
    letter_ids,
    question,
    state_as_text,
    user_text,
)

# 真实 checkpoint 的 `image_token_id`（`config.json`）：图块中间那个 token。
IMAGE_TOKEN_ID = 248056

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "torchcast_vendor_prompts.json").read_text(
        encoding="utf-8"
    )
)
LETTER_IDS = tuple(FIXTURE["letter_token_ids"])


class FakeTokenizer:
    """只够这条答面用：字母是独立 token（与 `decision_config.json` 一致），模板以思考关结尾。

    模板套用的原文要能被逐字比对，所以这里把消息直接拼成一段可读文本。
    """

    # 图块那三个是**原子** token：真分词器上它们各是一个 id（`added_tokens`）。
    specials = {"<|vision_start|>": 248053, "<|image_pad|>": IMAGE_TOKEN_ID,
                "<|vision_end|>": 248054}

    def __init__(self, letters: tuple[int, ...] = LETTER_IDS) -> None:
        self.letters = dict(zip(LETTERS, letters))
        self.calls: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool):
        """逐字符成 token：大写的 A..Z 用 `decision_config.json` 的 id，其余用 `ord`。

        于是 `题面 + "A"` 恰好是 `题面 id + [字母 id]` —— 与真分词器在同一条性质上。
        """
        assert add_special_tokens is False
        self.calls.append(text)
        ids: list[int] = []
        rest = text
        while rest:
            for token, value in self.specials.items():
                if rest.startswith(token):
                    ids.append(value)
                    rest = rest[len(token):]
                    break
            else:
                char, rest = rest[0], rest[1:]
                if char.strip():
                    ids.append(self.letters.get(char, ord(char)))
        return {"input_ids": ids}

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                            **kwargs):
        assert tokenize is False and add_generation_prompt is True
        assert kwargs["enable_thinking"] is False
        body = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return body + "\n" + torchcast_serve.THINKING_OFF


def config(**overrides) -> dict:
    base = {
        "prompt_format": "jev_render_v3",
        "attention_mode": "noncausal_full_attention",
        "temperature_by_type": {"choice": 0.9, "noul": 0.9, "score": 0.9},
        "noul_floor": 0.801,
        "letter_token_ids": list(LETTER_IDS),
    }
    base.update(overrides)
    return base


def serve(**overrides) -> Serve:
    return Serve(config(**overrides), FakeTokenizer())


# ------------------------------------------------------------------- 题面
def test_the_fixture_carries_the_whole_letter_table() -> None:
    """读出表是 A..Z 的 26 行 `lm_head`：少一个字母，`letter_rows` 就取不满。"""
    assert len(LETTER_IDS) == 26
    assert len(set(LETTER_IDS)) == 26


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_the_question_text_matches_the_vendor_byte_for_byte(case: dict) -> None:
    item = question(case["state"], case["spec"])
    assert item.kind == case["kinds"]
    assert item.ids == case["ids"]
    assert user_text(item) == case["user"]


def test_a_long_list_gets_an_index_like_the_vendor() -> None:
    """8 项起的列表每项补 `_index`：与厂商同一条规则，少这一层 35 道 dict state 的题会漂。"""
    text = state_as_text({"rows": [{"id": index} for index in range(9)]})
    assert '"_index"' in text
    assert state_as_text({"rows": [{"id": index} for index in range(7)]}).count(
        '"_index"') == 0


def test_an_empty_state_reads_as_none_and_a_string_state_is_verbatim() -> None:
    assert user_text(question("", {"type": "noul", "instructions": "q"})).startswith(
        "Evidence:\n(none)\n")
    assert state_as_text("  空格原样  ") == "  空格原样  "


# ------------------------------------------------------------------- 答面
def _same_shape(ours: object, theirs: object, rel: float = 1e-6) -> None:
    """逐项比：数用相对误差（我们这条路的概率是 float32），其余逐值相等。"""
    if isinstance(theirs, dict):
        assert isinstance(ours, dict) and set(ours) == set(theirs)
        for key in theirs:
            _same_shape(ours[key], theirs[key], rel)
    elif isinstance(theirs, (int, float)) and not isinstance(theirs, bool):
        assert ours == pytest.approx(theirs, rel=rel)
    else:
        assert ours == theirs


@pytest.mark.parametrize("probe", FIXTURE["answers"], ids=lambda probe: probe["name"])
def test_the_answer_object_matches_the_vendor(probe: dict) -> None:
    subject = serve()
    prepared = subject.prepare(probe["state"], probe["spec"])
    ours = subject._answer(prepared, np.asarray(probe["probabilities"], dtype=np.float32))
    theirs = probe["answer"]
    if theirs["type"] == "score":
        # 只省 `legend`：它是请求里档位文本的回声，而两家对非字符串档位的写法不同（与 mude
        # 那一侧同一个理由）。期望值与置信度照比。
        assert set(theirs) - set(ours) == {"legend"}
        theirs = {key: value for key, value in theirs.items() if key != "legend"}
    _same_shape(ours, theirs)


def test_the_wire_answer_carries_vendor_labels_and_confidence() -> None:
    """答面交给引擎的是 (判词, 标签, 概率, 置信度)：标签是**选项 id**，字母只是题面记号。"""
    subject = serve()
    prepared = subject.prepare("record", {
        "type": "choice", "instructions": "Which?",
        "criteria": {"payments": "Payments", "web": None},
    })
    # 温度 0.9 下 softmax([2, 0])：把 logits 喂进去，判词跟概率走
    value, labels, probs, confidence = subject.answer(prepared, [2.0, 0.0])
    assert value == "payments"
    assert labels == ("payments", "web")
    # softmax([2 / 0.9, 0]) —— 温度来自 decision_config，不在这里重算
    assert probs[0] == pytest.approx(0.9022274, rel=1e-6)
    assert confidence == pytest.approx(0.8044548, rel=1e-6)


def test_a_logit_count_that_does_not_match_the_labels_is_rejected() -> None:
    subject = serve()
    prepared = subject.prepare("record", {
        "type": "choice", "instructions": "Which?", "criteria": {"a": "A", "b": "B"}})
    with pytest.raises(Exception):
        subject.answer(prepared, [1.0])


# ------------------------------------------------------------------- 边界
def test_more_than_26_options_is_refused_instead_of_grouped() -> None:
    """厂商在 27 个选项上分两趟（`wide_choice`）；那条路的概率合并没有在真权重上对过。"""
    criteria = {f"opt{index}": f"option {index}" for index in range(27)}
    with pytest.raises(UnsupportedKindError):
        question("record", {"type": "choice", "instructions": "Which?", "criteria": criteria})


def test_a_score_question_over_ten_levels_is_refused() -> None:
    criteria = [f"level {index}" for index in range(11)]
    with pytest.raises(UnsupportedKindError):
        question("record", {"type": "score", "instructions": "How bad?", "criteria": criteria})


def test_a_prompt_over_the_budget_is_refused_rather_than_truncated() -> None:
    """厂商截 state 的首尾拼一行 `...`；我们不截（重新切 token 的边界不可能与厂商一致）。"""
    subject = serve()
    big = "x" * (BUDGET + 16)
    with pytest.raises(RequestTooLongError):
        subject.prepare(big, {"type": "choice", "instructions": "Which?",
                              "criteria": {"a": "A", "b": "B"}})


# --------------------------------------------------------------------- 图
@pytest.mark.parametrize("case", FIXTURE["images"], ids=lambda case: case["name"])
def test_the_image_block_lands_where_the_vendor_puts_it(case: dict) -> None:
    """图块的位置与厂商 `_state_with_images` 逐字节相同（金标由厂商那句生成一次）。

    三种情形都在里面：没有记号（图在前，多图是 `Image k:` 逐行）、记号数对得上（图跟着文字
    走）、以及 dict state（记号在压缩后的 JSON 里）。
    """
    assert torchcast_serve.with_images(case["state"], case["count"]) == case["text"]


def test_a_mark_count_that_disagrees_with_the_images_is_refused() -> None:
    """记号与图各说各话时不能"照旧覆盖"：那会把一张图的行盖到另一张的位置上。"""
    with pytest.raises(UnsupportedMediaError):
        torchcast_serve.with_images("one <image> only", 2)


def test_prepare_keeps_exactly_one_placeholder_per_image() -> None:
    """展开成 `counts[i]` 行是链的事（`merge.attach`）；这条面只放**一个**占位符。

    放两个就会让 `expand_image_tokens` 数不对而报错，或者更糟：两个占位符各取一半行。
    """
    subject = serve()
    prepared = subject.prepare("record", {"type": "choice", "instructions": "q",
                                          "criteria": {"a": "A", "b": "B"}}, images=2)
    assert subject.supports_images is True
    assert list(prepared.ids).count(IMAGE_TOKEN_ID) == 2
    # 两张图的题面比一张长两行左右的 `Image k: …`，且都以图块开头
    single = subject.prepare("record", {"type": "choice", "instructions": "q",
                                        "criteria": {"a": "A", "b": "B"}}, images=1)
    assert len(prepared.ids) > len(single.ids)


def test_a_negative_image_count_is_refused() -> None:
    subject = serve()
    with pytest.raises(UnsupportedMediaError):
        subject.prepare("record", {"type": "choice", "instructions": "q",
                                   "criteria": {"a": "A", "b": "B"}}, images=-1)


def test_the_letters_must_match_decision_config() -> None:
    """厂商构造时核对分词器的字母与 `decision_config.json`：错一个字母整张读出表就错位。"""
    shifted = tuple(value + 1 for value in LETTER_IDS)
    with pytest.raises(ConfigError):
        Serve(config(), FakeTokenizer(letters=shifted))


def test_a_foreign_prompt_format_or_attention_mode_is_refused() -> None:
    with pytest.raises(ConfigError):
        serve(prompt_format="jev_render_v2")
    with pytest.raises(ConfigError):
        serve(attention_mode="causal")


def test_the_chat_template_must_end_with_thinking_off() -> None:
    """模板悄悄带上思考前缀时题面 token 会整体漂移，必须当场报配置错。"""

    class BadTemplate(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "no thinking prefix here"

    with pytest.raises(ConfigError):
        chat_text(BadTemplate(), [{"role": "user", "content": "x"}])


def test_letter_ids_are_read_off_the_tokenizer() -> None:
    assert letter_ids(FakeTokenizer()) == LETTER_IDS


def test_open_needs_the_vendor_decision_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        Serve.open(tmp_path, tokenizer=FakeTokenizer())
