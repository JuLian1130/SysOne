"""torchcast 的图块在真分词器上落成一个占位符（要真 checkpoint，缺了就跳过）。

纯逻辑那几条在 `tests/unit/test_torchcast_serve.py`（金标由厂商 `_state_with_images` 生成）；
这里钉的是**分词**这一层：字符串拼接出的图块与模板为结构化 content 渲染出的，必须是同一串
token —— 否则"我们的题面"与"厂商训练时的题面"在图上是两回事，而答案看着仍然正常。

塔与链在门里（`.scratch/probe_torchcast_vision.py`，真图、MPS、26 份产物）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from sysone.models.tokenizer import ChatTokenizer
from sysone.models.vendors import torchcast_serve
from sysone.models.vendors.torchcast_serve import question, user_text, with_images

WEIGHTS = Path(os.environ.get("SYS1_TORCHCAST_WEIGHTS", ""))
STATE = "CLAIM FILE HX-2026-118804 — water damage at the kitchen ceiling."
SPEC = {
    "type": "choice",
    "instructions": "Which outcome fits the file?",
    "criteria": {"deny": "Repeated seepage is excluded", "pay": "One covered event"},
}


def _image_token() -> int:
    """`config.json` 的 `image_token_id`：占位符那个 token，链据此展开。"""
    return int(json.loads((WEIGHTS / "config.json").read_text(encoding="utf-8"))["image_token_id"])


def _skip_reason() -> str | None:
    if not (WEIGHTS / "tokenizer.json").is_file():
        return f"缺少真实分词器：SYS1_TORCHCAST_WEIGHTS={WEIGHTS}"
    return None


pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="只在 macOS 上跑（占位 token 来自真分词器）"),
    pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or ""),
]


@pytest.fixture(scope="module")
def tokenizer() -> ChatTokenizer:
    return ChatTokenizer.load(WEIGHTS)


def _our_ids(tokenizer: ChatTokenizer, images: int) -> list[int]:
    item = question(with_images(STATE, images), SPEC)
    text = torchcast_serve.chat_text(tokenizer, torchcast_serve.chat(item))
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _template_ids(tokenizer: ChatTokenizer, images: int) -> list[int]:
    """模板那条路：结构化 content + 厂商的占位符展开（一张图一个 `image_pad`，先不展开）。"""
    text = user_text(question(with_images(STATE, images), SPEC))
    pieces = text.split(torchcast_serve.VISION_BLOCK)
    content: list[dict] = []
    for index, piece in enumerate(pieces):
        if index:
            content.append({"type": "image"})
        if piece:
            content.append({"type": "text", "text": piece})
    messages = [{"role": "system", "content": torchcast_serve.SYSTEM},
                {"role": "user", "content": content}]
    return list(tokenizer(
        torchcast_serve.chat_text(tokenizer, messages), add_special_tokens=False)["input_ids"])


@pytest.mark.parametrize("images", [1, 2, 3])
def test_the_string_form_renders_the_same_tokens_as_the_template(tokenizer: ChatTokenizer,
                                                                images: int) -> None:
    """厂商把图交给模板展开，我们拼字符串 —— 两条路必须给出同一串 id。

    差一个 token（比如模板多加了 "Picture k:"）就足以让整条题面错位，而答案仍然"看起来正常"。
    """
    assert _our_ids(tokenizer, images) == _template_ids(tokenizer, images)


@pytest.mark.parametrize("images", [1, 2, 3])
def test_each_image_is_exactly_one_placeholder(tokenizer: ChatTokenizer, images: int) -> None:
    """合流靠这个数把图与行对齐：多一个占位符就会让第二张图的行盖到第一张上。"""
    image_token = _image_token()
    ids = _our_ids(tokenizer, images)
    assert ids.count(image_token) == images


def test_the_block_is_three_atomic_tokens() -> None:
    """`<|vision_start|><|image_pad|><|vision_end|>` 必须是三个独立 token：分词器把它们切开，
    链就会把占位符数算错。"""
    tokenizer = ChatTokenizer.load(WEIGHTS)
    ids = tokenizer(torchcast_serve.VISION_BLOCK, add_special_tokens=False)["input_ids"]
    assert len(ids) == 3
    assert ids[1] == _image_token()
