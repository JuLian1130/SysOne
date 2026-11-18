"""合流接进真实分词器与题面：占位符、前缀、位置（要真 checkpoint，缺了就跳过）。

纯函数那几条在 `tests/unit/test_vision_merge.py`；这里钉的是**渲染与分词**这一层：模板真的
会为一张图吐一个占位符、前缀测量把图块算进去、图数不同渲染也不同。塔与链在门里
（`.scratch/p3_gate.py`）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from sysone.models.families import mude
from sysone.models.tokenizer import ChatTokenizer
from sysone.models.vision import merge
from sysone.domain.decision import Question

MUDE_DIR = Path(os.environ.get("SYS1_MUDE_DIR", "/Users/nyonic/Documents/opensource/mude-bench"))
STATE = "A small office with a desk and a lamp."
QUESTION = Question(text="Is the light on?", options=("Yes, it is on.", "No, it is off."))


def _skip_reason() -> str | None:
    if not (MUDE_DIR / "tokenizer.json").is_file():
        return f"缺少真实分词器：{MUDE_DIR}"
    return None


pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="只在 macOS 上跑（占位 token 的 id 来自真分词器）"),
    pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or ""),
]


@pytest.fixture(scope="module")
def prompter() -> mude.MudePrompter:
    tokenizer = ChatTokenizer.load(MUDE_DIR)
    return mude.MudePrompter(tokenizer, mude.read_codes(MUDE_DIR))


def test_one_image_renders_exactly_one_placeholder(prompter: mude.MudePrompter) -> None:
    """模板把一张图渲染成 `<|vision_start|><|image_pad|><|vision_end|>`：**一个**占位符。

    合流靠这个数对齐图与行：渲染成两个，就会把第二张图的行放到第一张的位置上。
    """
    ids, _, _, _ = prompter.encode(STATE, QUESTION, images=1)
    token = int(mude.parse_config(MUDE_DIR).image_token_id)
    assert list(ids).count(token) == 1

    two, _, _, _ = prompter.encode(STATE, QUESTION, images=2)
    assert list(two).count(token) == 2
    assert len(two) > len(ids)


def test_the_probe_prefix_includes_the_image_block(prompter: mude.MudePrompter) -> None:
    """前缀测量必须带上同样的图块：漏掉它，`common_prefix` 会在图块处提前断开。

    多题共享前缀就静默退化成逐题整趟（答案还对，等于没复用）；而带图的请求里前缀长度还会与
    尾巴错位。
    """
    plain, plain_prefix, _, _ = prompter.encode(STATE, QUESTION)
    with_image, image_prefix, _, _ = prompter.encode(STATE, QUESTION, images=1)
    descriptor = mude.parse_config(MUDE_DIR)
    # 图块是三个原子 token（未展开时），所以前缀正好长三格；漏掉它就会短三格，
    # 而 `common_prefix` 会在图块处提前断开。
    assert image_prefix == plain_prefix + 3, (image_prefix, plain_prefix)
    # 图块在前缀**里面**（题面之前），三个 id 连续：vision_start / image_pad / vision_end
    at = list(with_image[:image_prefix]).index(descriptor.vision_start_token_id)
    assert list(with_image[at:at + 3]) == [descriptor.vision_start_token_id,
                                           descriptor.image_token_id,
                                           descriptor.vision_end_token_id]


def test_two_questions_share_the_image_prefix(prompter: mude.MudePrompter) -> None:
    """同一请求的两道题前缀逐位相同（图块 + state），尾巴不同 —— 共享前缀的前提。"""
    other = Question(text="Is the room empty?", options=("Yes, it is.", "No, it is not."))
    first, first_prefix, _, _ = prompter.encode(STATE, QUESTION, images=1)
    second, second_prefix, _, _ = prompter.encode(STATE, other, images=1)
    assert first_prefix == second_prefix
    assert first[:first_prefix] == second[:second_prefix]
    assert first[first_prefix:] != second[second_prefix:]


def test_the_prompt_expands_to_text_plus_four_rows_per_image(
    prompter: mude.MudePrompter,
) -> None:
    """`L/4 = 4`（64×64 的图，2×2 合并）：展开之后恰好比文本多 `4 + 2` 个 token。"""
    plain, _, _, _ = prompter.encode(STATE, QUESTION)
    with_image, _, _, _ = prompter.encode(STATE, QUESTION, images=1)
    descriptor = mude.parse_config(MUDE_DIR)
    rows = np.zeros((4, int(descriptor.vision.out_hidden_size)), np.float16)  # type: ignore[union-attr]
    media = merge.Media(counts=(4,), grids=((2, 2),), rows=rows)
    expanded, placement = merge.attach(with_image, media,
                                       image_token_id=int(descriptor.image_token_id or 0))
    assert len(expanded) == len(plain) + 4 + 2
    # 占位符自己的那一行就是要被盖掉的（它的 embedding 是垃圾，设计如此）
    token = int(descriptor.image_token_id or 0)
    assert placement.spans == ((list(with_image).index(token), 4),)
    # 位置表与展开后的长度一致，且图像那段是三维网格而不是一条直线
    assert placement.positions.shape == (3, len(expanded))
    start, count = placement.spans[0]
    block = placement.positions[:, start:start + count]
    # 减去第一个 token 之后是**合并后 2×2 网格**的 `indices`：(t, 行, 列) 三个轴各走一遍。
    # 这一条比"四个位置互不相同"强：patch 网格（4×4 截前 4 个）也满足互不相同，
    # 而它给出的是一行四列，这里要求的是一列一行各两格（票据 07 §7.4c）。
    assert (block - block[:, :1]).tolist() == [
        [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1],
    ]
