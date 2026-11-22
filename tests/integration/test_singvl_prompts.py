"""真发布包的分词器上，SingVL 的题面 id 与厂商逐位相同。

金标 `tests/fixtures/singvl/real_ids.json` 是发布包的 `tokenizer.json` 对
`tests/fixtures/singvl/prompts.json`（由厂商 `render()` 生成）分词的结果，所以这条钉的是
"我们把题面对了之后，切出来的 id 也是厂商那一串"：图块的三段占位、`<think>` 块、行尾与
选项行里的空格，任何一处不同都会在这里露出。

按 `SYS1_SINGVL_WEIGHTS` 指向检查点目录才会跑（只读 `tokenizer.json` 与 `config.json`，
不装载模型，所以内存开销可以忽略）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sysone.domain.decision import Question
from sysone.models.families import singvl
from sysone.models.tokenizer import ChatTokenizer
from sysone.service.protocol import _wire_question

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "singvl"
PROMPTS = json.loads((FIXTURES / "prompts.json").read_text(encoding="utf-8"))
IDS = {case["name"]: case["ids"] for case in
       json.loads((FIXTURES / "real_ids.json").read_text(encoding="utf-8"))["cases"]}


def test_the_prompt_ids_match_the_published_tokenizer() -> None:
    directory = os.environ.get("SYS1_SINGVL_WEIGHTS")
    if not directory or not Path(directory).is_dir():
        pytest.skip("设 SYS1_SINGVL_WEIGHTS 指向 SingVL S1 4B 的检查点目录才会跑")
    root = Path(directory)
    prompter = singvl.SingVLPrompter(ChatTokenizer.load(root), singvl.read_codes(root))
    for case in PROMPTS["cases"]:
        wire = _wire_question("q", case["question"])
        question = Question(text=wire.question.text, options=wire.question.options,
                            kind=wire.question.kind)
        ids, prefix, labels, indexes, readout = prompter.encode(
            case["state"], question, images=case["images"])
        assert list(ids) == IDS[case["name"]], case["name"]
        # 这一家只有 `noncausal_full` 且读出在末位：前缀量不出来，读出位置不特殊。
        assert prefix is None and readout is None
        assert len(labels) == len(indexes)
