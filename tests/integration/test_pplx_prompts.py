"""真 checkpoint 的 pplx 题面：真实 Qwen3.5 分词器上的 id 与读出位置逐位对拍。

金标 `tests/fixtures/pplx/real_prompts.json` 由厂商自己的 `protocol.py` + 发布包的
`tokenizer.json` 生成（生成器在 ``.tmp``，见 `docs/sysone/plan.md` 票据 22 的补片）。
单元测试用结构同形的 tiny tokenizer 钉题面；这一条钉**真实 BPE** 的切分，包括 v1.2
读出不在末位那件事。

按 `SYS1_PPLX12_WEIGHTS` / `SYS1_PPLX13_WEIGHTS` 指向检查点目录才会跑。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sysone.domain.decision import Question
from sysone.models.families import pplx
from sysone.models.tokenizer import ChatTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pplx"
GOLDEN = json.loads((FIXTURES / "real_prompts.json").read_text(encoding="utf-8"))
QUESTION = Question(text=GOLDEN["question"]["text"],
                    options=tuple(GOLDEN["question"]["options"]),
                    kind=GOLDEN["question"]["kind"])
CASES = {
    "v12": ("SYS1_PPLX12_WEIGHTS", pplx.V12, pplx.read_codes_12),
    "v13": ("SYS1_PPLX13_WEIGHTS", pplx.V13, pplx.read_codes_13),
}


@pytest.mark.parametrize("label", ("v12", "v13"))
def test_the_prompts_match_the_vendor_ids(label: str) -> None:
    variable, protocol, codes_for = CASES[label]
    directory = os.environ.get(variable)
    if not directory or not Path(directory).is_dir():
        pytest.skip(f"设 {variable} 指向 {protocol.family} 的检查点目录才会跑")
    prompter = pplx.PplxPrompter(ChatTokenizer.load(directory), codes_for(Path(directory)), protocol)
    for case in ("plain", "escaped"):
        entry = GOLDEN["cases"][case]
        segments, _tails = prompter.text(entry["state"], QUESTION)
        ids, prefix, _labels, _indexes, readout = prompter.encode(entry["state"], QUESTION)
        assert list(segments) == entry[label]["segments"], (label, case)
        assert list(ids) == entry[label]["ids"], (label, case)
        assert prefix == 0
        assert readout == entry[label]["readout"], (label, case)
