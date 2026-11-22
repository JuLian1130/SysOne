"""真 checkpoint 的 token id 金标：`transformers` 换成 `tokenizers` 之后题面字节不许变。

金标 `tests/fixtures/mude_tokens.json` 由 `.scratch/mude_token_golden.py` 用
`AutoTokenizer` 生成一次；这里用出货路径的 `MudePrompter` + `ChatTokenizer` 重算
(整条 prompt 的 id, 前缀长度) 并逐位比——模板差一个空格就会在这里红，所以不比文本。

金标与 checkpoint 绑定（mude-bench）：换分布会对不上，那时要重新生成。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sysone.domain.decision import Question
from sysone.models.families import mude
from sysone.models.tokenizer import ChatTokenizer

MUDE = os.environ.get("SYS1_MUDE_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not MUDE or not Path(MUDE).is_dir(),
    reason="设 SYS1_MUDE_WEIGHTS 指向 mude-bench 权重目录才会跑",
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = json.loads((FIXTURES / "mude_tokens.json").read_text(encoding="utf-8"))


def test_prompt_ids_match_the_golden() -> None:
    prompter = mude.MudePrompter(ChatTokenizer.load(MUDE), mude.read_codes(MUDE))
    for case in GOLDEN["cases"]:
        sequence, prefix, _labels, _indexes, _readout = prompter.encode(
            case["state"], Question(**case["question"])
        )
        label = case["state"][:40] or "empty state"
        assert list(sequence) == case["ids"], label
        assert prefix == case["prefix"], label
