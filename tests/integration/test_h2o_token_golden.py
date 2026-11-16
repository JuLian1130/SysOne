"""h2o 答面的 token 金标：分词器换成 `tokenizers` 直连之后题面 id 不许变。

金标 `tests/fixtures/h2o_tokens.json` 由 `.scratch/c2_h2o_token_golden.py` 用
`transformers.AutoTokenizer` 生成一次；这里用出货路径（`Serve.open` + `ChatTokenizer`，
`tokenizer.json` 直连，不装 transformers）重算同一批题面并逐位比。

金标与 checkpoint 绑定（h2o4b-int8）：换分布会对不上，那时要重新生成。只比长度 + 首尾各 16
个 id + `label_ids`：首尾足以发现任何位置的漂移，而整条 3952 token 的 id 会让 fixture 涨到
几百 KB。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sysone.domain.decision import Question
from sysone.models.vendors.base import question_spec
from sysone.models.vendors.h2o_serve import Serve
from sysone.service.protocol import describe

H2O = os.environ.get("SYS1_H2O_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not H2O or not Path(H2O).is_dir(),
    reason="设 SYS1_H2O_WEIGHTS 指向 h2o4b-int8 权重目录才会跑",
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
GOLDEN = json.loads((FIXTURES / "h2o_tokens.json").read_text(encoding="utf-8"))
TASKS = Path(__file__).resolve().parents[2] / "docs/collective-decision/tasks/hard.jsonl"


def _states() -> dict[str, tuple[str, dict]]:
    """题库里金标提到的那几道题 → (state, 题面 spec)。"""
    wanted = {case["id"] for case in GOLDEN["cases"]}
    out: dict[str, tuple[str, dict]] = {}
    for line in TASKS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in wanted:
            continue
        raw = row["question"]
        criteria = raw.get("criteria") or {}
        options = tuple(f"{key}: {describe(value)}" for key, value in criteria.items())
        question = Question(text=str(raw.get("instructions") or ""), options=options,
                            kind=raw.get("type", "choice"))
        out[row["id"]] = (row["state"], question_spec(question))
    return out


def test_prompt_ids_match_the_golden() -> None:
    serve = Serve.open(H2O)
    states = _states()
    head, tail = GOLDEN["head"], GOLDEN["tail"]
    for case in GOLDEN["cases"]:
        state, spec = states[case["id"]]
        prepared = serve.prepare(state, spec)
        label = case["id"]
        assert len(prepared.ids) == case["tokens"], label
        assert list(prepared.ids[:head]) == case["ids_head"], label
        assert list(prepared.ids[-tail:]) == case["ids_tail"], label
        assert list(prepared.label_ids) == case["label_ids"], label
        assert list(prepared.labels) == case["labels"], label
