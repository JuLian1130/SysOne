"""mude 的题面与答面逐字节等于厂商那次渲染 —— 金标由厂商模块生成一次。

`d3_format.py` / `d3_runtime.py` 是随 checkpoint 发布的厂商文件（在下载清单里，但推理路径不读
它们），所以这里不 import：`tests/fixtures/mude_vendor_prompts.json` 是把它们跑一次存下来的
结果。每个用例带三样东西 —— 请求里的那道题、厂商 `d3_format.messages()` 渲染出的整段轮次、
以及把一组概率喂给 `d3_runtime.product_answer()` 得到的答面 —— 这里用出货路径（协议层 →
`option_plan` → `render_messages` / `_answer_object`）重算并逐字节比。

以前的对账是自指的：金标用**我们自己的**渲染器生成，只能证明"我们的渲染没变过"，证明不了
"等于厂商"，所以题面丢判据、score 被拒、置信度另算这三件事一条都没红。换 checkpoint（码表变
了）或厂商改题面/答面时，用权重目录里那两个文件重跑一次、重新生成那份 fixture（见它的
`source` 字段）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sysone.domain.decision import Answer
from sysone.models.families import mude
from sysone.service import protocol

FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "mude_vendor_prompts.json").read_text(
        encoding="utf-8"
    )
)
CODES = tuple(FIXTURE["codes"])


def _wire(
    case: dict,
) -> tuple[protocol.WireQuestion, str, tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """请求里那道题 → 协议层的线格式 → 家族的选项计划（与一条真请求走同一条路）。"""
    request, wire = protocol.parse_request(
        {"state": case["state"], "questions": {"q": case["question"]}}
    )
    item = wire[0]
    texts, labels, indexes = mude.option_plan(item.question)
    return item, request.state, texts, labels, indexes


def test_the_fixture_carries_a_whole_code_table() -> None:
    """码表随 checkpoint 走：255 个互不相同的回答码，少一个读出表就没有对应的行。"""
    assert len(CODES) == mude.CODE_ROWS
    assert len(set(CODES)) == mude.CODE_ROWS


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_the_prompt_matches_the_vendor_byte_for_byte(case: dict) -> None:
    item, state, texts, _, _ = _wire(case)
    assert mude.render_messages(state, item.question, texts, CODES) == case["messages"]


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_the_answer_object_matches_the_vendor(case: dict) -> None:
    item, _, _, labels, indexes = _wire(case)
    # 厂商给的向量是**题面**顺序；引擎内部按标签顺序存，noul 的索引反着放（"yes" 在 false 行）
    probs = tuple(case["probabilities"][index] for index in indexes)
    best = max(range(len(probs)), key=probs.__getitem__)
    answer = Answer(
        question_index=0,
        kind=item.question.kind,
        value=labels[best],
        probs=probs,
        labels=labels,
        confidence=mude.confidence(probs, item.question.kind),
    )
    ours = protocol._answer_object(answer, item)
    theirs = case["answer"]
    if theirs["type"] == "score":
        # 只省 `legend`：它是请求里档位文本的回声，而两家对非字符串档位的写法不同 —— 厂商 mude
        # 把档位排键压紧成 `{"label":"低","level":1}`，题面里却是 `{"label": "低", "level": 1}`，
        # h2o 那一侧回的就是题面那种。一个共享字段没法同时是两家的。
        assert set(theirs) - set(ours) == {"legend"}
        theirs = {key: value for key, value in theirs.items() if key != "legend"}
    assert ours == theirs
