"""`/v1/systemone` 的线格式：路由表、状态码、错误形状、请求 → 领域对象。

这份契约逐字复刻自 mude-bench 的 `d3_server.py`（旧实现整个复用了那个壳），所以"协议兼容"
没有第二种解释：路由与白名单、401/405 的头、两种 422 形状、以及 `{"model", "answers",
"usage"}` 的回包都对着它。改动要有 `.scratch/service_parity.py` 的对拍背书。

请求里的 `questions` 是**对象**（题号 → 题目），不是数组；选项键是 `criteria` 的键，
描述缺省时选项行只写键，有描述时写 `键: 描述`——这与 `render_messages` 的拼接口径一致，
协议层负责拼好再交给题面。
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Iterable, Mapping, Sequence

from ..domain.attention import MaskMode
from ..domain.decision import Answer, DecisionRequest, DecisionResult, Question

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
SYSTEM_ONE_PATH = "/v1/systemone"
ROUTES = {HEALTH_PATH: "GET", MODELS_PATH: "GET", SYSTEM_ONE_PATH: "POST"}
# 需要 bearer key 的那两条；`/health` 永远开放给探针。
PROTECTED = (MODELS_PATH, SYSTEM_ONE_PATH)
KEY_VARIABLE = "DECISION_API_KEY"
MEDIA_FIELDS = ("images", "videos")
# `attention_mode` 是向后兼容的新增字段（architecture.md §9.2），其余与旧实现一致。
ACCEPTED_FIELDS = ("state", "questions", *MEDIA_FIELDS, "model", "attention_mode")
JSON_TYPE = "application/json"
DESCRIPTION = "Typed decisions with a probability for every option"

# mude 的线格式只有这两种读出方式（`d3_format` 不给 score）。
KINDS = ("choice", "noul")
MAX_OPTIONS = 255
MODALITIES = ["text"]


class Refusal(Exception):
    """前门用 `{"detail": ...}` 回的错误。"""

    def __init__(self, status: HTTPStatus, detail: Any, headers: dict[str, str] | None = None) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail
        self.headers = headers or {}


class Framing(Exception):
    """请求体的边界读不出来（Content-Length 或 chunk 坏）：400 且关连接。"""


def unprocessable(detail: Any) -> Refusal:
    return Refusal(HTTPStatus.UNPROCESSABLE_ENTITY, detail)


@dataclass(frozen=True)
class WireQuestion:
    """一道题的线格式：题号、交给引擎的 `Question`、回包里的选项键。"""

    key: str
    question: Question
    labels: tuple[str, ...]


# ---------------------------------------------------------------------- 入


def describe(value: Any) -> str:
    """state / 题干 / 选项描述的文字：字符串原样，其他按 JSON。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def parse_body(raw: bytes) -> Any:
    """把请求体读成 JSON；`NaN`/`Infinity` 不算 JSON。"""
    try:
        return json.loads(raw, parse_constant=_no_constants)
    except ValueError as exc:
        raise unprocessable(f"could not read the body as JSON ({exc})") from exc


def parse_request(body: Any) -> tuple[DecisionRequest, tuple[WireQuestion, ...]]:
    """请求体 → (`DecisionRequest`, 题目线格式)；不合法的题目一次全报出来，什么都不跑。"""
    if not isinstance(body, dict):
        raise unprocessable('expected a JSON object such as {"questions": {...}, "state": ...}')
    unknown = sorted(set(body) - set(ACCEPTED_FIELDS))
    if unknown:
        raise unprocessable(
            f"{', '.join(map(repr, unknown))} not understood; a request may carry "
            f"{', '.join(ACCEPTED_FIELDS)}"
        )
    for name in MEDIA_FIELDS:
        if body.get(name) is not None:
            raise unprocessable(f"{name} is not supported: this model answers text only")
    raw_questions = body.get("questions")
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise unprocessable(
            "questions is required: an object of one or more question IDs, each to a question"
        )
    mode = _attention_mode(body.get("attention_mode"))

    parsed: list[WireQuestion] = []
    errors: dict[str, str] = {}
    for key, raw in raw_questions.items():
        try:
            parsed.append(_wire_question(str(key), raw))
        except ValueError as exc:
            errors[str(key)] = str(exc)
    if errors:
        raise unprocessable({"invalid_questions": errors})

    request = DecisionRequest(
        state=_state_text(body.get("state")),
        questions=tuple(item.question for item in parsed),
        attention_mode=mode,
    )
    return request, tuple(parsed)


def _state_text(value: Any) -> str:
    """缺省与空串都表示"没有状态"（题面里会写成 `(empty)`）。"""
    return "" if value is None else describe(value)


def _wire_question(key: str, raw: Any) -> WireQuestion:
    if not isinstance(raw, dict):
        raise ValueError(f"question {key!r} must be an object")
    kind = raw.get("type")
    if kind not in KINDS:
        raise ValueError(
            f"question {key!r} has type {kind!r}; this model answers {', '.join(KINDS)}"
        )
    instructions = raw.get("instructions")
    text = "" if instructions is None else describe(instructions)
    if kind == "noul":
        return WireQuestion(key=key, question=Question(text=text, kind="noul"), labels=())
    criteria = raw.get("criteria")
    if not isinstance(criteria, Mapping) or not criteria:
        raise ValueError(f"question {key!r} needs criteria: an object of one or more option keys")
    if len(criteria) > MAX_OPTIONS:
        raise ValueError(f"question {key!r} has {len(criteria)} options; at most {MAX_OPTIONS}")
    labels = tuple(str(item) for item in criteria)
    options = tuple(
        label if criteria[label] is None else f"{label}: {describe(criteria[label])}"
        for label in labels
    )
    return WireQuestion(key=key, question=Question(text=text, options=options), labels=labels)


def _attention_mode(value: Any) -> MaskMode | None:
    """未知模式按 400 回（`architecture.md` §9.2：不在能力集就是坏请求，不是坏数据）。"""
    if value is None:
        return None
    try:
        return MaskMode(str(value))
    except ValueError as exc:
        raise Refusal(
            HTTPStatus.BAD_REQUEST,
            f"attention_mode {value!r} is not one of "
            f"{', '.join(item.value for item in MaskMode)}",
        ) from exc


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


# ---------------------------------------------------------------------- 出


def health_payload(*, model_name: str, max_input_tokens: int, authentication: bool) -> dict[str, Any]:
    return {
        "status": "ready",
        "model": model_name,
        "max_input_tokens": int(max_input_tokens),
        "modalities": list(MODALITIES),
        "authentication": bool(authentication),
    }


def models_payload(*, model_name: str, max_input_tokens: int) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": model_name,
                "description": DESCRIPTION,
                "max_input_tokens": int(max_input_tokens),
                "modalities": list(MODALITIES),
            }
        ]
    }


def answer_payload(
    result: DecisionResult,
    *,
    model_name: str,
    wire: Sequence[WireQuestion],
    debug_timings: bool = False,
) -> dict[str, Any]:
    """`{"model", "answers", "usage"}`，外加本引擎的 `attention_mode` / `attention_trained`。"""
    if len(result.answers) != len(wire):
        raise ValueError(f"expected {len(wire)} answers, got {len(result.answers)}")
    answers = {
        item.key: _answer_object(answer, item)
        for item, answer in zip(wire, result.answers)
    }
    payload: dict[str, Any] = {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": int(result.usage.input_tokens), "output_tokens": 0},
        "attention_mode": result.provenance.attention_mode.value,
        "attention_trained": bool(result.provenance.attention_trained),
    }
    if debug_timings and result.timings is not None:
        payload["timings"] = dataclasses.asdict(result.timings)
    return payload


def _answer_object(answer: Answer, wire: WireQuestion) -> dict[str, Any]:
    if wire.question.kind == "noul":
        # 引擎里的标签顺序是 ("yes", "no")，线格式要的是 P(true)。
        return {"type": "noul", "noul": _normalized(answer.probs, 2)[0]}
    shares = _normalized(answer.probs, len(wire.labels))
    best = max(range(len(shares)), key=shares.__getitem__)
    return {
        "type": "choice",
        "choice": wire.labels[best],
        "probabilities": dict(zip(wire.labels, shares)),
    }


def _normalized(values: Iterable[Any], expected: int) -> list[float]:
    """一道题的选项概率：有限、非负、和为正，然后按和归一（与 `d3_format.normalized` 同口径）。"""
    shares = [float(value) for value in values]
    if len(shares) != expected:
        raise ValueError(f"expected {expected} option probabilities, got {len(shares)}")
    if not all(math.isfinite(value) and value >= 0.0 for value in shares):
        raise ValueError("option probabilities must be finite and non-negative")
    total = sum(shares)
    if not total > 0.0:
        raise ValueError("option probabilities sum to zero")
    return [value / total for value in shares]
