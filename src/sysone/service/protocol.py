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
METRICS_PATH = "/metrics"
MODELS_PATH = "/v1/models"
SYSTEM_ONE_PATH = "/v1/systemone"
ROUTES = {
    HEALTH_PATH: "GET",
    MODELS_PATH: "GET",
    METRICS_PATH: "GET",
    SYSTEM_ONE_PATH: "POST",
}
# 需要 bearer key 的那两条；`/health`（探针）与 `/metrics`（本机诊断）永远开放。
PROTECTED = (MODELS_PATH, SYSTEM_ONE_PATH)
KEY_VARIABLE = "DECISION_API_KEY"
MEDIA_FIELDS = ("images", "videos")
# `attention_mode` 是向后兼容的新增字段（architecture.md §9.2），其余与旧实现一致。
ACCEPTED_FIELDS = ("state", "questions", *MEDIA_FIELDS, "model", "attention_mode")
JSON_TYPE = "application/json"
DESCRIPTION = "Typed decisions with a probability for every option"

# 线格式认的三种题型。**认**不等于**答得了**：score 要的是厂商自己的档位模板与
# `format_answer`（h2o / torchcast 的答面有），mude 的码表读出答不了 —— 那种情况由
# 家族在规划期明确拒绝（`families/mude.option_plan`），不是静默按 choice 猜一个。
KINDS = ("choice", "noul", "score")
# noul 的判据键：`{"true": <判据>, "false": <判据>}`（可选，缺省时厂商用默认描述）。
_NOUL_KEYS = ("true", "false")
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
        # 说"线格式有哪些"，不说"这个模型答得了哪些"：后者是家族的读出能力，协议层不知道。
        raise ValueError(
            f"question {key!r} has type {kind!r}; the wire format carries {', '.join(KINDS)}"
        )
    instructions = raw.get("instructions")
    text = "" if instructions is None else describe(instructions)
    if kind == "noul":
        return WireQuestion(
            key=key,
            question=Question(
                text=text, options=_noul_options(key, raw.get("criteria")), kind="noul"
            ),
            labels=(),
        )
    criteria = raw.get("criteria")
    if kind == "score":
        # score 的 criteria 是**有序档位数组**：档位名就是 `"0".."n-1"`，顺序即档位高低
        # （厂商按 `Σ i·p_i` 出期望值）。这里不接映射形态 —— 映射会把档位顺序丢掉，而
        # 顺序正是分数的全部含义（`h2o_lightning_shim.options_of` 只收数组）。
        if isinstance(criteria, (str, bytes)) or not isinstance(criteria, Sequence) or not criteria:
            raise ValueError(
                f"question {key!r} needs criteria: an ordered array of score levels"
            )
        if len(criteria) > MAX_OPTIONS:
            raise ValueError(f"question {key!r} has {len(criteria)} levels; at most {MAX_OPTIONS}")
        labels = tuple(str(index) for index in range(len(criteria)))
        levels = tuple(f"{label}: {describe(level)}" for label, level in zip(labels, criteria))
        return WireQuestion(
            key=key, question=Question(text=text, options=levels, kind="score"), labels=labels
        )
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


def _noul_options(key: str, criteria: Any) -> tuple[str, ...]:
    """noul 的可选判据 `{"true": <判据>, "false": <判据>}` → 两行选项文本。

    **必须读**：不读的时候厂商只能落回默认描述（"the statement holds" / "it does not"），
    题面就与厂商训练时不一致 —— 实测 38 道真题的题面因此**全都不一样**（票据 15 §34），
    与票据 12 那 7/105 条判词差异同类。

    顺序固定为 true 在前：厂商按 `true` / `false` 取键，不依赖这里的次序，但固定下来让同一
    请求渲染出的题面逐字可复现。
    """
    if criteria is None:
        return ()
    if not isinstance(criteria, Mapping):
        raise ValueError(
            f"question {key!r} has a criteria that is not an object; "
            "noul takes {'true': ..., 'false': ...}"
        )
    unknown = sorted(str(name) for name in criteria if name not in _NOUL_KEYS)
    if unknown:
        raise ValueError(
            f"question {key!r} has criteria keys {', '.join(map(repr, unknown))} not understood; "
            f"noul takes {', '.join(_NOUL_KEYS)}"
        )
    return tuple(
        f"{name}: {describe(criteria[name])}" for name in _NOUL_KEYS if name in criteria
    )


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
    if wire.question.kind == "score":
        # 厂商口径（`h2o_lightning_shim.format_answer` 的 score 分支）：档位序号上的期望值
        # `Σ i·p_i`，概率表按 `"0".."n-1"` 给。不带 `legend`：档位就是请求里那些，与 choice
        # 分支不回选项文本同口径。
        return {
            "type": "score",
            "score": sum(index * share for index, share in enumerate(shares)),
            "probabilities": dict(zip(wire.labels, shares)),
        }
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
