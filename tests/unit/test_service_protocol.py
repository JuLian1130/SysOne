"""`/v1/systemone` 的线格式：请求映射、两种 422 形状、回包形状。

纯函数，不需要起服务；HTTP 层的状态码与头在 `test_service_app.py` 里。
"""

from __future__ import annotations

import json

import pytest

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Answer, DecisionResult, Provenance, Timings, Usage
from sysone.domain.quant import QuantSpec
from sysone.service import protocol


def _body(**extra: object) -> dict:
    body: dict = {
        "state": {"battery": 8},
        "questions": {
            "q1": {
                "type": "choice",
                "instructions": "What now?",
                "criteria": {"1": "keep going", "2": None},
            }
        },
    }
    body.update(extra)
    return body


def _result(
    answers: tuple[Answer, ...],
    *,
    mode: MaskMode = MaskMode.PREFIX_NONCAUSAL,
    timings: Timings | None = None,
) -> DecisionResult:
    return DecisionResult(
        answers=answers,
        usage=Usage(input_tokens=123, prefill_calls=len(answers)),
        provenance=Provenance(
            family="mude",
            artifact_id="artifact",
            quant=QuantSpec(),
            attention_mode=mode,
            attention_trained=True,
        ),
        timings=timings,
    )


def _answer(
    index: int = 0,
    *,
    kind: str = "choice",
    value: str = "B",
    probs=(0.25, 0.75),
    labels=("A", "B"),
) -> Answer:
    return Answer(
        question_index=index,
        kind=kind,
        value=value,
        probs=probs,
        labels=labels,
        confidence=probs[0] if probs else 0.0,
    )


# ---------------------------------------------------------------------- 入


def test_parse_request_maps_each_option_to_its_rendered_text() -> None:
    request, wire = protocol.parse_request(_body())
    assert request.state == '{"battery": 8}'
    assert wire[0].key == "q1"
    assert wire[0].labels == ("1", "2")
    assert wire[0].question.text == "What now?"
    # 描述缺省的选项只写键：线格式里就是 `key`，不是 `key: null`。
    assert wire[0].question.options == ("1: keep going", "2")
    assert request.questions == (wire[0].question,)
    assert request.attention_mode is None


def test_parse_request_keeps_the_question_order() -> None:
    body = {
        "questions": {
            "second": {"type": "noul", "instructions": "要看吗？"},
            "first": {"type": "choice", "criteria": {"yes": "是"}},
        }
    }
    _, wire = protocol.parse_request(body)
    assert [item.key for item in wire] == ["second", "first"]
    assert wire[0].question.kind == "noul"
    assert wire[0].labels == ()
    assert wire[1].question.options == ("yes: 是",)


def test_parse_request_defaults_empty_state_and_instructions() -> None:
    request, wire = protocol.parse_request({"questions": {"q": {"type": "noul"}}})
    assert request.state == ""
    assert wire[0].question.text == ""


def test_parse_request_accepts_a_plain_string_state() -> None:
    request, _ = protocol.parse_request(_body(state="只有这一句"))
    assert request.state == "只有这一句"


def test_parse_request_reads_attention_mode() -> None:
    request, _ = protocol.parse_request(_body(attention_mode="noncausal_full"))
    assert request.attention_mode is MaskMode.NONCAUSAL_FULL


def test_unknown_attention_mode_is_a_bad_request() -> None:
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request(_body(attention_mode="causal_full"))
    assert info.value.status == 400
    assert isinstance(info.value.detail, str)


def test_parse_request_rejects_fields_outside_the_whitelist() -> None:
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request(_body(temperature=0.5))
    assert info.value.status == 422
    assert "temperature" in info.value.detail


def test_parse_request_rejects_media() -> None:
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request(_body(images=["data:image/png;base64,AAAA"]))
    assert info.value.status == 422
    assert "images" in info.value.detail


def test_parse_request_rejects_a_non_object_body() -> None:
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request([{"questions": {}}])
    assert info.value.status == 422


def test_parse_request_rejects_missing_or_empty_questions() -> None:
    for body in ({}, {"questions": {}}, {"questions": ["q1"]}):
        with pytest.raises(protocol.Refusal) as info:
            protocol.parse_request(body)
        assert info.value.status == 422


def test_every_bad_question_is_reported_at_once() -> None:
    body = {
        "questions": {
            "ok": {"type": "choice", "criteria": {"a": "A"}},
            "no_criteria": {"type": "choice"},
            "bad_type": {"type": "score"},
            "not_object": "试试",
        }
    }
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request(body)
    assert info.value.status == 422
    assert set(info.value.detail["invalid_questions"]) == {"no_criteria", "bad_type", "not_object"}


def test_too_many_options_is_rejected() -> None:
    criteria = {str(index): None for index in range(protocol.MAX_OPTIONS + 1)}
    with pytest.raises(protocol.Refusal) as info:
        protocol.parse_request({"questions": {"q": {"type": "choice", "criteria": criteria}}})
    assert "at most" in info.value.detail["invalid_questions"]["q"]


# ---- noul：criteria 是 {"true": 判据, "false": 判据}，不读就等于换了一套题面 -------------


def test_a_noul_question_carries_its_two_criteria_descriptions() -> None:
    """noul 的判据描述必须原样进题面。

    不读它的时候厂商只能落回默认描述（"the statement holds" / "it does not"），于是题面与
    厂商训练时不一致：实测 38 道真题的题面因此**全都不一样**（票据 15 §34）。答案标签仍然
    固定是 ("yes", "no")，与这两行无关。
    """
    _, wire = protocol.parse_request(
        {
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Hold?",
                    "criteria": {"true": "the notice was served", "false": "it was not"},
                }
            }
        }
    )
    assert wire[0].question.kind == "noul"
    assert wire[0].question.options == ("true: the notice was served", "false: it was not")
    assert wire[0].labels == ()


def test_a_noul_question_without_criteria_is_unchanged() -> None:
    """判据可缺省（旧请求照旧工作）；`false` 在前也按 true / false 的固定次序渲染。"""
    _, wire = protocol.parse_request({"questions": {"q": {"type": "noul"}}})
    assert wire[0].question.options == ()
    _, ordered = protocol.parse_request(
        {"questions": {"q": {"type": "noul", "criteria": {"false": "no", "true": "yes"}}}}
    )
    assert ordered[0].question.options == ("true: yes", "false: no")


def test_noul_criteria_must_be_a_true_false_object() -> None:
    """非映射或未知键都是坏请求：静默丢掉一个客户端显式给的判据，题面就悄悄换了。"""
    for criteria in (["a", "b"], "yes/no", {"yes": "y", "no": "n"}):
        with pytest.raises(protocol.Refusal) as info:
            protocol.parse_request({"questions": {"q": {"type": "noul", "criteria": criteria}}})
        assert info.value.status.value == 422


# ---- score：criteria 是**有序档位数组**，不是映射 -------------------------------------

def test_a_score_scale_keeps_its_order_and_names_the_levels_by_index() -> None:
    """档位顺序就是分数的含义：`0` 是最低档。线格式按序编号，回包的键也是 `"0".."n-1"`。"""
    _, wire = protocol.parse_request(
        {
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["SEV-4 (lowest)", "SEV-3", "SEV-2", "SEV-1 (highest)"],
                }
            }
        }
    )
    assert wire[0].labels == ("0", "1", "2", "3")
    assert wire[0].question.kind == "score"
    assert wire[0].question.options == (
        "0: SEV-4 (lowest)",
        "1: SEV-3",
        "2: SEV-2",
        "3: SEV-1 (highest)",
    )


def test_a_score_scale_must_be_an_ordered_array() -> None:
    """映射形态是 choice 的；收下它会把档位顺序丢掉（厂商 `options_of` 也只收数组）。"""
    for criteria in ({"0": "low", "1": "high"}, [], "low, high", 3):
        with pytest.raises(protocol.Refusal) as info:
            protocol.parse_request(
                {"questions": {"q": {"type": "score", "criteria": criteria}}}
            )
        assert "ordered array" in info.value.detail["invalid_questions"]["q"]


def test_a_score_answer_is_the_expected_level_and_its_distribution() -> None:
    """厂商口径（`h2o_lightning_shim.format_answer`）：score = Σ i·p_i，概率按档位序号给。"""
    _, wire = protocol.parse_request(
        {"questions": {"q": {"type": "score", "criteria": ["low", "mid", "high"]}}}
    )
    payload = protocol.answer_payload(
        _result((_answer(kind="score", value="0.750", probs=(0.25, 0.75, 0.0),
                         labels=("0", "1", "2")),)),
        model_name="h2o",
        wire=wire,
    )
    answer = payload["answers"]["q"]
    assert answer["type"] == "score"
    assert answer["probabilities"] == {"0": 0.25, "1": 0.75, "2": 0.0}
    # 0·0.25 + 1·0.75 + 2·0.0
    assert answer["score"] == pytest.approx(0.75)


def test_parse_body_reads_json_and_refuses_nan() -> None:
    assert protocol.parse_body(b'{"state": "x"}') == {"state": "x"}
    for raw in (b"{", b'{"state": NaN}', b'{"state": Infinity}'):
        with pytest.raises(protocol.Refusal) as info:
            protocol.parse_body(raw)
        assert info.value.status == 422


# ---------------------------------------------------------------------- 出


def test_answer_payload_choice_shape() -> None:
    request, wire = protocol.parse_request(_body())
    payload = protocol.answer_payload(
        _result((_answer(probs=(0.25, 0.75), labels=("1: keep going", "2")),)),
        model_name="mude",
        wire=wire,
    )
    assert payload["model"] == "mude"
    assert payload["answers"]["q1"] == {
        "type": "choice",
        "choice": "2",
        "probabilities": {"1": 0.25, "2": 0.75},
    }
    assert payload["usage"] == {"input_tokens": 123, "output_tokens": 0}
    assert payload["attention_mode"] == "prefix_noncausal"
    assert payload["attention_trained"] is True
    assert "timings" not in payload


def test_answer_payload_noul_shape_reports_p_true() -> None:
    _, wire = protocol.parse_request({"questions": {"q": {"type": "noul"}}})
    # 引擎里的 noul 标签顺序是 ("yes", "no")，所以 probs 的第一项就是 P(true)。
    payload = protocol.answer_payload(
        _result((_answer(kind="noul", value="yes", probs=(0.8, 0.2), labels=("yes", "no")),)),
        model_name="mude",
        wire=wire,
    )
    assert payload["answers"]["q"] == {"type": "noul", "noul": 0.8}


def test_answer_payload_renormalizes_the_shares() -> None:
    _, wire = protocol.parse_request({"questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}})
    payload = protocol.answer_payload(
        _result((_answer(probs=(1.0, 3.0), labels=("a: A", "b: B")),)),
        model_name="mude",
        wire=wire,
    )
    assert payload["answers"]["q"]["probabilities"] == {"a": 0.25, "b": 0.75}
    assert payload["answers"]["q"]["choice"] == "b"


def test_answer_payload_adds_timings_only_when_debugging() -> None:
    _, wire = protocol.parse_request({"questions": {"q": {"type": "noul"}}})
    timings = Timings(prepare=0.1, prefill=2.0, readout=1.5, total=2.2)
    payload = protocol.answer_payload(
        _result((_answer(kind="noul", labels=("yes", "no")),), timings=timings),
        model_name="mude",
        wire=wire,
        debug_timings=True,
    )
    assert payload["timings"] == {
        "prepare": 0.1,
        "prefill": 2.0,
        "readout": 1.5,
        "total": 2.2,
    }


def test_answer_payload_refuses_a_bad_readout() -> None:
    _, wire = protocol.parse_request(
        {"questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}
    )
    for probs in ((0.5,), (-0.1, 1.1), (0.0, 0.0)):
        with pytest.raises(ValueError):
            protocol.answer_payload(
                _result((_answer(probs=probs, labels=("a: A", "b: B")),)),
                model_name="mude",
                wire=wire,
            )


def test_answer_payload_refuses_a_short_answer_list() -> None:
    _, wire = protocol.parse_request({"questions": {"q": {"type": "noul"}}})
    with pytest.raises(ValueError):
        protocol.answer_payload(_result(()), model_name="mude", wire=wire)


def test_health_and_models_payloads() -> None:
    health = protocol.health_payload(model_name="mude", max_input_tokens=16384, authentication=False)
    assert health == {
        "status": "ready",
        "model": "mude",
        "max_input_tokens": 16384,
        "modalities": ["text"],
        "authentication": False,
    }
    models = protocol.models_payload(model_name="mude", max_input_tokens=16384)
    assert models["models"][0]["name"] == "mude"
    assert models["models"][0]["modalities"] == ["text"]
    assert models["models"][0]["description"] == protocol.DESCRIPTION


def test_routes_match_the_upstream_contract() -> None:
    assert protocol.ROUTES == {
        "/health": "GET",
        "/v1/models": "GET",
        "/v1/systemone": "POST",
    }
    assert protocol.PROTECTED == ("/v1/models", "/v1/systemone")
    assert set(protocol.ACCEPTED_FIELDS) == {
        "state",
        "questions",
        "images",
        "videos",
        "model",
        "attention_mode",
    }


def test_payload_is_json_serializable_as_compact_json() -> None:
    _, wire = protocol.parse_request(_body())
    payload = protocol.answer_payload(_result((_answer(probs=(0.5, 0.5), labels=("1: keep going", "2")),)),
                                      model_name="mude", wire=wire)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert '"model":"mude"' in encoded
