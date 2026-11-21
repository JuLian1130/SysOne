"""前门：路由、状态码、头、就绪门与准入（真 `http.server`，绑端口 0）。

协议体（两种 422 形状、回包字段）在 `test_service_protocol.py` 里逐字节断言；这里只验证
"HTTP 这一层的行为"，也就是旧实现那套壳的语义：405 带 `Allow`、401 带 `WWW-Authenticate`、
尾部斜杠 307、`Content-Length` 与 chunked 只能二选一、未就绪 503、满容量 429。
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time

import pytest

from sysone.domain.attention import MaskMode
from sysone.domain.decision import Answer, DecisionResult, Provenance, Usage
from sysone.domain.quant import QuantSpec
from sysone.domain.work import CompilationTimes
from sysone.engine.stats import RequestStats, RunStats, StageStats, StartupStats
from sysone.errors import ConfigError, NotReadyError
from sysone.service import app as service_app


class FakeEngine:
    """按题目数产出答案；`hold` 给定时会按住不放，用来把准入占满。"""

    def __init__(self, *, hold: threading.Event | None = None) -> None:
        self.calls = 0
        self._hold = hold

    def run(self, request: DecisionRequest) -> DecisionResult:
        self.calls += 1
        if self._hold is not None:
            self._hold.wait(10.0)
        answers = []
        for index, question in enumerate(request.questions):
            if question.kind == "noul":
                answers.append(
                    Answer(question_index=index, kind="noul", value="yes", probs=(0.8, 0.2),
                           labels=("yes", "no"), confidence=0.8)
                )
            else:
                answers.append(
                    Answer(question_index=index, kind="choice", value="b", probs=(0.25, 0.75),
                           labels=("a", "b"), confidence=0.75)
                )
        return DecisionResult(
            answers=tuple(answers),
            usage=Usage(input_tokens=10, prefill_calls=len(answers)),
            provenance=Provenance(
                family="fake",
                artifact_id="artifact",
                quant=QuantSpec(),
                attention_mode=MaskMode.PREFIX_NONCAUSAL,
                attention_trained=True,
            ),
        )

    def stats(self) -> RunStats:
        """`/metrics` 要的三层数字；请求数就是跑过几道。"""
        return RunStats(
            request=RequestStats(requests=self.calls),
            stage=StageStats(),
            startup=StartupStats(),
        )

    def image_contract(self) -> dict:
        """这个替身没有视觉：协议层据此拒图，并把 `unavailable` 原样回出去。"""
        from sysone.models.vision.contract import image_contract

        return image_contract(unavailable="这个替身没有视觉塔")

    def vision_stats(self) -> dict[str, object]:
        return {"available": False}

    def decision_stats(self) -> dict[str, object] | None:
        """这个替身不认决策 runner 的契约：`/metrics` 里就该是空的，而不是一排伪造的 0。"""
        return None


def _decider(engine: FakeEngine | None = None) -> service_app.EngineDecider:
    return service_app.EngineDecider(
        engine or FakeEngine(), model_name="fake", max_input_tokens=16384
    )


@pytest.fixture
def start_server():
    servers: list[service_app.SystemOneServer] = []

    def start(*, load=None, engine=None, api_key=None, queue=2, workers=1, debug=False):
        decider = None if load else _decider(engine)
        server, gate = service_app.serve(
            load=load or (lambda: (decider, CompilationTimes(0.0, 0.0, 0.0))),
            host="127.0.0.1",
            port=0,
            api_key=api_key,
            workers=workers,
            queue=queue,
            debug_timings=debug,
        )
        threading.Thread(target=server.serve_forever, name="test-http", daemon=True).start()
        servers.append(server)
        return server, gate

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def _call(server, method, path, *, body=None, headers=None):
    host, port = server.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        payload = None
        if body is not None:
            payload = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        conn.request(method, path, body=payload, headers=headers or {})
        response = conn.getresponse()
        raw = response.read()
        return response.status, {name.lower(): value for name, value in response.getheaders()}, raw
    finally:
        conn.close()


def _json(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


def _raw(server, request: bytes) -> bytes:
    host, port = server.server_address[:2]
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                return b"".join(chunks)
            chunks.append(data)


def _wait_until(predicate, timeout: float = 5.0, step: float = 0.005) -> bool:
    """轮询到条件成立；准入的计数是在回包之后才还的，所以客户端返回时可能还没落。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        threading.Event().wait(step)
    return predicate()


CHOICE = {"questions": {"q1": {"type": "choice", "instructions": "选哪个？", "criteria": {"a": None, "b": None}}}}


# ---------------------------------------------------------------------- 读路由


def test_health_reports_ready(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "GET", "/health")
    assert status == 200
    assert headers["content-type"] == "application/json"
    body = _json(raw)
    assert body["status"] == "ready"
    assert body["model"] == "fake"
    assert body["max_input_tokens"] == 16384
    assert body["modalities"] == ["text"]
    assert body["authentication"] is False


def test_health_answers_get_only(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "HEAD", "/health")
    assert status == 405
    assert headers["allow"] == "GET"
    # HEAD 不写 body（与上游一致），只留下 Content-Length。
    assert raw == b""
    assert int(headers["content-length"]) > 0


def test_models_lists_the_served_model(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, _, raw = _call(server, "GET", "/v1/models")
    assert status == 200
    entry = _json(raw)["models"][0]
    assert entry["name"] == "fake"
    assert entry["modalities"] == ["text"]


def test_unknown_path_is_404_json(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "GET", "/nope")
    assert status == 404
    assert headers["content-type"] == "application/json"
    assert "nothing is served" in _json(raw)["detail"]


def test_trailing_slash_redirects(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, headers, _ = _call(server, "POST", "/v1/systemone/", body=CHOICE)
    assert status == 307
    assert headers["location"].endswith("/v1/systemone")


def test_wrong_method_is_405(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, headers, _ = _call(server, "GET", "/v1/systemone")
    assert status == 405
    assert headers["allow"] == "POST"


# ---------------------------------------------------------------------- 决策


def test_systemone_answers_a_choice_question(start_server) -> None:
    engine = FakeEngine()
    server, gate = start_server(engine=engine)
    assert gate.wait(5.0)
    status, _, raw = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert status == 200
    body = _json(raw)
    assert body["model"] == "fake"
    assert body["answers"]["q1"] == {
        "type": "choice",
        "choice": "b",
        "probabilities": {"a": 0.25, "b": 0.75},
        "confidence": 0.75,
    }
    assert body["usage"] == {"input_tokens": 10, "output_tokens": 0}
    assert body["attention_mode"] == "prefix_noncausal"
    assert body["attention_trained"] is True
    assert engine.calls == 1


def test_systemone_answers_a_noul_question(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, _, raw = _call(server, "POST", "/v1/systemone",
                           body={"questions": {"q1": {"type": "noul", "instructions": "要看吗？"}}})
    assert status == 200
    assert _json(raw)["answers"]["q1"] == {"type": "noul", "noul": 0.8}


def test_timings_are_returned_only_when_debugging(start_server) -> None:
    server, gate = start_server(debug=True)
    assert gate.wait(5.0)
    _, _, raw = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert "timings" not in _json(raw)  # FakeEngine 不给 timings，也不该凭空造一份


def test_unreadable_body_is_422(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, _, raw = _call(server, "POST", "/v1/systemone", body=b"{ not json")
    assert status == 422
    detail = _json(raw)["detail"]
    assert isinstance(detail, str) and "JSON" in detail


def test_unknown_field_is_422(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    status, _, raw = _call(server, "POST", "/v1/systemone", body={"questions": {}, "x": 1})
    assert status == 422
    assert isinstance(_json(raw)["detail"], str)


def test_invalid_question_is_422_with_ids(start_server) -> None:
    engine = FakeEngine()
    server, gate = start_server(engine=engine)
    assert gate.wait(5.0)
    body = {"questions": {"bad": {"type": "score"}, "good": {"type": "noul"}}}
    status, _, raw = _call(server, "POST", "/v1/systemone", body=body)
    assert status == 422
    assert set(_json(raw)["detail"]["invalid_questions"]) == {"bad"}
    assert engine.calls == 0  # 有坏题就什么都不跑


def test_engine_errors_become_their_status(start_server) -> None:
    class NotReady(FakeEngine):
        def run(self, request):
            raise NotReadyError("这个模型还没准备好", mode="causal")

    server, gate = start_server(engine=NotReady())
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert status == 503
    assert _json(raw)["detail"] == "这个模型还没准备好"
    assert headers["retry-after"] == "1"


def test_fatal_engine_errors_do_not_ask_the_client_to_retry(start_server) -> None:
    class Broken(FakeEngine):
        def run(self, request):
            raise ConfigError("几何与权重不符", cout=5120)

    server, gate = start_server(engine=Broken())
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert status == 500
    assert _json(raw)["detail"] == "几何与权重不符"
    assert "retry-after" not in headers


# ---------------------------------------------------------------------- 鉴权


def test_bearer_key_is_required_for_protected_routes(start_server) -> None:
    server, gate = start_server(api_key="s3cret")
    assert gate.wait(5.0)
    status, headers, raw = _call(server, "GET", "/v1/models")
    assert status == 401
    assert headers["www-authenticate"] == "Bearer"
    assert _json(raw)["detail"].endswith("bearer key is required")

    status, _, _ = _call(server, "GET", "/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert status == 200
    # 摘要比较：错一个字符也不行。
    status, _, _ = _call(server, "GET", "/v1/models", headers={"Authorization": "Bearer s3cret "})
    assert status == 401


def test_health_stays_open_and_reports_authentication(start_server) -> None:
    server, gate = start_server(api_key="k")
    assert gate.wait(5.0)
    status, _, raw = _call(server, "GET", "/health")
    assert status == 200
    assert _json(raw)["authentication"] is True


# ---------------------------------------------------------------------- 就绪与准入


def test_not_ready_is_503_with_retry_after(start_server) -> None:
    release = threading.Event()

    def load():
        release.wait(10.0)
        return _decider(), CompilationTimes(0.0, 0.0, 0.0)

    server, gate = start_server(load=load)
    for path, body in (
        ("/health", None),
        ("/v1/systemone", CHOICE),
        ("/v1/models", None),
        ("/metrics", None),
    ):
        status, headers, raw = _call(server, "POST" if body else "GET", path, body=body)
        assert status == 503, path
        assert headers["retry-after"] == "1"
        assert _json(raw)["detail"] == "引擎尚未装载完成"

    release.set()
    assert gate.wait(5.0)
    status, _, _ = _call(server, "GET", "/health")
    assert status == 200


def test_load_failure_keeps_the_gate_failed(start_server) -> None:
    def load():
        raise ConfigError("权重目录里没有 safetensors", path="/x")

    server, gate = start_server(load=load)
    assert gate.wait(5.0) is False
    assert gate.status == "failed"
    assert isinstance(gate.error, ConfigError)
    status, _, raw = _call(server, "GET", "/health")
    assert status == 503


def test_capacity_is_enforced_before_the_body(start_server) -> None:
    hold = threading.Event()
    server, gate = start_server(engine=FakeEngine(hold=hold), queue=0)
    assert gate.wait(5.0)
    runner = threading.Thread(
        target=lambda: _call(server, "POST", "/v1/systemone", body=CHOICE), daemon=True
    )
    runner.start()
    assert _wait_until(lambda: server.admission.snapshot()[0] == 1)

    status, headers, raw = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert status == 429
    assert headers["retry-after"] == "1"
    assert headers["connection"] == "close"
    assert _json(raw)["detail"].startswith("at capacity")

    hold.set()
    runner.join(5.0)
    assert not runner.is_alive()
    assert _wait_until(lambda: server.admission.snapshot() == (0, 0))


def test_a_queue_waits_instead_of_refusing(start_server) -> None:
    hold = threading.Event()
    server, gate = start_server(engine=FakeEngine(hold=hold), queue=1)
    assert gate.wait(5.0)
    first = threading.Thread(
        target=lambda: _call(server, "POST", "/v1/systemone", body=CHOICE), daemon=True
    )
    first.start()
    assert _wait_until(lambda: server.admission.snapshot()[0] == 1)
    second = threading.Thread(
        target=lambda: _call(server, "POST", "/v1/systemone", body=CHOICE), daemon=True
    )
    second.start()
    assert _wait_until(lambda: server.admission.snapshot() == (1, 1))  # 一个在跑、一个在等

    hold.set()
    first.join(5.0)
    second.join(5.0)
    assert _wait_until(lambda: server.admission.snapshot() == (0, 0))


# ---------------------------------------------------------------------- 请求体


def _chunked_request(payload: bytes) -> bytes:
    return (
        b"POST /v1/systemone HTTP/1.1\r\nHost: t\r\nTransfer-Encoding: chunked\r\n"
        b"Connection: close\r\n\r\n" + b"%x\r\n" % len(payload) + payload + b"\r\n0\r\n\r\n"
    )


def _status_line(response: bytes) -> int:
    return int(response.split(b"\r\n", 1)[0].split()[1])


def test_chunked_bodies_are_accepted(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    payload = json.dumps(CHOICE).encode("utf-8")
    response = _raw(server, _chunked_request(payload))
    assert _status_line(response) == 200
    assert b'"choice":"b"' in response


def test_content_length_and_chunked_together_is_400(start_server) -> None:
    server, gate = start_server()
    assert gate.wait(5.0)
    request = (
        b"POST /v1/systemone HTTP/1.1\r\nHost: t\r\nContent-Length: 5\r\n"
        b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
    )
    response = _raw(server, request)
    assert _status_line(response) == 400
    assert b"either Content-Length" in response


# ---------------------------------------------------------------------- 启动报告


def test_ready_line_reports_the_startup_breakdown() -> None:
    """就绪行带三段拆分。冷产物目录上编译占掉绝大部分、第二次启动它接近 0，只印总耗时
    看不出这个差异（`plan.md` R6 要的就是这一条）。"""
    note = service_app.startup_report(
        CompilationTimes(load_s=0.85, compile_s=10.62, warmup_s=0.57)
    )
    assert note == " (load 0.85s, compile 10.62s, warmup 0.57s)"
    # 装载还没结束时 `Gate.startup` 是 None：不印假的 0.00，宁可什么都不加。
    assert service_app.startup_report(None) == ""


# ---------------------------------------------------------------------- /metrics


def test_metrics_report_the_engine_layers_and_the_admission(start_server) -> None:
    """一个诊断口，两块数字：准入是服务这一层的账，三层统计是引擎的。"""
    server, gate = start_server()
    assert gate.wait(5.0)
    assert _call(server, "POST", "/v1/systemone", body=CHOICE)[0] == 200
    assert _wait_until(lambda: server.admission.snapshot() == (0, 0))

    status, _, raw = _call(server, "GET", "/metrics")
    assert status == 200
    payload = _json(raw)
    # 默认 workers=1 + queue=2，一条请求进来跑完：在册数归零、累计 1、没有拒绝。
    assert payload["admission"] == {
        "active": 0,
        "waiting": 0,
        "entered": 1,
        "rejected": 0,
        "capacity": 3,
    }
    assert payload["engine"]["request"]["requests"] == 1
    assert set(payload["engine"]["request"]) == {"requests", "questions", "tokens"}
    assert set(payload["engine"]["stage"]) == {"prefill_s", "readout_s"}
    assert set(payload["engine"]["startup"]) == {"load_s", "compile_s", "warmup_s"}
    # 不认决策 runner 契约的替身：这一栏是空的，不是一排伪造的 0。
    assert payload["engine"]["decision"] is None


def test_metrics_count_admission_refusals(start_server) -> None:
    """被准入挡掉的那一条记进 `rejected`：429 是"门没让进"，不是"跑失败"。"""
    hold = threading.Event()
    server, gate = start_server(engine=FakeEngine(hold=hold), queue=0)
    assert gate.wait(5.0)
    running = threading.Thread(
        target=lambda: _call(server, "POST", "/v1/systemone", body=CHOICE), daemon=True
    )
    running.start()
    assert _wait_until(lambda: server.admission.snapshot()[0] == 1)

    status, _, _ = _call(server, "POST", "/v1/systemone", body=CHOICE)
    assert status == 429

    hold.set()
    running.join(5.0)
    assert _wait_until(lambda: server.admission.snapshot() == (0, 0))
    status, _, raw = _call(server, "GET", "/metrics")
    assert status == 200
    admission = _json(raw)["admission"]
    assert admission["entered"] == 1
    assert admission["rejected"] == 1
    assert admission["capacity"] == 1
    assert admission["active"] == 0


def test_metrics_carry_the_decision_counters_and_capabilities(start_server) -> None:
    """`/metrics` 的 `decision` 一栏来自执行器：能力位与命中数一起给出，读的人不用去翻代码。"""

    class DecisionEngine(FakeEngine):
        def decision_stats(self) -> dict[str, object]:
            return {
                "shared_hit": 1, "shared_miss": 2, "batch_hit": 0, "batch_miss": 0,
                "batch_fallback": 0, "media_skip": 1,
                "supports_shared": True, "supports_batch": False,
            }

    server, gate = start_server(engine=DecisionEngine())
    assert gate.wait(5.0)
    status, _, raw = _call(server, "GET", "/metrics")
    assert status == 200
    decision = _json(raw)["engine"]["decision"]
    assert decision["supports_shared"] is True and decision["supports_batch"] is False
    assert decision["shared_hit"] == 1 and decision["media_skip"] == 1


def test_an_engine_that_cannot_batch_is_not_given_a_batcher() -> None:
    """`batch_limit == 1` 时不该建攒批器：它会绕过 `Engine.run`，把每条请求送进 `Engine.batch`，
    而那条路要求执行器有 `answers_batch`（h2o / vendor 没有）—— 建了就是每条请求都 500。
    """

    class SerialEngine(FakeEngine):
        def batch(self, requests):
            raise AssertionError("batch_limit = 1 的引擎不该被送进合批路径")

        @property
        def batch_limit(self) -> int:
            return 1

    decider = service_app.EngineDecider(SerialEngine(), model_name="m", max_input_tokens=16384)
    assert decider.batcher is None
    assert decider.metrics()["batching"] is None
    assert decider.answer(CHOICE)["answers"]
