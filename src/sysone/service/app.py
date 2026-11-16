"""前门：`ThreadingHTTPServer` + 准入 + 就绪门，把请求交给一个已装载的引擎。

HTTP 层从 mude-bench 的 `d3_server.py` 逐字搬来（标准库 `http.server`，零依赖），只在三处加
我们自己的语义：

1. **就绪门**：引擎在后台线程里装载，装好之前所有路由回 503 + `Retry-After`（`/health` 同），
   于是探针拿到的是"还没好"而不是"连接被拒"；
2. **准入**：`/v1/systemone` 在**读 body 之前**先要一个执行位，满了立刻 429 + `Retry-After` +
   `Connection: close`（body 没读，这条连接不能复用）；
3. **引擎错误**：`SysOneError` 按自带的 `status` 回 `{"detail": message}`。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Protocol

from ..domain.decision import DecisionRequest, DecisionResult
from ..domain.work import CompilationTimes
from ..engine.stats import RunStats
from ..errors import NotReadyError, SysOneError
from . import protocol
from .admission import Admission

IDLE_SECONDS = 120
RETRY_AFTER_SECONDS = 1


def _startup_report(times: CompilationTimes | None) -> str:
    """`ready in` 那一行的后缀：装载 / 编译 / 预热三段（秒）。

    只印总耗时看不出首启为什么要等那么久：冷产物目录上编译占掉绝大部分，第二次启动它
    接近 0，两次的 `ready in` 差一个数量级却看不出原因（`plan.md` 的 R6）。
    """
    if times is None:
        return ""
    return (
        f" (load {times.load_s:.2f}s, compile {times.compile_s:.2f}s,"
        f" warmup {times.warmup_s:.2f}s)"
    )


class EngineSource(Protocol):
    """`EngineDecider` 从引擎要的两件事：跑一道请求、给三层统计。"""

    def run(self, request: DecisionRequest) -> DecisionResult: ...

    def stats(self) -> RunStats: ...


class EngineDecider:
    """把引擎包成前门要的形状：健康信息、模型信息、跑一道请求、报三层统计。"""

    def __init__(
        self,
        engine: EngineSource,
        *,
        model_name: str,
        max_input_tokens: int,
        debug_timings: bool = False,
    ) -> None:
        self.engine = engine
        self.model_name = model_name
        self.max_input_tokens = int(max_input_tokens)
        self.debug_timings = bool(debug_timings)

    def health(self, *, authentication: bool) -> dict[str, Any]:
        return protocol.health_payload(
            model_name=self.model_name,
            max_input_tokens=self.max_input_tokens,
            authentication=authentication,
        )

    def metrics(self) -> dict[str, Any]:
        """引擎那三层的数字（`RunStats`）。准入的计数属于服务这一层，在 `_answer` 里合并。"""
        stats = self.engine.stats()
        return {
            "request": {
                "requests": stats.request.requests,
                "questions": stats.request.questions,
                "tokens": stats.request.tokens,
            },
            "stage": {
                "prefill_s": round(stats.stage.prefill_s, 3),
                "readout_s": round(stats.stage.readout_s, 3),
            },
            "startup": {
                "load_s": round(stats.startup.load_s, 3),
                "compile_s": round(stats.startup.compile_s, 3),
                "warmup_s": round(stats.startup.warmup_s, 3),
            },
        }

    def models(self) -> dict[str, Any]:
        return protocol.models_payload(
            model_name=self.model_name, max_input_tokens=self.max_input_tokens
        )

    def answer(self, body: Any) -> dict[str, Any]:
        """一次 `/v1/systemone`：解析 → 引擎 → 回包（引擎的异常原样往上抛）。"""
        request, wire = protocol.parse_request(body)
        result = self.engine.run(request)  # type: ignore[attr-defined]
        return protocol.answer_payload(
            result,
            model_name=self.model_name,
            wire=wire,
            debug_timings=self.debug_timings,
        )


class Gate:
    """装载进度：`loading` → `ready` / `failed`。前门靠它区分 503 与 200。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._status = "loading"
        self._error: BaseException | None = None
        self._decider: EngineDecider | None = None
        self.startup: CompilationTimes | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def decider(self) -> EngineDecider:
        with self._lock:
            if self._decider is None:
                raise NotReadyError("引擎尚未装载完成", status=self._status)
            return self._decider

    def start(self, load: Callable[[], tuple[EngineDecider, CompilationTimes]]) -> None:
        """后台装载：`load()` 返回 (decider, CompilationTimes)；失败记下错误并置为 failed。"""
        thread = threading.Thread(target=self._load, args=(load,), name="sysone-load", daemon=True)
        thread.start()

    def wait(self, timeout: float | None = None) -> bool:
        """等到装载结束；返回是否装载成功。"""
        self._ready.wait(timeout)
        return self.status == "ready"

    def _load(self, load: Callable[[], tuple[EngineDecider, CompilationTimes]]) -> None:
        try:
            decider, startup = load()
        except BaseException as exc:  # noqa: BLE001 - 装载失败要作为状态回出去
            with self._lock:
                self._status = "failed"
                self._error = exc
            self._ready.set()
            return
        with self._lock:
            self._decider = decider
            self.startup = startup
            self._status = "ready"
        self._ready.set()


class SystemOneServer(ThreadingHTTPServer):
    """一个监听口；每个连接一个线程；引擎那一侧由准入限流。"""

    daemon_threads = True
    # 标准库默认的 backlog 只有 5，压测时会排队在 TCP 层而不是我们的准入里。
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        gate: Gate,
        *,
        api_key: str | None = None,
        admission: Admission | None = None,
        debug_timings: bool = False,
    ) -> None:
        super().__init__(address, SystemOneHandler)
        self.gate = gate
        self.key_digest = hashlib.sha256(api_key.encode("utf-8")).digest() if api_key else None
        self.admission = admission or Admission()
        self.debug_timings = bool(debug_timings)


class SystemOneHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "sysone"
    sys_version = ""
    timeout = IDLE_SECONDS
    # 头与 body 是两次写；没有 TCP_NODELAY 时 body 要等客户端的延迟 ACK。
    disable_nagle_algorithm = True
    default_request_version = "HTTP/1.1"
    error_content_type = "text/plain; charset=utf-8"
    error_message_format = "%(code)d: %(explain)s\n"

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        # 不支持的 HTTP 版本按坏请求回，与上游一致。
        if code == HTTPStatus.HTTP_VERSION_NOT_SUPPORTED:
            code = HTTPStatus.BAD_REQUEST
        super().send_error(code, message, explain)

    def do_GET(self) -> None:
        self.route("GET")

    def do_POST(self) -> None:
        self.route("POST")

    def do_PUT(self) -> None:
        self.route("PUT")

    def do_PATCH(self) -> None:
        self.route("PATCH")

    def do_DELETE(self) -> None:
        self.route("DELETE")

    def do_OPTIONS(self) -> None:
        self.route("OPTIONS")

    def do_HEAD(self) -> None:
        self.route("HEAD")

    # ------------------------------------------------------------------ 路由

    def route(self, method: str) -> None:
        """先过准入（读 body 之前），再读 body，最后分派 / 重定向 / 405 / 404。"""
        # 目标按原样取：http.server 会把开头的 "//" 折叠掉。
        path, mark, query = self.requestline.split()[1].partition("?")
        admitted = False
        if path == protocol.SYSTEM_ONE_PATH and method == protocol.ROUTES[protocol.SYSTEM_ONE_PATH]:
            if not self.server.admission.enter():
                on_duty, waiting = self.server.admission.snapshot()
                self.close_connection = True
                self._reply(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"detail": f"at capacity: {on_duty} in flight, {waiting} waiting"},
                    {"Retry-After": str(self.server.admission.retry_after)},
                )
                return
            admitted = True
        try:
            try:
                raw = self.read_body()
            except protocol.Framing as exc:
                self.close_connection = True
                self.send_text(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._dispatch(method, path, mark, query, raw)
        finally:
            if admitted:
                self.server.admission.leave()

    def _dispatch(self, method: str, path: str, mark: str, query: str, raw: bytes) -> None:
        try:
            allowed = protocol.ROUTES.get(path)
            if allowed is None:
                target = path.rstrip("/")
                if path.endswith("/") and target in protocol.ROUTES:
                    self.redirect(target + mark + query)
                    return
                raise protocol.Refusal(HTTPStatus.NOT_FOUND, f"nothing is served at {path}")
            if method != allowed:
                raise protocol.Refusal(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    f"{path} answers {allowed} only",
                    {"Allow": allowed},
                )
            if path in protocol.PROTECTED and not self.authorized():
                raise protocol.Refusal(
                    HTTPStatus.UNAUTHORIZED,
                    "a valid bearer key is required",
                    {"WWW-Authenticate": "Bearer"},
                )
            reply = self._answer(path, raw)
        except protocol.Refusal as refusal:
            self._reply(refusal.status, {"detail": refusal.detail}, refusal.headers)
            return
        except SysOneError as exc:
            # 只有"还没好"这一类才带 `Retry-After`；配置/形状错误重试也没用。
            headers = (
                {"Retry-After": str(RETRY_AFTER_SECONDS)}
                if exc.status == HTTPStatus.SERVICE_UNAVAILABLE
                else {}
            )
            self._reply(exc.status, {"detail": exc.message}, headers)
            return
        except ValueError as exc:  # 读出形状不合法（`answer_payload` 的校验）
            self._reply(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - 客户端拿到纯文本 500，服务继续
            self.log_error("system one request failed: %s: %s", type(exc).__name__, exc)
            self.send_text(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"the server failed on this request ({type(exc).__name__})",
            )
            return
        self._reply(HTTPStatus.OK, reply)

    def _answer(self, path: str, raw: bytes) -> dict[str, Any]:
        if path == protocol.SYSTEM_ONE_PATH:
            # 未就绪时先回 503：此时连模型信息都还没定下来，答案更不可能对。
            return self.server.gate.decider.answer(self.parse(raw))
        if path == protocol.MODELS_PATH:
            return self.server.gate.decider.models()
        if path == protocol.METRICS_PATH:
            # 准入是服务这一层的账（引擎不知道门的存在），所以在这里合并两边的数字。
            return {
                "admission": self.server.admission.stats(),
                "engine": self.server.gate.decider.metrics(),
            }
        return self.server.gate.decider.health(authentication=self.server.key_digest is not None)

    def authorized(self) -> bool:
        """没配 key 就放行；配了就必须是 `Authorization: Bearer <key>`（摘要恒定时间比较）。"""
        expected = self.server.key_digest
        if expected is None:
            return True
        scheme, space, credential = self.headers.get("Authorization", "").partition(" ")
        offered = hashlib.sha256(credential.encode("utf-8")).digest()
        return scheme == "Bearer" and space == " " and hmac.compare_digest(offered, expected)

    # ------------------------------------------------------------------ 读写

    def read_body(self) -> bytes:
        """请求体：Content-Length、chunked，或干脆没有。"""
        coding = self.headers.get("Transfer-Encoding")
        size = self.headers.get("Content-Length")
        if coding is not None:
            if size is not None or coding.strip().lower() != "chunked":
                raise protocol.Framing("use either Content-Length or Transfer-Encoding: chunked")
            return self.read_chunks()
        if size is None:
            return b""
        size = size.strip()
        if not size.isdigit():
            raise protocol.Framing("Content-Length must be a byte count")
        raw = self.rfile.read(int(size))
        if len(raw) != int(size):
            raise protocol.Framing("the connection closed before Content-Length bytes arrived")
        return raw

    def read_chunks(self) -> bytes:
        parts = []
        while True:
            line = self.rfile.readline(1 << 16)
            digits = line.split(b";", 1)[0].strip()
            if not digits or any(char not in b"0123456789abcdefABCDEF" for char in digits):
                raise protocol.Framing("a chunk must start with its size in hexadecimal")
            size = int(digits, 16)
            if size == 0:
                while self.rfile.readline(1 << 16) not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            chunk = self.rfile.read(size)
            if len(chunk) != size or self.rfile.readline(3) not in (b"\r\n", b"\n"):
                raise protocol.Framing("a chunk is shorter than its size")
            parts.append(chunk)

    def parse(self, raw: bytes) -> Any:
        return protocol.parse_body(raw)

    def redirect(self, target: str) -> None:
        host = self.headers.get("Host") or "%s:%d" % self.server.server_address[:2]
        self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
        self.send_header("Location", f"http://{host}{target}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_text(self, status: HTTPStatus, text: str) -> None:
        data = f"{int(status)}: {text}\n".encode("utf-8")
        self._emit(status, "text/plain; charset=utf-8", data)

    def _reply(self, status: HTTPStatus, body: Any, headers: dict[str, str] | None = None) -> None:
        try:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode(
                "utf-8"
            )
        except ValueError:
            status, headers = HTTPStatus.INTERNAL_SERVER_ERROR, None
            data = b'{"detail":"the reply holds a number that JSON cannot carry"}'
        self._emit(status, protocol.JSON_TYPE, data, headers)

    def _emit(
        self,
        status: HTTPStatus,
        content_type: str,
        data: bytes,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


def serve(
    *,
    load: Callable[[], tuple[EngineDecider, CompilationTimes]],
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    workers: int = 1,
    queue: int = 2,
    retry_after: int = RETRY_AFTER_SECONDS,
    admit_timeout: float = 120.0,
    debug_timings: bool = False,
) -> tuple[SystemOneServer, Gate]:
    """绑好监听口、后台开始装载，返回 (server, gate)；调用方负责 `serve_forever()`。"""
    gate = Gate()
    server = SystemOneServer(
        (host, int(port)),
        gate,
        api_key=api_key,
        admission=Admission(
            workers=workers, queue=queue, retry_after=retry_after, timeout=admit_timeout
        ),
        debug_timings=debug_timings,
    )
    gate.start(load)
    return server, gate


def run(
    *,
    load: Callable[[], tuple[EngineDecider, CompilationTimes]],
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str | None = None,
    workers: int = 1,
    queue: int = 2,
    retry_after: int = RETRY_AFTER_SECONDS,
    admit_timeout: float = 120.0,
    debug_timings: bool = False,
    announce: bool = True,
) -> int:
    """前台服务：边装载边回 503，装载完成后（或失败时）由调用方决定退出码。"""
    server, gate = serve(
        load=load,
        host=host,
        port=port,
        api_key=api_key,
        workers=workers,
        queue=queue,
        retry_after=retry_after,
        admit_timeout=admit_timeout,
        debug_timings=debug_timings,
    )
    if announce:
        hostname, portname = server.server_address[:2]
        access = "bearer key required" if api_key else "open (set DECISION_API_KEY to require a key)"
        print(
            f"sysone serve: http://{hostname}:{portname}{protocol.SYSTEM_ONE_PATH}; {access}",
            flush=True,
        )
    started = time.perf_counter()
    http = threading.Thread(target=server.serve_forever, name="sysone-http", daemon=True)
    http.start()
    try:
        if not gate.wait():
            error = gate.error
            if isinstance(error, SysOneError):
                raise error
            raise NotReadyError("引擎装载失败", reason=repr(error))
        if announce:
            elapsed = time.perf_counter() - started
            print(
                f"sysone serve: ready in {elapsed:.2f}s{_startup_report(gate.startup)}",
                flush=True,
            )
        threading.Event().wait()  # 交给 Ctrl-C
    except KeyboardInterrupt:
        pass
    finally:
        # `shutdown()` 只能在 serve_forever 跑着的时候调，所以它一定在 http 线程里。
        server.shutdown()
        server.server_close()
    return 0
