"""split 策略的管线测试（假链 + 假分词器，不依赖 ANE / transformers）。

真链只在设了 `SYS1_MUDE_WEIGHTS` 的集成测试里跑；这里验证的是"题面 → id → 一次整链
前向 → 答案"这段接线的语义：线格式的字节、前缀切在哪、noul 的序号、模式门禁、
超长与预热失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from helpers import OPTIONS, load_tiny_descriptor, make_request

from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.work import CompilationTimes
from sysone.engine.engine import Engine
from sysone.engine.strategies.split import SplitRunner
from sysone.errors import (
    ConfigError,
    NotReadyError,
    RequestTooLongError,
    UnsupportedKindError,
    UnsupportedModeError,
)
from sysone.models.families import mude

CODES = tuple(f"K{index:03d}" for index in range(mude.CODE_ROWS))


class FakeTokenizer:
    """字符级假分词器（`token = ord(ch)`）：chat 模板只加定位用的标记，user 内容原样保留。"""

    def __init__(self) -> None:
        self.chat_calls = 0

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.chat_calls += 1
        return f"<|system|>{conversation[0]['content']}<|user|>{conversation[1]['content']}<|assistant|>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(char) for char in text]}


class FakeChain:
    """只记账、按 `options` 给均匀概率；`shares` 可覆盖成不等概率。"""

    def __init__(self, *, shares: tuple[float, ...] | None = None) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int]] = []
        self.closed = False
        self._shares = shares

    def forward(self, tokens, *, options, prefix=0, causal=False):
        self.calls.append((tuple(tokens), int(options), int(prefix), int(bool(causal))))
        if self._shares is not None:
            values = list(self._shares[:options])
            return [value / sum(values) for value in values]
        return [1.0 / options] * options

    def close(self) -> None:
        self.closed = True


def _prompter(tokenizer: FakeTokenizer | None = None) -> tuple[mude.MudePrompter, FakeTokenizer]:
    tokenizer = tokenizer or FakeTokenizer()
    return mude.MudePrompter(tokenizer, CODES), tokenizer


def _config(mode: MaskMode | None = None, max_len: int = 4096) -> EngineConfig:
    return EngineConfig(
        model=ModelSelection(name="fake", dir="."),
        scheduler=SchedulerConfig(attention_mode=mode, max_len=max_len),
        service=ServiceConfig(debug_timings=True),
    )


def _runner(
    chain: FakeChain, *, max_len: int = 4096, tokenizer: FakeTokenizer | None = None
) -> SplitRunner:
    prompter, _ = _prompter(tokenizer)
    return SplitRunner(lambda: chain, prompter, max_len=max_len)


def _engine(chain: FakeChain, config: EngineConfig | None = None, *, max_len: int = 4096) -> Engine:
    from sysone.devices.reference.encoding import CharEncoder

    descriptor = load_tiny_descriptor()
    return Engine(
        descriptor=descriptor,
        runner=_runner(chain, max_len=max_len),
        encoder=CharEncoder(vocab=descriptor.vocab),
        config=config or _config(),
    )


# ---------------------------------------------------------------------- 线格式


def test_option_plan_choice_keeps_options_in_order() -> None:
    question = Question(text="选一个", options=("甲", "乙", "丙"))
    texts, labels, indexes = mude.option_plan(question)
    assert texts == ("甲", "乙", "丙")
    assert labels == ("甲", "乙", "丙")
    assert indexes == (0, 1, 2)


def test_option_plan_noul_reverses_the_prompt_order() -> None:
    """prompt 里是 "No / false, Yes / true"，答案标签仍是 ("yes", "no") ⇒ 序号反着放。"""
    texts, labels, indexes = mude.option_plan(Question(text="要看这个文件吗？", kind="noul"))
    assert texts == mude._NOUL_TEXTS
    assert labels == ("yes", "no")
    assert indexes == (1, 0)


def test_option_plan_refuses_a_kind_the_code_table_cannot_answer() -> None:
    """码表读出没有 score 的档位模板：这是**坏请求**（400），不是我们的配置坏了（500）。"""
    with pytest.raises(UnsupportedKindError) as info:
        mude.option_plan(Question(text="打分", kind="score"))
    assert info.value.status == 400
    assert info.value.context == {"kind": "score", "supported": ["choice", "noul"]}
    with pytest.raises(ConfigError):
        mude.option_plan(Question(text="选一个"))


def test_render_messages_matches_the_wire_format() -> None:
    question = Question(text="选哪个？", options=("甲", "乙"))
    messages = mude.render_messages("只有这一句", question, ("甲", "乙"), CODES)
    assert messages[0] == {"role": "system", "content": mude.SYSTEM_PROMPT}
    assert messages[1]["content"] == (
        "State:\n"
        "只有这一句\n"
        "\n"
        "Question:\n"
        "选哪个？\n"
        "\n"
        "Options:\n"
        "K000: 甲\n"
        "K001: 乙\n"
        "\n"
        "Reply with only the code of the best option."
    )


def test_render_messages_falls_back_for_empty_state_and_text() -> None:
    messages = mude.render_messages("", Question(text="", options=("甲",)), ("甲",), CODES)
    assert "State:\n(empty)\n" in messages[1]["content"]
    assert "Question:\nChoose the best matching option.\n" in messages[1]["content"]


# ---------------------------------------------------------------------- 题面


def test_prompter_cuts_the_prefix_right_before_the_question() -> None:
    prompter, _ = _prompter()
    question = Question(text="选哪个？", options=("甲", "乙"))
    sequence, prefix_len, labels, indexes = prompter.encode("只有这一句", question)
    assert labels == ("甲", "乙")
    assert indexes == (0, 1)
    # 前缀停在"Question:"换行之后，所以前缀之后第一个 token 就是题干的第一个字符。
    assert prefix_len > 0
    assert sequence[prefix_len] == ord(question.text[0])
    assert sequence[prefix_len - 10:prefix_len] == tuple(ord(char) for char in "Question:\n")


def test_prompter_caches_the_prefix_for_one_state() -> None:
    prompter, tokenizer = _prompter()
    question = Question(text="选哪个？", options=("甲", "乙"))
    prompter.encode("同一个 state", question)
    calls = tokenizer.chat_calls
    prompter.encode("同一个 state", question)
    # 同一 state 的第二题不再重渲染哨兵题（第二题只多一次正常模板）。
    assert tokenizer.chat_calls - calls == 1


def test_prompter_requires_the_sentinel_to_survive_the_template() -> None:
    class StrippingTokenizer(FakeTokenizer):
        def apply_chat_template(self, conversation, **kwargs):
            text = super().apply_chat_template(conversation, **kwargs)
            return text.replace(mude.PREFIX_SENTINEL, "")

    prompter, _ = _prompter(StrippingTokenizer())
    with pytest.raises(ConfigError):
        prompter.encode("s", Question(text="q?", options=("甲",)))


# ---------------------------------------------------------------------- 装配


def test_answers_cover_every_question_and_forward_the_prefix() -> None:
    chain = FakeChain()
    engine = _engine(chain)
    engine.start()

    request = make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=2)
    result = engine.run(request)

    assert len(chain.calls) == 3  # 预热一次 + 两道题
    assert len(result.answers) == 2
    assert result.answers[0].question_index == 0
    assert result.answers[0].labels == ("A", "B", "C", "D")
    assert result.answers[0].probs == (0.25, 0.25, 0.25, 0.25)
    assert result.answers[0].confidence == 0.25
    assert result.answers[0].value == "A"  # 平分时取第一个
    assert result.usage.prefill_calls == 2
    assert all(options == 4 for _, options, _, _ in chain.calls[1:])
    assert all(prefix > 0 for _, _, prefix, _ in chain.calls)
    assert result.timings is not None


def test_noncausal_full_sends_a_zero_prefix() -> None:
    chain = FakeChain()
    engine = _engine(chain)
    engine.start()
    engine.run(make_request(mode=MaskMode.NONCAUSAL_FULL, questions=1))
    assert chain.calls[-1][2] == 0


def test_the_winning_option_is_the_largest_share() -> None:
    chain = FakeChain(shares=(0.1, 0.6, 0.2, 0.1))
    engine = _engine(chain)
    engine.start()
    result = engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=1))
    answer = result.answers[0]
    assert answer.value == "B"
    assert answer.confidence == pytest.approx(0.6)


def test_noul_question_reports_yes_no_in_engine_order() -> None:
    chain = FakeChain(shares=(0.25, 0.75))  # prompt 顺序是 No, Yes
    engine = _engine(chain)
    engine.start()
    request = make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=1)
    request = type(request)(state=request.state, questions=(Question(text="要看这个文件吗？", kind="noul"),))
    result = engine.run(request)
    answer = result.answers[0]
    assert answer.labels == ("yes", "no")
    assert answer.probs == (0.75, 0.25)
    assert answer.value == "yes"
    assert chain.calls[-1][1] == 2  # 两道选项: No / Yes


def test_causal_reaches_the_chain_as_a_triangular_mask() -> None:
    """M3：causal 不再是 400 —— 它作为 `causal=True` 传到链上（掩码在链里）。"""
    chain = FakeChain()
    engine = _engine(chain)
    engine.start()
    result = engine.run(make_request(mode=MaskMode.CAUSAL, questions=1))
    assert result.provenance.attention_mode is MaskMode.CAUSAL
    assert chain.calls[-1][3] == 1


def test_an_untrained_mode_warns_once_and_says_so_in_provenance(caplog) -> None:
    """M3：`attention_trained` 逐请求带；日志按 mode 只报一次（服务日志不能被它淹掉）。"""
    chain = FakeChain()
    engine = _engine(chain)
    engine.start()
    with caplog.at_level("WARNING", logger="sysone.engine.engine"):
        first = engine.run(make_request(mode=MaskMode.CAUSAL, questions=1))
        engine.run(make_request(mode=MaskMode.CAUSAL, questions=1))
        again = engine.run(make_request(mode=MaskMode.NONCAUSAL_FULL, questions=1))
    assert first.provenance.attention_trained is False
    assert first.provenance.attention_mode is MaskMode.CAUSAL
    assert again.provenance.attention_trained is False
    warnings = [record for record in caplog.records if "attention_mode" in record.getMessage()]
    assert len(warnings) == 2, f"两个未训练 mode 应该正好两条告警，实测 {len(warnings)}"
    assert "causal" in warnings[0].getMessage()
    assert "noncausal_full" in warnings[1].getMessage()


def test_the_configured_mode_applies_when_the_request_has_none() -> None:
    """`--attention-mode` 走 `SchedulerConfig`：请求没带 mode 时它必须生效，而不是被忽略。

    这一条守的是"静默忽略"回归：配置里写着 causal、请求不指定、结果按 `default_mode` 跑了
    一路，provenance 也报 `prefix_noncausal` —— 开关只在 `cache_key` 里留痕。
    """
    chain = FakeChain()
    engine = _engine(chain, _config(mode=MaskMode.CAUSAL))
    engine.start()
    result = engine.run(make_request(questions=1))
    assert result.provenance.attention_mode is MaskMode.CAUSAL
    assert chain.calls[-1][3] == 1


def test_the_request_mode_beats_the_configured_mode() -> None:
    chain = FakeChain()
    engine = _engine(chain, _config(mode=MaskMode.CAUSAL))
    engine.start()
    result = engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=1))
    assert result.provenance.attention_mode is MaskMode.PREFIX_NONCAUSAL
    assert chain.calls[-1][3] == 0


def test_the_trained_mode_does_not_warn(caplog) -> None:
    engine = _engine(FakeChain())
    engine.start()
    with caplog.at_level("WARNING", logger="sysone.engine.engine"):
        result = engine.run(make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=1))
    assert result.provenance.attention_trained is True
    assert not [r for r in caplog.records if "attention_mode" in r.getMessage()]


def test_causal_shares_the_prefix_and_keeps_the_triangular_mask() -> None:
    """causal 下前缀照样共享：前缀相自己成三角，尾巴行看 `[0, q]` 已含整个前缀。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2, mode=MaskMode.CAUSAL), MaskMode.CAUSAL)
    assert len(chain.shared) == 1
    assert chain.shared[0][3] is True
    assert chain.calls == []


def test_prompt_over_max_len_is_422() -> None:
    """上限是链的缓冲宽度：超了在 `answers_for` 里就是 422，不是让链抛裸 ValueError。"""
    from sysone.domain.decision import DecisionRequest

    chain = FakeChain()
    runner = _runner(chain, max_len=8)
    runner.load_weights()
    with pytest.raises(RequestTooLongError) as info:
        runner.answers_for(
            DecisionRequest(state="s", questions=(Question(text="q", options=("甲", "乙")),)),
            MaskMode.PREFIX_NONCAUSAL,
        )
    assert info.value.status == 422
    assert info.value.context["max_len"] == 8
    assert chain.calls == []  # 一次前向都没发生过


def test_warmup_failure_blocks_start() -> None:
    class UnnormalizedChain(FakeChain):
        def forward(self, tokens, *, options, prefix=0, causal=False):
            self.calls.append((tuple(tokens), int(options), int(prefix)))
            return [0.1] * options  # 和为 0.1，不是归一化的概率

    engine = _engine(UnnormalizedChain())
    with pytest.raises(NotReadyError):
        engine.start()


def test_bad_readout_is_reported_not_rounded() -> None:
    class NaNChain(FakeChain):
        def forward(self, tokens, *, options, prefix=0, causal=False):
            self.calls.append((tuple(tokens), int(options), int(prefix)))
            return [float("nan")] + [0.3] * (options - 1)

    runner = _runner(NaNChain())
    runner.load_weights()
    from sysone.domain.decision import DecisionRequest

    with pytest.raises(Exception) as info:
        runner.answers_for(
            DecisionRequest(state="s", questions=(Question(text="q", options=("甲", "乙")),)),
            MaskMode.PREFIX_NONCAUSAL,
        )
    assert "概率" in str(info.value)


def test_shutdown_closes_the_chain_and_requires_load() -> None:
    chain = FakeChain()
    runner = _runner(chain)
    with pytest.raises(ConfigError) as info:
        runner.answers_for(make_request(), MaskMode.PREFIX_NONCAUSAL)
    assert "尚未装载" in str(info.value)
    runner.load_weights()
    assert runner.check_health() is False  # 还没预热
    times = runner.compile_or_warm_up()
    assert isinstance(times, CompilationTimes)
    assert runner.check_health() is True
    runner.shutdown()
    assert chain.closed is True
    assert runner.check_health() is False


def test_execute_is_not_the_split_entry() -> None:
    with pytest.raises(ConfigError):
        _runner(FakeChain()).execute(plan=None, session=None)


# ---------------------------------------------------------------------- 码表


def _write_decision(tmp_path: Path, payload: dict) -> Path:
    root = tmp_path / "mude"
    root.mkdir()
    (root / "decision_config.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_read_codes_takes_the_trained_table(tmp_path: Path) -> None:
    root = _write_decision(tmp_path, {"format_id": mude.FORMAT_ID, "codes": list(CODES)})
    assert mude.read_codes(root) == CODES


def test_read_codes_rejects_another_wire_format(tmp_path: Path) -> None:
    root = _write_decision(tmp_path, {"format_id": "other-v1", "codes": list(CODES)})
    with pytest.raises(ConfigError) as info:
        mude.read_codes(root)
    assert info.value.context["expected"] == mude.FORMAT_ID


def test_read_codes_rejects_a_short_table(tmp_path: Path) -> None:
    root = _write_decision(tmp_path, {"format_id": mude.FORMAT_ID, "codes": list(CODES[:-1])})
    with pytest.raises(ConfigError) as info:
        mude.read_codes(root)
    assert info.value.context["rows"] == mude.CODE_ROWS


# ---------------------------------------------------------------------- 共享前缀


class SharedChain(FakeChain):
    """多一条 `forward_shared`；返回均匀概率，和 `forward` 的默认行为一致。"""

    def __init__(self, *, shares: tuple[float, ...] | None = None) -> None:
        super().__init__(shares=shares)
        self.shared: list[tuple[tuple[int, ...], list[tuple[int, ...]], tuple[int, ...]]] = []

    def forward_shared(self, prefix, tails, *, options, causal=False):
        self.shared.append(
            (tuple(prefix), [tuple(tail) for tail in tails], tuple(int(value) for value in options),
             bool(causal))
        )
        if self._shares is not None:
            rows = []
            for count in options:
                values = list(self._shares[: int(count)])
                rows.append([value / sum(values) for value in values])
            return rows
        return [[1.0 / int(count)] * int(count) for count in options]


class ScriptedPrompter:
    """按调用次序给固定 `(sequence, prefix_len, labels, indexes)`：用来造"前缀不同"的请求。"""

    def __init__(self, plans) -> None:
        self.plans = list(plans)
        self.calls = 0

    def encode(self, state, question):
        plan = self.plans[self.calls]
        self.calls += 1
        return plan


def _forced_runner(plans, chain: FakeChain, *, max_len: int = 4096) -> SplitRunner:
    return SplitRunner(lambda: chain, ScriptedPrompter(plans), max_len=max_len)


def test_two_questions_with_one_state_share_the_prefix() -> None:
    """两题一次请求：前缀只跑一遍，每题只跑自己的尾巴。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    request = make_request(questions=2)
    runner.answers_for(request, MaskMode.PREFIX_NONCAUSAL)
    assert len(chain.shared) == 1, "两题共用一条前缀时只该有一条共享路径"
    prefix, tails, options, causal = chain.shared[0]
    assert len(tails) == 2 and len(prefix) > 0
    assert options == (len(OPTIONS), len(OPTIONS))
    assert causal is False, "`prefix_noncausal` 的掩码仍是双向" 
    assert chain.calls == [], "共享路径不该再逐题整趟"
    # 每条尾巴都是"整条 prompt 去掉前缀"（两题的题面不同，尾巴也就不同）
    assert tails[0] != tails[1]


def test_a_lone_question_runs_one_full_pass() -> None:
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=1), MaskMode.PREFIX_NONCAUSAL)
    assert chain.shared == [], "只有一题时不建前缀缓存"
    assert len(chain.calls) == 1


def test_a_different_prefix_per_question_falls_back() -> None:
    """分词边界把某题的前缀和别的题错开时，不共用前缀（复用会改那一题的数值）。"""
    chain = SharedChain()
    labels = ("甲", "乙")
    plans = [((10, 11, 12, 13), 2, labels, (0, 1)), ((10, 99, 12, 13), 2, labels, (0, 1))]
    runner = _forced_runner(plans, chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2), MaskMode.PREFIX_NONCAUSAL)
    assert chain.shared == []
    assert len(chain.calls) == 2


def test_noncausal_full_never_shares() -> None:
    """`noncausal_full` 把前缀当 0：掩码不同，不能复用前缀那一段。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2, mode=MaskMode.NONCAUSAL_FULL), MaskMode.NONCAUSAL_FULL)
    assert chain.shared == []
    assert len(chain.calls) == 2


def test_the_shared_path_gives_the_same_answers_and_usage() -> None:
    """共享前缀只是省算力：答案与 `input_tokens` 与逐题整趟一致。"""
    shared_chain = SharedChain()
    serial_chain = FakeChain()
    request = make_request(questions=3)
    shared_runner, serial_runner = _runner(shared_chain), _runner(serial_chain)
    shared_runner.load_weights()
    serial_runner.load_weights()
    shared = shared_runner.answers_for(request, MaskMode.PREFIX_NONCAUSAL)
    serial = serial_runner.answers_for(request, MaskMode.PREFIX_NONCAUSAL)
    assert shared[0] == serial[0]
    assert shared[1] == serial[1]
