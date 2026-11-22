"""split 策略的管线测试（假链 + 假分词器，不依赖 ANE / transformers）。

真链只在设了 `SYS1_MUDE_WEIGHTS` 的集成测试里跑；这里验证的是"题面 → id → 一次整链
前向 → 答案"这段接线的语义：线格式的字节、前缀切在哪、noul 的序号、模式门禁、
超长与预热失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import OPTIONS, load_tiny_descriptor, make_request

from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, ServiceConfig
from sysone.domain.attention import MaskMode
from sysone.domain.decision import Question
from sysone.domain.work import CompilationTimes
from sysone.engine.engine import Engine
from sysone.engine.strategies.decision import DecisionMetrics
from sysone.engine.strategies.split import SplitRunner
from sysone.models.families.mude import confidence as mude_confidence
from sysone.errors import (
    ConfigError,
    NotReadyError,
    RequestTooLongError,
    UnsupportedKindError,
    UnsupportedMediaError,
)
from sysone.models.families import mude
from sysone.models.vision import merge

CODES = tuple(f"K{index:03d}" for index in range(mude.CODE_ROWS))
# 真分词表里那三个特殊 token 的 id（两个出货 checkpoint 相同）。字符级假分词器要像真的一样
# 把整段特殊文本收成**一个** id，否则"一位图一个占位符"这件事在这里根本表示不出来。
SPECIAL_TOKENS = (
    ("<|vision_start|>", 248053),
    ("<|vision_end|>", 248054),
    ("<|image_pad|>", 248056),
)


class FakeTokenizer:
    """字符级假分词器（`token = ord(ch)`）：chat 模板只加定位用的标记，user 内容原样保留。

    带图时 user 轮是 **content parts**（`[{"type": "image"}, {"type": "text", ...}]`）：真模板
    （`chat_template.jinja` 的 `render_content`）把每个 image part 渲染成
    `<|vision_start|><|image_pad|><|vision_end|>`，文字 part 原样拼上 —— 这里照做，否则
    "带图的前缀量不出来"这类问题在单元测试里根本不会出现。
    """

    def __init__(self) -> None:
        self.chat_calls = 0

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        self.chat_calls += 1
        content = conversation[1]["content"]
        if isinstance(content, list):
            content = "".join(
                "<|vision_start|><|image_pad|><|vision_end|>" if part["type"] == "image"
                else part["text"]
                for part in content
            )
        return f"<|system|>{conversation[0]['content']}<|user|>{content}<|assistant|>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        ids: list[int] = []
        index = 0
        while index < len(text):
            for literal, token in SPECIAL_TOKENS:
                if text.startswith(literal, index):
                    ids.append(token)
                    index += len(literal)
                    break
            else:
                ids.append(ord(text[index]))
                index += 1
        return {"input_ids": ids}


class FakeChain:
    """只记账、按 `options` 给均匀概率；`shares` 可覆盖成不等概率。"""

    # runner 从链上取这两段装载耗时（真链由 ANE 侧量），个别测试会改成别的数。
    load_s = 0.0
    compile_s = 0.0

    def __init__(self, *, shares: tuple[float, ...] | None = None) -> None:
        self.calls: list[tuple[tuple[int, ...], int, int]] = []
        self.closed = False
        self._shares = shares

    def forward(self, tokens, *, options, prefix=0, causal=False, mm=None):
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
    return SplitRunner(lambda: chain, prompter, confidence=mude_confidence, max_len=max_len)


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


def test_noul_criteria_reach_the_code_table_prompt() -> None:
    """请求给了判据，题面就用它 —— 厂商 `d3_format._noul_options` 就是这么渲染的。

    这一段曾经被刻意忽略（"改协议层不许动出货路径"），但厂商的题面本来就吃判据：不读的时候
    带判据的 noul 题题面**全都**不一样（票据 15 §34 那 38 道真题），等于在一套厂商从没见过的
    题面上读概率。判据顺序也照厂商：false 在前。
    """
    carried = Question(
        text="要看这个文件吗？",
        options=("true: 已经送达", "false: 尚未送达"),
        kind="noul",
    )
    texts, labels, indexes = mude.option_plan(carried)
    assert texts == ("尚未送达", "已经送达")
    assert labels == ("yes", "no")
    assert indexes == (1, 0)
    body = mude.render_messages("状态", carried, texts, CODES)[1]["content"]
    assert "K000: 尚未送达\nK001: 已经送达\n" in body


def test_noul_without_criteria_falls_back_to_the_fixed_texts() -> None:
    """没给判据、或判据给了空串时，题面就是厂商那两句缺省的话。"""
    empty = Question(text="要看这个文件吗？", options=("true: ",), kind="noul")
    for question in (Question(text="要看这个文件吗？", kind="noul"), empty):
        texts, _, _ = mude.option_plan(question)
        assert texts == mude._NOUL_TEXTS


def test_option_plan_reads_score_levels_in_order() -> None:
    """score 在题面上就是一道"一个档位一行"的 choice，行里不带序号前缀。

    厂商 `normalize_question` 的改写法：criteria 的有序档位原样成为选项文本，标签是
    `"0".."n-1"`，于是"码序 = 档位序"，答案由概率折成期望档位 `Σ i·p_i`。协议层给的那段
    `"0: 差"` 里的序号是给答面拼标签用的，题面里不能留。
    """
    question = Question(text="多正面？", options=("0: 差", "1: 中", "2: 好"), kind="score")
    texts, labels, indexes = mude.option_plan(question)
    assert texts == ("差", "中", "好")
    assert labels == ("0", "1", "2")
    assert indexes == (0, 1, 2)


def test_option_plan_refuses_a_scale_the_vendor_would_refuse() -> None:
    """档位数与重复档位按厂商 `SCORE_LEVELS` / `normalize_question` 的判据挡掉。"""
    for options in (
        ("0: 只有一档",),
        tuple(f"{index}: 档{index}" for index in range(11)),
        ("0: 一样", "1: 一样"),
    ):
        with pytest.raises(ConfigError):
            mude.option_plan(Question(text="打分", options=options, kind="score"))


def test_option_plan_refuses_a_kind_outside_the_wire_format() -> None:
    """线格式只有 choice / noul / score：直接构造 `Question` 的调用方拿到 400，而不是被当 choice。"""
    with pytest.raises(UnsupportedKindError) as info:
        mude.option_plan(Question(text="排个序", kind="ranking"))
    assert info.value.status == 400
    assert info.value.context == {"kind": "ranking", "supported": ["choice", "noul", "score"]}
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
    sequence, prefix_len, labels, indexes, _readout = prompter.encode("只有这一句", question)
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
    # 平分 ⇒ 置信度 0：码表读出这一家用的是 `1 - H/log n`，不是最大概率 0.25
    assert result.answers[0].confidence == pytest.approx(0.0, abs=1e-12)
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
    # 不是最大概率 0.6：厂商 `d3_runtime._confidence` 是 1 - H/log 4
    assert answer.confidence == pytest.approx(0.2145, abs=1e-4)


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


def test_a_score_question_reads_its_scale_through_the_same_code_table() -> None:
    """score 不新开一条读出：档位按 0..n-1 占前几行码，标签就是档位序号。"""
    chain = FakeChain(shares=(0.1, 0.2, 0.7))
    engine = _engine(chain)
    engine.start()
    request = make_request(mode=MaskMode.PREFIX_NONCAUSAL, questions=1)
    question = Question(text="item 0", options=("0: 差", "1: 中", "2: 好"), kind="score")
    request = type(request)(state=request.state, questions=(question,))
    result = engine.run(request)
    answer = result.answers[0]
    assert answer.labels == ("0", "1", "2")
    assert answer.probs == pytest.approx((0.1, 0.2, 0.7))
    assert answer.value == "2"
    assert chain.calls[-1][1] == 3  # 三个档位 ⇒ 三个候选


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
        def forward(self, tokens, *, options, prefix=0, causal=False, mm=None):
            self.calls.append((tuple(tokens), int(options), int(prefix)))
            return [0.1] * options  # 和为 0.1，不是归一化的概率

    engine = _engine(UnnormalizedChain())
    with pytest.raises(NotReadyError):
        engine.start()


def test_bad_readout_is_reported_not_rounded() -> None:
    class NaNChain(FakeChain):
        def forward(self, tokens, *, options, prefix=0, causal=False, mm=None):
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


def test_startup_times_come_from_the_chain() -> None:
    """装载/编译由链自己量（只有它知道哪几层是从头构建的产物），预热由 runner 量。"""
    chain = FakeChain()
    chain.load_s, chain.compile_s = 1.25, 9.5
    runner = _runner(chain)
    runner.load_weights()
    times = runner.compile_or_warm_up()
    assert (times.load_s, times.compile_s) == (1.25, 9.5)
    assert times.warmup_s > 0.0


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

    def forward_shared(self, prefix, tails, *, options, causal=False, mm=None):
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
    """按调用次序给固定 `(sequence, prefix_len, labels, indexes, readout)`：造"前缀不同"的请求。"""

    def __init__(self, plans) -> None:
        self.plans = list(plans)
        self.calls = 0

    def encode(self, state, question, *, images=0):
        plan = self.plans[self.calls]
        self.calls += 1
        return plan


def _forced_runner(plans, chain: FakeChain, *, max_len: int = 4096) -> SplitRunner:
    return SplitRunner(lambda: chain, ScriptedPrompter(plans),
                       confidence=mude_confidence, max_len=max_len)


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
    plans = [((10, 11, 12, 13), 2, labels, (0, 1), None),
             ((10, 99, 12, 13), 2, labels, (0, 1), None)]
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


def test_the_engine_hands_the_runner_decision_stats_to_metrics() -> None:
    """`/metrics` 的 `decision` 一栏由执行器给：Engine 只转交，不改数字。"""
    chain = SharedChain()
    engine = _engine(chain)
    engine.start()
    engine.run(make_request(questions=2))
    stats = engine.decision_stats()
    assert stats is not None and stats["supports_shared"] is True
    # 两题一起走批那条路（`Engine.run` 对 `batch_limit > 1` 的单条也走批），共用前缀。
    assert stats["batch_hit"] == 1


# ---------------------------------------------------------------------- 计数


def test_the_shared_path_is_counted_as_a_hit() -> None:
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2), MaskMode.PREFIX_NONCAUSAL)
    stats = runner.decision_stats()
    assert stats["shared_hit"] == 1 and stats["shared_miss"] == 0
    # 能力位问的是"这条链上有没有那条原语"：这条替身只有 `forward_shared`，没有块对角那条。
    assert stats["supports_shared"] is True and stats["supports_batch"] is False


def test_the_batch_limit_comes_from_the_buffer_width() -> None:
    """块对角一趟最多并排 `width // 128` 条：一条序列至少占一槽（槽宽是 128 的整数倍）。

    引擎取"执行器的形状上限"与 `--batch`（默认 4）的小值：默认负载下 `batch_limit` 仍是 4，
    与 Stage 1 之前一致。
    """
    runner = _runner(TiledChain(), max_len=2048)
    runner.load_weights()                         # 能力位问的是**装载后**那条链
    assert runner.batch_limit == 16
    engine = _engine(TiledChain(), max_len=2048)
    engine.start()
    assert engine.batch_limit == 4


def test_a_chain_without_any_batch_primitive_reports_a_limit_of_one() -> None:
    """两条原语都没有：引擎不攒批，单条也不绕进 `answers_batch`。"""
    runner = _runner(FakeChain())
    runner.load_weights()
    assert runner.batch_limit == 1


def test_a_lone_question_is_neither_a_hit_nor_a_miss() -> None:
    """单题没有"与谁共享"这件事：记成 miss 会把最常见的负载涂成失败。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=1), MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.shared_hit == 0 and runner.decisions.shared_miss == 0


def test_a_different_prefix_per_question_is_counted_as_a_miss() -> None:
    chain = SharedChain()
    labels = ("甲", "乙")
    plans = [((10, 11, 12, 13), 2, labels, (0, 1), None),
             ((10, 99, 12, 13), 2, labels, (0, 1), None)]
    runner = _forced_runner(plans, chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2), MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.shared_miss == 1 and runner.decisions.shared_hit == 0


def test_noncausal_full_has_no_shared_opportunity_to_miss() -> None:
    """双向前向里前缀行与题面耦合，本来就不能复用：那是"没这个语义"，不是"漏做了"。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2, mode=MaskMode.NONCAUSAL_FULL),
                       MaskMode.NONCAUSAL_FULL)
    assert runner.decisions.shared_miss == 0


def test_an_over_long_prompt_is_refused_before_the_counters_move() -> None:
    """超长是 422：编码那一关没过，共享的账就不该动。"""
    chain = SharedChain()
    runner = _runner(chain, max_len=8)
    runner.load_weights()
    with pytest.raises(RequestTooLongError):
        runner.answers_for(make_request(questions=2), MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.snapshot() == DecisionMetrics().snapshot()


class TiledChain(SharedChain):
    """再补一条块对角原语：它的形状判定归链自己，这里只记账、按选项数给等概率。"""

    bucket = 512
    width = 2048

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[tuple[tuple[int, ...], ...]] = []
        self.prefixes: tuple[int, ...] | None = None

    def forward_batched(self, sequences, *, options, prefixes, causal=False):
        self.rows.append(tuple(tuple(sequence) for sequence in sequences))
        self.prefixes = tuple(int(value) for value in prefixes)
        return [[1.0 / int(count)] * int(count) for count in options]


def test_two_requests_that_share_a_prefix_count_as_a_batch_hit() -> None:
    chain = TiledChain()
    runner = _runner(chain)
    runner.load_weights()
    request = make_request(questions=1)
    runner.answers_batch([request, request], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_hit == 1
    assert chain.rows == [], "同前缀该走共享，不该走块对角"
    assert runner.supports_shared and runner.supports_batch, "这条链两条原语都有"


def test_two_independent_requests_tile_into_one_pass_and_count_as_a_hit() -> None:
    chain = TiledChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_batch([make_request(questions=1, state="甲"),
                          make_request(questions=1, state="乙")], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_hit == 1 and len(chain.rows) == 1


def test_a_batch_that_cannot_share_nor_tile_counts_as_a_fallback() -> None:
    """两条不同 state 的请求，链却没有块对角原语：退回逐题 —— 以前这件事没有计数。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_batch([make_request(questions=1, state="甲"),
                          make_request(questions=1, state="乙")], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_fallback == 1 and runner.decisions.batch_hit == 0


def test_a_single_sequence_through_the_batch_route_counts_as_a_miss() -> None:
    """引擎把单条也送进这条路（`Engine.run` 的"免得两条代码路"），它没有可合的东西。"""
    chain = SharedChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_batch([make_request(questions=1)], MaskMode.PREFIX_NONCAUSAL)
    assert runner.decisions.batch_miss == 1 and runner.decisions.batch_hit == 0


def test_images_are_refused_by_a_model_without_a_tower() -> None:
    """没有视觉塔的装配收到图：当场拒（422 那一类），不能变成"占位符数不对"的裸 ValueError。"""
    chain = FakeChain()
    runner = SplitRunner(lambda: chain, _prompter()[0], confidence=mude_confidence,
                         max_len=4096)      # 没给 image_token_id
    runner.load_weights()
    media = merge.Media(counts=(4,), grids=((2, 2),), rows=np.zeros((4, 8), np.float32))
    with pytest.raises(UnsupportedMediaError):
        runner.answers_for(make_request(questions=1), MaskMode.PREFIX_NONCAUSAL, media=media)
    assert chain.calls == [], "拒收之前不该发生任何前向"


# ------------------------------------------------------- 前门那条真路径（P4）

class FakeSlot:
    """`VisionSlot` 的替身：只回答"被调用了几次、给了什么、收尾了几次"。"""

    def __init__(self, rows: int = 4) -> None:
        self.rows = rows
        self.calls: list[object] = []
        self.finished = 0
        self.closed = 0
        self.released = 0

    def media(self, refs):
        self.calls.append(tuple(refs))
        return merge.Media(counts=(self.rows,), grids=((2, 2),),
                           rows=np.zeros((self.rows, 8), np.float32))

    def finish(self) -> None:
        self.finished += 1

    def close(self) -> None:
        self.closed += 1

    def release(self) -> None:
        self.released += 1

    def warm(self) -> None:
        self.calls.append("warm")

    def stats(self) -> dict:
        return {"loaded": bool(self.calls), "bytes": 1, "reuse": [26, 26],
                "loads": len(self.calls), "unloads": 0, "load_s": 0.0, "idle_s": 60.0,
                "evals": 0, "ane_ms": 0.0, "gpu_ms": 0.0}


def _vision_runner(chain, slot: FakeSlot | None, *, metal: bool = True, missing: int = 0):
    """一条"有塔"的装配：`image_token_id`、工厂与产物账都给了，链上挂着 `metal_runtime`。"""
    prompter, _ = _prompter()
    if metal:
        chain.metal_runtime = object()          # 只用来回答"有没有 Metal 运行时"
    state = None if slot is None else (lambda held: {"reusable": 26 - missing, "total": 26})
    return SplitRunner(
        lambda: chain, prompter, confidence=mude_confidence, max_len=4096,
        image_token_id=248056,
        load_tower=(lambda held: slot) if slot is not None else None,
        vision_state=state,
    )


def test_a_tower_less_assembly_refuses_images_with_the_reason() -> None:
    chain = FakeChain()
    runner = _vision_runner(chain, None)
    runner.load_weights()
    request = make_request(questions=1)
    object.__setattr__  # noqa: B018 - 只是提醒：DecisionRequest 是 frozen 的
    request = _with_media(request)
    with pytest.raises(UnsupportedMediaError) as caught:
        runner.answers_for(request, MaskMode.PREFIX_NONCAUSAL)
    assert "--vision" in caught.value.message, caught.value.message
    assert chain.calls == []


def test_request_images_go_through_the_slot_and_are_released_afterwards() -> None:
    """请求自带的图走真路径：`slot.media(request.media)` → 前向 → `finish()`（TTL 从这儿起算）。"""
    chain = FakeChain()
    slot = FakeSlot()
    runner = _vision_runner(chain, slot)
    runner.load_weights()
    request = _with_media(make_request(questions=2))
    answers, tokens, _ = runner.answers_for(request, MaskMode.PREFIX_NONCAUSAL)
    assert len(answers) == 2 and tokens > 0
    assert slot.calls == [(request.media[0],)], "slot 拿到的应当就是请求里那些图"
    assert slot.finished == 1, "一次请求结束要调 finish（idle = 0 时那就是卸载点）"
    assert chain.calls, "带图的请求照样要前向"
    assert runner.decisions.media_skip == 1, "带图请求不进共享/合批，要记这一笔"


def test_shutdown_closes_the_tower_before_the_chain() -> None:
    chain = FakeChain()
    slot = FakeSlot()
    runner = _vision_runner(chain, slot)
    runner.load_weights()
    runner.shutdown()
    assert slot.closed == 1 and chain.closed is True


def test_the_contract_says_which_of_the_four_reasons_applies() -> None:
    chain = FakeChain()

    text_only = SplitRunner(lambda: chain, _prompter()[0], confidence=mude_confidence,
                            max_len=4096)
    text_only.load_weights()
    assert text_only.vision_unavailable() == "这个模型没有视觉塔，收不了图"
    assert text_only.image_contract()["supported"] is False
    assert text_only.image_contract()["unavailable"] == "这个模型没有视觉塔，收不了图"

    switched_off = _vision_runner(chain, None)
    switched_off.load_weights()
    assert "--vision" in switched_off.vision_unavailable()

    # 链是 numpy 形态：塔装了也跑不了注意力核
    numpy_chain = FakeChain()
    numpy_form = _vision_runner(numpy_chain, FakeSlot(), metal=False)
    numpy_form.load_weights()
    assert "Metal" in numpy_form.vision_unavailable()

    ready = _vision_runner(FakeChain(), FakeSlot())
    ready.load_weights()
    assert ready.vision_unavailable() is None
    contract = ready.image_contract()
    assert contract["supported"] is True and "unavailable" not in contract
    assert ready.vision_stats()["available"] is True


def test_vision_warmup_reports_the_artifacts_of_the_tower() -> None:
    """首启预处理与显式预热共用这一条：装上（= 编译产物），报 26 份里命中几个。"""
    chain = FakeChain()
    slot = FakeSlot()
    runner = _vision_runner(chain, slot)
    runner.load_weights()
    report = runner.vision_warmup()
    assert report is not None
    assert slot.calls == ["warm"]
    assert report["programs"] == {"reusable": 26, "total": 26}
    assert SplitRunner(lambda: chain, _prompter()[0], confidence=mude_confidence,
                       max_len=4096).vision_warmup() is None


def _with_media(request, *, data: bytes = b"\x89PNG\r\n\x1a\n"):
    import dataclasses

    from sysone.domain.decision import MediaRef

    return dataclasses.replace(request, media=(MediaRef(kind="image", data=data),))


def test_a_chain_without_the_primitive_is_declared_incapable_and_runs_serially() -> None:
    """能力位说真话：链上没有 `forward_shared` 就报 False，多题退回逐题而不是报错。"""
    chain = FakeChain()
    runner = _runner(chain)
    runner.load_weights()
    runner.answers_for(make_request(questions=2), MaskMode.PREFIX_NONCAUSAL)
    assert len(chain.calls) == 2, "没有共享原语就逐题整趟"
    assert runner.decisions.shared_miss == 1, "机会在、能力不在：这一笔是 Stage 3 的立项依据"
    assert runner.decision_stats()["supports_shared"] is False


def test_prepare_vision_burns_the_26_programs_then_parks_the_tower() -> None:
    """首启：缺产物才装塔，编完立刻归还（0 常驻）；第二次启动不构造塔。"""
    chain = FakeChain()
    slot = FakeSlot()
    runner = _vision_runner(chain, slot, missing=26)
    runner.load_weights()

    report = runner.prepare_vision()
    assert report is not None
    assert report["programs"] == {"reusable": 26, "total": 26}
    assert slot.calls == ["warm"]
    assert slot.released == 1

    runner = _vision_runner(FakeChain(), FakeSlot(), missing=0)
    runner.load_weights()
    assert runner.prepare_vision() is None
    assert runner._vision.calls == [] and runner._vision.released == 0


def test_warmup_uses_the_checkpoints_default_mode() -> None:
    """预热与默认读法同掩码：只训练过双向的检查点不该用 `prefix_noncausal` 白热一遍。

    `noncausal_full` 没有前缀行（`use_prefix=False`），所以链收到的是 0；默认形态仍是
    `prefix_noncausal`（mude 这类检查点），题面前的公共前缀长度照常传下去。
    """
    chain = FakeChain()
    prompter, _ = _prompter()
    runner = SplitRunner(lambda: chain, prompter, confidence=mude_confidence, max_len=4096,
                         default_mode=MaskMode.NONCAUSAL_FULL)
    runner.load_weights()
    runner.compile_or_warm_up()
    assert chain.calls[0][2] == 0
    assert runner.check_health() is True

    chain.calls.clear()
    default_runner = _runner(chain)
    default_runner.load_weights()
    default_runner.compile_or_warm_up()
    assert chain.calls[0][2] > 0
