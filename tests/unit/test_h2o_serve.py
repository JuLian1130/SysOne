"""`Serve` 的答面接线：题面怎么渲染、标签怎么落到 token、logits 怎么交给厂商判。

不依赖 transformers / torch：shim、`serve_config` 契约与分词器都是假的。真 shim 与真分词器
在 `.scratch` 的对拍里跑（那里两边都用同一份厂商代码）——包括分词器换掉之后那一次逐位对照
（`.scratch/c2_tokenizer_parity.py`，105 道真题 + 三档合成拉伸）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sysone.errors import ConfigError, ReadoutError
from sysone.models.tokenizer import ChatTokenizer
from sysone.models.vendors import h2o_serve
from sysone.models.vendors.h2o_serve import Serve

CONFIG = {
    "labels": ["A", "B", "C"],
    "max_options": 3,
    "temperature": {"choice": 0.5, "noul": 2.0, "score": 1.0},
    "noul_floor": 0.0,
}


class FakeContract:
    """`serve_config.json` 的等价物：只留 `Serve` 用到的那几项。"""

    def __init__(self, cfg: dict) -> None:
        self.state_compact = True
        self.temperature_by_type = dict(cfg["temperature"])
        self.temperature = self.temperature_by_type["choice"]
        self.noul_floor = float(cfg["noul_floor"])
        self.labels = list(cfg["labels"])

    def question_prompt(self, state_text: str, spec: dict):
        """把选项按字母排好：名字原样，标签取 CONFIG 里的前 n 个。"""
        names = list(spec["criteria"])
        descs = [spec["criteria"][name] for name in names]
        labels = self.labels[: len(names)]
        prompt = f"{state_text}|{'/'.join(labels)}|Answer:"
        return prompt, names, descs, labels


class FakeShim:
    """最小 shim：题面原样返回 state，答案用 `probabilities`（softmax 除温度）。"""

    Contract = FakeContract

    def render_state(self, state: str, compact: bool) -> str:
        assert compact is True
        return state

    def noul_threshold(self, spec: dict):
        return None

    def format_answer(self, typ, names, descs, logprobs, temperature, noul_floor, threshold):
        scaled = [value / temperature for value in logprobs]
        top = max(scaled)
        weights = [pow(2.718281828459045, value - top) for value in scaled]
        total = sum(weights)
        probs = [value / total for value in weights]
        best = max(range(len(probs)), key=probs.__getitem__)
        return {
            "type": "choice",
            "choice": names[best],
            "probabilities": {name: value for name, value in zip(names, probs)},
            "confidence": probs[best],
        }


class FakeTokenizer:
    """空格直接丢掉，其余逐字符成 token —— 于是 `prompt + " " + 单字符标签` 恰好多一个 token。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str, *, add_special_tokens: bool):
        assert add_special_tokens is False
        self.calls.append(text)
        return {"input_ids": [ord(char) for char in text if char != " "]}


def spec(options=("A", "B"), instructions="pick one") -> dict:
    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {option: f"description of {option}" for option in options},
    }


def serve(**kwargs) -> tuple[Serve, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    return Serve(CONFIG, FakeShim(), tokenizer, **kwargs), tokenizer


# ------------------------------------------------------------------ prepare
def test_prepare_carries_prompt_labels_and_temperature():
    """题面、回答槽上的标签 token、每题型的温度 —— 都是厂商契约给的，不许在别处重算。"""
    subject, _ = serve()
    prepared = subject.prepare("record", spec(options=("A", "B")))

    assert prepared.ids == tuple(ord(char) for char in "record|A/B|Answer:")
    # 标签落在回答槽上那一个 token：ord("A") / ord("B")
    assert prepared.label_ids == (ord("A"), ord("B"))
    assert prepared.labels == ("A", "B")
    assert prepared.names == ("A", "B")
    assert prepared.descs == ("description of A", "description of B")
    assert prepared.temperature == 0.5
    assert prepared.kind == "choice"


def test_prepare_uses_the_type_specific_temperature():
    subject, _ = serve()
    noul = spec(options=("true", "false"))
    noul["type"] = "noul"
    assert subject.prepare("record", noul).temperature == 2.0


def test_prepare_refuses_a_label_that_is_not_one_token():
    """多 token 的标签在回答槽上读不出来：宁可报错，也不要退化成读最后一个子 token。

    标签来自厂商契约（`serve_config.json` 的 `labels`），不来自选项名 —— h2o 就是拿这份
    字母表去问"回答槽上落的是哪个字母"。
    """
    config = {**CONFIG, "labels": ["A", "BB"]}
    subject = Serve(config, FakeShim(), FakeTokenizer())
    with pytest.raises(ReadoutError):
        subject.prepare("record", spec(options=("A", "B")))


# ------------------------------------------------------------------- answer
def test_answer_hands_the_raw_logits_to_the_vendor():
    """`answer` 不再自己归一化：温度是厂商的，我们只负责把 logits 原样递过去。"""
    subject, _ = serve()
    prepared = subject.prepare("record", spec(options=("A", "B")))
    value, labels, probs, confidence = subject.answer(prepared, [1.0, 0.0])

    # T=0.5 下的 softmax([1, 0]) = softmax([2, 0])
    assert value == "A"
    assert labels == ("A", "B")
    assert probs[0] == pytest.approx(0.8807970779778823, rel=1e-12)
    assert sum(probs) == pytest.approx(1.0, rel=1e-12)
    # 置信度是厂商 `format_answer` 自己给的那个数（这个假 shim 取最大概率），原样带出去
    assert confidence == probs[0]


def test_answer_rejects_a_logit_count_that_does_not_match_the_labels():
    subject, _ = serve()
    prepared = subject.prepare("record", spec(options=("A", "B")))
    with pytest.raises(ReadoutError):
        subject.answer(prepared, [1.0])


# --------------------------------------------------------------------- open
def test_open_needs_the_vendor_serve_config(tmp_path):
    """目录里没有 `serve_config.json` 就报配置错，而不是拿空的标签表去跑。"""
    with pytest.raises(ConfigError):
        Serve.open(tmp_path, tokenizer=FakeTokenizer())


def _write_tokenizer(directory: Path) -> None:
    """一个词表级的小分词器：`tokenizer.json` + `tokenizer_config.json`（模板留空）。"""
    from tokenizers import Tokenizer, models, pre_tokenizers

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "record": 1, "A/B": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    backend.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": ""}), encoding="utf-8"
    )


def test_open_reads_the_tokenizer_file_without_transformers(tmp_path, monkeypatch):
    """答面只要"文本 → id"：走目录里的 `tokenizer.json`，不为它装 transformers。

    `transformers` 为这件事要多装 ≈0.28 GB 的底座（`models/tokenizer.py` 的实测），而这条
    答面一次也没用到它的模板与解码 —— 模板是厂商 shim 自己渲染的。换成与 mude 出货路径
    同一个 `ChatTokenizer` 之后，h2o 的出货路径才进得了没有 transformers 的门禁。
    """
    (tmp_path / "serve_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    _write_tokenizer(tmp_path)
    monkeypatch.setattr(h2o_serve, "_load_shim", lambda root: FakeShim)
    monkeypatch.setitem(sys.modules, "transformers", None)      # 真导入会炸

    subject = Serve.open(tmp_path)

    assert isinstance(subject._tokenizer, ChatTokenizer)
    # 与 `tokenizer.json` 直编逐位相同（也就是"真用了那个文件"，而不是回落到别的东西）
    from tokenizers import Tokenizer

    expected = Tokenizer.from_file(str(tmp_path / "tokenizer.json"))
    assert subject._ids("record A/B") == tuple(expected.encode("record A/B").ids)


def test_open_names_the_missing_tokenizer_file(tmp_path, monkeypatch):
    """目录里没有 `tokenizer.json` 就指名道姓地报，不要静默退化成"按字符切"。"""
    (tmp_path / "serve_config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    monkeypatch.setattr(h2o_serve, "_load_shim", lambda root: FakeShim)
    with pytest.raises(ConfigError, match="tokenizer.json"):
        Serve.open(tmp_path)
