"""首启（D9）：每个出货家族都有发布来源，且下载下来的分片目录能直接装载。

判据不靠"跑一次真下载"：真下载要 9–10 GB 与网络，CI 与本地都不该有那条依赖。这里用注入
的下载器把 `source.files` **逐个**写成发布包同形的文件（分片 + 索引 + 配置），再走
`open_checkpoint` / 家族解析 / 分词器 —— 首启真正会碰到的每一步都在这儿。清单里多出一个
用例不认识的文名时直接失败，这样给仓库加文件必须是有意的。

判据的另一半是"没有来源就是漏接"：D9 说任何接入的模型都要纳进首启，所以能由 CLI 装配的
家族必须都在 `SOURCES` 里。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers import FIXTURES, write_safetensors, write_sharded_checkpoint
from sysone.cli import _CLI_FAMILIES
from sysone.models.families import d3, torchcast
from sysone.models.sources import SOURCES, WeightsSource, source_for
from sysone.models.tokenizer import ChatTokenizer
from sysone.weights.checkpoint import open_checkpoint
from sysone.weights.fetch import ensure_weights

HIDDEN = 2560
ROWS = 255

_D3_SHARDS = {
    "model-00001-of-00002.safetensors": {
        "language_model.embed_tokens.weight": np.arange(HIDDEN * 2, dtype=np.float32).reshape(2, HIDDEN),
    },
    "model-00002-of-00002.safetensors": {
        "language_model.layers.0.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
    },
}
_TORCHCAST_SHARDS = {
    "model-00001-of-00003.safetensors": {
        "model.language_model.embed_tokens.weight": np.zeros((2, HIDDEN), dtype=np.float32),
        "lm_head.weight": np.arange(HIDDEN * 3, dtype=np.float32).reshape(3, HIDDEN),
    },
    "model-00002-of-00003.safetensors": {
        "model.language_model.layers.0.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
    },
    "model-00003-of-00003.safetensors": {
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": np.zeros((32, HIDDEN), dtype=np.float32),
    },
}


class Publisher:
    """假下载器：按发布包的结构写出 `source.files` 里的每一个文件。

    它不认识的文件名直接失败 —— 清单长出新文件时，这条用例要求有人先想清楚那是什么。
    """

    def __init__(self, shards: dict[str, dict[str, np.ndarray]], config: Path, decision: dict) -> None:
        self.shards = shards
        self.config = config
        self.decision = decision
        self.calls = 0

    def __call__(self, source: WeightsSource, destination: Path) -> None:
        self.calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        write_sharded_checkpoint(destination, self.shards)
        write_safetensors(
            destination / "readout.safetensors",
            {"weight": np.zeros((ROWS, HIDDEN), dtype=np.float32)},
        )
        shutil.copy(self.config, destination / "config.json")
        (destination / "decision_config.json").write_text(json.dumps(self.decision), encoding="utf-8")
        _write_tokenizer(destination)
        for name in source.files:
            path = destination / name
            if path.exists():
                continue
            if name in {"LICENSE", "NOTICE", "LICENSE-qwen-Apache-2.0"}:
                path.write_text("license", encoding="utf-8")
                continue
            raise AssertionError(f"清单里有这条用例不认识的文件：{name}")


def _write_tokenizer(destination: Path) -> None:
    """一份最小可用的分词器：单词表 + 内嵌的模板（首启清单里的两个文件必须真能读）。"""
    from tokenizers import Tokenizer, models, pre_tokenizers

    backend = Tokenizer(models.WordLevel(vocab={"hello": 0, "<unk>": 1}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.save(str(destination / "tokenizer.json"))
    (destination / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{% for message in messages %}{{ message.content }}\n{% endfor %}"}),
        encoding="utf-8",
    )


def _never(*_args, **_kwargs) -> None:
    raise AssertionError("这份权重已经齐了，不该再下载")


def test_every_cli_family_has_a_published_source() -> None:
    """D9：能由 CLI 装配的家族，首启都必须能下载 —— 少一个就是漏接。"""
    missing = sorted(name for name in _CLI_FAMILIES if name not in SOURCES)
    assert not missing, f"这些家族没有发布来源：{missing}"


def test_the_d3_source_is_enough_to_first_run_and_read(tmp_path: Path) -> None:
    """d3 的清单下载完就是一个能装载的检查点：两个分片都读到，配置解析成纯文本 4B。"""
    source = source_for("d3")
    publisher = Publisher(
        _D3_SHARDS,
        FIXTURES / "h2o" / "config.json",
        {"format_id": "d3-code-readout-v1", "codes": [f"K{index:03d}" for index in range(ROWS)],
         "temperature": 1.0, "attention_mode": "noncausal_full"},
    )
    target = ensure_weights(source, tmp_path, download=publisher)
    assert publisher.calls == 1
    store = open_checkpoint(target)
    try:
        assert store.read("language_model.embed_tokens.weight").shape == (2, HIDDEN)
        np.testing.assert_allclose(store.read("language_model.layers.0.input_layernorm.weight"), 1.0)
    finally:
        store.close()
    descriptor = d3.parse_config(Path(target))
    assert (descriptor.hidden, descriptor.ffn, descriptor.num_layers) == (2560, 9216, 32)
    assert descriptor.vision is None
    assert descriptor.readout.rows == ROWS and descriptor.readout.temperature == pytest.approx(1.0)
    tokenizer = ChatTokenizer.load(target)
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False,
                                        add_generation_prompt=True) == "hello\n"


def test_the_torchcast_source_is_enough_to_first_run_and_read(tmp_path: Path) -> None:
    """torchcast 的清单下载完也是一个能装载的检查点：三个分片、解绑的 `lm_head` 都在。"""
    source = source_for("torchcast")
    publisher = Publisher(
        _TORCHCAST_SHARDS,
        FIXTURES / "torchcast" / "config.json",
        {"format": "StartLux-Decision-v1", "letter_token_ids": list(range(32, 58))},
    )
    target = ensure_weights(source, tmp_path, download=publisher)
    assert publisher.calls == 1
    store = open_checkpoint(target)
    try:
        assert store.read("model.language_model.embed_tokens.weight").shape == (2, HIDDEN)
        np.testing.assert_allclose(store.read("model.language_model.layers.0.input_layernorm.weight"), 1.0)
        # 读出在解绑的 `lm_head`（它在**第一个**分片里，词表在第二个分片里）。
        np.testing.assert_array_equal(
            store.read_rows("lm_head.weight", [1, 0]),
            np.arange(HIDDEN * 3, dtype=np.float32).reshape(3, HIDDEN)[[1, 0]],
        )
    finally:
        store.close()
    descriptor = torchcast.parse_config(Path(target))
    assert descriptor.tied_embeddings is False
    assert descriptor.readout.kind == "letter_rows"
    assert descriptor.readout.weight == "lm_head.weight"
    assert descriptor.vision is not None


def test_a_second_start_reads_the_marker_instead_of_the_network(tmp_path: Path) -> None:
    """首启扫一遍清单；之后只认 `source.json`，离线也一样能用。"""
    source = source_for("d3")
    publisher = Publisher(
        _D3_SHARDS,
        FIXTURES / "h2o" / "config.json",
        {"format_id": "d3-code-readout-v1", "codes": [f"K{index:03d}" for index in range(ROWS)],
         "temperature": 1.0, "attention_mode": "noncausal_full"},
    )
    target = ensure_weights(source, tmp_path, download=publisher)
    assert ensure_weights(source, tmp_path, download=_never) == target
    assert ensure_weights(source, tmp_path, offline=True, download=_never) == target
    assert (Path(target) / "source.json").is_file()
