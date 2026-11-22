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
from sysone.domain.quant import DType
from sysone.models.families import d3, pplx, torchcast
from sysone.models.sources import SOURCES, WeightsSource, source_for
from sysone.models.tokenizer import ChatTokenizer
from sysone.weights.checkpoint import open_checkpoint
from sysone.weights.fetch import ensure_weights

HIDDEN = 2560
ROWS = 255
LICENCES = {"LICENSE", "NOTICE", "LICENSE-qwen-Apache-2.0"}

_D3_SHARDS = {
    "model-00001-of-00002.safetensors": {
        "language_model.embed_tokens.weight": np.arange(HIDDEN * 2, dtype=np.float32).reshape(2, HIDDEN),
    },
    "model-00002-of-00002.safetensors": {
        "language_model.layers.0.input_layernorm.weight": np.ones(HIDDEN, dtype=np.float32),
    },
}
_PPLX_SHARDS = {
    "model-00001-of-00002.safetensors": {
        "language_model.embed_tokens.weight": np.zeros((2, HIDDEN), dtype=np.float32),
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

    `files` 是"用例要的那几份"（fixture 路径或要写进去的 JSON）；其余按名字补齐：分片来自
    `shards`、读出表按 `ROWS × HIDDEN` 写、license 写一行占位文本。清单里出现用例不认识的
    文件名就失败 —— 那是清单变了而没有人看。
    """

    def __init__(self, shards: dict[str, dict[str, np.ndarray]], files: dict[str, Path | dict]) -> None:
        self.shards = shards
        self.files = files
        self.calls = 0

    def __call__(self, source: WeightsSource, destination: Path) -> None:
        self.calls += 1
        destination.mkdir(parents=True, exist_ok=True)
        write_sharded_checkpoint(destination, self.shards)
        write_safetensors(
            destination / "readout.safetensors",
            {"weight": np.zeros((ROWS, HIDDEN), dtype=np.float32)},
        )
        for name, value in self.files.items():
            if isinstance(value, Path):
                shutil.copy(value, destination / name)
            else:
                (destination / name).write_text(json.dumps(value), encoding="utf-8")
        if not (destination / "tokenizer.json").exists():
            _write_tokenizer(destination)
        for name in source.files:
            path = destination / name
            if path.exists():
                continue
            if name in LICENCES:
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


def _d3_publisher() -> Publisher:
    return Publisher(_D3_SHARDS, {
        "config.json": FIXTURES / "h2o" / "config.json",
        "decision_config.json": {
            "format_id": "d3-code-readout-v1",
            "codes": [f"K{index:03d}" for index in range(ROWS)],
            "temperature": 1.0,
            "attention_mode": "noncausal_full",
        },
    })


def test_every_cli_family_has_a_published_source() -> None:
    """D9：能由 CLI 装配的家族，首启都必须能下载 —— 少一个就是漏接。"""
    missing = sorted(name for name in _CLI_FAMILIES if name not in SOURCES)
    assert not missing, f"这些家族没有发布来源：{missing}"


def test_the_d3_source_is_enough_to_first_run_and_read(tmp_path: Path) -> None:
    """d3 的清单下载完就是一个能装载的检查点：两个分片都读到，配置解析成纯文本 4B。"""
    source = source_for("d3")
    publisher = _d3_publisher()
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
    publisher = Publisher(_TORCHCAST_SHARDS, {
        "config.json": FIXTURES / "torchcast" / "config.json",
        "decision_config.json": {"format": "StartLux-Decision-v1",
                                 "letter_token_ids": list(range(32, 58))},
    })
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


@pytest.mark.parametrize(("label", "parse", "codes_for", "temperature"), (
    ("v12", pplx.parse_config_12, pplx.read_codes_12, 1.5539092221462472),
    ("v13", pplx.parse_config_13, pplx.read_codes_13, 1.3302962111834522),
))
def test_the_pplx_sources_are_enough_to_first_run_and_load(
    tmp_path: Path, label: str, parse, codes_for, temperature: float
) -> None:
    """两版 pplx 的清单下载完都能装载；模板只有兄弟 `.jinja` 那一条路（配置里是 null）。"""
    directory = FIXTURES / "pplx" / label
    files = {name: directory / name for name in (
        "config.json", "decision_config.json", "joint_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    publisher = Publisher(_PPLX_SHARDS, files)
    target = ensure_weights(source_for(f"pplx{label[1:]}"), tmp_path, download=publisher)
    assert publisher.calls == 1
    store = open_checkpoint(target)
    try:
        np.testing.assert_allclose(store.read("language_model.layers.0.input_layernorm.weight"), 1.0)
    finally:
        store.close()
    config = json.loads((Path(target) / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert config["chat_template"] is None            # 模板只能来自兄弟文件
    tokenizer = ChatTokenizer.load(target)
    assert tokenizer.special_tokens                    # 转义表是从分词器里读出来的
    descriptor = parse(Path(target))
    assert descriptor.vision is None
    assert descriptor.readout.dtype is DType.BF16
    assert descriptor.readout.temperature == pytest.approx(temperature)
    assert len(codes_for(Path(target))) == ROWS


def test_a_second_start_reads_the_marker_instead_of_the_network(tmp_path: Path) -> None:
    """首启扫一遍清单；之后只认 `source.json`，离线也一样能用。"""
    source = source_for("d3")
    publisher = _d3_publisher()
    target = ensure_weights(source, tmp_path, download=publisher)
    assert ensure_weights(source, tmp_path, download=_never) == target
    assert ensure_weights(source, tmp_path, offline=True, download=_never) == target
    assert (Path(target) / "source.json").is_file()
