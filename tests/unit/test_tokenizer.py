"""模型目录的文本侧（`tokenizers` 直连 + `jinja2` 聊天模板）。

真 checkpoint 的 token id 逐位对拍在 `tests/integration/test_tokenizer_golden.py`（要
`SYS1_MUDE_WEIGHTS`）；这里用词表级的小分词器 + 一份形似 Qwen 的模板，钉住适配器自己那层
契约：模板环境与 transformers 一致（`trim_blocks`/`lstrip_blocks`、`tojson`、
`raise_exception`）、`add_special_tokens` 照传、缺文件按配置错报。
"""

from __future__ import annotations

import json
from pathlib import Path

import jinja2
import pytest

from sysone.errors import ConfigError
from sysone.models.tokenizer import ChatTokenizer

VOCAB = {"[BOS]": 0, "<eos>": 1, "hello": 2, "world": 3, "<unk>": 4}

TEMPLATE = """{% for message in messages %}
{{ message.role }}={{ message.content }}
{% endfor %}
{% if add_generation_prompt %}<<generate>>{% if enable_thinking is defined and enable_thinking is true %}think{% endif %}{% endif %}
"""

CONVERSATION = [
    {"role": "system", "content": " system prompt "},
    {"role": "user", "content": "请问 hello world?"},
]


def _model(directory: Path, *, template: object = TEMPLATE) -> Path:
    """写一个单词表分词器（后处理会加 `[BOS]` / `<eos>`）与一份配置。"""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    backend = Tokenizer(models.WordLevel(vocab=VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A <eos>", special_tokens=[("[BOS]", 0), ("<eos>", 1)]
    )
    backend.save(str(directory / "tokenizer.json"))
    config = {} if template is None else {"chat_template": template}
    (directory / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    return directory


def test_chat_template_trims_blocks_and_follows_generation_prompt(tmp_path: Path) -> None:
    tokenizer = ChatTokenizer.load(_model(tmp_path))
    rendered = tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    assert rendered == "system= system prompt \nuser=请问 hello world?\n<<generate>>"


def test_chat_template_passes_template_variables(tmp_path: Path) -> None:
    tokenizer = ChatTokenizer.load(_model(tmp_path))
    plain = tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=False)
    thinking = tokenizer.apply_chat_template(
        CONVERSATION, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert plain == "system= system prompt \nuser=请问 hello world?\n"
    assert thinking == "system= system prompt \nuser=请问 hello world?\n<<generate>>think"


def test_chat_template_tojson_keeps_non_ascii_and_html(tmp_path: Path) -> None:
    tokenizer = ChatTokenizer.load(_model(tmp_path, template="{{ payload|tojson }}"))
    payload = {"state": "电池 <ok> & \"quoted\"", "steps": [1, "一二三"]}
    rendered = tokenizer.apply_chat_template([], tokenize=False, payload=payload)
    assert rendered == json.dumps(payload, ensure_ascii=False)


def test_chat_template_raise_exception_is_a_template_error(tmp_path: Path) -> None:
    template = "{% if not messages %}{{ raise_exception('No messages provided.') }}{% endif %}"
    tokenizer = ChatTokenizer.load(_model(tmp_path, template=template))
    with pytest.raises(jinja2.TemplateError, match="No messages provided"):
        tokenizer.apply_chat_template([], tokenize=False)


def test_call_encodes_ids_and_respects_add_special_tokens(tmp_path: Path) -> None:
    tokenizer = ChatTokenizer.load(_model(tmp_path))
    assert tokenizer("hello world")["input_ids"] == [0, 2, 3, 1]
    assert tokenizer("hello world", add_special_tokens=False)["input_ids"] == [2, 3]


def test_load_requires_tokenizer_json(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match="tokenizer.json"):
        ChatTokenizer.load(tmp_path)


def test_load_requires_tokenizer_config(tmp_path: Path) -> None:
    _model(tmp_path)
    (tmp_path / "tokenizer_config.json").unlink()
    with pytest.raises(ConfigError, match="tokenizer_config.json"):
        ChatTokenizer.load(tmp_path)


def test_load_rejects_named_template(tmp_path: Path) -> None:
    _model(tmp_path, template=[{"name": "default", "template": TEMPLATE}])
    with pytest.raises(ConfigError, match="chat_template"):
        ChatTokenizer.load(tmp_path)


def test_apply_chat_template_rejects_tokenize(tmp_path: Path) -> None:
    tokenizer = ChatTokenizer.load(_model(tmp_path))
    with pytest.raises(ConfigError, match="只渲染文本"):
        tokenizer.apply_chat_template(CONVERSATION, tokenize=True)


def test_a_sibling_chat_template_file_is_used_when_the_config_has_null(tmp_path: Path) -> None:
    """`pplx` 的发布包把模板放在 `chat_template.jinja` 里，配置项是 `null`。

    transformers 认这个兄弟文件，所以这里也按同一条顺序找：否则一份完全合法的检查点会被判成
    坏配置，而它本来的模型文件都是齐的。
    """
    _model(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": None}), encoding="utf-8"
    )
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    tokenizer = ChatTokenizer.load(tmp_path)
    rendered = tokenizer.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
    assert rendered == "system= system prompt \nuser=请问 hello world?\n<<generate>>"


def test_a_null_template_without_a_sibling_file_is_a_config_error(tmp_path: Path) -> None:
    _model(tmp_path)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": None}), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="chat_template"):
        ChatTokenizer.load(tmp_path)


def test_a_named_template_is_not_replaced_by_a_sibling_file(tmp_path: Path) -> None:
    """命名模板（list）仍是坏配置：挑哪一条当默认是 transformers 的约定，`.jinja` 不顶替它。"""
    _model(tmp_path, template=[{"name": "default", "template": TEMPLATE}])
    (tmp_path / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
    with pytest.raises(ConfigError, match="chat_template"):
        ChatTokenizer.load(tmp_path)
