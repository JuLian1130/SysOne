"""模型目录的文本侧：`tokenizers` 直连 + `jinja2` 聊天模板，不装 `transformers`。

出货路径只需要两件事（`models/families/mude.py` 里 `MudePrompter` 的鸭子类型）：把文本编成
id，以及渲染 chat 模板。`transformers` 为这两件事要多装 ≈0.28 GB 的底座（`AutoTokenizer` 走到
0.3635 GB，`tokenizers` 直连 `tokenizer.json` 是 0.0814 GB，票据 03 的实测）；模板取
`tokenizer_config.json` 里的 jinja 字符串，或同目录的 `chat_template.jinja`，渲染环境照
transformers 的 `_cached_compile_jinja_template` 复制——同一套过滤器与全局函数，否则模板
渲染差一个空格就会静默改变 token。

`transformers` 仍然留给 h2o / torchcast 两条 vendor 路径，它们要的是模型结构不是文本。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from ..errors import ConfigError


def _environment():
    """transformers `_cached_compile_jinja_template` 的那套环境（只去掉 `{% generation %}`）。"""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise jinja2.TemplateError(message)

    def tojson(value, ensure_ascii: bool = False, indent=None, separators=None, sort_keys: bool = False) -> str:
        # transformers 覆盖了 jinja 自带的 `tojson`：那个会转义 HTML 字符，模型看不到原文。
        return json.dumps(
            value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    def strftime_now(format: str) -> str:
        return datetime.now().strftime(format)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    environment.filters["tojson"] = tojson
    environment.globals["raise_exception"] = raise_exception
    environment.globals["strftime_now"] = strftime_now
    return environment


def _template_text(directory: Path, config: dict) -> str:
    """模板从哪来：`tokenizer_config.json` 的内嵌字符串，或同目录的 `chat_template.jinja`。

    `pplx` 的发布包把模板放在单独的 `.jinja` 里（配置项是 `null`），transformers 也认这个
    兄弟文件，所以这里按同样的顺序找 —— 否则一份完全合法的检查点会因为"模板不在配置里"被判
    成坏配置。命名模板（dict/list）仍然不猜：挑哪一条当默认是 transformers 的约定。
    """
    template = config.get("chat_template")
    if isinstance(template, str):
        return template
    sibling = directory / "chat_template.jinja"
    if template is None and sibling.is_file():
        return sibling.read_text(encoding="utf-8")
    raise ConfigError(
        "tokenizer_config.json 的 chat_template 不是单条字符串",
        kind=type(template).__name__,
        searched=str(sibling),
    )


class ChatTokenizer:
    """`tokenizer.json` + `tokenizer_config.json` 的聊天分词器（transformers 那层的替身）。

    鸭子类型只有两件事，与 `MudePrompter.__init__` 的说明一致：
    `__call__(text, add_special_tokens=False)["input_ids"]` 与
    `apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, **模板变量)`。
    模板取 `tokenizer_config.json` 的 `chat_template`，缺它时取同目录的 `chat_template.jinja`。
    """

    def __init__(self, backend: object, template: str) -> None:
        self._backend = backend
        self._template = _environment().from_string(template)

    @classmethod
    def load(cls, model_dir: str | Path) -> "ChatTokenizer":
        """读模型目录的两个文件；缺了按配置错报。"""
        directory = Path(model_dir)
        tokenizer_path = directory / "tokenizer.json"
        config_path = directory / "tokenizer_config.json"
        if not tokenizer_path.exists():
            raise ConfigError("模型目录没有 tokenizer.json", path=str(tokenizer_path))
        if not config_path.exists():
            raise ConfigError("模型目录没有 tokenizer_config.json", path=str(config_path))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        template = _template_text(directory, config)
        from tokenizers import Tokenizer as Backend

        return cls(Backend.from_file(str(tokenizer_path)), template)

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """与 transformers 的默认一致（`add_special_tokens=True`）；出货路径显式传 False。"""
        ids = self._backend.encode(text, add_special_tokens=add_special_tokens).ids
        return {"input_ids": [int(value) for value in ids]}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        **variables: object,
    ) -> str:
        """渲染一条对话；`tokenize=True` 不支持（模板变量会是 id，与文本不是一回事）。"""
        if tokenize:
            raise ConfigError("聊天模板只渲染文本", tokenize=tokenize)
        return self._template.render(
            messages=conversation,
            tools=None,
            documents=None,
            add_generation_prompt=add_generation_prompt,
            **variables,
        )


__all__ = ["ChatTokenizer"]
