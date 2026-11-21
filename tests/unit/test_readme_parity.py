"""成对的公开文档必须同步：英文版是主稿，中文版是翻译。

覆盖 `README.md`/`README.zh.md` 与 `API.md`/`API.zh.md`。只改一份忘了另一份时，这条
测试会指出分叉点：围栏代码块必须逐字节一致，标题层级、表格行数与形状、行内代码 token
（参数名/环境变量/路径/状态码的出现次数）与链接必须一致。措辞可以不同。

链接按"一对文档"归一：`README.md` 与 `README.zh.md` 都记作 `@readme`，`API.md` 与
`API.zh.md` 都记作 `@api`，因此语言切换链接和跨文档链接在两边都能对上。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAIRS = (("README.md", "README.zh.md"), ("API.md", "API.zh.md"))
PAIRED = {
    "README.md": "@readme",
    "README.zh.md": "@readme",
    "API.md": "@api",
    "API.zh.md": "@api",
}

_CODE = re.compile(r"```.*?```", re.S)
_HEADING = re.compile(r"^(#{1,6}) ", re.M)
_INLINE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _read(name: str) -> str:
    path = ROOT / name
    assert path.is_file(), f"缺少 {name}"
    return path.read_text(encoding="utf-8")


def _body(text: str) -> str:
    """去掉围栏代码块：行内代码、标题与表格只对正文检查。"""
    return _CODE.sub("", text)


def _blocks(text: str) -> list[str]:
    return _CODE.findall(text)


def _headings(text: str) -> list[str]:
    return _HEADING.findall(text)


def _tables(text: str) -> list[int]:
    """每张连续表格的行数，按出现顺序。"""
    runs: list[int] = []
    current = 0
    for line in _body(text).splitlines():
        if line.startswith("|"):
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _inline(text: str) -> Counter[str]:
    return Counter(_INLINE.findall(_body(text)))


def _links(text: str) -> list[str]:
    return [PAIRED.get(target, target) for target in _LINK.findall(text)]


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_the_pair_links_to_itself(en_name: str, zh_name: str) -> None:
    assert f"({zh_name})" in _read(en_name)
    assert f"({en_name})" in _read(zh_name)


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_code_blocks_are_byte_identical(en_name: str, zh_name: str) -> None:
    en, zh = _blocks(_read(en_name)), _blocks(_read(zh_name))
    assert len(en) == len(zh), f"{en_name}: {len(en)} vs {zh_name}: {len(zh)} blocks"
    for index, (left, right) in enumerate(zip(en, zh)):
        assert left == right, f"{en_name} block {index} differs:\n{left}\n---\n{right}"


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_heading_levels_match(en_name: str, zh_name: str) -> None:
    assert _headings(_read(en_name)) == _headings(_read(zh_name))


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_tables_keep_the_same_shape(en_name: str, zh_name: str) -> None:
    assert _tables(_read(en_name)) == _tables(_read(zh_name))


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_inline_code_tokens_match(en_name: str, zh_name: str) -> None:
    en, zh = _inline(_read(en_name)), _inline(_read(zh_name))
    assert en == zh, {
        "missing_in_zh": en - zh,
        "extra_in_zh": zh - en,
    }


@pytest.mark.parametrize(("en_name", "zh_name"), PAIRS)
def test_links_match(en_name: str, zh_name: str) -> None:
    assert _links(_read(en_name)) == _links(_read(zh_name))
