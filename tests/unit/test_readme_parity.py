"""英文与中文 README 必须同步：代码块、标题层级、表格、行内代码与链接逐项对账。

英文 `README.md` 是主稿，中文 `README.zh.md` 是翻译。措辞可以不同，但读者照着做的部分
不能分叉：命令与 JSON 逐字节一致，参数名、环境变量、路径的出现次数一致，标题层级与表格
行数一致。只改一份忘了另一份时，这条测试会指出分叉点。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EN = ROOT / "README.md"
ZH = ROOT / "README.zh.md"

_CODE = re.compile(r"```.*?```", re.S)
_HEADING = re.compile(r"^(#{1,6}) ", re.M)
_INLINE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\]\(([^)\s]+)\)")
# 语言切换链接在两边互为镜像，不参与对账。
_SWITCHER = {"README.md", "README.zh.md"}


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
    return [target for target in _LINK.findall(text) if target not in _SWITCHER]


def _read(path: Path) -> str:
    assert path.is_file(), f"缺少 {path.name}"
    return path.read_text(encoding="utf-8")


def test_both_readmes_link_to_each_other() -> None:
    assert "(README.zh.md)" in _read(EN)
    assert "(README.md)" in _read(ZH)


def test_code_blocks_are_byte_identical() -> None:
    en, zh = _blocks(_read(EN)), _blocks(_read(ZH))
    assert len(en) == len(zh), f"{len(en)} vs {len(zh)} fenced blocks"
    for index, (left, right) in enumerate(zip(en, zh)):
        assert left == right, f"block {index} differs:\n{left}\n---\n{right}"


def test_heading_levels_match() -> None:
    assert _headings(_read(EN)) == _headings(_read(ZH))


def test_tables_keep_the_same_shape() -> None:
    assert _tables(_read(EN)) == _tables(_read(ZH))


def test_inline_code_tokens_match() -> None:
    en, zh = _inline(_read(EN)), _inline(_read(ZH))
    assert en == zh, {
        "missing_in_zh": en - zh,
        "extra_in_zh": zh - en,
    }


def test_links_match() -> None:
    assert _links(_read(EN)) == _links(_read(ZH))
