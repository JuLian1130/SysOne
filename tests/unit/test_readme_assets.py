"""README 里的图：引用得到，且画出来的比例与它自己标的数字一致。

这条防的是**图在撒谎**：柱状图的条宽是手算的，改了秒数却忘了重算条宽，页面上的文字说 8.4×
而图看上去是 5×，两边都"有数据"，谁也不会发现。SVG 是手写的（没有生成脚本），所以比例这条
只能由测试守着。

第二种失效是引用：`assets/` 下的图被改名或删掉，GitHub 上就是一个断图，本地看 markdown 源
却是正常的。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SVG = "http://www.w3.org/2000/svg"
CHART = ROOT / "assets" / "h2o-speed.svg"

_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")


def _readme_images() -> list[str]:
    """两份 README 引用的图片路径（去重，保持出现顺序）。"""
    found: list[str] = []
    for name in ("README.md", "README.zh.md"):
        for target in _IMAGE.findall((ROOT / name).read_text(encoding="utf-8")):
            if target not in found:
                found.append(target)
    return found


def test_every_image_the_readmes_show_exists() -> None:
    images = _readme_images()
    assert "assets/h2o-speed.svg" in images
    for target in images:
        assert (ROOT / target).is_file(), f"README 引用了不存在的图：{target}"


def _seconds(root: ET.Element) -> list[float]:
    """图里标的两个总秒数（`136.03 s` / `1137.30 s`）。"""
    values = []
    for node in root.iter(f"{{{SVG}}}text"):
        match = re.fullmatch(r"([0-9.]+) s", (node.text or "").strip())
        if match:
            values.append(float(match.group(1)))
    return sorted(values)


def _bar_widths(root: ET.Element) -> list[float]:
    """两根数据条：各有自己的填充色，且不是卡片或底轨。"""
    tracks = {"#ffffff", "#f0f3f6"}
    return sorted(
        float(rect.get("width", "0"))
        for rect in root.iter(f"{{{SVG}}}rect")
        if (rect.get("fill") or "") not in tracks
    )


def test_the_chart_bars_are_proportional_to_their_numbers() -> None:
    root = ET.parse(CHART).getroot()
    seconds, widths = _seconds(root), _bar_widths(root)
    assert len(seconds) == 2 and len(widths) == 2
    labelled = round(seconds[1] / seconds[0], 1)
    drawn = round(widths[1] / widths[0], 1)
    # 一像素的取整在短条上是 2 % 的量级；差到 0.1 就说明有人只改了一边。
    assert drawn == labelled, {"labelled": labelled, "drawn": drawn}


def test_the_ratio_note_matches_the_bars() -> None:
    root = ET.parse(CHART).getroot()
    seconds = _seconds(root)
    notes = [
        (node.text or "").strip()
        for node in root.iter(f"{{{SVG}}}text")
        if (node.text or "").strip().endswith("faster")
    ]
    assert len(notes) == 1
    assert notes[0].startswith(f"{seconds[1] / seconds[0]:.1f}×")
