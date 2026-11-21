"""h2o 的答面：`serve_config.json` + 厂商 shim + 分词器。

**这里没有前向**。两条前向共用这一份"题面怎么渲染、答案怎么格式化"：

- `vendors/h2o_bridge.py`：厂商自己的 torch 模型；
- `engine/strategies/pooled.py`：我们的 ANE 链。

也**不需要 `transformers`**：这条面只做"文本 → id"，用的是与 mude 出货路径同一个
`ChatTokenizer`（`tokenizer.json` 直连，见 `models/tokenizer.py`）。要厂商的模型结构时才装
它（`h2o_bridge` 的前向），那时它已经在环境里。

共用不是省事，而是对拍的前提：两条路要是各有各的模板，概率差多少都说不清是谁的问题
（票据 12 量过，只给选项键不给判据描述时 105 题里 7 题判词不同，把厂商原装的 criteria
喂回去这 7 题全部复现 —— 差的不是权重，是题面）。

读出为什么不能只写在设备层：h2o 的答案是**回答槽上那个标签 token 的 lm_head 行**，而选项
字母怎么落到 token 上、按题型换哪个温度、noul 的阈值与地板怎么算，全在厂商的
`format_answer` 里（`h2o_lightning_shim.py:750`）。设备层只该回答"这串 token 过完整条链
之后，某几行词表权重的点积是多少"，所以 `Serve.answer` 只吃一个 logits 向量。

载体与接口（`Prepared` / `AnswerSurface`）在 `vendors/base.py`：`pooled` 策略按那份契约调
答面，所以这里只剩 h2o 自己的实现。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Sequence

from ...errors import ConfigError, ReadoutError
from ..tokenizer import ChatTokenizer
from .base import Prepared, wire_answer

SHIM_FILE = "h2o_lightning_shim.py"
SERVE_FILE = "serve_config.json"
# 一个占位符一张图。厂商模板（`chat_template.jinja` 与 mude 同源）把 `{"type": "image"}` 展开成
# 这三个 id；我们这里没有模板可走（`Serve` 只收一段文本），所以直接拼字符串 —— 分词器保证
# 这三个是原子 token（实测 248053 / 248056 / 248054）。
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def _load_shim(model_dir: Path):
    path = model_dir / SHIM_FILE
    if not path.exists():
        raise ConfigError("厂商目录缺少 shim", path=str(path))
    spec = importlib.util.spec_from_file_location("h2o_lightning_shim", path)
    if spec is None or spec.loader is None:
        raise ConfigError("无法加载 shim", path=str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Serve:
    """厂商的答面。`config` / `shim` / `tokenizer` 都能注入，单测不需要真模型目录。"""

    # 图块由 `prepare` 自己拼在 state 之前（`VISION_BLOCK`），所以装配要为它建塔。
    supports_images = True

    def __init__(self, config: dict, shim: object, tokenizer: object) -> None:
        self._shim = shim
        try:
            self._contract = shim.Contract(config)
        except (ValueError, KeyError) as exc:
            raise ConfigError("serve_config.json 与 shim 不匹配", reason=str(exc)) from exc
        self._tokenizer = tokenizer

    @classmethod
    def open(cls, model_dir: Path | str, tokenizer: object | None = None) -> Serve:
        """从厂商目录装：`serve_config.json`、shim、分词器（不给就读目录里的 `tokenizer.json`）。"""
        root = Path(model_dir)
        serve = root / SERVE_FILE
        if not serve.exists():
            raise ConfigError("厂商目录缺少 serve_config.json", path=str(serve))
        config = json.loads(serve.read_text(encoding="utf-8"))
        if tokenizer is None:
            tokenizer = ChatTokenizer.load(root)
        return cls(config, _load_shim(root), tokenizer)

    # ---------------------------------------------------------------- 题面
    def prepare(self, state: str, spec: dict, *, images: int = 0) -> Prepared:
        """题面 → 整条 prompt 的 token 与回答槽上的标签 token。

        `spec` 是已经转成厂商线格式的问题（`vendors.base.question_spec`）：渲染与判答案都吃
        这一种形态，两条前向的调用点才可能逐字相同。

        `images > 0` 时**图块由我们拼在 `state_text` 之前**，一个 `<|image_pad|>` 一张图：厂商
        的 `Serve` 只收一段 `state_text`（`serve_config.json` 没有 `image` 块），它的
        `question_prompt` 不认图。位置与 mude 那边一致（题面之前），于是两个家族只差渲染方式。
        """
        state_text = self._shim.render_state(state, self._contract.state_compact)
        if images < 0:
            raise ConfigError("图数不能为负", images=images)
        if images:
            state_text = VISION_BLOCK * images + state_text
        prompt, names, descs, labels = self._contract.question_prompt(state_text, spec)
        ids = self._ids(prompt)
        if not ids:
            raise ReadoutError("渲染出的 prompt 是空的")
        kind = spec.get("type", "choice")
        return Prepared(
            ids=ids,
            label_ids=self._label_ids(prompt, ids, labels),
            labels=tuple(labels),
            names=tuple(names),
            descs=tuple(descs),
            kind=kind,
            temperature=float(
                self._contract.temperature_by_type.get(kind, self._contract.temperature)
            ),
            threshold=self._shim.noul_threshold(spec),
        )

    def answer(
        self, prepared: Prepared, logits: Sequence[float]
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], float]:
        """回答槽上的 logits（未除温度）→ (值, 标签, 概率, 置信度)。判答案的是厂商代码。"""
        values = [float(value) for value in logits]
        if len(values) != len(prepared.label_ids):
            raise ReadoutError(
                "logits 数与标签数不一致",
                logits=len(values),
                labels=len(prepared.label_ids),
            )
        result = self._shim.format_answer(
            prepared.kind,
            prepared.names,
            prepared.descs,
            values,
            prepared.temperature,
            self._contract.noul_floor,
            prepared.threshold,
        )
        return wire_answer(result)

    # ---------------------------------------------------------------- 内部
    def _ids(self, text: str) -> tuple[int, ...]:
        values = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return tuple(int(value) for value in values)

    def _label_ids(
        self, prompt: str, ids: tuple[int, ...], labels: Sequence[str]
    ) -> tuple[int, ...]:
        """每个选项标签在回答槽上的单 token id（必须是恰好一个新 token）。"""
        base = len(ids)
        picked: list[int] = []
        for label in labels:
            probe = self._ids(prompt + " " + label)
            if len(probe) != base + 1:
                raise ReadoutError(
                    "选项标签不是单 token，无法在回答槽上读出",
                    label=label,
                    tokens=len(probe) - base,
                )
            picked.append(probe[-1])
        return tuple(picked)


__all__ = ["Serve"]
