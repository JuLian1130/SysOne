"""torchcast 的答面：`decision_config.json` + `jev_render_v3` 题面 + 厂商判答。

移植自厂商 Apache-2.0 包 `torchcast_decision`（`render.py` 与 `model.py`，随 checkpoint 发布，
见 `LICENSE`）。**不 import 厂商模块**：`model.py` 顶层 `import torch`（它要做前向），而这条面
只做"文本 → id"与判答案，该与 `h2o_serve` 一样是一条不装 torch 的路径。题面那两个纯函数与
判答那几十行照抄过来，让两条前向（厂商桥与我们的 ANE 链）共用同一份实现 —— 各有各的模板时，
概率差多少都说不清是谁的问题（票据 12 在 h2o 上量过这件事）。

**这里没有前向。** `vendors/torchcast_bridge.py`（厂商 torch 引擎，对拍基线）与
`engine/strategies/pooled.py`（我们的 ANE 链）都走这一份答面。

**图**：厂商带图时把消息改成结构化 content 交给模板展开图块，这里是字符串拼接
（`VISION_BLOCK`）—— 两者渲染出的文本逐字节相同，所以 token 也相同。占位符展开成
`counts[i]` 行、M-RoPE 三轴位置与行覆盖都在链上（`models/vision/merge`、`chain._mrope`），
这条面只负责"图放在题面的哪儿"。

与厂商有意的两处差别，都会出声而不是静默改变数值：

1. 题面超过 16 384 token 时厂商把 state 首尾留下、中间插一行 `\\n...\\n`；我们不改写（重新切
   token 的边界不可能与厂商一致），按 `RequestTooLongError` 拒掉。真实题集最长 3 952 token
   （111 道），而全双向本来就要整条 prompt 落在一个桶里。
2. 选项多于 26 个时厂商分两趟（`wide_choice`）并把第一轮的概率按 `residual` 合并；本机题集最
   多 6 个选项，那条路的概率合并没有在真权重上对过，所以这里报 `UnsupportedKindError`。
"""

from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...errors import (
    ConfigError,
    ReadoutError,
    RequestTooLongError,
    UnsupportedKindError,
    UnsupportedMediaError,
)
from ..tokenizer import ChatTokenizer
from .base import Prepared, wire_answer

VERSION = "jev_render_v3"
CONFIG_FILE = "decision_config.json"
# 一张图一个占位符。分词器模板对 `{"type": "image"}` 渲染出的就是这三个特殊 token（`add_vision_id`
# 未定义，所以模板不加 "Picture k:"），实测与字符串拼接**逐字节相同**；`image_pad` 的 id 与
# `config.json` 的 `image_token_id` 一致，链据此把占位符展开成 `counts[i]` 行。
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
# state 里可以用它标"图放这儿"（厂商 `_state_with_images` 同一记号）；不写就放在 state 之前。
IMAGE_MARK = "<image>"
SYSTEM = (
    "Apply the criterion to the evidence. Choose exactly one listed option. "
    "Answer with its letter only."
)
LETTERS = string.ascii_uppercase
MAX_OPTIONS = len(LETTERS)
MAX_LEVELS = 10
POSITIONS_FROM = 8       # 列表至少这么长时，每一项补一个 `_index`（厂商同值）
BUDGET = 16384
NO_STATE = "(none)"
DEFAULT_NOUL_QUESTION = "Which answer fits the evidence?"
# 分词器在 thinking 关掉之后必须以这一段结尾：不检查的话，模板悄悄带上思考前缀而题面还是
# "看起来正常"的 token 串，判词会整体漂移。
THINKING_OFF = "<think>\n\n</think>\n\n"
_LABEL_ONLY = re.compile(r"\(?[a-z][).]?|\(?\d{1,3}[).]?|opt(?:ion)?[_ ]?\d{1,3}", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    """厂商 `render.Question` 的替身：题型、state 文本、instruction、选项（id, 描述）。"""

    kind: str
    state: str
    instructions: str
    options: tuple

    @property
    def ids(self) -> list[str]:
        return [option_id for option_id, _ in self.options]


# ---------------------------------------------------------------- 请求 → Question
def _positions(value: object) -> object:
    """厂商那份渲染对长列表的处理：8 项起每项补一个 `_index`。

    这不是装饰 —— 厂商 harness 就是这么渲染 state 的，少这一层，35 道 dict state 的题（111 道
    里）题面会与对拍基准差一截。
    """
    if isinstance(value, dict):
        return {key: _positions(item) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) < POSITIONS_FROM:
            return [_positions(item) for item in value]
        return [
            {"_index": index, **_positions(item)}
            if isinstance(item, dict)
            else {"_index": index, "value": _positions(item)}
            for index, item in enumerate(value)
        ]
    return value


def state_as_text(state: object) -> str:
    """state 的线格式：字符串原样，其余压成紧凑 JSON；空 state 由 `user_text` 换成 `(none)`。"""
    if isinstance(state, str):
        return state
    return "" if state in (None, {}, []) else json.dumps(_positions(state), ensure_ascii=False)


def with_images(state: object, count: int) -> str:
    """带图请求的 state 文本：按厂商 `_state_with_images` 的同一套位置规则拼进图块。

    三种情形，与厂商逐条对应：

    - state 里有 `count` 个 `<image>` 记号 → 每个记号换成一块（图跟着文字走）；
    - 有记号但数对不上 → 拒（`422`：请求自相矛盾，重试也一样）；
    - 没有记号 → 一张图放在 state 之前；多张按 `Image k:` 逐行列出，再接空行与 state。

    模板那条路是"结构化 content 交给 jinja 展开"，这里是字符串拼接 —— `add_vision_id` 未定义
    时两者渲染出的**字符串逐字节相同**，所以 token 也相同（见 `test_torchcast_serve.py`）。
    """
    if count < 1:
        raise UnsupportedMediaError("图数必须为正", images=count)
    text = state_as_text(state)
    marks = text.count(IMAGE_MARK)
    if marks == count:
        return text.replace(IMAGE_MARK, VISION_BLOCK)
    if marks:
        raise UnsupportedMediaError(
            "state 里的 <image> 记号数与图数不一致", marks=marks, images=count)
    if count == 1:
        lead = VISION_BLOCK
    else:
        lead = "\n".join(f"Image {index}: {VISION_BLOCK}" for index in range(1, count + 1))
    return lead if not text.strip() else lead + "\n\n" + text


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = (value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)).strip()
    return text or None


def level_keys(scale: object) -> list:
    """dict 形态的 score 档位键，按档位高低（数值）排；解析不成就按给的顺序。"""
    try:
        return sorted(scale, key=float)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return list(scale)  # type: ignore[arg-type]


def question(state: object, spec: dict) -> Question:
    """一道题的线格式 → 检查过的 `Question`（与厂商 `render.question` 同一条规则）。"""
    kind = spec.get("type", "choice")
    kind = "noul" if kind == "bool" else kind
    criteria = spec["criteria"] if "criteria" in spec else spec.get("options")
    instructions = _text_or_none(
        spec["instructions"] if "instructions" in spec else spec.get("question")) or ""
    if kind == "choice":
        pairs = (
            [(str(item), None) for item in dict.fromkeys(str(item) for item in criteria)]
            if isinstance(criteria, (list, tuple))
            else [(str(key), _text_or_none(value)) for key, value in (criteria or {}).items()]
        )
    elif kind == "score":
        levels = ([criteria[key] for key in level_keys(criteria)]
                  if isinstance(criteria, dict) else list(criteria or ()))
        pairs = [(str(index), _text_or_none(value)) for index, value in enumerate(levels)]
    elif kind == "noul":
        criteria = criteria if isinstance(criteria, dict) else {}
        yes = criteria["true"] if "true" in criteria else criteria.get(True)
        no = criteria["false"] if "false" in criteria else criteria.get(False)
        pairs = [("true", _text_or_none(yes)), ("false", _text_or_none(no))]
        instructions = instructions or DEFAULT_NOUL_QUESTION
    else:
        raise UnsupportedKindError("torchcast 答不了这种题型", kind=kind)
    result = Question(kind, state_as_text(state), instructions, tuple(pairs))
    _check(result)
    return result


def _check(item: Question) -> None:
    """厂商 `_check`：这些不合法都是**坏请求**，客户端不该重试。"""
    if not item.instructions.strip():
        raise ReadoutError("这道题没有 instructions", kind=item.kind)
    count = len(item.options)
    if not 2 <= count <= MAX_OPTIONS:
        raise UnsupportedKindError(
            "torchcast 最多 26 个选项", kind=item.kind, options=count, limit=MAX_OPTIONS)
    for option_id, description in item.options:
        if (not option_id.strip() or "\n" in option_id or "\r" in option_id
                or len(option_id) > 200):
            raise UnsupportedKindError("选项 id 必须是 1 到 200 字符的一行", kind=item.kind,
                                       option=option_id)
        if description is not None and not isinstance(description, str):
            raise UnsupportedKindError("选项描述必须是文本", kind=item.kind, option=option_id)
    if len(set(item.ids)) < count:
        raise UnsupportedKindError("选项 id 有重复", kind=item.kind, options=count)
    if item.kind == "score" and count > MAX_LEVELS:
        raise UnsupportedKindError(
            "torchcast 的 score 最多 10 档", kind=item.kind, options=count, limit=MAX_LEVELS)


# ---------------------------------------------------------------- Question → prompt
def _squash(text: object) -> str:
    return re.sub(r"[\s_\-]+", " ", str(text)).strip().lower()


def option_lines(item: Question) -> list[str]:
    """选项行：`字母) 内容`。描述缺失或只是重复 id 时只写 id，noul 写 yes/no。

    选项 id 全是裸标签（字母、数字、`option 3`）且都有描述时只写描述：否则 `A) C: Paris` 这种
    行会让那个字母指谁变得含糊。
    """
    descs = dict(item.options)
    labels_only = (
        item.kind == "choice"
        and all(_LABEL_ONLY.fullmatch(option_id) for option_id in item.ids)
        and all(descs[option_id] for option_id in item.ids)
    )
    lines = []
    for letter, (option_id, description) in zip(LETTERS, item.options):
        shown = ("yes" if option_id == "true" else "no") if item.kind == "noul" else option_id
        if labels_only:
            body = description
        elif (description is None or not description.strip()
              or _squash(description) == _squash(shown)):
            body = shown
        else:
            body = f"{shown}: {description}"
        lines.append(f"{letter}) {body}")
    return lines


def user_text(item: Question) -> str:
    state = item.state if item.state.strip() else NO_STATE
    return (
        f"Evidence:\n{state}\n\nQuestion: {item.instructions.strip()}\nOptions:\n"
        + "\n".join(option_lines(item))
    )


def chat(item: Question) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(item)}]


def chat_text(tokenizer: object, messages: list[dict]) -> str:
    """套分词器自己的 chat 模板（thinking 关）；模板结尾不是思考关前缀就报配置错。"""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINKING_OFF):
        raise ConfigError("分词器的 chat 模板结尾不是 thinking-off 前缀", tail=text[-40:])
    return text


def letter_ids(tokenizer: object) -> tuple[int, ...]:
    """A..Z 的 token id，并验证每个字母在题面之后都是一个独立 token。"""
    probe = chat_text(tokenizer, chat(question("", {"type": "noul", "instructions": "q"})))
    base = _ids(tokenizer, probe)
    picked: list[int] = []
    for letter in LETTERS:
        one = _ids(tokenizer, letter)
        if len(one) != 1 or _ids(tokenizer, probe + letter) != base + one:
            raise ConfigError("字母在题面之后不是一个独立 token", letter=letter, tokens=len(one))
        picked.append(one[0])
    if len(set(picked)) != len(picked):
        raise ConfigError("26 个字母的 token id 有重复")
    return tuple(picked)


def _ids(tokenizer: object, text: str) -> tuple[int, ...]:
    values = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tuple(int(value) for value in values)


# ---------------------------------------------------------------- 判答（厂商口径）
def choice_certainty(probabilities: Sequence[float]) -> float:
    """选择题的置信度：`(n·p_max − 1) / (n − 1)`，截在 [0, 1]。"""
    if len(probabilities) < 2:
        return 1.0
    count = len(probabilities)
    value = (count * max(probabilities) - 1) / (count - 1)
    return 0.0 if value < 0 else 1.0 if value > 1 else float(value)


def score_certainty(probabilities: Sequence[float]) -> float:
    """score 的置信度：1 − 期望档位距离 ÷ 均匀分布摊在中间时的同一个距离。"""
    levels = range(len(probabilities))
    mode = max(levels, key=lambda index: probabilities[index])
    spread = sum(probabilities[index] * abs(index - mode) for index in levels)
    centre = (len(probabilities) - 1) / 2
    even = sum(abs(index - centre) for index in levels) / len(probabilities)
    if not even:
        return 1.0
    return max(0.0, 1 - spread / even)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


class Serve:
    """厂商的答面。`config` / `tokenizer` 都能注入，单测不需要真模型目录。"""

    # 图块由 `prepare` 拼进 state（`with_images`），链负责展开占位符与 M-RoPE 位置，装配据此建塔。
    supports_images = True

    def __init__(self, config: dict, tokenizer: object) -> None:
        prompt_format = config.get("prompt_format", VERSION)
        if prompt_format != VERSION:
            raise ConfigError("decision_config.json 的题面格式与这个适配不一致",
                              prompt_format=prompt_format, expected=VERSION)
        mode = config.get("attention_mode")
        if mode != "noncausal_full_attention":
            raise ConfigError("decision_config.json 的注意力模式与这个适配不一致",
                              attention_mode=mode, expected="noncausal_full_attention")
        floor = float(config.get("noul_floor", 0.0))
        if floor and not 0.5 < floor < 1.0:
            raise ConfigError("noul_floor 必须在 0.5 与 1 之间", noul_floor=floor)
        self._noul_floor = floor
        self._temperature = {
            kind: float(value)
            for kind, value in (config.get("temperature_by_type") or {}).items()
        }
        self._tokenizer = tokenizer
        self._letters = letter_ids(tokenizer)
        declared = config.get("letter_token_ids")
        if declared is not None and tuple(int(value) for value in declared) != self._letters:
            raise ConfigError("分词器给出的字母 token 与 decision_config.json 不一致",
                              declared=[int(value) for value in declared],
                              tokenizer=list(self._letters))

    @classmethod
    def open(cls, model_dir: Path | str, tokenizer: object | None = None) -> Serve:
        """从 checkpoint 目录装：`decision_config.json` 与分词器（不给就读目录里的）。"""
        root = Path(model_dir)
        path = root / CONFIG_FILE
        if not path.exists():
            raise ConfigError("checkpoint 缺少 decision_config.json", path=str(path))
        config = json.loads(path.read_text(encoding="utf-8"))
        if tokenizer is None:
            tokenizer = ChatTokenizer.load(root)
        return cls(config, tokenizer)

    # ---------------------------------------------------------------- 题面
    def prepare(self, state: object, spec: dict, *, images: int = 0) -> Prepared:
        """题面 → 整条 prompt 的 token 与回答槽上那 26 行字母里的前 n 行。

        `label_ids` 是字母 token 的 id（读出表按它取 `lm_head` 的行），`labels` 是**选项 id**
        —— 字母只是题面里的记号，答面直接给引擎的判词口径，不用把字母再翻回 id。

        `images > 0` 时图块已由 `with_images` 拼进 state，这里照常渲染：返回的 id 里每张图只有
        **一个** `image_pad`，展开成 `counts[i]` 行是链的事（`merge.attach`）。
        """
        if images < 0:
            raise UnsupportedMediaError("图数不能为负", images=images)
        if images:
            state = with_images(state, images)
        item = question(state, spec)
        ids = _ids(self._tokenizer, chat_text(self._tokenizer, chat(item)))
        if len(ids) > BUDGET:
            raise RequestTooLongError("题面超过厂商预算（厂商会截 state，我们不截）",
                                      tokens=len(ids), budget=BUDGET)
        count = len(item.options)
        return Prepared(
            ids=ids,
            label_ids=self._letters[:count],
            labels=tuple(item.ids),
            names=tuple(item.ids),
            descs=tuple(description or "" for _, description in item.options),
            kind=item.kind,
            temperature=self._temperature.get(item.kind, 1.0),
            threshold=None,
        )

    # ---------------------------------------------------------------- 判答
    def answer(
        self, prepared: Prepared, logits: Sequence[float]
    ) -> tuple[str, tuple[str, ...], tuple[float, ...], float]:
        """回答槽上的 logits（未除温度）→ (值, 标签, 概率, 置信度)。

        温度由厂商配置按题型给，概率在这里只归一化一次：`pooled` 策略把 logits 原样递过来，
        中间再归一化一遍就成了两套温度。
        """
        if len(logits) != len(prepared.label_ids):
            raise ReadoutError("logits 数与标签数不一致", logits=len(logits),
                               labels=len(prepared.label_ids))
        scaled = np.asarray([float(value) for value in logits], dtype=np.float32)
        probabilities = _softmax(scaled / np.float32(prepared.temperature))
        return wire_answer(self._answer(prepared, probabilities))

    def _answer(self, prepared: Prepared, probabilities: np.ndarray) -> dict:
        """厂商 `TorchcastDecision._answer` 的同一套输出。"""
        values = [float(value) for value in probabilities]
        if prepared.kind == "noul":
            yes = values[prepared.labels.index("true")]
            # 厂商 JevBench 口径：0.2 < P(yes) < 0.8 视为没答出来，压到地板（按它自己那一侧）
            if self._noul_floor and max(yes, 1.0 - yes) < self._noul_floor:
                yes = self._noul_floor if yes >= 0.5 else 1.0 - self._noul_floor
            return {"type": "noul", "noul": yes}
        if prepared.kind == "score":
            expected = sum(level * probability for level, probability in enumerate(values))
            return {
                "type": "score",
                "score": expected,
                "confidence": score_certainty(values),
                "probabilities": {str(level): value for level, value in enumerate(values)},
            }
        dist = dict(zip(prepared.labels, values))
        return {
            "type": "choice",
            "choice": max(dist, key=dist.get),
            "confidence": choice_certainty(values),
            "probabilities": dist,
        }


__all__ = ["Serve", "chat", "chat_text", "letter_ids", "option_lines", "question", "state_as_text",
           "user_text", "with_images"]
