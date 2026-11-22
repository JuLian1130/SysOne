"""inclusionAI SingVL S1 4B 家族：绑定 embedding 的 256 码读出 + 厂商自拼的题面。

与 d3 那条 `split` 链路**同一条执行器**（投影/FFN 走 ANE、注意力与 GDN 走 Metal），差别只在
这一家声明的三件事：

- **读出**：`tie_word_embeddings: true`，没有 `lm_head`，所以 256 个答案码的 logits 就是
  `embed_tokens.weight` 里那 256 个 id 的行（`ReadoutPlan.kind="embed_rows"`）。厂商
  `inference/model.py` 的 `logits[:, choice_ids[:n]]` 与这里按 id 切出来的表做 `table @ hidden`
  是同一个乘法 —— 不需要往权重目录写一份派生表。
- **题面**：不走 chat 模板。厂商 `render()` 自己拼 `<|im_start|>user\\n` + 固定指令 +
  `### Question:` / `### Options:` + 固定的 assistant 前缀（`ANSWER_PREFIX`），所以读出在
  **末位**（前缀最后是 `</think>\\n\\n`），读出位置 `None`。
- **掩码**：config 写 `jev_prefix_attention_mode: "prefix_bidir"`，但厂商 `model.py` 的做法是把
  全注意力层的 `is_causal` 整体关掉、只留 padding 可见性（`bidirectional_prompt`），GDN 保持
  因果。那正好是 `noncausal_full`：题与题之间没有可复用的前缀行（`use_prefix=False`），只有
  跨请求合批这一条收益。

**题面的每一处细节都是契约**（厂商 `render` / `questions_for_state` 逐条移植，改一个字符就换
一批答案）：`<image>` 占位符的编号与重写（`[Image k]`）、结尾图块前移、结尾 `Choices:` 截断、
控制标记拒收、`(key) <image>` 形式的选项图引用。这些都在下面按厂商的顺序执行。

**两种上限与本引擎不同，按厂商的来**（它们是模型的读出能力，不是我们的配置）：
选项 2..256（我们协议层另有 255 的 `MAX_OPTIONS`，所以实际撞不到 256）、`score` 2..10 档。

**没接的部分**（出声，不静默）：视频（塔只做单帧，与 d3 同一条边界）；厂商的
`TEXT_MAX_TOKENS 65 536` / `VISION_MAX_TOKENS 10 000` 只是它自己的预算 —— 我们能跑的宽度是
链的 `--width`（默认 32 768），更长的请求按 `RequestTooLongError` 422 下线。

**置信度**：厂商只发概率，没有 confidence 字段。这里沿用 mude / d3 的熵口径
（`1 - H/log n`），并在文档里写明这是我们的口径，不是厂商的。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

from ...config import EngineConfig
from ...domain.attention import MaskMode
from ...domain.decision import Question
from ...domain.quant import DType
from ...errors import ConfigError, NotReadyError, UnsupportedKindError, UnsupportedMediaError
from ...weights.checkpoint import stored_quant
from ..descriptor import ModelDescriptor, ReadoutPlan
from ..layer_plan import LayerPlan, WeightPlan
from ..recipe import SplitRecipe
from ..registry import register_family
from ..vendors.base import question_spec
from . import mude, qwen35

# 256 个两字母回答码（`config.json` 的 `jev_soft_choice.token_ids`）。
CODE_ROWS = 256
# 张量名前缀：`model.language_model.*`（与 torchcast 同形）；塔在 `model.visual.*`，由
# `weights/vision.tower_root` 按锚点自己找根，不从这里做名字手术。
PREFIX = "model.language_model"
# 绑定 embedding 的那张表：`embed_rows` 从这里按 id 切行。
EMBED = "model.language_model.embed_tokens.weight"
# 与 h2o / d3 / torchcast 同一套 Qwen3.5 结构：norm 零中心存，真值 `w + 1`。
NORM_OFFSET = 1.0

MODES = frozenset({MaskMode.NONCAUSAL_FULL})

# 厂商 `model.py` 的题面常量，逐字照抄。
INSTRUCTION = (
    "Solve the multiple-choice question and output ONLY the index token of the correct option.\n\n"
)
ANSWER_PREFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"
TRAILING_IMAGES = re.compile(r"(?:\s*<image>\s*)+$")
# 用户内容里的模型控制标记：厂商整条拒收（不转义），理由是它们能改写题面结构。
CONTROL_MARKER = re.compile(r"<video>|<\|(?:im_|vision_|image_pad|video_pad)")
# 厂商 `questions_for_state` 的档位上限（`SCORE_LEVELS`）。
SCORE_LEVELS = (2, 10)

# 判据缺省（厂商 `{"true": "Yes", "false": "No"}`）。
_NOUL_KEYS = ("true", "false")
_NOUL_TEXTS = ("Yes", "No")


# ---------------------------------------------------------------------- 读配置


def code_ids(config: Mapping[str, object]) -> tuple[int, ...]:
    """`config.json` 声明的 256 个回答码 id（顺序即读出表行序）。

    三项声明一起验：版本、读出位置与 id 表。只核 id 数不够 —— 换一份 `readout` 含义不同的
    检查点（比如改成读某个标记位置）时，id 表照样是 256 个，而我们的"末位读出"就变成假的了。
    """
    choice = config.get("jev_soft_choice")
    if not isinstance(choice, Mapping):
        raise ConfigError("config.json 没有 jev_soft_choice 段", family="singvl")
    version = choice.get("version")
    readout = choice.get("readout")
    if version != "soft-choice-full-vocab-v2" or readout != "last-valid-input-position":
        raise ConfigError(
            "这份检查点不是 SingVL 的软选择读出",
            version=version,
            readout=readout,
            expected=["soft-choice-full-vocab-v2", "last-valid-input-position"],
        )
    ids = tuple(int(value) for value in choice.get("token_ids") or ())
    if len(ids) != CODE_ROWS or len(set(ids)) != CODE_ROWS:
        raise ConfigError(
            "回答码不是 256 个互不相同的 id",
            count=len(ids),
            unique=len(set(ids)),
            rows=CODE_ROWS,
        )
    return ids


def _tokenizer_backend(model_dir: str | Path) -> object:
    """`tokenizer.json` 的 `tokenizers` 后端（解回答码文本要用它，不需要 transformers）。"""
    path = Path(model_dir) / "tokenizer.json"
    if not path.is_file():
        raise ConfigError("模型目录没有 tokenizer.json", path=str(path))
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # 装了权重但没装这层的依赖：环境未就绪，不是模型问题
        raise NotReadyError("解 SingVL 的回答码需要 tokenizers", reason=str(exc)) from exc
    return Tokenizer.from_file(str(path))


def read_codes(model_dir: str | Path) -> tuple[str, ...]:
    """256 个回答码的**文本**：厂商用 `decode([id])` 生成，并要求每个都编码回同一个 id。

    码表随 checkpoint 走（`config.json` 的 id 表），不从分词器重算：重算错了会把行序弄反，
    而概率看起来仍然像模像样。这里只把 id 翻成文本，并验一遍它对得上 —— 对不上说明这个目录
    的 `tokenizer.json` 与 config 不是同一份，那时整条题面的选项行都会是错的。
    """
    directory = Path(model_dir)
    config = qwen35.load_json(directory / "config.json")
    ids = code_ids(config)
    backend = _tokenizer_backend(directory)
    labels: list[str] = []
    for token in ids:
        label = str(backend.decode([token]))  # type: ignore[attr-defined]
        # 厂商的同一条判据：答案码是两字母（`[A-Z]{2}`）。带空格的解码结果会让选项行变成
        # `AA:  foo`，与训练时的字节不同，而形状不会崩。
        if not re.fullmatch(r"[A-Z]{2}", label):
            raise ConfigError("回答码不是两字母", token=token, label=label)
        encoded = backend.encode(label, add_special_tokens=False).ids  # type: ignore[attr-defined]
        if list(encoded) != [token]:
            raise ConfigError(
                "回答码与分词器对不上", label=label, token=token, encoded=list(encoded)
            )
        labels.append(label)
    return tuple(labels)


def parse_config(model_dir: Path) -> ModelDescriptor:
    """按 SingVL 的读出与掩码读描述符；塔的几何照读 `vision_config`。"""
    directory = Path(model_dir)
    config = qwen35.load_json(directory / "config.json")
    ids = code_ids(config)
    return qwen35.descriptor_from(
        config,
        family="singvl",
        model_id=directory.name,
        readout=ReadoutPlan(
            kind="embed_rows",
            weight=EMBED,
            ids=ids,
            rows=len(ids),
            # 厂商推理不做温度缩放（`logits.softmax(-1)`，T=1），所以这里不给温度。
            temperature=None,
            # 发布包的分片里这张表是 BF16（`dtype: bfloat16`）；读进来一律 fp32，
            # 这里报的是**存**的形态。
            dtype=DType.BF16,
        ),
        supported_modes=MODES,
        trained_modes=MODES,
        default_mode=MaskMode.NONCAUSAL_FULL,
        quant=stored_quant(directory),
    )


def build_layer_plan(descriptor: ModelDescriptor, config: EngineConfig) -> LayerPlan:
    return qwen35.layer_plan(descriptor, config)


def weight_plan(model_dir: Path) -> WeightPlan:
    config = qwen35.load_json(Path(model_dir) / "config.json")
    return WeightPlan(
        descriptor=parse_config(Path(model_dir)),
        prefix=PREFIX,
        norm_offset=NORM_OFFSET,
        rope_theta=qwen35.rope_theta(config),
    )


# ---------------------------------------------------------------------- 题面


def option_plan(question: Question) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...]]:
    """(渲染用的选项文本, 答案标签, 每个标签对应的选项序号)。

    choice 三者同序（标签是选项键，文本是判据描述 —— 厂商 `questions_for_state` 的
    `options = {key: key if value is None else as_text(value)}`）；noul 的文本是
    `Yes` / `No`，引擎标签仍是 ("yes", "no")，所以序号直接同序；score 的标签是 `"0".."n-1"`，
    顺序即档位高低，文本是档位描述。

    两种上限是**模型的读出能力**（厂商 `questions_for_state`），所以越界是坏请求（400）：
    `score` 2..10 档、choice 2..256 个选项。我们的协议层另有 255 的上限，所以 256 走不到。
    """
    if question.kind == "noul":
        given = question_spec(question)["criteria"]
        texts = tuple(given.get(key) or fallback for key, fallback in zip(_NOUL_KEYS, _NOUL_TEXTS))
        return texts, ("yes", "no"), (0, 1)
    if question.kind == "score":
        levels = tuple(question_spec(question)["criteria"])
        low, high = SCORE_LEVELS
        if not low <= len(levels) <= high:
            raise UnsupportedKindError(
                "score 的档位数超出这个模型的范围",
                levels=len(levels),
                range=list(SCORE_LEVELS),
            )
        labels = tuple(str(index) for index in range(len(levels)))
        return levels, labels, tuple(range(len(levels)))
    if question.kind != "choice":
        raise UnsupportedKindError(
            "SingVL 的码表读出答不了这种题型",
            kind=question.kind,
            supported=["choice", "noul", "score"],
        )
    if not question.options:
        raise ConfigError("choice 题缺少选项")
    # 选项键（协议层的线格式键）即答案标签，文本是判据描述 —— 厂商 `questions_for_state` 的
    # `options = {key: key if value is None else as_text(value)}`：行里只写描述，不写键。
    criteria = question_spec(question)["criteria"]
    keys = tuple(criteria)
    return tuple(criteria[key] for key in keys), keys, tuple(range(len(keys)))


class SingVLPrompter:
    """厂商 `render()` 的逐字移植：一段文本、不走 chat 模板、读出在末位。

    `tokenizer` 是 `ChatTokenizer` 那一层（只要有 `__call__(text, add_special_tokens=False)`）。
    这里的题面**不用模板**，所以 `special_tokens` 一个都不转义 —— 厂商的做法是整条拒收含控制
    标记的请求（`CONTROL_MARKER`），而不是把它们转成全角括号。
    """

    def __init__(self, tokenizer: object, codes: Sequence[str]) -> None:
        self._tokenizer = tokenizer
        self._codes = tuple(codes)

    def encode(
        self, state: str, question: Question, *, images: int = 0
    ) -> tuple[tuple[int, ...], int, tuple[str, ...], tuple[int, ...], int | None]:
        """(整条 prompt 的 id, 前缀长度, 答案标签, 标签序号, 读出位置)。

        前缀长度恒为 `None`：这份检查点只有 `noncausal_full`，题与题之间没有可复用的前缀行
        （那条掩码里前缀行与题面耦合）。声明 0 会让编排以为可以按槽切；`None` 是"量不出边界"，
        两条消费方（`shared_prefix` 与 `encode_request`）都把它当"没有前缀"处理。

        读出位置是 `None`（末位）：`ANSWER_PREFIX` 之后没有别的 token。

        `images` 是这道题题面里的图数：一张图在 id 里仍只占**一个** `<|image_pad|>` 占位符，
        展开成 `gh*gw` 个是合流那一步的事（`models/vision/merge`）。
        """
        _texts, labels, indexes, _keys = _plan(question)
        return self._ids(self.text(state, question, images=images)), None, labels, indexes, None

    def text(self, state: str, question: Question, *, images: int = 0) -> str:
        """整条 prompt 的**文本**（厂商 `render()` 的逐字复刻）。

        这是题面的可测契约：单元测试逐字节比它，不需要真分词器；`encode` 只是把它分词。
        厂商的同名函数也分这三段（`<|im_start|>user\\n` + user 内容 + `ANSWER_PREFIX`）。
        """
        texts, _labels, _indexes, keys = _plan(question)
        room = min(CODE_ROWS, len(self._codes))
        if not 1 <= len(texts) <= room:
            raise ConfigError("选项数超出码表", options=len(texts), room=room)
        state = self._state(state, images)
        if any(CONTROL_MARKER.search(text) for text in texts):
            raise UnsupportedMediaError("选项里含模型控制标记", markers=CONTROL_MARKER.pattern)
        if CONTROL_MARKER.search(question.text or ""):
            raise UnsupportedMediaError("题干里含模型控制标记", markers=CONTROL_MARKER.pattern)
        body, texts = self._question(state, question, texts, keys)
        # 厂商在这里做唯一一次占位符替换：整段 user 内容（题干与选项行）里的 `<image>` 都换成
        # 模板那三段 id，一个 `<|image_pad|>` 一张图。
        user = (
            INSTRUCTION + "### Question:\n" + body + "\n\n### Options:\n" + self._lines(texts)
        ).strip()
        return "<|im_start|>user\n" + user.replace("<image>", IMAGE_SLOT) + ANSWER_PREFIX

    # -- 题面 ----------------------------------------------------------------

    def _state(self, state: str, images: int) -> str:
        """state 的 `<image>` 契约：占位符数必须等于图数，不给就按图数补在最前面。"""
        if images < 0:
            raise ConfigError("图数不能为负", images=images)
        if CONTROL_MARKER.search(state):
            raise UnsupportedMediaError("state 里含模型控制标记", markers=CONTROL_MARKER.pattern)
        count = state.count("<image>")
        if images and count == 0:
            state = "<image>\n" * images + state
            count = images
        if count != images:
            # 厂商同一条：题面里的占位符数与图数各说各话时整条拒收。照旧往下跑只会把一张图
            # 的行盖到另一张的位置上 —— 形状全对、答案是错的。
            raise UnsupportedMediaError(
                "state 里的 <image> 占位符数量与请求的图数不一致",
                placeholders=count,
                images=images,
            )
        return state

    def _question(
        self, state: str, question: Question, texts: Sequence[str], keys: Sequence[str]
    ) -> tuple[str, tuple[str, ...]]:
        """厂商 `render()` 中间那一段：选项图引用、结尾图块前移、结尾 `Choices:` 截断。

        返回 `(题干, 选项文本)` —— 选项引用的重写要跟着回到选项行里，厂商是原地改
        `options[key]` 的。
        """
        values = tuple(texts)
        if any("<image>" in text for text in values):
            state, values = _reference_images(state, values, keys)
        body = state + (question.text or "")
        trailing = TRAILING_IMAGES.search(body)
        if trailing and body[: trailing.start()].rstrip():
            body = trailing.group().strip() + "\n" + body[: trailing.start()].rstrip()
        body = body.rstrip()
        if body.endswith("Choices:"):
            body = body[: -len("Choices:")].rstrip()
        return body, values

    def _lines(self, texts: Sequence[str]) -> str:
        return "\n".join(f"{code}: {text}" for code, text in zip(self._codes, texts))

    def _ids(self, text: str) -> tuple[int, ...]:
        values = self._tokenizer(text, add_special_tokens=False)["input_ids"]  # type: ignore[operator]
        return tuple(int(value) for value in values)


def _reference_images(
    state: str, texts: Sequence[str], keys: Sequence[str]
) -> tuple[str, tuple[str, ...]]:
    """选项引用的图（`(key) <image>` 写在 state 里）：把选项文本里的 `<image>` 换成
    `[Image k above]`，再给 state 里每个占位符编上号（厂商 `reference_images` 逐条）。

    只有在某个选项文本含 `<image>` 时才走这条路（厂商同一条判据）：state 里没有对应
    `(key) <image>` 时**报错**，不猜 —— 猜错就是把另一张图的编号写进题面。
    """
    values = list(texts)
    for index, (key, value) in enumerate(zip(keys, values)):
        if "<image>" not in value:
            continue
        needle = f"({key}) {value}"
        count = state.count(needle)
        if count != 1:
            raise UnsupportedMediaError(
                "选项引用的图在 state 里找不到唯一的落点",
                option=key, needle=needle, count=count,
            )
        offset = state.index(needle) + len(f"({key}) ")
        first = state[:offset].count("<image>") + 1
        references = iter(range(first, first + value.count("<image>")))
        values[index] = re.sub(
            r"<image>", lambda _: f"[Image {next(references)} above]", value
        )
    numbered = iter(range(1, state.count("<image>") + 1))
    return re.sub(r"<image>", lambda _: f"[Image {next(numbered)}] <image>", state), tuple(values)


def _plan(
    question: Question,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[str, ...]]:
    """`option_plan` + 选项键（选项图引用要按厂商的 `(key)` 形式去 state 里找）。"""
    texts, labels, indexes = option_plan(question)
    if question.kind == "score":
        keys = tuple(str(index) for index in range(len(texts)))
    elif question.kind == "noul":
        keys = _NOUL_KEYS
    else:
        keys = tuple(question_spec(question)["criteria"])
    return texts, labels, indexes, keys


# 这份检查点的张量前缀、norm 存法与置信度口径。形状与读出方案来自 `parse_config`。
RECIPE = SplitRecipe(labels=read_codes, prompter=SingVLPrompter, confidence=mude.confidence)

# 参数量按发布包的两个分片实测：4 539 265 536（与 d3 / pplx 同一个 4B 底座 + 视觉塔）。
register_family("singvl", parse_config, build_layer_plan, strategy="split",
                weight_plan=weight_plan, recipe=RECIPE, params="4.54B", source=__name__)
