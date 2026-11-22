"""四份发布权重从哪来：仓库、固定修订与引擎读的文件清单。

这份清单原先只写在 README 的 `hf download` 命令里，现在它是唯一事实来源：首启自动下载
按它取文件，`DownloadError` 按它拼出手动命令。修订必须钉死 —— follow branch 会换权重，
而产物缓存只认权重文件的指纹，换了等于悄悄重编一遍。license 文件也在清单里，随包保留。

按**引擎真正读的文件**列，不抄仓库全量：`d3` / `pplx` / `torchcast` 都把权重发成
`model-0000N-of-000M.safetensors` 加 `model.safetensors.index.json`（首启直接读分片，
不合并），仓库里的厂商参考脚本、示例图、README 与 `merges.txt`/`vocab.json`（`tokenizer.json`
自带）都不在清单里。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ConfigError


@dataclass(frozen=True)
class WeightsSource:
    """一个发布权重包的来源。

    `directory` 是权重落点的子目录名，同时决定 `model_id`（家族解析取目录名）与产物缓存
    路径，所以取规范的仓库名而不是家族名：**改它等于换一份缓存**。
    `size_hint` 只用于首启日志里的"约多少 GB"，不参与完整性判断（那由下载后记录的真实
    字节数负责）。
    """

    family: str
    repo_id: str
    revision: str
    directory: str
    files: tuple[str, ...]
    size_hint: int


SOURCES: dict[str, WeightsSource] = {
    "mude": WeightsSource(
        family="mude",
        repo_id="qybaihe/mude-1.0",
        revision="0b6a5c4ea9ca2267707a28ff568737f54ae219c5",
        directory="mude-1.0",
        files=(
            "model.safetensors",
            "readout.safetensors",
            "decision_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            # 厂商自己的题面/答面定义。推理路径不跑它们，但
            # `tests/fixtures/mude_vendor_prompts.json` 是用这两个文件生成的 —— 随权重一起
            # 下，换 checkpoint 后"重新生成 fixture"才不用另找来源（合计不到 100 KB）。
            "d3_runtime.py",
            "d3_format.py",
            "LICENSE",
            "NOTICE",
            "LICENSES/Apache-2.0.txt",
        ),
        size_hint=4_000_000_000,
    ),
    "h2o": WeightsSource(
        family="h2o",
        repo_id="h2oai/h2o-lightning-4b",
        revision="659fc412de8486dbc2477b0d1cb4121e97e72471",
        directory="h2o-lightning-4b",
        files=(
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "serve_config.json",
            "h2o_lightning_shim.py",
            "box.py",
            "assets.py",
            "card_common.py",
            "pooled_readout.py",
            "selector_v2.py",
            "label_rows.lightning",
            "LICENSE",
        ),
        size_hint=9_300_000_000,
    ),
    "d3": WeightsSource(
        family="d3",
        repo_id="vllm-sr/d3-mini",
        revision="df99bf4ca4418bc6908ee68e9b7479af1f3b8800",
        directory="d3-mini",
        files=(
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "readout.safetensors",
            "decision_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "LICENSE",
            "NOTICE",
        ),
        size_hint=9_100_000_000,
    ),
    "pplx12": WeightsSource(
        family="pplx12",
        repo_id="perplexity-ai/pplx-decider-flash-v1.2",
        revision="9ff7cf9838e59df461c276b7699d56b244d2af58",
        directory="pplx-decider-flash-v1.2",
        files=(
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "readout.safetensors",
            "decision_config.json",
            # 协议（版本 + system 题面 + 分隔符）只在它里面；缺了裁决不出是哪一版。
            "joint_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            # 这一份的模板不在 tokenizer_config.json 里（是 null），只有这个兄弟文件。
            "chat_template.jinja",
            "LICENSE",
            "NOTICE",
        ),
        size_hint=9_100_000_000,
    ),
    "pplx13": WeightsSource(
        family="pplx13",
        repo_id="perplexity-ai/pplx-decider-flash-v1.3",
        revision="d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
        directory="pplx-decider-flash-v1.3",
        files=(
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "readout.safetensors",
            "decision_config.json",
            "joint_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "LICENSE",
            "NOTICE",
        ),
        size_hint=9_100_000_000,
    ),
    "torchcast": WeightsSource(
        family="torchcast",
        repo_id="torchcast-ai/torchcast-decision-4b",
        revision="492bfb19662f3c6e601cc9c4bd2911ac95609ed4",
        directory="torchcast-decision-4b",
        files=(
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "decision_config.json",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "LICENSE",
            "LICENSE-qwen-Apache-2.0",
            "NOTICE",
        ),
        size_hint=10_400_000_000,
    ),
}


def source_for(family: str) -> WeightsSource:
    """按家族取发布来源；没有发布权重的家族要求显式 `--weights`（或走 vendor 策略）。"""
    try:
        return SOURCES[family]
    except KeyError:
        raise ConfigError(
            "这个家族没有发布权重的下载来源",
            family=family,
            available=sorted(SOURCES),
            hint="用 --weights 指定本地权重；vendor 策略不读发布来源",
        ) from None
