"""两份发布权重从哪来：仓库、固定修订与引擎读的文件清单。

这份清单原先只写在 README 的 `hf download` 命令里，现在它是唯一事实来源：首启自动下载
按它取文件，`DownloadError` 按它拼出手动命令。修订必须钉死 —— follow branch 会换权重，
而产物缓存只认权重文件的指纹，换了等于悄悄重编一遍。license 文件也在清单里，随包保留。
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
