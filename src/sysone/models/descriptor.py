"""模型元数据：从厂商 config 读出的、与设备无关的结构描述。

`ModelDescriptor` 只回答"这模型长什么样、怎么读出答案"，不含权重也不含设备句柄。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..domain.attention import AttentionRole, MaskMode
from ..domain.quant import DType, QuantSpec, Scales, Scheme, Storage
from ..errors import ConfigError

READOUT_KINDS = ("code_rows", "letter_rows", "pooled")


@dataclass(frozen=True)
class NoulThresholds:
    """yes/no 题的两个决策阈值（mude 的 `noul` 语义）。"""

    yes: float
    no: float


@dataclass(frozen=True)
class ReadoutPlan:
    """读出方案：读哪张表、多少行、什么温度。

    - `code_rows`：读固定行数的码表（mude，255 行），再做温度 softmax；
    - `letter_rows`：读 `lm_head` 中候选字母对应的行（torchcast）；
    - `pooled`：厂商自己的 pooled readout + selector（h2o），M1 由 vendor 策略实现。
    """

    kind: str
    weight: str | None = None
    file: str | None = None
    rows: int | None = None
    labels: tuple[str, ...] | None = None
    temperature: float | None = None
    noul: NoulThresholds | None = None
    dtype: DType = DType.FP32

    def validate(self) -> None:
        if self.kind not in READOUT_KINDS:
            raise ConfigError("读出方案类型未知", kind=self.kind, allowed=READOUT_KINDS)
        if self.kind == "code_rows":
            if not self.weight or not self.file or not self.rows:
                raise ConfigError("code_rows 需要 weight / file / rows", kind=self.kind)
        if self.kind == "letter_rows" and not self.weight:
            raise ConfigError("letter_rows 需要 weight", kind=self.kind)
        if self.temperature is not None and self.temperature <= 0:
            raise ConfigError("temperature 必须为正", temperature=self.temperature)
        if self.rows is not None and self.rows <= 0:
            raise ConfigError("rows 必须为正", rows=self.rows)


@dataclass(frozen=True)
class VisionSpec:
    """视觉塔的几何（M0 只记录，不加载）。"""

    model_type: str
    depth: int = 0
    hidden: int = 0
    patch_size: int = 0
    spatial_merge: int = 2


@dataclass(frozen=True)
class ModelDescriptor:
    """一个模型的完整结构描述。

    `supported_modes` 是"能不能跑"，`trained_modes` 是"训练时见过没有"；二者分开，
    后者只影响 provenance 里的 `attention_trained`，不会静默改结果。
    """

    family: str
    model_id: str
    hidden: int
    num_layers: int
    layer_types: tuple[AttentionRole, ...]
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    max_position: int
    readout: ReadoutPlan
    supported_modes: frozenset[MaskMode]
    trained_modes: frozenset[MaskMode]
    default_mode: MaskMode
    # 线性注意力层（GDN）的几何。**v 与 k 分开记**：4B 的 `linear_value_heads` 是
    # `linear_key_heads` 的两倍（32 对 16），于是 `value_dim = 4096 ≠ hidden 2560`。在这些
    # 数字上做"v 就是 k 的副本"的假设会把缓冲大小算错。没有线性层的模型（reference tiny）
    # 留 `None`；有线性层却留 `None` 在 `validate()` 里报错，不猜。
    linear_key_heads: int | None = None
    linear_key_dim: int | None = None
    linear_value_heads: int | None = None
    linear_value_dim: int | None = None
    conv_kernel: int = 4
    # 旋转只覆盖每个头的这一部分通道（Qwen3.5 是 0.25），设备层据此算 `rotary_dim`。
    partial_rotary_factor: float = 1.0
    quant: QuantSpec = field(default_factory=QuantSpec)
    tied_embeddings: bool = True
    mtp_layers: int = 0
    rms_eps: float = 1e-6
    vision: VisionSpec | None = None

    def validate(self) -> None:
        """结构自检；不做"参数猜谜"，只查能让引擎崩掉的不一致。

        注意：**不检查 `heads * head_dim == hidden`** —— Qwen3.5 系列的 `head_dim`
        与 `hidden` 相互独立（4B：16×256=4096 ≠ 2560），旧文档里的这条断言是错的。
        """
        if not self.family or not self.model_id:
            raise ConfigError("family / model_id 不能为空")
        if self.num_layers < 1:
            raise ConfigError("num_layers 必须 >= 1", num_layers=self.num_layers)
        if len(self.layer_types) != self.num_layers:
            raise ConfigError(
                "layer_types 长度与 num_layers 不一致",
                layers=len(self.layer_types),
                num_layers=self.num_layers,
            )
        if AttentionRole.FULL not in self.layer_types:
            raise ConfigError("至少需要一个全注意力层", family=self.family)
        if self.hidden < 1 or self.ffn < 1 or self.head_dim < 1 or self.vocab < 1:
            raise ConfigError("hidden/ffn/head_dim/vocab 必须为正")
        if self.kv_heads < 1 or self.kv_heads > self.heads:
            raise ConfigError("kv_heads 必须在 [1, heads]", kv_heads=self.kv_heads, heads=self.heads)
        if AttentionRole.LINEAR in self.layer_types:
            missing = [
                name
                for name in (
                    "linear_key_heads",
                    "linear_key_dim",
                    "linear_value_heads",
                    "linear_value_dim",
                )
                if getattr(self, name) is None
            ]
            if missing:
                raise ConfigError(
                    "模型有线性注意力层，descriptor 却没带线性层几何",
                    missing=missing,
                    hint="家族 parser 要从 config 的 linear_num_key_heads / linear_key_head_dim / "
                         "linear_num_value_heads / linear_value_head_dim 填这四项",
                )
        if not self.supported_modes:
            raise ConfigError("supported_modes 不能为空")
        if self.default_mode not in self.supported_modes:
            raise ConfigError(
                "default_mode 必须在 supported_modes 内",
                default_mode=self.default_mode.value,
                supported=sorted(mode.value for mode in self.supported_modes),
            )
        self.readout.validate()
        self.quant.validate()

    def summary(self) -> dict[str, object]:
        """给 `sysone inspect` 用的 JSON 友好快照。"""
        return {
            "family": self.family,
            "model_id": self.model_id,
            "hidden": self.hidden,
            "num_layers": self.num_layers,
            "layer_types": [role.value for role in self.layer_types],
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "linear_key_heads": self.linear_key_heads,
            "linear_key_dim": self.linear_key_dim,
            "linear_value_heads": self.linear_value_heads,
            "linear_value_dim": self.linear_value_dim,
            "conv_kernel": self.conv_kernel,
            "partial_rotary_factor": self.partial_rotary_factor,
            "ffn": self.ffn,
            "vocab": self.vocab,
            "max_position": self.max_position,
            "tied_embeddings": self.tied_embeddings,
            "mtp_layers": self.mtp_layers,
            "rms_eps": self.rms_eps,
            "supported_modes": sorted(mode.value for mode in self.supported_modes),
            "trained_modes": sorted(mode.value for mode in self.trained_modes),
            "default_mode": self.default_mode.value,
            "quant": str(self.quant),
            "readout": {
                "kind": self.readout.kind,
                "weight": self.readout.weight,
                "file": self.readout.file,
                "rows": self.readout.rows,
                "labels": list(self.readout.labels) if self.readout.labels else None,
                "temperature": self.readout.temperature,
                "dtype": self.readout.dtype.value,
            },
            "vision": (
                None
                if self.vision is None
                else {
                    "model_type": self.vision.model_type,
                    "depth": self.vision.depth,
                    "hidden": self.vision.hidden,
                    "patch_size": self.vision.patch_size,
                    "spatial_merge": self.vision.spatial_merge,
                }
            ),
        }

    def to_mapping(self) -> dict[str, object]:
        """可回读的 JSON 形态（`summary()` 的超集，供 `run --descriptor` 往返）。"""
        data = self.summary()
        data["supported_modes"] = [mode.value for mode in self.supported_modes]
        data["trained_modes"] = [mode.value for mode in self.trained_modes]
        data["quant"] = {
            "scheme": self.quant.scheme.value,
            "scales": self.quant.scales.value,
            "storage": self.quant.storage.value,
        }
        data["layer_types"] = [role.value for role in self.layer_types]
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ModelDescriptor":
        """从 `to_mapping()` 的结果恢复，供 `run --descriptor` 使用。"""
        try:
            readout_raw = data["readout"]  # type: ignore[index]
            quant_raw = data.get("quant") or {}
            vision_raw = data.get("vision")
            noul_raw = (readout_raw or {}).get("noul")  # type: ignore[union-attr]
        except (KeyError, TypeError) as exc:
            raise ConfigError("descriptor 结构不合法", field=str(exc)) from None

        readout = ReadoutPlan(
            kind=str(readout_raw["kind"]),  # type: ignore[index]
            weight=_opt_str(readout_raw.get("weight")),  # type: ignore[union-attr]
            file=_opt_str(readout_raw.get("file")),  # type: ignore[union-attr]
            rows=_opt_int(readout_raw.get("rows")),  # type: ignore[union-attr]
            labels=_opt_labels(readout_raw.get("labels")),  # type: ignore[union-attr]
            temperature=_opt_float(readout_raw.get("temperature")),  # type: ignore[union-attr]
            noul=NoulThresholds(float(noul_raw["yes"]), float(noul_raw["no"])) if noul_raw else None,
            dtype=DType(str(readout_raw.get("dtype", DType.FP32.value))),  # type: ignore[union-attr]
        )
        quant = QuantSpec(
            scheme=Scheme(str(quant_raw.get("scheme", Scheme.NONE.value))),
            scales=Scales(str(quant_raw.get("scales", Scales.NONE.value))),
            storage=Storage(str(quant_raw.get("storage", Storage.TENSOR.value))),
        )
        vision = None
        if vision_raw:
            vision = VisionSpec(
                model_type=str(vision_raw.get("model_type", "")),
                depth=int(vision_raw.get("depth", 0)),
                hidden=int(vision_raw.get("hidden", 0)),
                patch_size=int(vision_raw.get("patch_size", 0)),
                spatial_merge=int(vision_raw.get("spatial_merge", 2)),
            )
        return cls(
            family=str(data["family"]),
            model_id=str(data["model_id"]),
            hidden=int(data["hidden"]),  # type: ignore[arg-type]
            num_layers=int(data["num_layers"]),  # type: ignore[arg-type]
            layer_types=tuple(AttentionRole(str(role)) for role in data["layer_types"]),  # type: ignore[union-attr]
            heads=int(data["heads"]),  # type: ignore[arg-type]
            kv_heads=int(data["kv_heads"]),  # type: ignore[arg-type]
            head_dim=int(data["head_dim"]),  # type: ignore[arg-type]
            linear_key_heads=_opt_int(data.get("linear_key_heads")),
            linear_key_dim=_opt_int(data.get("linear_key_dim")),
            linear_value_heads=_opt_int(data.get("linear_value_heads")),
            linear_value_dim=_opt_int(data.get("linear_value_dim")),
            conv_kernel=int(data.get("conv_kernel", 4)),  # type: ignore[arg-type]
            partial_rotary_factor=float(data.get("partial_rotary_factor", 1.0)),  # type: ignore[arg-type]
            ffn=int(data["ffn"]),  # type: ignore[arg-type]
            vocab=int(data["vocab"]),  # type: ignore[arg-type]
            max_position=int(data["max_position"]),  # type: ignore[arg-type]
            readout=readout,
            supported_modes=frozenset(MaskMode(str(mode)) for mode in data["supported_modes"]),  # type: ignore[union-attr]
            trained_modes=frozenset(MaskMode(str(mode)) for mode in data["trained_modes"]),  # type: ignore[union-attr]
            default_mode=MaskMode(str(data["default_mode"])),
            quant=quant,
            tied_embeddings=bool(data.get("tied_embeddings", True)),
            mtp_layers=int(data.get("mtp_layers", 0)),
            rms_eps=float(data.get("rms_eps", 1e-6)),
            vision=vision,
        )


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _opt_int(value: object) -> int | None:
    return None if value is None else int(value)  # type: ignore[arg-type]


def _opt_float(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


def _opt_labels(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(str(item) for item in value)  # type: ignore[union-attr]
