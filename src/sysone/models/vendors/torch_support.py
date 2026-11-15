"""torch 侧装载：先在 meta 上建出厂商模型，再按 row-int8 的 codes/scale 就地填满。

这样 4B 的 bf16 常驻（约 9.3 GB）从不出现在内存里：权重以 int8 码表 + 每行
fp32 scale 的形式常驻，反量化发生在每次前向的线性层里。

两条都对着实测定的（`.scratch/mem_budget_h2o.py`）：

- **`skip_prefixes` 是"整块不装"，不是"缺了就算"**。视觉塔在纯文本 prefill 里一次都不碰，
  但它在权重文件里存在，于是"只在缺失时跳过"会把 0.62 GiB 白填进设备；
- **上传不复制**。`read_raw` 给的是只读的 mmap 视图，旧实现为了可写先 `copy()` 一份，
  32 层 / 3.3 GiB 的一趟实测多花 45 % 的时间（`.scratch/stream_cost_h2o.py`）。

只被 vendor 桥 import，核心面（`tests/unit/test_import_light.py`）不碰 torch。
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...errors import ConfigError, MissingTensorError
from ...weights.packing.row_int8 import RowInt8Store

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查用
    from pathlib import Path

_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

# 前向里按位置自己算的派生张量，检查点里通常不存
_DERIVED_SUFFIXES = ("rotary_emb.inv_freq", "rotary_emb.cos_cached", "rotary_emb.sin_cached")


def torch_dtype(name: str) -> torch.dtype:
    """把配置里的 bf16/fp16/fp32 转成 torch dtype。"""
    try:
        return _DTYPES[name]
    except KeyError as exc:
        raise ConfigError("未知的权重精度", dtype=name, allowed=sorted(_DTYPES)) from exc


class Int8Linear(torch.nn.Module):
    """int8 常驻、前向时逐行反量化的线性层。"""

    def __init__(
        self,
        codes: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("codes", codes.detach().to(torch.int8), persistent=False)
        self.register_buffer("scale", scale.detach().to(torch.float32), persistent=False)
        self.in_features = int(codes.shape[1])
        self.out_features = int(codes.shape[0])
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.codes.to(x.dtype) * self.scale.to(x.dtype)[:, None]
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return torch.nn.functional.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, int8"


def build_model(model_dir: Path, *, dtype: torch.dtype) -> tuple[torch.nn.Module, object]:
    """在 meta 设备上构造厂商架构；返回 (model, config)。"""
    import transformers

    config = transformers.AutoConfig.from_pretrained(str(model_dir))
    architectures = getattr(config, "architectures", None) or []
    if not architectures:
        raise ConfigError("config.json 缺少 architectures", model_dir=str(model_dir))
    name = architectures[0]
    cls = getattr(transformers, name, None)
    if cls is None:
        raise ConfigError("当前 transformers 版本没有这个架构类", architecture=name)
    with torch.device("meta"):
        model = cls(config)
    return model, config


def _assign(module: torch.nn.Module, name: str, tensor: torch.Tensor, *, param: bool) -> None:
    if param:
        module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
    else:
        module._buffers[name] = tensor


_NP_TO_TORCH = {
    np.dtype(np.int8): torch.int8,
    np.dtype(np.uint8): torch.uint8,
    np.dtype(np.int16): torch.int16,
    np.dtype(np.float16): torch.float16,
    np.dtype(np.float32): torch.float32,
    np.dtype(np.float64): torch.float64,
}


def _view(tensor: np.ndarray) -> torch.Tensor:
    """零拷贝把 numpy 缓冲交给 torch —— 只读的 mmap 视图也能直接看。

    `torch.frombuffer` 收不可写缓冲时会警告"你能写进这块只读内存"。我们不写（只做 dtype
    转换与上传），所以就地屏蔽那条警告；换来的是一整趟装载少一次 host 拷贝。
    """
    array = np.ascontiguousarray(tensor)
    target = _NP_TO_TORCH.get(array.dtype)
    if target is None:
        raise ConfigError("torch 侧不认识的 numpy dtype", dtype=str(array.dtype))
    flat = array.reshape(-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        view = torch.frombuffer(flat.data, dtype=target, count=int(flat.size))
    return view.view(tuple(array.shape))


def _to(tensor: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _view(tensor).to(device=device, dtype=dtype)


def fill_model(
    model: torch.nn.Module,
    config: object,
    store: RowInt8Store,
    *,
    device: torch.device,
    dtype: torch.dtype,
    skip_prefixes: tuple[str, ...] = ("model.visual", "multi_modal_projector", "mtp."),
) -> torch.nn.Module:
    """用 int8 码表与 keep 张量就地填满 meta 模型。

    线性层的权重换成 `Int8Linear`；其余参数与 buffer 逐个赋值。落在 `skip_prefixes`
    下的张量**整块不装**（视觉塔、MTP 草稿模块，纯文本 prefill 一次都不碰）——它们留在
    meta 上，权重文件里有也不填；其余张量缺失则按缺失张量报错。
    """
    modules = dict(model.named_modules())
    handled: set[str] = set()
    stored = set(store.names())

    for entry_name in store.manifest.quantized():
        if not entry_name.endswith(".weight"):
            continue
        if entry_name.startswith(skip_prefixes):
            continue
        module_path = entry_name[: -len(".weight")]
        parent_path, _, leaf = module_path.rpartition(".")
        parent = modules.get(parent_path)
        module = modules.get(module_path)
        if parent is None or module is None:
            raise ConfigError("权重名对不上模型模块", tensor=entry_name)
        codes, scale = store.read_parts(entry_name)
        bias = None
        bias_name = f"{module_path}.bias"
        if bias_name in stored:
            bias = _to(store.read_raw(bias_name), device, dtype)
        parent._modules[leaf] = Int8Linear(
            _to(codes, device, torch.int8),
            _to(scale, device, torch.float32),
            bias,
        )
        handled.add(entry_name)
        if bias is not None:
            handled.add(bias_name)

    names = stored
    tied = bool(getattr(config, "tie_word_embeddings", True))
    for module_path, module in modules.items():
        for name, param in list(module._parameters.items()):
            full = f"{module_path}.{name}" if module_path else name
            if param is None or full in handled:
                continue
            if full.startswith(skip_prefixes):
                continue
            if tied and full.startswith("lm_head."):
                continue  # 稍后与 embedding 共用同一个 Parameter
            if full not in names:
                raise MissingTensorError("权重文件里缺这张量", tensor=full)
            _assign(module, name, _to(store.read_raw(full), device, dtype), param=True)
            handled.add(full)
        for name, buf in list(module._buffers.items()):
            full = f"{module_path}.{name}" if module_path else name
            if buf is None or full in handled:
                continue
            if full.startswith(skip_prefixes):
                continue
            if name in getattr(module, "_non_persistent_buffers_set", frozenset()):
                continue  # 派生 buffer（rotary 的 cos/sin 等），前向里会自己算
            if full.endswith(_DERIVED_SUFFIXES):
                continue
            if full not in names:
                raise MissingTensorError("权重文件里缺这张量", tensor=full)
            _assign(module, name, _to(store.read_raw(full), device, buf.dtype), param=False)
            handled.add(full)

    _materialize_derived(model, device=device, skip_prefixes=skip_prefixes)
    _tie_embeddings(model, config, dtype)
    return model


def _materialize_derived(
    model: torch.nn.Module, *, device: torch.device, skip_prefixes: tuple[str, ...] = ()
) -> None:
    """把派生 buffer（rotary 的 `inv_freq` / `original_inv_freq`）从 meta 变成真实值。

    它们是 `persistent=False`，权重文件里没有，装载时只能由模块自己算出来 —— meta 建出来的
    只是个空壳。留着它在 meta 上不是"省一次计算"，是**必然的崩溃点**：
    `Qwen3_5TextRotaryEmbedding` 只在序列超过 `max_seq_len_cached`（= `max_position_embeddings`）
    时才重算（`dynamic_frequency_update` 的增长分支），静态 rope 永远走不到那条分支，于是
    前向里的 `inv_freq` 还是 meta 张量。实测的症状不指向这里：torchcast 的厂商引擎在
    `autocast(bf16)` 里跑前向，meta 张量没被 autocast 转换、真实张量被转了，报出来的是
    "expected scalar type torch.float32 but found torch.bfloat16"（`meta_bmm` 的 dtype 校验）。

    做法是最小的一步：用模块自己的构造器在真实设备上再建一个同类实例，把它算出来的同名
    buffer 拷过来。这样连 rope 的具体公式都不必知道，跟着 transformers 的版本走。
    """
    for module_path, module in list(model.named_modules()):
        if module_path.startswith(skip_prefixes):
            continue      # 整块不装的模块（视觉塔）没必要实化，它根本不会被调用
        pending = [name for name, buffer in module._buffers.items()
                   if buffer is not None and buffer.device.type == "meta"]
        if not pending:
            continue
        config = getattr(module, "config", None)
        if config is None:
            raise MissingTensorError("模块留有 meta buffer 且无 config 可重建",
                                     module=module_path, buffers=pending)
        fresh = type(module)(config)
        for name in pending:
            built = fresh._buffers.get(name)
            if built is None:
                raise MissingTensorError("重建的模块没有同名 buffer", module=module_path,
                                         buffer=name)
            module._buffers[name] = built.detach().to(device)


def _tie_embeddings(model: torch.nn.Module, config: object, dtype: torch.dtype) -> None:
    """tied 模型里 lm_head 与 embed_tokens 必须共用同一个 Parameter 对象。"""
    if not getattr(config, "tie_word_embeddings", True):
        return
    embedding = getattr(model, "language_model", model)
    embed = getattr(embedding, "embed_tokens", None)
    if embed is None:
        embed = model.get_input_embeddings()
    head = getattr(model, "lm_head", None)
    if head is None:
        head = model.get_output_embeddings()
    if embed is None or head is None or embed is head:
        return
    weight = embed._parameters.get("weight")
    if weight is None:
        return
    head._parameters["weight"] = weight


def text_backbone(model: torch.nn.Module) -> torch.nn.Module:
    """找到吃 input_ids 的文本主干（旧实现靠扫 `named_modules()` 找）。"""
    for _, module in model.named_modules():
        layers = getattr(module, "layers", None)
        layer_types = getattr(getattr(module, "config", None), "layer_types", None)
        if isinstance(layers, torch.nn.ModuleList) and layer_types:
            return module
    return model
