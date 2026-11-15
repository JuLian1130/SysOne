"""H2O-Lightning-4B 桥：模板、读出与双向读法全部来自厂商自带代码。

厂商包给的三个入口在这里被原样调用：
- `h2o_lightning_shim.Contract(cfg)` 从 `serve_config.json` 读服务设置；
- `Contract.question_prompt(state_text, q)` 渲染题面（包含选项标签与回答槽）；
- `h2o_lightning_shim.format_answer(...)` 把答案槽上的 logprobs 转成线格式。

我们只负责：用 int8 码表把权重装进 transformers 的 Qwen3.5 结构，以及按请求的
注意力模式决定要不要让全注意力层读双向。
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ...domain.attention import MaskMode
from ...domain.work import CompilationTimes
from ...errors import ConfigError, NotReadyError, ReadoutError, UnsupportedModeError
from ...weights.packing.row_int8 import RowInt8Store
from .base import VendorAnswer, wire_answer
from .torch_support import build_model, fill_model, output_weight, text_backbone, torch_dtype

_SHIM_FILE = "h2o_lightning_shim.py"
_SERVE_FILE = "serve_config.json"
# 预热题：只用两个选项，尽量便宜，目的只是把冷启动那次前向烧掉
_WARMUP_STATE = "warmup"
_WARMUP_QUESTION = {
    "type": "choice",
    "instructions": "Which option is correct?",
    "criteria": {"A": "A", "B": "B"},
}
# 三种长度：进程内头几次 prefill 才是坏的，长短各来一遍才能都烧掉
_WARMUP_STATES = (_WARMUP_STATE, _WARMUP_STATE * 40, _WARMUP_STATE * 120)


def _load_shim(model_dir: Path):
    path = model_dir / _SHIM_FILE
    if not path.exists():
        raise ConfigError("厂商目录缺少 shim", path=str(path))
    spec = importlib.util.spec_from_file_location("h2o_lightning_shim", path)
    if spec is None or spec.loader is None:
        raise ConfigError("无法加载 shim", path=str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_precision(device: torch.device) -> str:
    """默认计算精度：加速器 bf16（W8A16），CPU fp32。

    两半都是实测的。CPU 上 bf16 会在首个全注意力层读出 NaN、同一输入的概率在 0.75 与 0.97
    之间跳（`f85dcf3` 的记录），所以 CPU 保持 fp32。MPS 上 bf16 完全正常且明显更快更省：
    13 767 token 的整条 prompt 上，峰值驱动内存 10.28 GiB vs fp32 的 14.44 GiB（−29 %），
    墙钟 135 s vs 665 s（同一轮对照，票据 13）。短题上两者打平（每 1k token 7.16 vs 7.19 ms），
    所以这条默认在短序列上不亏、在长序列上净赚。
    """
    return "fp32" if device.type == "cpu" else "bf16"


class H2OBridge:
    """H2O-Lightning-4B 的 vendor 桥。

    默认 fp32：实测这个 hybrid 检查点在 CPU 上用 bf16 算会出 NaN 与大幅漂移
    （首个全注意力层读出 NaN，同一输入的概率在 0.75 与 0.97 之间跳），fp32
    首访即稳定。权重常驻的是 int8，切 fp32 只加计算量、不加权重内存。
    """

    name = "h2o"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "cpu",
        precision: str | None = None,
    ) -> None:
        self.dir = Path(model_dir)
        self.device = torch.device(device)
        # 计算精度默认"跟随设备"（`_default_precision`）：加速器上 bf16，CPU 上 fp32。
        self.dtype = torch_dtype(precision or _default_precision(self.device))
        self._shim = None
        self._contract = None
        self._tokenizer = None
        self._model = None
        self._backbone = None
        self._lm = None
        self._store = None
        self._bidir = False
        self._load_s = 0.0

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        started = time.perf_counter()
        serve = self.dir / _SERVE_FILE
        if not serve.exists():
            raise ConfigError("厂商目录缺少 serve_config.json", path=str(serve))
        cfg = json.loads(serve.read_text(encoding="utf-8"))
        self._shim = _load_shim(self.dir)
        try:
            self._contract = self._shim.Contract(cfg)
        except (ValueError, KeyError) as exc:
            raise ConfigError("serve_config.json 与 shim 不匹配", reason=str(exc)) from exc
        import transformers

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.dir))
        self._store = RowInt8Store(self.dir)
        model, config = build_model(self.dir, dtype=self.dtype)
        self._model = fill_model(
            model,
            config,
            self._store,
            device=self.device,
            dtype=self.dtype,
        )
        self._backbone = text_backbone(self._model)
        self._lm = output_weight(self._model)
        self._load_s = time.perf_counter() - started

    # ---------------------------------------------------------------- warmup
    def compile_or_warm_up(self) -> CompilationTimes:
        """烧掉冷启动前向。

        实测该检查点第一次 prefill 会在第一个全注意力层读出 NaN（第二次起正常），
        所以装载完必须先跑一次并把结果校验掉：非法概率在这里以 `NotReadyError`
        暴露，而不是变成第一道题的答案。
        """
        started = time.perf_counter()
        for state in _WARMUP_STATES:
            self._warm_once(state)
        return CompilationTimes(
            load_s=self._load_s,
            compile_s=0.0,
            warmup_s=time.perf_counter() - started,
        )

    def _warm_once(self, state: str) -> None:
        try:
            answer = self.decide(state, _WARMUP_QUESTION, mode=MaskMode.CAUSAL)
        except Exception as exc:  # noqa: BLE001 — 预热失败就是要按未就绪报出去
            raise NotReadyError("厂商桥预热前向失败", bridge=self.name, reason=str(exc)) from exc
        probs = tuple(float(value) for value in answer.probs)
        if not probs or not all(math.isfinite(value) for value in probs):
            raise NotReadyError("厂商桥预热前向产生了非法概率", bridge=self.name, probs=list(probs))
        if abs(sum(probs) - 1.0) > 1e-3:
            raise NotReadyError("厂商桥预热前向概率未归一", bridge=self.name, total=sum(probs))

    # ---------------------------------------------------------------- decide
    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        if mode is MaskMode.PREFIX_NONCAUSAL:
            raise NotImplementedError("h2o 的两种读法是 causal 与 noncausal_full，没有 prefix 语义")
        started = time.perf_counter()
        self._apply_mode(mode)

        state_text = self._shim.render_state(state, self._contract.state_compact)
        prompt, names, descs, labels = self._contract.question_prompt(state_text, question)
        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ReadoutError("渲染出的 prompt 是空的")
        label_ids = self._label_ids(prompt, ids, labels)

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        hidden = out.last_hidden_state[:, -1].float()[0]
        rows = self._lm[label_ids].to(torch.float32)
        logprobs = (hidden @ rows.T).detach().cpu().tolist()

        kind = question.get("type", "choice")
        temperature = self._contract.temperature_by_type.get(kind, self._contract.temperature)
        result = self._shim.format_answer(
            kind,
            names,
            descs,
            logprobs,
            temperature,
            self._contract.noul_floor,
            self._shim.noul_threshold(question),
        )
        answer = wire_answer(result)
        return VendorAnswer(
            value=answer[0],
            labels=answer[1],
            probs=answer[2],
            input_tokens=len(ids),
            readout_s=time.perf_counter() - started,
        )

    def artifact_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self._model = None
        self._backbone = None
        self._lm = None
        if self._store is not None:
            self._store.close()
            self._store = None

    # ---------------------------------------------------------------- 内部
    def _apply_mode(self, mode: MaskMode) -> None:
        """双向读法靠 jeff 的 bidir 把全注意力层改成无因果掩码；causal 要能**切回去**。

        这个开关是进程级状态（`is_causal` 写在层上），而 `attention_mode` 是逐请求的：只开不关
        的话，一条 `noncausal_full` 之后所有 `causal` 请求实际都跑双向，而且 `provenance` 还
        报 `causal` —— 静默错。（票据 12 的探针里撞到过：先跑完 105 题双向再跑 causal，
        同一批题里 3/24 的判词对不上自己的"原装问法"。）

        jeff 只提供 `enable()`，没有 `disable()`；把它自己的 `causal()` 上下文管理器做的事
        原样做一遍：把 `is_causal` 置回真、清掉模式标记。
        """
        if mode is MaskMode.CAUSAL:
            self._leave_bidirectional()
            return
        if mode is not MaskMode.NONCAUSAL_FULL:
            raise UnsupportedModeError("h2o 桥只支持 causal / noncausal_full", mode=mode.value)
        if self._bidir:
            return
        try:
            from jeff import bidir  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 取决于是否装了 jeff
            raise UnsupportedModeError(
                "双向读法需要 jeff 包（bidir.enable）",
                mode=mode.value,
                hint="把 jev-coll 的 src_jeff 加到 sys.path",
            ) from exc
        bidir.enable(self._backbone, "full")
        self._bidir = True

    def _leave_bidirectional(self) -> None:
        """把全注意力层改回因果（jeff 的 `causal()` 上下文管理器在被调用时做的事）。"""
        if not self._bidir:
            return
        for layer in getattr(self._backbone, "layers", ()):
            attention = getattr(layer, "self_attn", None)
            if attention is not None and getattr(attention, "is_causal", True) is False:
                attention.is_causal = True
        self._backbone._jev_bidirectional = None
        self._bidir = False

    def _label_ids(self, prompt: str, ids: list[int], labels: list[str]) -> torch.Tensor:
        """每个选项标签在回答槽上的单 token id（必须是恰好一个新 token）。"""
        base = len(ids)
        picked: list[int] = []
        for label in labels:
            probe = self._tokenizer(prompt + " " + label, add_special_tokens=False)["input_ids"]
            if len(probe) != base + 1:
                raise ReadoutError(
                    "选项标签不是单 token，无法在回答槽上读出",
                    label=label,
                    tokens=len(probe) - base,
                )
            picked.append(int(probe[-1]))
        return torch.tensor(picked, dtype=torch.long, device=self.device)


__all__ = ["H2OBridge"]
