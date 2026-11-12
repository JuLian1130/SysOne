"""Torchcast Decision 4B 桥：完全交给厂商引擎渲染与读出。

厂商引擎只跑 `noncausal_full_attention`（它的 `DECISION_CONFIG` 与内核都只做
这一种），因此这里不做模式映射，选别的模式直接按不支持处理。

厂商装载走 `model_class.from_pretrained(path, dtype=..., device_map=...)`；我们
把该方法临时替换成返回自己的 int8 常驻模型，这样不用把 9.3 GB 的 bf16 实化。
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import torch

from ...domain.attention import MaskMode
from ...errors import ConfigError, ReadoutError, UnsupportedModeError
from ...weights.packing.row_int8 import RowInt8Store
from .base import VendorAnswer, wire_answer
from .torch_support import build_model, fill_model, torch_dtype


class TorchcastBridge:
    """Torchcast Decision 4B 的 vendor 桥。"""

    name = "torchcast"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        device: str = "cpu",
        precision: str = "bf16",
        max_length: int = 16384,
    ) -> None:
        self.dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = torch_dtype(precision)
        self.max_length = int(max_length)
        self._model = None
        self._engine = None
        self._store = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        package_dir = self.dir / "torchcast_decision"
        if not package_dir.exists():
            raise ConfigError("厂商目录缺少 torchcast_decision 包", path=str(package_dir))
        sys.path.insert(0, str(self.dir))
        try:
            module = importlib.import_module("torchcast_decision.model")
        except ImportError as exc:
            raise ConfigError("无法导入 torchcast_decision", reason=str(exc)) from exc

        self._store = RowInt8Store(self.dir)
        model, config = build_model(self.dir, dtype=self.dtype)
        self._model = fill_model(model, config, self._store, device=self.device, dtype=self.dtype)

        import transformers

        cls = getattr(transformers, config.architectures[0])
        original = cls.from_pretrained
        cls.from_pretrained = staticmethod(lambda *args, **kwargs: self._model)
        try:
            self._engine = module.TorchcastDecision(
                str(self.dir),
                device=self.device,
                max_length=self.max_length,
                graphs=False,
                images=False,
            )
        except TypeError as exc:
            raise ConfigError("TorchcastDecision 构造参数与本包不匹配", reason=str(exc)) from exc
        finally:
            cls.from_pretrained = original
            sys.path.remove(str(self.dir))

    # ---------------------------------------------------------------- decide
    def decide(self, state: str, question: dict, *, mode: MaskMode) -> VendorAnswer:
        if mode is not MaskMode.NONCAUSAL_FULL:
            raise NotImplementedError("torchcast 只跑 noncausal_full_attention")
        started = time.perf_counter()
        answers, usage = self._engine.decide(state, {"q": question})
        item = self._first(answers)
        value, labels, probs = wire_answer(item)
        tokens = int(usage.get("input_tokens", 0)) if isinstance(usage, dict) else 0
        return VendorAnswer(
            value=value,
            labels=tuple(labels),
            probs=tuple(probs),
            input_tokens=tokens,
            readout_s=time.perf_counter() - started,
        )

    def artifact_bytes(self) -> int:
        return 0

    def close(self) -> None:
        self._engine = None
        self._model = None
        if self._store is not None:
            self._store.close()
            self._store = None

    @staticmethod
    def _first(answers: object) -> dict:
        if isinstance(answers, dict):
            if not answers:
                raise ReadoutError("厂商没有返回答案")
            return next(iter(answers.values()))
        if isinstance(answers, (list, tuple)):
            if not answers:
                raise ReadoutError("厂商没有返回答案")
            return answers[0]
        raise ReadoutError("厂商返回的答案形态不认识", kind=type(answers).__name__)


__all__ = ["TorchcastBridge"]
