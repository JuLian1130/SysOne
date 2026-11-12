"""row-int8 的纯数学与检查点读取（不需要 torch）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sysone.errors import BudgetExceededError, ConfigError, MissingTensorError
from sysone.weights.packing import (
    QUANT_SUFFIX,
    SCALE_SUFFIX,
    RowInt8Manifest,
    RowInt8Store,
    dequantize,
    pack_row,
    pack_tensors,
)
from sysone.weights.store import TensorStore, write_file


def test_pack_row_roundtrip_is_close() -> None:
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(8, 16)).astype(np.float32)
    codes, scale = pack_row(weight, min_elems=1)
    assert codes.dtype == np.int8 and codes.shape == weight.shape
    assert scale.shape == (8,)
    restored = dequantize(codes, scale)
    # 每行误差上界是半个步长：row_max / 254
    row_max = np.abs(weight).max(axis=1)
    error = np.abs(restored - weight).max()
    assert error <= (row_max / 254.0).max() + 1e-6


def test_pack_row_keeps_small_tensors() -> None:
    weight = np.ones((4, 4), dtype=np.float32)
    assert pack_row(weight, min_elems=1 << 20) is None


def test_pack_row_zero_rows_stay_zero() -> None:
    weight = np.zeros((3, 5), dtype=np.float32)
    weight[1, 2] = 1.0
    codes, scale = pack_row(weight, min_elems=1)
    assert np.all(dequantize(codes, scale)[0] == 0.0)
    assert np.isclose(dequantize(codes, scale)[1, 2], 1.0, atol=1e-2)


def test_pack_tensors_manifest_counts() -> None:
    tensors = {
        "big": np.full((32, 32), 0.5, dtype=np.float32),
        "small": np.arange(4, dtype=np.float32),
    }
    packed, manifest = pack_tensors(tensors, min_elems=1)
    assert "big" + QUANT_SUFFIX in packed and "big" + SCALE_SUFFIX in packed
    assert "small" in packed and "small" + QUANT_SUFFIX not in packed
    assert manifest.quantized() == frozenset({"big"})
    assert manifest.source_bf16_bytes == tensors["big"].nbytes + tensors["small"].nbytes
    assert manifest.int8_bytes == packed["big" + QUANT_SUFFIX].nbytes + packed["big" + SCALE_SUFFIX].nbytes + packed["small"].nbytes


def test_manifest_roundtrip_json(tmp_path: Path) -> None:
    tensors = {"w": np.full((16, 16), 0.25, dtype=np.float32),
               "n": np.ones(16, dtype=np.float32)}
    packed, manifest = pack_tensors(tensors, min_elems=1)
    write_file(tmp_path / "model.safetensors", packed)
    payload = {
        "tensors": {n: {"kind": e.kind, "shape": list(e.shape)} for n, e in manifest.entries.items()},
        "source_bf16_bytes": manifest.source_bf16_bytes,
        "int8_bytes": manifest.int8_bytes,
    }
    (tmp_path / "int8_manifest.json").write_text(json.dumps(payload), "utf-8")
    loaded = RowInt8Manifest.load(tmp_path)
    assert loaded.names() == manifest.names()
    assert loaded.quantized() == frozenset({"w"})
    assert loaded.shape("w") == (16, 16)


def test_row_int8_store_dequantizes(tmp_path: Path) -> None:
    tensors = {"w": np.full((8, 8), 0.5, dtype=np.float32),
               "n": np.arange(8, dtype=np.float32)}
    packed, manifest = pack_tensors(tensors, min_elems=1)
    write_file(tmp_path / "model.safetensors", packed)
    payload = {
        "tensors": {n: {"kind": e.kind, "shape": list(e.shape)} for n, e in manifest.entries.items()},
        "source_bf16_bytes": manifest.source_bf16_bytes,
        "int8_bytes": manifest.int8_bytes,
    }
    (tmp_path / "int8_manifest.json").write_text(json.dumps(payload), "utf-8")
    store = RowInt8Store(tmp_path)
    assert store.read("w").shape == (8, 8)
    assert np.isclose(store.read("w"), 0.5, atol=1e-2).all()
    assert np.allclose(store.read("n"), tensors["n"])
    assert store.missing({"absent"}) == {"absent"}
    assert store.artifact_bytes == manifest.int8_bytes
    store.close()


def test_row_int8_store_requires_manifest(tmp_path: Path) -> None:
    write_file(tmp_path / "model.safetensors", {"w": np.zeros((2, 2), dtype=np.float32)})
    with pytest.raises(MissingTensorError):
        RowInt8Store(tmp_path)


def test_manifest_rejects_unknown_kind() -> None:
    with pytest.raises(ConfigError):
        RowInt8Manifest.from_dict({"tensors": {"w": {"kind": "fp8", "shape": [2, 2]}}})


def test_lazy_store_reads_slices_without_eager_load(tmp_path: Path) -> None:
    write_file(tmp_path / "model.safetensors", {"w": np.arange(12, dtype=np.float32).reshape(3, 4)})
    eager = TensorStore([tmp_path / "model.safetensors"])
    lazy = TensorStore([tmp_path / "model.safetensors"], lazy=True)
    assert np.array_equal(lazy.read("w"), eager.read("w"))
    assert lazy.names() == eager.names()
    lazy.close()


def test_memory_plan_budget_rejects() -> None:
    from sysone.config import EngineConfig, ModelSelection, SchedulerConfig, StorageConfig
    from sysone.engine.memory import plan_memory
    from sysone.models.descriptor import ModelDescriptor, ReadoutPlan

    descriptor = ModelDescriptor.from_mapping(
        {
            "family": "reference",
            "model_id": "tiny",
            "hidden": 16,
            "num_layers": 2,
            "layer_types": ["full", "full"],
            "heads": 2,
            "kv_heads": 1,
            "head_dim": 8,
            "ffn": 32,
            "vocab": 16,
            "max_position": 128,
            "readout": {"kind": "code_rows", "weight": "readout", "rows": 4, "labels": ["A", "B"]},
            "supported_modes": ["causal", "noncausal_full", "prefix_noncausal"],
            "trained_modes": ["prefix_noncausal"],
            "default_mode": "prefix_noncausal",
        }
    )
    config = EngineConfig(
        model=ModelSelection(name="tiny", dir=Path("/tmp/tiny")),
        scheduler=SchedulerConfig(max_len=64, ring=2),
        storage=StorageConfig(artifacts_budget=1),
    )
    with pytest.raises(BudgetExceededError) as excinfo:
        plan_memory(descriptor, config, artifacts_bytes=0)
    assert "breakdown" in str(excinfo.value) or excinfo.value.context

    ok = plan_memory(descriptor, config.__class__(
        model=config.model, scheduler=config.scheduler, storage=StorageConfig(artifacts_budget=0)
    ))
    assert ok.total > 0 and ok.fits()


def test_readout_plan_requires_rows_for_code_rows() -> None:
    from sysone.models.descriptor import ReadoutPlan

    with pytest.raises(ConfigError):
        ReadoutPlan(kind="code_rows", weight="w").validate()
