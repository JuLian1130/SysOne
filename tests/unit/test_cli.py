"""CLI：inspect 走家族解析，run 走参考后端。"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import FIXTURES, make_request, make_tiny
from sysone.cli import _artifacts_dir, _geometry_report, main
from sysone.devices.apple import platform as apple_platform
from sysone.errors import ConfigError


def test_inspect_prints_descriptor(capsys) -> None:
    code = main(["inspect", "--model", "mude", "--weights", str(FIXTURES / "mude")])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["family"] == "mude"
    assert payload["num_layers"] == 20
    assert payload["default_mode"] == "prefix_noncausal"


def test_run_with_tiny_model(tmp_path: Path, capsys) -> None:
    descriptor, store = make_tiny(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor.to_mapping()), encoding="utf-8")
    request = make_request(mode=None, questions=1)
    input_path = tmp_path / "request.json"
    input_path.write_text(
        json.dumps(
            {
                "state": request.state,
                "questions": [{"text": request.questions[0].text, "options": list(request.questions[0].options)}],
                "attention_mode": "causal",
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--descriptor",
            str(descriptor_path),
            "--weights",
            str(tmp_path / "tiny.safetensors"),
            "--input",
            str(input_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answers"][0]["value"] in {"A", "B", "C", "D"}
    assert payload["provenance"]["attention_mode"] == "causal"
    assert payload["provenance"]["attention_trained"] is False


def test_run_rejects_unknown_mode(tmp_path: Path, capsys) -> None:
    descriptor, _ = make_tiny(tmp_path)
    descriptor_path = tmp_path / "descriptor.json"
    descriptor_path.write_text(json.dumps(descriptor.to_mapping()), encoding="utf-8")
    code = main(
        [
            "run",
            "--descriptor",
            str(descriptor_path),
            "--weights",
            str(tmp_path / "tiny.safetensors"),
            "--input",
            str(tmp_path / "request.json"),
            "--attention-mode",
            "bogus",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"


# ---- doctor：几何白名单的**查询**形态（不装载权重）--------------------------------------


def test_doctor_reports_whether_a_models_geometry_is_measured() -> None:
    """`doctor --model … --weights …` 回答"这套几何在这台机器上验过没有"，不用装权重。"""
    report = _geometry_report("mude", FIXTURES / "mude")
    assert (report["hidden"], report["ffn"], report["num_layers"]) == (2048, 6144, 20)
    assert report["measured"] is True
    assert "note" not in report


def test_doctor_marks_an_unmeasured_geometry_as_an_experimental_reading(monkeypatch) -> None:
    """表外几何不是"不能跑"，是"数值没对拍过"：报告要这么写，与装链时的告警同一判据。"""
    monkeypatch.setattr(apple_platform, "MEASURED_GEOMETRIES", frozenset())
    report = _geometry_report("mude", FIXTURES / "mude")
    assert report["measured"] is False
    assert "实验读数" in report["note"]


def test_doctor_needs_both_the_model_and_its_weights() -> None:
    with pytest.raises(ConfigError):
        _geometry_report("mude", None)
    with pytest.raises(ConfigError):
        _geometry_report(None, FIXTURES / "mude")


# ---- 产物落点：命令行 > 环境变量 > 平台缓存目录 ----------------------------------------


def test_artifacts_dir_precedence(tmp_path: Path, monkeypatch) -> None:
    """落点决定首启要不要重编，所以三档优先级都是契约，不是实现细节。"""
    from sysone.config.storage import default_artifacts_dir

    monkeypatch.delenv("SYS1_ARTIFACTS_DIR", raising=False)
    assert _artifacts_dir() == default_artifacts_dir()

    monkeypatch.setenv("SYS1_ARTIFACTS_DIR", str(tmp_path / "env"))
    assert _artifacts_dir() == tmp_path / "env"
    assert _artifacts_dir(tmp_path / "flag") == tmp_path / "flag"


# ---- compile：与 serve 同一条装载路径，只是不起服务 -------------------------------------


def test_compile_refuses_the_reference_backend(capsys) -> None:
    """参考后端没有产物可编：报错要说清给哪两个后端，而不是让它走到 `--descriptor` 缺失。"""
    code = main(["compile", "--weights", str(FIXTURES / "mude")])
    assert code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "config"
    assert payload["context"]["backend"] == "reference"
    assert payload["context"]["available"] == ["split", "pooled"]


def test_compile_reports_the_startup_split_and_shuts_down(monkeypatch, capsys) -> None:
    """装完就卸：这条命令是部署时跑一次的，不该留一条链占着 ANE。"""
    from sysone import cli
    from sysone.config import StorageConfig
    from sysone.domain.work import CompilationTimes

    calls: list[str] = []

    class FakeEngine:
        config = SimpleNamespace(storage=StorageConfig(artifacts_dir=Path("/var/art")))

        def start(self) -> CompilationTimes:
            calls.append("start")
            return CompilationTimes(load_s=1.0, compile_s=12.5, warmup_s=0.5)

        def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setattr(cli, "_build_engine", lambda args, mode: FakeEngine())
    code = cli.main(["compile", "--backend", "split", "--weights", str(FIXTURES / "mude")])
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "sysone compile: 14.00s (load 1.00s, compile 12.50s, warmup 0.50s)"
    assert lines[1] == "artifacts: /var/art"
    assert calls == ["start", "shutdown"]

