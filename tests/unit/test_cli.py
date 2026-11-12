"""CLI：inspect 走家族解析，run 走参考后端。"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import FIXTURES, make_request, make_tiny
from sysone.cli import main


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
