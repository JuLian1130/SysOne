"""首启下载：marker 复用、失败重下、离线错误与换入恢复。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sysone.errors import ConfigError, DownloadError
from sysone.models.sources import WeightsSource, source_for
from sysone.weights.fetch import MARKER_NAME, ensure_weights, manual_command

SOURCE = WeightsSource(
    family="mude",
    repo_id="qybaihe/mude-1.0",
    revision="rev-a",
    directory="mude-1.0",
    files=("config.json", "model.safetensors"),
    size_hint=2_000,
)


class Recorder:
    """假下载器：按文件清单写小文件，记录调用次数。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def __call__(self, source: WeightsSource, destination: Path) -> None:
        self.calls += 1
        if self.fail:
            raise OSError("network down")
        destination.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(source.files):
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{source.revision}:{name}:{index}", encoding="utf-8")


def _never(*_args, **_kwargs) -> None:
    raise AssertionError("this call must not download")


def test_second_run_uses_the_marker_without_touching_the_network(tmp_path: Path) -> None:
    downloader = Recorder()
    first = ensure_weights(SOURCE, tmp_path, download=downloader)
    second = ensure_weights(SOURCE, tmp_path, download=_never)

    assert first == second == tmp_path / "mude-1.0"
    assert downloader.calls == 1
    marker = json.loads((second / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["repo_id"] == SOURCE.repo_id
    assert marker["revision"] == SOURCE.revision
    assert set(marker["files"]) == set(SOURCE.files)


def test_missing_file_forces_a_download(tmp_path: Path) -> None:
    downloader = Recorder()
    target = ensure_weights(SOURCE, tmp_path, download=downloader)
    (target / "config.json").unlink()

    fresh = Recorder()
    ensure_weights(SOURCE, tmp_path, download=fresh)
    assert fresh.calls == 1
    assert (target / "config.json").is_file()


def test_size_mismatch_forces_a_download(tmp_path: Path) -> None:
    downloader = Recorder()
    target = ensure_weights(SOURCE, tmp_path, download=downloader)
    (target / "model.safetensors").write_text("longer than before", encoding="utf-8")

    fresh = Recorder()
    ensure_weights(SOURCE, tmp_path, download=fresh)
    assert fresh.calls == 1


def test_revision_change_forces_a_download(tmp_path: Path) -> None:
    downloader = Recorder()
    ensure_weights(SOURCE, tmp_path, download=downloader)

    other = WeightsSource(
        family=SOURCE.family,
        repo_id=SOURCE.repo_id,
        revision="rev-b",
        directory=SOURCE.directory,
        files=SOURCE.files,
        size_hint=SOURCE.size_hint,
    )
    fresh = Recorder()
    target = ensure_weights(other, tmp_path, download=fresh)
    assert fresh.calls == 1
    marker = json.loads((target / MARKER_NAME).read_text(encoding="utf-8"))
    assert marker["revision"] == "rev-b"


def test_offline_without_weights_says_how_to_download_them(tmp_path: Path) -> None:
    with pytest.raises(DownloadError) as captured:
        ensure_weights(SOURCE, tmp_path, offline=True)

    context = captured.value.context
    assert captured.value.code == "download"
    assert context["repo_id"] == SOURCE.repo_id
    assert context["revision"] == SOURCE.revision
    assert "hf download" in context["command"]
    assert "rev-a" in context["command"]
    assert "--weights" in context["hint"]


def test_offline_with_local_weights_is_silent(tmp_path: Path) -> None:
    target = ensure_weights(SOURCE, tmp_path, download=Recorder())
    assert ensure_weights(SOURCE, tmp_path, offline=True) == target


def test_a_crashed_swap_restores_the_previous_directory(tmp_path: Path) -> None:
    target = ensure_weights(SOURCE, tmp_path, download=Recorder())
    old = target.with_name(target.name + ".old")
    target.rename(old)

    recovered = ensure_weights(SOURCE, tmp_path, offline=True)
    assert recovered == target
    assert (target / "model.safetensors").is_file()
    assert not old.exists()


def test_failed_download_cleans_the_partial_directory(tmp_path: Path) -> None:
    def half_then_fail(source: WeightsSource, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.safetensors").write_text("half", encoding="utf-8")
        raise OSError("connection reset")

    with pytest.raises(DownloadError):
        ensure_weights(SOURCE, tmp_path, download=half_then_fail)

    assert not (tmp_path / "mude-1.0").exists()
    assert not (tmp_path / "mude-1.0.partial").exists()


def test_a_family_without_a_published_source_asks_for_explicit_weights() -> None:
    with pytest.raises(ConfigError):
        source_for("torchcast")


def test_manual_command_names_the_pinned_revision(tmp_path: Path) -> None:
    command = manual_command(SOURCE, tmp_path / SOURCE.directory)
    assert command.startswith("hf download qybaihe/mude-1.0")
    assert "--revision rev-a" in command
    assert "--local-dir" in command


def test_the_first_download_is_announced_on_stderr(tmp_path: Path, capsys) -> None:
    target = ensure_weights(SOURCE, tmp_path, download=Recorder())
    err = capsys.readouterr().err
    assert "weights: downloading" in err
    assert SOURCE.repo_id in err
    assert SOURCE.revision[:8] in err
    assert f"weights: ready {target}" in err
