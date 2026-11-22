"""首启下载：marker 复用、失败重下、离线错误与换入恢复。"""

from __future__ import annotations

import json
from dataclasses import replace
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
    """假下载器：按文件清单写小文件，记录调用次数与最近一次收到的清单和落点。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail
        self.files: tuple[str, ...] = ()
        self.destination: Path | None = None

    def __call__(self, source: WeightsSource, destination: Path) -> None:
        self.calls += 1
        self.files = tuple(source.files)
        self.destination = destination
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


def test_a_file_added_to_the_manifest_is_fetched_into_the_existing_directory(
    tmp_path: Path,
) -> None:
    """清单只是多出文件时（补两个厂商参考文件）只下那几份，不整份重下。"""
    target = ensure_weights(SOURCE, tmp_path, download=Recorder())

    grown = replace(SOURCE, files=(*SOURCE.files, "d3_format.py"))
    additions = Recorder()
    assert ensure_weights(grown, tmp_path, download=additions) == target

    assert additions.calls == 1
    assert additions.files == ("d3_format.py",)  # 只下新的那一份
    assert additions.destination == target  # 就地补，不落到新的 partial
    assert (target / "d3_format.py").is_file()
    marker = json.loads((target / MARKER_NAME).read_text(encoding="utf-8"))
    assert set(marker["files"]) == set(grown.files)


def test_a_damaged_or_removed_entry_still_redownloads_everything(tmp_path: Path) -> None:
    """已记录的文件坏了、或清单少了条目，就不是"只增"，整份重下。"""
    target = ensure_weights(SOURCE, tmp_path, download=Recorder())
    (target / "model.safetensors").write_text("longer than before", encoding="utf-8")
    grown = replace(SOURCE, files=(*SOURCE.files, "d3_format.py"))

    damaged = Recorder()
    ensure_weights(grown, tmp_path, download=damaged)
    assert damaged.destination is not None
    assert damaged.destination.name.endswith(".partial")

    # 清单少了条目：旧目录里多出来的那个文件不在清单里，旧目录不再可信
    shrunk = Recorder()
    ensure_weights(replace(SOURCE, files=("config.json",)), tmp_path, download=shrunk)
    assert shrunk.destination is not None
    assert shrunk.destination.name.endswith(".partial")


def test_offline_never_takes_the_additions_path(tmp_path: Path) -> None:
    """离线 + 只缺新增的那几份：按离线报错，不借口"只补几份"去联网。"""
    ensure_weights(SOURCE, tmp_path, download=Recorder())
    grown = replace(SOURCE, files=(*SOURCE.files, "d3_format.py"))

    with pytest.raises(DownloadError):
        ensure_weights(grown, tmp_path, offline=True, download=_never)


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
    """出货家族都有来源之后，这条仍是新家族的默认：没来源就要显式 `--weights`。"""
    with pytest.raises(ConfigError) as caught:
        source_for("unpublished")
    assert caught.value.context["available"] == [
        "d3", "h2o", "mude", "pplx12", "pplx13", "singvl", "torchcast"]


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
