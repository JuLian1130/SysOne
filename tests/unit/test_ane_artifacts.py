"""产物缓存：sidecar 的身份比对，以及 `ARTIFACT_VERSION` 的失效。

这条判断是 `build_blob`（本机 6 s）与"读一遍 fp16 权重"（2.4 s）能不能被跳过的唯一依据，
所以它必须只认"确实还是这份权重"的情形；真链的装载在集成测试里验证。
"""

from __future__ import annotations

import json
from pathlib import Path

from sysone.devices.apple.compile import (
    ARTIFACT_VERSION,
    BLOB_NAME,
    MIL_NAME,
    SIDECAR_NAME,
    ProgramFiles,
    cached_program,
    write_program,
)

HASHES = ("a" * 64, "b" * 64, "c" * 64)


class FakeRuntime:
    """只记调用序，不碰 ANE：装载路径要能被单测钉住，真装载在集成测试里守。"""

    def __init__(self, *, fast: bool = False, hashes: tuple | None = HASHES) -> None:
        self.calls: list[str] = []
        self.fast = fast
        self.hashes = hashes

    def load(self, name, mil, blob_path) -> None:
        self.calls.append("load")

    def load_warm(self, name, mil, blob_path) -> None:
        self.calls.append("load_warm")

    def load_fast(self, name, mil, blob_path, hashes) -> bool:
        self.calls.append("load_fast")
        return self.fast

    def descriptor_hashes(self, mil, blob_path) -> tuple:
        self.calls.append("descriptor_hashes")
        if self.hashes is None:
            from sysone.errors import NotReadyError

            raise NotReadyError("dylib 没有 e8_descriptor_hashes")
        return self.hashes

MIL = "program(1.3)\nfunc main<ios18>(tensor<fp16, [1, 8, 1, 32]> x) {\n    return x;\n}\n"
SOURCE = {
    "weights": "/models/mude/model.safetensors",
    "bytes": 10,
    "mtime_ns": 20,
    "layer": 0,
    "type": "full",
    "cout": 16,
    "hidden": 8,
    "ffn": 24,
    "s": 32,
    "quantize": True,
}


def _write(directory: Path, *, source: dict | None = SOURCE, s: int = 32) -> Path:
    return write_program(directory, mil=MIL, blob=b"abc", s=s, meta={"cout": 16}, source=source)


def test_cache_misses_before_anything_is_written(tmp_path: Path) -> None:
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=32) is None


def test_cache_hits_after_a_matching_write(tmp_path: Path) -> None:
    _write(tmp_path / "layer0")
    files = cached_program(tmp_path / "layer0", source=SOURCE, s=32)
    assert files is not None
    assert files.reused is True
    assert files.mil == MIL
    assert files.blob_path.name == BLOB_NAME
    assert files.sidecar_path.name == SIDECAR_NAME
    assert files.mil_path.name == MIL_NAME


def test_cache_misses_when_the_source_identity_changes(tmp_path: Path) -> None:
    _write(tmp_path / "layer0")
    for field, value in (
        ("weights", "/models/other/model.safetensors"),
        ("bytes", 11),
        ("mtime_ns", 21),
        ("layer", 1),
        ("type", "linear"),
        ("cout", 32),
        ("hidden", 16),
        ("ffn", 48),
        ("s", 16),
        ("quantize", False),
    ):
        changed = dict(SOURCE, **{field: value})
        assert cached_program(tmp_path / "layer0", source=changed, s=32) is None, field


def test_cache_misses_when_the_bucket_width_changes(tmp_path: Path) -> None:
    _write(tmp_path / "layer0", s=32)
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=16) is None


def test_cache_misses_after_the_version_salt_moves(tmp_path: Path) -> None:
    _write(tmp_path / "layer0")
    sidecar = tmp_path / "layer0" / SIDECAR_NAME
    payload = json.loads(sidecar.read_text())
    payload["version"] = ARTIFACT_VERSION + 1
    sidecar.write_text(json.dumps(payload))
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=32) is None


def test_cache_misses_when_the_blob_or_the_sidecar_is_unusable(tmp_path: Path) -> None:
    _write(tmp_path / "layer0")
    (tmp_path / "layer0" / BLOB_NAME).unlink()
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=32) is None

    _write(tmp_path / "layer0")
    (tmp_path / "layer0" / SIDECAR_NAME).write_text("{ not json")
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=32) is None


def test_writes_without_a_source_never_hit_the_cache(tmp_path: Path) -> None:
    """旧产物（没有 source 字段）也算未命中，这样升级后第一次启动会自己重建。"""
    _write(tmp_path / "layer0", source=None)
    assert cached_program(tmp_path / "layer0", source=SOURCE, s=32) is None


def test_sidecar_is_rewritten_when_only_the_identity_is_new(tmp_path: Path) -> None:
    """内容一模一样、只是身份刚补上时也要落 sidecar，否则缓存永远学不到它。"""
    directory = tmp_path / "layer0"
    write_program(directory, mil=MIL, blob=b"abc", s=32)  # 没有 source 的"老产物"
    files = write_program(directory, mil=MIL, blob=b"abc", s=32, meta={"cout": 16}, source=SOURCE)
    assert files.reused is True  # MIL 与 blob 都没变，不重写它们
    assert cached_program(directory, source=SOURCE, s=32) is not None


def _files(directory: Path) -> ProgramFiles:
    files = cached_program(directory, source=SOURCE, s=32)
    assert files is not None
    return files


def test_load_goes_fast_when_the_hashes_are_known(tmp_path: Path) -> None:
    """identity 已记过：一次 `load_fast` 装完，不碰 `load_warm`。"""
    _write(tmp_path / "layer0")
    files = _files(tmp_path / "layer0")
    files = ProgramFiles(
        directory=files.directory, mil_path=files.mil_path, blob_path=files.blob_path,
        sidecar_path=files.sidecar_path, mil=files.mil, s=files.s, reused=True,
        descriptor_hashes=HASHES,
    )
    runtime = FakeRuntime(fast=True)
    files.load(runtime, "layer")
    assert runtime.calls == ["load_fast"]


def test_load_falls_back_when_the_fast_path_refuses(tmp_path: Path) -> None:
    """aned 缓存被驱逐（rc≠0）：回落 `load_warm`，不抛异常。"""
    _write(tmp_path / "layer0")
    files = _files(tmp_path / "layer0")
    runtime = FakeRuntime(fast=False)
    ProgramFiles(
        directory=files.directory, mil_path=files.mil_path, blob_path=files.blob_path,
        sidecar_path=files.sidecar_path, mil=files.mil, s=files.s, reused=True,
        descriptor_hashes=HASHES,
    ).load(runtime, "layer")
    assert runtime.calls == ["load_fast", "load_warm"]


def test_first_load_records_the_hashes_and_the_next_one_goes_fast(tmp_path: Path) -> None:
    """第一次装载算一次哈希写进 sidecar；同内容再装载就是快路。"""
    _write(tmp_path / "layer0")
    files = _files(tmp_path / "layer0")
    runtime = FakeRuntime(fast=False)
    files.load(runtime, "layer")
    assert runtime.calls == ["load_warm", "descriptor_hashes"]
    again = _files(tmp_path / "layer0")
    assert again.descriptor_hashes == HASHES
    fast = FakeRuntime(fast=True)
    again.load(fast, "layer")
    assert fast.calls == ["load_fast"]


def test_write_program_keeps_a_recorded_identity(tmp_path: Path) -> None:
    """内容没变就继承哈希：下一次构建不会把它抹掉。"""
    _write(tmp_path / "layer0")
    files = _files(tmp_path / "layer0")
    files._record_hashes(HASHES)
    _write(tmp_path / "layer0")
    assert _files(tmp_path / "layer0").descriptor_hashes == HASHES


def test_write_program_drops_the_identity_when_the_blob_changes(tmp_path: Path) -> None:
    """blob 换了：aned 给的三段哈希就不认了。"""
    _write(tmp_path / "layer0")
    files = _files(tmp_path / "layer0")
    files._record_hashes(HASHES)
    write_program(tmp_path / "layer0", mil=MIL, blob=b"abd", s=32, meta={"cout": 16},
                  source=SOURCE)
    assert _files(tmp_path / "layer0").descriptor_hashes is None
