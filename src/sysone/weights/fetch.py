"""首启把发布权重拉下来：完整性 marker、单进程锁、临时目录与原子换入。

判据只有目录里的 `source.json`：仓库、修订、以及引擎所读每个文件的实际字节数。刻意不
算内容哈希 —— 9 GB 读一遍正是"第二次启动"要省掉的那一段，而修订已经固定，文件由
Hugging Face 的 etag 保证。

下载走 `snapshot_download`（README 的 `hf download` 同一条路径），先落临时目录，写完
marker 再换入正式目录；换入前的中间态也有恢复路径，崩溃不会把一份旧权重变成半份新权重。
"""

from __future__ import annotations

import fcntl
import json
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..errors import DownloadError
from ..models.sources import WeightsSource

MARKER_NAME = "source.json"
# `snapshot_download(local_dir=...)` 会在目录里自建一份元数据缓存；它不参与完整性判据。
Downloader = Callable[[WeightsSource, Path], None]



def ensure_weights(
    source: WeightsSource,
    root: str | Path,
    *,
    offline: bool = False,
    download: Downloader | None = None,
) -> Path:
    """返回 `<root>/<source.directory>`；本地齐全就直接用，否则下载。

    `download` 只为测试注入：生产路径用 `_snapshot`。`offline=True` 时缺权重直接抛
    `DownloadError`，并把可复制的 `hf download` 命令写进 context。
    """
    weights_root = Path(root)
    target = weights_root / source.directory
    # 锁定前只读判断：`_recover` 会改目录，不能与另一个进程的换入并发。
    if _complete(target, source):
        return target

    with _locked(weights_root):
        # 等锁期间可能已经有另一个进程把它下完了：拿到锁后必须重新判一次。
        _recover(target)
        if _complete(target, source):
            return target
        if offline:
            raise DownloadError(
                "本地没有这份发布权重，且当前是离线模式",
                repo_id=source.repo_id,
                revision=source.revision,
                directory=str(target),
                command=manual_command(source, target),
            )
        partial = _partial(target)
        shutil.rmtree(partial, ignore_errors=True)
        print(
            f"weights: downloading {source.repo_id} @ {source.revision[:8]} "
            f"({_size_label(source.size_hint)}, {len(source.files)} files) -> {target}",
            file=sys.stderr,
        )
        try:
            (download or _snapshot)(source, partial)
        except DownloadError:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        except Exception as exc:  # 网络/磁盘各家的异常都在这里归成一个可读错误
            shutil.rmtree(partial, ignore_errors=True)
            raise DownloadError(
                "权重下载失败",
                repo_id=source.repo_id,
                revision=source.revision,
                directory=str(target),
                error=f"{type(exc).__name__}: {exc}",
                command=manual_command(source, target),
            ) from exc
        missing = [name for name in source.files if not (partial / name).is_file()]
        if missing:
            shutil.rmtree(partial, ignore_errors=True)
            raise DownloadError(
                "下载结束后仍缺少文件",
                repo_id=source.repo_id,
                revision=source.revision,
                directory=str(target),
                missing=missing,
                command=manual_command(source, target),
            )
        _write_marker(partial, source)
        _swap(partial, target)
    print(f"weights: ready {target}", file=sys.stderr)
    return target


def manual_command(source: WeightsSource, target: Path) -> str:
    """离线/失败时给用户的等价手命令（与 README 的下载段同一形状）。"""
    files = " ".join(source.files)
    return (
        f"hf download {source.repo_id} --revision {source.revision} "
        f"{files} --local-dir {target}"
    )


def _snapshot(source: WeightsSource, destination: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DownloadError(
            "自动下载需要 huggingface_hub",
            hint='pip install -e "./src"',
        ) from exc
    snapshot_download(
        repo_id=source.repo_id,
        revision=source.revision,
        allow_patterns=list(source.files),
        local_dir=str(destination),
    )


def _complete(target: Path, source: WeightsSource) -> bool:
    marker = _read_marker(target)
    if marker is None:
        return False
    if marker.get("repo_id") != source.repo_id or marker.get("revision") != source.revision:
        return False
    recorded = marker.get("files")
    if not isinstance(recorded, dict) or set(recorded) != set(source.files):
        return False
    for name in source.files:
        path = target / name
        try:
            size = path.stat().st_size
        except OSError:
            return False
        try:
            expected = int(recorded[name])
        except (TypeError, ValueError):
            return False
        if size != expected:
            return False
    return True


def _read_marker(target: Path) -> dict | None:
    path = target / MARKER_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_marker(target: Path, source: WeightsSource) -> None:
    files = {name: (target / name).stat().st_size for name in source.files}
    payload = {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "files": files,
    }
    (target / MARKER_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _partial(target: Path) -> Path:
    return target.with_name(target.name + ".partial")


def _swap(partial: Path, target: Path) -> None:
    """`target` 可能已有一份旧修订：先挪开，再换入，最后删旧。"""
    old = target.with_name(target.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if target.exists():
        target.rename(old)
    partial.rename(target)
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _recover(target: Path) -> None:
    """收拾换入中途崩溃留下的 `.old` / `.partial`。

    只有一种中间态会造成损失：旧目录已挪走、新目录还没换入。这时把它挪回来，下一次会
    按 marker 判为不完整并重下；半份的 `.partial` 永远不作为权重使用。
    """
    old = target.with_name(target.name + ".old")
    partial = _partial(target)
    if not target.exists() and old.is_dir():
        old.rename(target)
    if target.exists() and old.exists():
        shutil.rmtree(old, ignore_errors=True)
    if target.exists() and partial.exists():
        shutil.rmtree(partial, ignore_errors=True)


@contextmanager
def _locked(root: Path) -> Iterator[None]:
    """两个进程同时首启时，后一个等前一个把权重落盘。

    锁文件放在权重根而不是每个模型目录里：一个进程一次只下一次，挂在根上最简单。等锁是
    阻塞的，先打印一行日志，避免看起来像卡死。
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".fetch.lock"
    with open(path, "w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("weights: waiting for another process to finish downloading", file=sys.stderr)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _size_label(size: int) -> str:
    return f"{size / 1e9:.1f} GB"
