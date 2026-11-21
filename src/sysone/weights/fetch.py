"""首启把发布权重拉下来：完整性 marker、单进程锁、临时目录与原子换入。

判据只有目录里的 `source.json`：仓库、修订、以及引擎所读每个文件的实际字节数。刻意不
算内容哈希 —— 9 GB 读一遍正是"第二次启动"要省掉的那一段，而修订已经固定，文件由
Hugging Face 的 etag 保证。

清单**只会多出文件**（例如补两个厂商参考文件）。那种情况走"只补缺的那几份"，补进现有
目录再重写 marker：为两个 100 KB 的文件整份重下 4 GB，是改一行清单不该有的代价。修订换了、
已记录的文件坏了、或者清单少了条目，仍然整份重下 —— 那几种情况下目录里的东西不再可信。

下载走 `snapshot_download`（与 `hf download` 同一条路径），先落临时目录，写完
marker 再换入正式目录；换入前的中间态也有恢复路径，崩溃不会把一份旧权重变成半份新权重。
"""

from __future__ import annotations

import fcntl
import json
import shutil
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from ..errors import DownloadError
from ..models.sources import WeightsSource

MARKER_NAME = "source.json"
# `snapshot_download(local_dir=...)` 会在目录里自建一份元数据缓存；它不参与完整性判据。
Downloader = Callable[[WeightsSource, Path], None]

# `hf download` 只把文件放到位、不写 `source.json`，所以手动命令之后必须走显式 `--weights`
# ——否则下一次启动仍判为不完整并重下。
MANUAL_HINT = (
    "跑完这条命令后用 --weights 指向该目录启动：显式路径只读，不查 source.json；"
    "整份复制缓存时要连 source.json 一起带上"
)


def ensure_weights(
    source: WeightsSource,
    root: str | Path,
    *,
    offline: bool = False,
    download: Downloader | None = None,
) -> Path:
    """返回 `<root>/<source.directory>`；本地齐全就直接用，否则下载。

    `download` 只为测试注入：生产路径用 `_snapshot`。`offline=True` 时缺权重直接抛
    `DownloadError`，并把可复制的 `hf download` 命令写进 context。清单里只是多出几项时
    只补那几项，不整份重下。
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
                hint=MANUAL_HINT,
            )
        # 清单只是多出文件（例如补两个厂商参考文件）：补进现有目录，不整份重下。
        awaited = _additions(target, source)
        if awaited:
            _fetch_additions(source, awaited, target, download)
            return target
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
                hint=MANUAL_HINT,
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
                hint=MANUAL_HINT,
            )
        _write_marker(partial, source)
        _swap(partial, target)
    print(f"weights: ready {target}", file=sys.stderr)
    return target


def manual_command(source: WeightsSource, target: Path) -> str:
    """离线/失败时给用户的等价手命令：把文件放到 `target`；之后的启动见 `MANUAL_HINT`。"""
    files = " ".join(source.files)
    return (
        f"hf download {source.repo_id} --revision {source.revision} "
        f"{files} --local-dir {target}"
    )


def _additions(target: Path, source: WeightsSource) -> tuple[str, ...]:
    """清单里只是多出文件时，返回要补的那几个；其他任何不一致都返回空元组。

    判据与 `_complete` 同源：仓库与修订对上、已记录的文件都还在且字节数没变、并且清单是
    已记录集合的**超集**（只增不减）。这样"给清单补两个厂商参考文件"不用把 4 GB 重下一遍；
    修订换了、文件坏了、清单删了条目都仍然整份重下 —— 那时目录里的旧文件不再可信。
    """
    marker = _read_marker(target)
    if marker is None:
        return ()
    if marker.get("repo_id") != source.repo_id or marker.get("revision") != source.revision:
        return ()
    recorded = marker.get("files")
    if not isinstance(recorded, dict):
        return ()
    if not set(recorded) <= set(source.files):
        return ()
    awaited = tuple(name for name in source.files if name not in recorded)
    if not awaited:
        return ()
    for name, expected in recorded.items():
        try:
            if (target / name).stat().st_size != int(expected):
                return ()
        except (OSError, TypeError, ValueError):
            return ()
    return awaited


def _fetch_additions(
    source: WeightsSource, awaited: tuple[str, ...], target: Path,
    download: Downloader | None,
) -> None:
    """把缺的那几个文件补进现有目录，再按全量清单重写 marker。

    失败不清理目录里已有的文件：marker 还是旧的，下一次启动会原样再算出同一份 `awaited`，
    幂等重试 —— 比整份下载那条"删掉 partial 重来"的路更适合就地补文件。
    """
    print(f"weights: adding {len(awaited)} file(s) to {target}", file=sys.stderr)
    only = replace(source, files=awaited)
    try:
        (download or _snapshot)(only, target)
    except DownloadError:
        raise
    except Exception as exc:  # 与整份下载同一套归错
        raise DownloadError(
            "权重补下载失败",
            repo_id=source.repo_id,
            revision=source.revision,
            directory=str(target),
            files=list(awaited),
            error=f"{type(exc).__name__}: {exc}",
            command=manual_command(source, target),
            hint=MANUAL_HINT,
        ) from exc
    missing = [name for name in awaited if not (target / name).is_file()]
    if missing:
        raise DownloadError(
            "补下载结束后仍缺少文件",
            repo_id=source.repo_id,
            revision=source.revision,
            directory=str(target),
            missing=missing,
            command=manual_command(source, target),
            hint=MANUAL_HINT,
        )
    _write_marker(target, source)
    print(f"weights: ready {target}", file=sys.stderr)


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
