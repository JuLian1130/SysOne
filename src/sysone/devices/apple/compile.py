"""ANE program 的生成与装载：M2.1 的第一个可执行单元（单层投影 conv）。

MIL 文本与旧引擎 `e8_ane.layer_mil` 的 proc 0 同形（`docs/ANE/anebench/e8_ane.py`），
区别只在于把 hidden / 通道数 / S 参数化 —— 旧引擎把 `HID=2048`、`FFN=6144` 写死在文本里。
形状参数化是 4B 复用的前提，也让单层对拍能用真实的 (2048 → 6144) 权重。

编译缓存分两层，别混：
- **aned 的 lowering 缓存**：`e8_compiled_exists` 能查到，`e8_load_warm` 命中即跳过编译。
  它跨进程、不跨 aned 重启，驱逐策略不透明，所以只能当"省时间"，不能当产物存储。
- **我们自己的产物目录**（`ProgramFiles`）：`program.mil` + `weights.bin` + `sidecar.json`，
  sidecar 记录 MIL 与 blob 的 sha256，用来判断"这份产物还是不是当前这份 MIL"。

M2.1 只做 `main`（投影）这一个 func，FFN 的 3 路 K 切分与 attention 留在 M2.2 的 pipeline。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...errors import NotReadyError
from ...weights.packing.ane_blob import BlobEntry
from .runtime import AneRuntime, Surface

# 旧引擎的 buildInfo；ANE 的 MIL 前端要它，缺了会 compile 失败。
BUILD_INFO = (
    '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, '
    '{"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, '
    '{"coremltools-version", "9.0"}})]\n'
)

# ANE 的 program 形状是静态的，所以 S 分桶：1024 是单桶原生尺寸（4096 由 4 个 1024 拼，
# 见 `docs/ANE/prefill-engine-design.md`）。M2.1 只用 1024。
S_NATIVE = 1024

SIDECAR_NAME = "sidecar.json"
MIL_NAME = "program.mil"
BLOB_NAME = "weights.bin"


def _conv(
    tag: str,
    *,
    cin: int,
    cout: int,
    s: int,
    entry: BlobEntry,
    src: str,
    out: str,
) -> list[str]:
    """一个从 blob 取权重的 1×1 conv；int8 走向量化形式。"""
    lines = [
        f'        string {tag}_pt = const()[name = string("{tag}_pt"), val = string("valid")];',
        f'        tensor<int32, [2]> {tag}_st = const()[name = string("{tag}_st"), '
        f"val = tensor<int32, [2]>([1, 1])];",
        f'        tensor<int32, [4]> {tag}_pd = const()[name = string("{tag}_pd"), '
        f"val = tensor<int32, [4]>([0, 0, 0, 0])];",
        f'        tensor<int32, [2]> {tag}_dl = const()[name = string("{tag}_dl"), '
        f"val = tensor<int32, [2]>([1, 1])];",
        f'        int32 {tag}_g = const()[name = string("{tag}_g"), val = int32(1)];',
    ]
    path = 'string("@model_path/weights.bin")'
    if entry.quantized:
        if entry.scale_offset is None:
            raise NotReadyError("int8 权重缺少 scale 条目", tensor=entry.name)
        lines.append(
            f'        tensor<fp16, [{cout}, {cin}, 1, 1]> {tag}_w = '
            f'constexpr_affine_dequantize()[axis = int32(0), name = string("{tag}_w"), '
            f"quantized_data = tensor<int8, [{cout}, {cin}, 1, 1]>("
            f"BLOBFILE(path = {path}, offset = uint64({entry.offset}))), "
            f"scale = tensor<fp16, [{cout}]>(BLOBFILE(path = {path}, "
            f"offset = uint64({entry.scale_offset}))), zero_point = int8(0)];"
        )
    else:
        lines.append(
            f'        tensor<fp16, [{cout}, {cin}, 1, 1]> {tag}_w = const()[name = '
            f'string("{tag}_w"), val = tensor<fp16, [{cout}, {cin}, 1, 1]>('
            f"BLOBFILE(path = {path}, offset = uint64({entry.offset})))];"
        )
    lines.append(
        f'        tensor<fp16, [1, {cout}, 1, {s}]> {out} = conv(dilations = {tag}_dl, '
        f"groups = {tag}_g, pad = {tag}_pd, pad_type = {tag}_pt, strides = {tag}_st, "
        f'weight = {tag}_w, x = {src})[name = string("{out}")];'
    )
    return lines


def projection_mil(*, hidden: int, cout: int, s: int, entry: BlobEntry) -> str:
    """单层投影 program 的 MIL：`[1, hidden, 1, s]` 输入，`[1, cout, 1, s]` 输出。"""
    if hidden <= 0 or cout <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, cout=cout, s=s)
    body = _conv("p", cin=hidden, cout=cout, s=s, entry=entry, src="x", out="y")
    out = [f"program(1.3)\n{BUILD_INFO}{{\n"]
    out.append(f"    func main<ios18>(tensor<fp16, [1, {hidden}, 1, {s}]> x) {{\n")
    out.extend(line + "\n" for line in body)
    out.append("    } -> (y);\n")
    out.append("}\n")
    return "".join(out)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProgramFiles:
    """一份落在磁盘上的编译产物。`reused` 表示 sidecar 与当前 MIL/blob 一致。"""

    directory: Path
    mil_path: Path
    blob_path: Path
    sidecar_path: Path
    mil: str
    s: int
    reused: bool

    def load(self, runtime: AneRuntime, name: str) -> None:
        """装载。产物已是最新就尽量走 `load_warm`（aned 缓存命中时省掉 lowering）。"""
        if self.reused:
            runtime.load_warm(name, self.mil, self.blob_path)
        else:
            runtime.load(name, self.mil, self.blob_path)


def write_program(
    directory: str | Path,
    *,
    mil: str,
    blob: bytes,
    s: int,
    meta: dict | None = None,
) -> ProgramFiles:
    """把 MIL 与 blob 落盘，sidecar 写内容哈希；目录里内容没变就复用。

    `reused=True` 不代表 aned 一定命中，只代表"这份文件就是这份 MIL 的产物"。
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    mil_path = root / MIL_NAME
    blob_path = root / BLOB_NAME
    sidecar_path = root / SIDECAR_NAME

    mil_hash = _sha256(mil.encode())
    blob_hash = _sha256(blob)
    payload = {"s": int(s), "mil_sha256": mil_hash, "blob_sha256": blob_hash, "meta": meta or {}}

    reused = False
    if mil_path.is_file() and blob_path.is_file() and sidecar_path.is_file():
        try:
            previous = json.loads(sidecar_path.read_text())
        except (OSError, ValueError):
            previous = {}
        reused = (
            previous.get("mil_sha256") == mil_hash
            and previous.get("blob_sha256") == blob_hash
            and previous.get("s") == int(s)
        )
    if not reused:
        mil_path.write_text(mil)
        blob_path.write_bytes(blob)
        sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return ProgramFiles(
        directory=root,
        mil_path=mil_path,
        blob_path=blob_path,
        sidecar_path=sidecar_path,
        mil=mil,
        s=int(s),
        reused=reused,
    )


class AneProjection:
    """装载好的单层投影：输入/输出各一张 IOSurface，零拷贝进出。

    `run()` 返回 (输出 fp16 `[cout, s]`, 设备毫秒数)。调用方负责比较数值。
    """

    def __init__(
        self,
        runtime: AneRuntime,
        files: ProgramFiles,
        *,
        name: str,
        hidden: int,
        cout: int,
        s: int,
    ) -> None:
        self._runtime = runtime
        self.name = name
        self.hidden = int(hidden)
        self.cout = int(cout)
        self.s = int(s)
        files.load(runtime, name)
        self._in: Surface = runtime.surface(self.hidden * self.s * 2)
        try:
            self._out: Surface = runtime.surface(self.cout * self.s * 2)
        except NotReadyError:
            self._in.free()
            raise
        runtime.attach(name, 0, self._in, self._out)

    def run(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """`x` 是 `[hidden, s]` fp16；返回 `[cout, s]` fp16 与设备时间（ms）。"""
        view = self._in.array(np.float16, (1, self.hidden, 1, self.s))
        view[...] = np.asarray(x, dtype=np.float16).reshape(1, self.hidden, 1, self.s)
        ms = self._runtime.eval(self.name, 0)
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, reason=self._runtime.error())
        out = self._out.array(np.float16, (1, self.cout, 1, self.s)).reshape(self.cout, self.s)
        return out.copy(), ms

    def close(self) -> None:
        self._runtime.unload(self.name)
        self._in.free()
        self._out.free()
