"""ANE program 的生成与装载：一层的三段 ANE 计算（投影 / 输出投影+残差 / 融合 FFN）。

MIL 文本与旧引擎 `e8_ane.layer_mil` 的三个 func 同形（`docs/ANE/anebench/e8_ane.py`），
区别只在于把 hidden / ffn / S 参数化 —— 旧引擎把 `HID=2048`、`FFN=6144` 写死在文本里。
形状参数化是 4B 复用的前提，也让对拍能用真实的 (2048 → 6144) 权重。

这里只生成**单 func** 的 program（每个 proc 一个 program），因为 M2.1 的验收是逐 proc 的
数值对拍；把三段合进一个 program（一次 `load`、三个 procedure）留给 M2.2 的 pipeline，
那里的收益是省掉两次装载，不是正确性。

编译缓存分两层，别混：
- **aned 的 lowering 缓存**：`e8_compiled_exists` 能查到，`e8_load_warm` 命中即跳过编译。
  它跨进程、不跨 aned 重启，驱逐策略不透明，所以只能当"省时间"，不能当产物存储。
- **我们自己的产物目录**（`ProgramFiles`）：`program.mil` + `weights.bin` + `sidecar.json`，
  sidecar 记录 MIL 与 blob 的 sha256，用来判断"这份产物还是不是当前这份 MIL"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from ...errors import ConfigError, NotReadyError
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

_TORCH_TO_MIL_DTYPE = "fp16"


def ffn_parts(hidden: int, ffn: int) -> int:
    """FFN 的 down 投影要把 K 维切成几段 —— 旧引擎取 3（6144 / 2048）。

    ANE 的 conv 一次吃下 6144 个输入通道偏大，旧实现按 `hidden` 一段一段切。这里要求
    `ffn` 能被 `hidden` 整除：mude 是 6144/2048 = 3；4B 是 9216/2560 = 3.6，需要变长切分，
    留到 M4 的 4B 实验再说，现在显式报错而不是悄悄生成一个坏 program。
    """
    if hidden <= 0 or ffn <= 0:
        raise ConfigError("FFN 切分需要正的 hidden/ffn", hidden=hidden, ffn=ffn)
    if ffn % hidden:
        raise ConfigError(
            "FFN 的 down 投影要求 ffn 能被 hidden 整除",
            hidden=hidden,
            ffn=ffn,
            hint="4B（9216/2560）需要变长切分，M2 只支持 mude 的 6144/2048",
        )
    return ffn // hidden


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


def _slice(tag: str, *, begin: int, size: int, s: int, src: str, out: str) -> list[str]:
    return [
        f'        tensor<int32, [4]> {tag}_b = const()[name = string("{tag}_b"), '
        f"val = tensor<int32, [4]>([0, {begin}, 0, 0])];",
        f'        tensor<int32, [4]> {tag}_sz = const()[name = string("{tag}_sz"), '
        f"val = tensor<int32, [4]>([1, {size}, 1, {s}])];",
        f'        tensor<fp16, [1, {size}, 1, {s}]> {out} = slice_by_size(begin = {tag}_b, '
        f' size = {tag}_sz, x = {src})[name = string("{out}")];',
    ]


def _program(*, func_input: str, body: list[str]) -> str:
    """把一段 func 体包成完整的 MIL 文本（单 func `main`）。"""
    out = [f"program(1.3)\n{BUILD_INFO}{{\n"]
    out.append(f"    func main<ios18>({func_input}) {{\n")
    out.extend(line + "\n" for line in body)
    out.append("    } -> (y);\n")
    out.append("}\n")
    return "".join(out)


def projection_mil(*, hidden: int, cout: int, s: int, entry: BlobEntry) -> str:
    """投影 program（旧引擎 proc 0）：`[1, hidden, 1, s]` → `[1, cout, 1, s]`。"""
    if hidden <= 0 or cout <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, cout=cout, s=s)
    body = _conv("p", cin=hidden, cout=cout, s=s, entry=entry, src="x", out="y")
    return _program(func_input=f"tensor<fp16, [1, {hidden}, 1, {s}]> x", body=body)


def out_mil(*, hidden: int, s: int, entry: BlobEntry) -> str:
    """输出投影 + 残差（旧引擎 proc 1）：输入是 `[attn_out | residual]` 两段拼接。"""
    if hidden <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, s=s)
    body = (
        _slice("sv", begin=0, size=hidden, s=s, src="x", out="v")
        + _slice("sr", begin=hidden, size=hidden, s=s, src="x", out="r")
        + _conv("o", cin=hidden, cout=hidden, s=s, entry=entry, src="v", out="ov")
        + [f'        tensor<fp16, [1, {hidden}, 1, {s}]> y = '
           f'add(x = ov, y = r)[name = string("y")];']
    )
    return _program(func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x", body=body)


def ffn_mil(*, hidden: int, ffn: int, s: int, entries: Mapping[str, BlobEntry]) -> str:
    """融合 FFN（旧引擎 proc 2）：gate/up → silu×up → K 切分的 down → 残差。

    `entries` 的键是 `gate`、`up`、`down0`..`down{parts-1}`；parts 由 `ffn_parts()` 定。
    """
    if hidden <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, s=s)
    parts = ffn_parts(hidden, ffn)
    missing = [f"down{i}" for i in range(parts) if f"down{i}" not in entries]
    if "gate" not in entries or "up" not in entries or missing:
        raise NotReadyError("FFN 权重不全", missing=sorted(set(missing)) or ["gate/up"])

    body = (
        _slice("sx", begin=0, size=hidden, s=s, src="x", out="xn")
        + _slice("sr", begin=hidden, size=hidden, s=s, src="x", out="r")
        + _conv("g", cin=hidden, cout=ffn, s=s, entry=entries["gate"], src="xn", out="gt")
        + _conv("u", cin=hidden, cout=ffn, s=s, entry=entries["up"], src="xn", out="up")
        + [f'        tensor<fp16, [1, {ffn}, 1, {s}]> gs = '
           f'silu(x = gt)[name = string("gs")];',
           f'        tensor<fp16, [1, {ffn}, 1, {s}]> h = '
           f'mul(x = gs, y = up)[name = string("h")];']
    )
    for i in range(parts):
        body += _slice(f"d{i}", begin=i * hidden, size=hidden, s=s, src="h", out=f"h{i}")
        body += _conv(f"k{i}", cin=hidden, cout=hidden, s=s, entry=entries[f"down{i}"],
                      src=f"h{i}", out=f"y{i}")
    acc = "y0"
    for i in range(1, parts):
        nxt = f"s{i}" if i < parts - 1 else "s"
        body.append(f'        tensor<fp16, [1, {hidden}, 1, {s}]> {nxt} = '
                    f'add(x = {acc}, y = y{i})[name = string("{nxt}")];')
        acc = nxt
    body.append(f'        tensor<fp16, [1, {hidden}, 1, {s}]> y = '
                f'add(x = {acc}, y = r)[name = string("y")];')
    return _program(func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x", body=body)


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


class AneProgram:
    """装载好的单 func program：输入/输出各一张 IOSurface，零拷贝进出。

    `cin` / `cout` 是 MIL 里声明的通道数（投影是 hidden→cout，out/ffn 是 2×hidden→hidden）。
    `run()` 返回 (输出 fp16 `[cout, s]`, 设备毫秒数)，调用方负责比较数值。
    """

    def __init__(
        self,
        runtime: AneRuntime,
        files: ProgramFiles,
        *,
        name: str,
        cin: int,
        cout: int,
        s: int,
    ) -> None:
        self._runtime = runtime
        self.name = name
        self.cin = int(cin)
        self.cout = int(cout)
        self.s = int(s)
        files.load(runtime, name)
        self._in: Surface = runtime.surface(self.cin * self.s * 2)
        try:
            self._out: Surface = runtime.surface(self.cout * self.s * 2)
        except NotReadyError:
            self._in.free()
            raise
        runtime.attach(name, 0, self._in, self._out)

    def run(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """`x` 是 `[cin, s]` fp16；返回 `[cout, s]` fp16 与设备时间（ms）。"""
        view = self._in.array(np.float16, (1, self.cin, 1, self.s))
        view[...] = np.asarray(x, dtype=np.float16).reshape(1, self.cin, 1, self.s)
        ms = self._runtime.eval(self.name, 0)
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, reason=self._runtime.error())
        out = self._out.array(np.float16, (1, self.cout, 1, self.s)).reshape(self.cout, self.s)
        return out.copy(), ms

    def close(self) -> None:
        self._runtime.unload(self.name)
        self._in.free()
        self._out.free()
