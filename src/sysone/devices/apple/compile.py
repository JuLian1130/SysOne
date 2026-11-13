"""ANE program 的生成与装载：一层的三段 ANE 计算（投影 / 输出投影+残差 / 融合 FFN）。

MIL 文本与旧引擎 `e8_ane.layer_mil` 的三个 func 同形（`docs/ANE/anebench/e8_ane.py`），
区别只在于把 hidden / ffn / S 参数化 —— 旧引擎把 `HID=2048`、`FFN=6144` 写死在文本里。
形状参数化是 4B 复用的前提，也让对拍能用真实的 (2048 → 6144) 权重。

两种装载形态：
- `AneProgram`：一个 program 一个 func，M2.1 的逐段对拍用它（每段独立装载/执行，便于隔离误差）。
- `AneLayer`：一个 program 三个 procedure（`main` / `p1` / `p2`，与旧引擎同序），M2.2 的
  pipeline 用它 —— 一次 `load` 拿到整层，省掉两次装载与两次 program 切换。

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

from ...domain.work import ffn_parts  # 规则只有一份，装配（models）与编译（devices）共用
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
# 见 `docs/ANE/prefill-engine-design.md`）。
S_NATIVE = 1024

SIDECAR_NAME = "sidecar.json"
MIL_NAME = "program.mil"
BLOB_NAME = "weights.bin"

# 一层一个 program 时的 procedure 顺序（与旧引擎的 `PROCS` 一致）。
PROCS = {"proj": 0, "out": 1, "ffn": 2}


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


def _func(name: str, *, func_input: str, body: list[str], ret: str = "y") -> str:
    out = [f"    func {name}<ios18>({func_input}) {{\n"]
    out.extend(line + "\n" for line in body)
    out.append(f"    }} -> ({ret});\n")
    return "".join(out)


def _program(funcs: list[str]) -> str:
    return f"program(1.3)\n{BUILD_INFO}{{\n" + "".join(funcs) + "}\n"


def _proj_body(*, hidden: int, cout: int, s: int, entry: BlobEntry) -> list[str]:
    return _conv("p", cin=hidden, cout=cout, s=s, entry=entry, src="x", out="y")


def _out_body(*, hidden: int, s: int, entry: BlobEntry) -> list[str]:
    return (
        _slice("sv", begin=0, size=hidden, s=s, src="x", out="v")
        + _slice("sr", begin=hidden, size=hidden, s=s, src="x", out="r")
        + _conv("o", cin=hidden, cout=hidden, s=s, entry=entry, src="v", out="ov")
        + [f'        tensor<fp16, [1, {hidden}, 1, {s}]> y = '
           f'add(x = ov, y = r)[name = string("y")];']
    )


def _ffn_body(*, hidden: int, ffn: int, s: int, entries: Mapping[str, BlobEntry]) -> list[str]:
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
    return body


def projection_mil(*, hidden: int, cout: int, s: int, entry: BlobEntry) -> str:
    """投影 program（旧引擎 proc 0）：`[1, hidden, 1, s]` → `[1, cout, 1, s]`。"""
    if hidden <= 0 or cout <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, cout=cout, s=s)
    return _program([_func("main", func_input=f"tensor<fp16, [1, {hidden}, 1, {s}]> x",
                           body=_proj_body(hidden=hidden, cout=cout, s=s, entry=entry))])


def out_mil(*, hidden: int, s: int, entry: BlobEntry) -> str:
    """输出投影 + 残差（旧引擎 proc 1）：输入是 `[attn_out | residual]` 两段拼接。"""
    if hidden <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, s=s)
    return _program([_func("main", func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x",
                           body=_out_body(hidden=hidden, s=s, entry=entry))])


def ffn_mil(*, hidden: int, ffn: int, s: int, entries: Mapping[str, BlobEntry]) -> str:
    """融合 FFN（旧引擎 proc 2）：gate/up → silu×up → K 切分的 down → 残差。

    `entries` 的键是 `gate`、`up`、`down0`..`down{parts-1}`；parts 由 `ffn_parts()` 定。
    """
    if hidden <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, s=s)
    return _program([_func("main", func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x",
                           body=_ffn_body(hidden=hidden, ffn=ffn, s=s, entries=entries))])


def layer_mil(
    *,
    hidden: int,
    cout: int,
    ffn: int,
    s: int,
    entries: Mapping[str, BlobEntry],
) -> str:
    """一整层的三段 MIL：`main`（投影）、`p1`（输出投影+残差）、`p2`（融合 FFN）。

    `entries` 的键是 `proj`、`o`、`gate`、`up`、`down0`..`down{parts-1}`。三个 func 的
    procedure 序号与旧引擎一致（`PROCS`），`AneLayer` 用它来 `attach` / `eval`。
    """
    if hidden <= 0 or cout <= 0 or s <= 0:
        raise NotReadyError("MIL 形状必须为正", hidden=hidden, cout=cout, s=s)
    if "proj" not in entries or "o" not in entries:
        raise NotReadyError("整层 MIL 缺少投影或输出投影权重", keys=sorted(entries))
    return _program([
        _func("main", func_input=f"tensor<fp16, [1, {hidden}, 1, {s}]> x",
              body=_proj_body(hidden=hidden, cout=cout, s=s, entry=entries["proj"])),
        _func("p1", func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x",
              body=_out_body(hidden=hidden, s=s, entry=entries["o"])),
        _func("p2", func_input=f"tensor<fp16, [1, {2 * hidden}, 1, {s}]> x",
              body=_ffn_body(hidden=hidden, ffn=ffn, s=s, entries=entries)),
    ])


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


class AneLayer:
    """一层装成一个 program、三个 procedure：投影 / 输出投影+残差 / 融合 FFN。

    六张 IOSurface 各管一个 stream 边界：ANE 写投影结果 → GPU 读它、写注意力输出 →
    ANE 读 `[attn | residual]` 算 out → GPU 再读 → ANE 读 `[xn | residual]` 算 FFN。
    与 `AneProgram` 相比省掉两次 `load` 与两次 program 切换（旧引擎也是这个形态）。
    """

    def __init__(
        self,
        runtime: AneRuntime,
        files: ProgramFiles,
        *,
        name: str,
        hidden: int,
        cout: int,
        ffn: int,
        s: int,
    ) -> None:
        self._runtime = runtime
        self.name = name
        self.hidden = int(hidden)
        self.cout = int(cout)
        self.ffn = int(ffn)
        self.s = int(s)
        files.load(runtime, name)
        self._surfaces: dict[str, Surface] = {}
        try:
            for key, items in (
                ("proj_in", hidden), ("proj_out", cout),
                ("out_in", 2 * hidden), ("out_out", hidden),
                ("ffn_in", 2 * hidden), ("ffn_out", hidden),
            ):
                self._surfaces[key] = runtime.surface(items * self.s * 2)
            runtime.attach(name, PROCS["proj"], self._surfaces["proj_in"],
                           self._surfaces["proj_out"])
            runtime.attach(name, PROCS["out"], self._surfaces["out_in"],
                           self._surfaces["out_out"])
            runtime.attach(name, PROCS["ffn"], self._surfaces["ffn_in"],
                           self._surfaces["ffn_out"])
        except NotReadyError:
            self.close()
            raise

    def _write(self, key: str, items: int, x: np.ndarray) -> None:
        view = self._surfaces[key].array(np.float16, (1, items, 1, self.s))
        view[...] = np.asarray(x, dtype=np.float16).reshape(1, items, 1, self.s)

    def _read(self, key: str, items: int) -> np.ndarray:
        view = self._surfaces[key].array(np.float16, (1, items, 1, self.s))
        return view.reshape(items, self.s).copy()

    def _eval(self, proc: str, cout: int) -> tuple[np.ndarray, float]:
        ms = self._runtime.eval(self.name, PROCS[proc])
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, procedure=proc,
                                reason=self._runtime.error())
        out_key = f"{proc}_out"
        return self._read(out_key, cout), ms

    def project(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """`[hidden, s]` → `[cout, s]`。"""
        self._write("proj_in", self.hidden, x)
        return self._eval("proj", self.cout)

    def output(self, attn: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
        """`[hidden, s]` 注意力输出 + `[hidden, s]` 残差 → `[hidden, s]`（含残差）。"""
        self._write("out_in", 2 * self.hidden,
                    np.concatenate([attn, residual], axis=0))
        return self._eval("out", self.hidden)

    def feed_forward(self, xn: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float]:
        """`[hidden, s]` 归一化输入 + `[hidden, s]` 残差 → `[hidden, s]`（含残差）。"""
        self._write("ffn_in", 2 * self.hidden,
                    np.concatenate([xn, residual], axis=0))
        return self._eval("ffn", self.hidden)

    def close(self) -> None:
        self._runtime.unload(self.name)
        for surface in self._surfaces.values():
            surface.free()
        self._surfaces.clear()
