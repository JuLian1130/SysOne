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
from collections.abc import Sequence
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

# 一个 (program, procedure) 能绑几套面。dylib 把请求按 `proc * 8 + slot` 键起来
# （`e8_ane.m:386`），所以槽位号只能落在 0..7 —— 写成 8 会撞到下一个 procedure。
MAX_SLOTS = 8


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


# 每个 procedure 用哪两张面：`(输入, 输出)`，键都进 `SurfaceSet.surfaces`。
SET_KEYS = {
    "proj": ("proj_in", "proj_out"),
    "out": ("out_in", "out_out"),
    "ffn": ("ffn_in", "ffn_out"),
}


class SurfaceSet:
    """一层三段计算要的六张 IOSurface，绑成一套。

    这套面**不归某个 program 所有**，因为所有权换来的东西没有用：ANE 只有一条固件队列、
    一次只有一个 eval 在飞，所以 20 层可以轮流用同一套面。旧引擎正是这么做的
    （`docs/ANE/anebench/e8_ane.py:564-591`：20 层共享一套，省掉 0.9 GB）。池里放多套的
    唯一理由是**相邻两层不能同时用同一套**：第 i 层的 ANE 还在读 `proj_in` 时，第 i+1 层的
    `norm` 要往另一套里写 —— 双链驱动按 (层, 瓦片奇偶) 各分一套，见第 3 步。

    谁开谁关：`AneLayer` 借用的池由调用方 `free()`。`close()` 里先卸载 program 再释放面
    （卸载会 CFRelease 掉每次 `attach` 时给面加的引用，面才真的回收）。
    """

    def __init__(self, runtime: AneRuntime, *, hidden: int, cout: int, s: int) -> None:
        self.hidden = int(hidden)
        self.cout = int(cout)
        self.s = int(s)
        self.surfaces: dict[str, Surface] = {}
        for key, items in (
            ("proj_in", self.hidden), ("proj_out", self.cout),
            ("out_in", 2 * self.hidden), ("out_out", self.hidden),
            ("ffn_in", 2 * self.hidden), ("ffn_out", self.hidden),
        ):
            try:
                self.surfaces[key] = runtime.surface(items * self.s * 2)
            except NotReadyError:
                self.free()
                raise

    def free(self) -> None:
        for surface in self.surfaces.values():
            surface.free()
        self.surfaces.clear()


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

    `pool` 给定时这六张面是借来的：池里的第 k 套绑到 ANE 槽位 k，调用方按链选槽位
    （`eval_proc(..., slot=k)`）。不给就自己开一套（M2.1 的逐段对拍与 numpy 链用这个形态）。
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
        pool: Sequence[SurfaceSet] | None = None,
    ) -> None:
        self._runtime = runtime
        self.name = name
        self.hidden = int(hidden)
        self.cout = int(cout)
        self.ffn = int(ffn)
        self.s = int(s)
        self._owned: SurfaceSet | None = None
        self._sets: list[SurfaceSet] = []
        self._surfaces: dict[str, Surface] = {}
        try:
            files.load(runtime, name)
            self._owned = None if pool else SurfaceSet(
                runtime, hidden=self.hidden, cout=self.cout, s=self.s
            )
            self._sets = [self._owned] if self._owned is not None else list(pool or ())
            if len(self._sets) > MAX_SLOTS:
                raise ConfigError("表面池超过 ANE 的槽位数", sets=len(self._sets),
                                  max_slots=MAX_SLOTS)
            self._surfaces = self._sets[0].surfaces
            for slot, st in enumerate(self._sets):
                for proc, pair in SET_KEYS.items():
                    runtime.attach_slot(name, PROCS[proc], slot,
                                        st.surfaces[pair[0]], st.surfaces[pair[1]])
        except BaseException:
            self.close()
            raise

    @property
    def surfaces(self) -> dict[str, Surface]:
        """六张 IOSurface（`proj_in` / `proj_out` / `out_in` / `out_out` / `ffn_in` / `ffn_out`）。

        Metal 侧要直接 wrap 这几张面做零拷贝：GPU 读 ANE 刚写过的面、写 ANE 下一步要读的
        面，中间不该经过 numpy。给出去的是**同一批对象**，调用方不许释放；借用池的时候它
        就是池里第 0 套（所有共享它的层拿到的是同一个 dict）。
        """
        return self._surfaces

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

    def eval_proc(self, proc: str, *, slot: int = 0) -> float:
        """只执行一个 procedure，不读回结果（流水线里结果留在面上给 GPU 核用）。"""
        ms = self._runtime.eval_slot(self.name, PROCS[proc], slot)
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, procedure=proc, slot=slot,
                                reason=self._runtime.error())
        return ms

    @property
    def pending(self) -> bool:
        """是否还有一次已发射、未回收的 eval 在飞。"""
        return self._runtime.pending

    def issue_proc(self, proc: str, *, slot: int = 0) -> None:
        """异步发射一个 procedure（不等待），随后用 `reap_proc()` 回收。

        发射之后到回收之前，CPU 可以去 GPU 命令缓冲里编码与该 procedure 无关的 kernel ——
        这是旧引擎 `ISSUE_EVAL` 的做法。违反"只有一个 eval 在飞"会抛 `NotReadyError`。
        """
        self._runtime.issue(self.name, PROCS[proc], slot=slot)

    def reap_proc(self) -> float:
        """回收上一次 `issue_proc()` 的 eval，返回毫秒数。"""
        ms = self._runtime.reap()
        if ms < 0:
            raise NotReadyError("ANE 执行失败", program=self.name, reason=self._runtime.error())
        return ms

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
        if self._owned is not None:
            self._owned.free()
            self._owned = None
        # 用的是重新绑定而不是 `clear()`：借来的池第 0 套就是别的层的那张 dict，
        # 就地清空会把它们的面一起搬走。
        self._surfaces = {}
        self._sets = []
