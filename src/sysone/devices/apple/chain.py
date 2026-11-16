"""Qwen3.5 形状的分层前向链：ANE 跑三段程序，numpy 只跑 norm / RoPE / 注意力 / GDN。

`SplitChain` 不认识具体 checkpoint：形状来自家族递来的 `WeightPlan`（经
`Geometry.from_descriptor`），张量名前缀、norm 偏移、RoPE 角基也来自它。两个出货模型
（mude 2B 的 20 层、h2o 4B 的 32 层）走的是同一条链。

M2.2 第 2 片先求打通与逐层对齐，不做性能优化：全注意力层与线性注意力层的"热点"仍是
numpy，第 3 片再换 Metal。精度路径与旧引擎一致（int8 权重 + fp16 累加），所以与 torch
bf16 参照不会逐位相同；链路里也不能出现 torch（`src/` 只用 numpy）。
"""

from __future__ import annotations

import tempfile
import time
from itertools import count
from pathlib import Path
from typing import Sequence

import numpy as np

from ...domain.attention import AttentionRole
from ...errors import ConfigError, NotReadyError
from ...models.descriptor import ModelDescriptor
from ...models.layer_plan import WeightPlan
from ...weights.checkpoint import WeightStore, int8_parts, open_checkpoint
from ...weights.packing.ane_blob import build as build_blob, build_quantized
from ...weights.store import SafeTensorFile
from .blobplan import ane_layer_quantized, ane_layer_weights
from .compile import (
    S_NATIVE,
    AneLayer,
    ProgramFiles,
    SurfaceSet,
    cached_program,
    layer_mil,
    write_program,
)
from .geometry import Geometry
from .metal import MetalRuntime
from .pipeline import MudePipeline, pool_slots
from .residency import Residency, targets_for
from .schedule import blocks

_CHAIN_IDS = count()
from .ops import (
    apply_rotary,
    attention,
    causal_depthwise_conv,
    delta_rule,
    rms_norm,
    rms_norm_gated,
    sigmoid,
    softplus,
    softmax,
)
from .runtime import AneRuntime


def _rms(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """`[hidden, s]` 布局的 RMSNorm：通道在前，所以转成 `[s, hidden]` 再复用 `ops.rms_norm`。"""
    return rms_norm(x.T, weight).T


def _place(sequence_first: np.ndarray, hidden: int, s: int) -> np.ndarray:
    """`[L, hidden]` 的行序列放进 ANE 要求的 `[hidden, s]` 通道优先布局，其余列清零。"""
    out = np.zeros((hidden, s), np.float32)
    out[:, : sequence_first.shape[0]] = np.asarray(sequence_first, np.float32).T
    return out


def _rope_table(theta: float, s: int, rotary_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """位置 0..s-1 的 RoPE 正余弦表，各 `[s, rotary_dim/2]`。

    文本 RoPE 的 cos/sin 两个半区相同（`Qwen3_5TextRotaryEmbedding` 里 `cat(freqs, freqs)`），
    而 `ops.apply_rotary` 按前半/后半配对，所以这里只留前一半的频率。
    """
    frequencies = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    angles = np.arange(s, dtype=np.float64)[:, None] * frequencies[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class SplitChain:
    """分层链：`forward()` 给出前 `options` 个码的概率，`logits()` 给全部读出码的原始 logits。

    `s` 是 ANE 程序的静态列宽，prompt 长度必须不超过它；一层的三段 ANE 程序在 `load()` 里
    编译并常驻，`close()` 统一卸载。参数全部 fp32（numpy 一侧）与 fp16（surfaces）混合，
    与旧引擎的数值路径一致。
    """

    def __init__(
        self,
        *,
        runtime: AneRuntime,
        layers: list[AneLayer],
        params: list[dict[str, np.ndarray]],
        roles: list[AttentionRole],
        store: WeightStore,
        embed_name: str,
        final_norm: np.ndarray,
        readout: np.ndarray | None,
        readout_kind: str,
        temperature: float,
        cos: np.ndarray,
        sin: np.ndarray,
        geometry: Geometry,
        s: int,
        width: int,
        own_runtime: bool = True,
        pipeline: object | None = None,
        pool: list[SurfaceSet] | None = None,
        own_metal: bool = False,
        metal_runtime: object | None = None,
        load_s: float = 0.0,
        compile_s: float = 0.0,
    ) -> None:
        self._load_s = float(load_s)
        self._compile_s = float(compile_s)
        self._runtime = runtime
        self._own_runtime = own_runtime
        self._pipeline = pipeline
        self._pool = pool
        self._own_metal = own_metal
        self._metal_runtime = metal_runtime
        self._layers = layers
        self._params = params
        self._roles = roles
        self._store = store
        self._embed_name = embed_name
        self._final_norm = final_norm
        self._readout = readout
        self._readout_kind = str(readout_kind)
        self._temperature = float(temperature)
        self._cos = cos
        self._sin = sin
        # 形状只有一个来源：几何对象。它同时被 MIL 生成、缓冲分配与下面这些 numpy 段用，
        # 抄成局部标量就会有两份真相（GDN 的 v 头数就曾经这样被写死成 k 头数）。
        self._geometry = geometry
        self._s = int(s)
        # GPU 侧缓冲的列数（= 一轮要算的 token 总数）；numpy 链把它当画布宽度用。
        self._width = int(width)
        self._closed = False

    # ------------------------------------------------------------------ 装载耗时

    @property
    def load_s(self) -> float:
        """这次装载花了几秒，**不含**产物编译的那部分。"""
        return self._load_s

    @property
    def compile_s(self) -> float:
        """这次装载里有多少秒花在**从头构建产物**上；产物全部命中缓存时是 0。

        它和 `load_s` 是两段、不重叠：冷产物目录上编译是分钟级、装载只有一秒量级，
        第二次启动反过来。分开报是为了让"首启慢"这件事能归因到编译（`plan.md` R6）。
        """
        return self._compile_s

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        weights: WeightPlan,
        device: str = "ane",
        s: int = S_NATIVE,
        width: int | None = None,
        quantize: bool = True,
        scratch_dir: str | Path | None = None,
        runtime: AneRuntime | None = None,
        metal: bool = False,
        resident: int | None = None,
    ) -> "SplitChain":
        """编译 20 层程序并从 `model_dir` 读权重。

        权重容器两种都收：bf16 的 `model.safetensors` 单文件，或 row-int8 目录（`__q` 码 +
        fp32 标度 + `int8_manifest.json`，4B 的 h2o 是这种）。选择在
        `weights.checkpoint.open_checkpoint` 里做，`read` 一律给 float32，所以下面这些
        读权重的地方都不认识容器。

        `device` 目前只有 `"ane"`（这条链就是 ANE 链）；`scratch_dir` 放 MIL/blob 产物，
        默认在系统临时目录，产物按内容哈希复用。`metal=True` 时额外建一条 GPU 流水线
        （`MudePipeline`）：norm / 注意力 / GDN 走 Metal 核，中间量全程留在 IOSurface 上，
        不再经 numpy 中转；此时 `logits()` 不提供逐层 trace。

        `s` 是 ANE program 的列数（桶宽，受 program 的 S 上限约束）；`width` 是 GPU 侧缓冲
        的列数，默认等于 `s`（一桶跑完），16k 这类长序列要 `width = 16384, s = 4096`
        （四段 ANE、注意力一次看 16k 键）。

        `resident` 是待命常驻的层数，`None` 表示 20 层全装（今天的行为）。给值时不能小于
        最大块长 —— `blocks()` 里一个块内的层是同时在飞的，整块必须常驻；窗口在块边界换窗、
        重装在线程上预取，省掉窗口之外的 ANE 权重，代价是每层约 66 ms（见 `residency.py`）。

        `weights` 是家族递过来的 `WeightPlan`：形状（`descriptor`，经
        `Geometry.from_descriptor` 投影）、张量名前缀、norm 的**存法偏移**（真值 `w + offset`；
        两个出货 checkpoint 都是零中心存、`+1`）、RoPE 角基。这四项是"这份 checkpoint 怎么
        存的"，设备层不认识任何具体 checkpoint，所以它**没有默认值**：给了错的偏移不会崩，
        只会让每一层都差一点点 —— 默认零正是让"忘了给"看起来像跑通的写法。

        读出可以是 `None`：`descriptor.readout.kind` 不是 `code_rows` 的模型（h2o 是
        `pooled`）没有码表，这条链就只提供逐层 hidden，`logits()` 会报
        `NotReadyError` 指出来（pooled 读出还没实现，票据 15 的 G6）。
        """
        started = time.perf_counter()
        # 产物从头构建的那一段单独累计：它是"首启 vs 二次启动"差异的全部来源。
        compile_s = 0.0
        if device != "ane":
            raise ValueError(f"这条链只有 ANE 设备，收到 {device!r}")
        root = Path(model_dir)
        prefix = weights.prefix
        # 设备侧的形状只从几何取（architecture.md §13 R1）：MIL 生成、缓冲分配、kernel 网格。
        # 几何从 descriptor 投影，descriptor 由家族的 parser 从 config 填 —— 设备层不再自己
        # 读 config 拼形状（那会有第二份真相，且 h2o 的 v 头曾被抄成 k 头）。
        geometry = Geometry.from_descriptor(weights.descriptor)
        hidden, ffn = geometry.hid, geometry.ffn
        readout = _readout(root, weights.descriptor)
        roles = list(weights.descriptor.layer_types)
        # 窗口在装载之前算：比块还短时在这里就报错，不必先装完 20 层再失败。
        targets: tuple[tuple[int, ...], ...] = ()
        if resident is not None:
            if not metal:
                raise ConfigError("常驻窗口只在 Metal 流水线上有意义", resident=int(resident))
            if int(resident) < len(roles):
                targets = targets_for(blocks(roles), len(roles), int(resident))
        width = int(s if width is None else width)
        if width < s:
            raise ValueError(f"缓冲宽度 {width} 小于桶宽 {s}：桶不能跨在缓冲之外")
        # 待命窗口：只有这几层留在设备上。窗口外的层在装载时**装一次就卸**（下面循环里），
        # 目的是让 aned 把 lowering 与身份哈希做好（之后的换窗是 `load_fast` ~1 ms），而不是
        # 让它们常驻 —— 32 层同时落地是 3.6 GB 的瞬态峰值（票据 15 §36）。
        standby: frozenset[int] | None = frozenset(targets[0]) if targets else None

        store = open_checkpoint(root)
        scratch = (Path(scratch_dir) if scratch_dir is not None
                   else Path(tempfile.gettempdir()) / "sysone-mude-chain")
        # dylib 的 program 缓存按名字键，进程内允许多条链共存（对拍时 numpy 与 Metal 各一条），
        # 所以每条链的 program 名必须带自己的编号。
        chain_id = next(_CHAIN_IDS)
        own_runtime = runtime is None
        runtime = runtime if runtime is not None else AneRuntime()
        pipeline = None
        metal_runtime = None
        layers: list[AneLayer] = []
        # 表面池归这条链所有（Metal 形态才有）：20 层共用一个池，因为一次只有一条 ANE 固件
        # 队列、一次只有一个 eval 在飞 —— 每层自带一套在 4096 桶下是 3.57 GB 对 0.20 GB。
        # 池里放几套取决于同时有几条链在跑，今天是一条；双链要 NCH × block 层数（第 3 步）。
        # `close()` 里必须在卸载 program **之后**才 `free()`：每次 `attach` 都给面加了引用。
        pool: list[SurfaceSet] | None = None
        if metal:
            pool = [SurfaceSet(runtime, hidden=hidden, core_out=geometry.core_out,
                               cout=max(geometry.cout_full, geometry.cout_gdn), s=s)
                    for _ in range(pool_slots(roles))]
        try:
            params = [_layer_params(store, prefix, index, role, weights.norm_offset)
                      for index, role in enumerate(roles)]
            # 产物目录按 (模型名, 桶宽) 分家：不同桶宽的产物互不覆盖 —— 否则一次 `s=256`
            # 的测试就会把 `s=1024` 的缓存顶掉，下一次装载要多花 9 s 重算权重与量化。
            program_root = scratch / root.name / f"s{s}"
            for index, role in enumerate(roles):
                cout = geometry.cout_full if role is AttentionRole.FULL else geometry.cout_gdn
                # 全注意力层把投影拆成 q 段 + kv 段（票据 06）：q 只活一个瓦片，`qg` 才能成环。
                split = (geometry.qg_rows, geometry.kv_rows) if role is AttentionRole.FULL else None
                directory = program_root / f"layer{index:02d}"
                source = _artifact_source(
                    store, prefix=prefix, layer=index, role=role, cout=cout,
                    core_out=geometry.core_out, hidden=hidden, ffn=ffn, s=s, quantize=quantize,
                )
                files = cached_program(directory, source=source, s=s)
                if files is None:
                    building = time.perf_counter()
                    files = _build_layer_program(
                        store, prefix=prefix, directory=directory, index=index, role=role,
                        cout=cout, core_out=geometry.core_out, hidden=hidden, ffn=ffn, s=s,
                        quantize=quantize, source=source, split=split,
                    )
                    compile_s += time.perf_counter() - building
                layer = AneLayer(runtime, files, name=f"mude{chain_id}_layer_{index}",
                                 hidden=hidden, core_out=geometry.core_out,
                                 cout=cout, ffn=ffn, s=s, pool=pool,
                                 split_rows=split)
                layers.append(layer)
                if standby is not None and index not in standby:
                    # 装载只为 aned 的 lowering 与身份哈希（下一次是 `load_fast`），装完就还回去：
                    # 32 层一起落地是 3.6 GB 的瞬态，而它们要到后面的块才用得上。
                    layer.unload()
            final_norm = _norm(store.read(f"{prefix}.norm.weight"), weights.norm_offset)
            temperature = float(weights.descriptor.readout.temperature or 1.0)
            cos, sin = _rope_table(weights.rope_theta, width, geometry.rotary_dim)
            if metal:
                # 待命集已经由上面的循环保证：启动时刻在设备上的就是 `targets[0]` 那几层，
                # 所以这里传 `resident` 而不是让窗口再卸一遍。
                residency = Residency(layers, targets, resident=targets[0]) if targets else None
                metal_runtime = MetalRuntime(geometry=geometry)
                pipeline = MudePipeline(
                    runtime=metal_runtime, layers=layers, roles=roles, params=params,
                    cos=cos, sin=sin, geometry=geometry, bucket=s, width=width, pool=pool,
                    residency=residency,
                )
        except BaseException:
            if pipeline is not None:
                pipeline.close()
            if metal_runtime is not None:
                metal_runtime.close()
            for layer in layers:
                layer.close()
            for surface_set in pool or ():
                surface_set.free()
            if own_runtime:
                runtime.close()
            store.close()
            raise

        return cls(
            runtime=runtime, layers=layers, params=params, roles=roles, store=store,
            embed_name=f"{prefix}.embed_tokens.weight", final_norm=final_norm,
            readout=readout, readout_kind=weights.descriptor.readout.kind,
            temperature=temperature, cos=cos, sin=sin,
            geometry=geometry, s=s, width=width,
            own_runtime=own_runtime, pipeline=pipeline, pool=pool,
            own_metal=metal_runtime is not None, metal_runtime=metal_runtime,
            load_s=time.perf_counter() - started - compile_s, compile_s=compile_s,
        )

    # ------------------------------------------------------------------ 前向

    def _ids_and_length(self, tokens: Sequence[int], prefix: int) -> tuple[np.ndarray, int]:
        """三条前向入口共用的门禁：token 数组、长度，与 `prefix` 的越界检查。"""
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        length = int(ids.size)
        if length == 0:
            raise ValueError("prompt 至少要有一个 token")
        if length > self._width:
            raise ValueError(
                f"prompt 长度 {length} 超过链的缓冲宽度 width={self._width}；"
                "用更大的 width 重新 load")
        if not 0 <= prefix <= length:
            raise ValueError(f"prefix 必须在 0..{length}，收到 {prefix}")
        return ids, length

    def logits(self, tokens: Sequence[int], *, trace: list[np.ndarray] | None = None,
               prefix: int = 0, causal: bool = False,
               debug: dict | None = None) -> np.ndarray:
        """整条 prompt 走 20 层，返回 `[rows]` 原始 logits（rows = 读出表的行数）（fp32，未除温度）。

        `trace` 非空时把每层输出（最后一条 token 的 `[hidden]`）按顺序追加进去，
        供探针与 torch 的逐层 hidden 对比。`prefix` 是 `prefix_noncausal` 的公共前缀长度
        （厂商 `Prepared.prefix()` 给出的那个数）：全注意力层的掩码是
        `(q >= prefix) | (k < prefix)`，额外把 padding 列当不可见。单问题、无 padding 时
        prefix=0 就退化为整段双向。

        `causal=True` 换成三角掩码 `k <= q`。它与 `prefix` 是**两个轴**：causal 下每一行
        只看自己之前（含自己），所以 `prefix` 不再改变掩码 —— 前缀行本来就是行内三角的一部
        分，尾巴行看到的 `[0, prefix)` 也在 `k <= q` 之内。

        只有最后一条 token 的行进读出，所以 Metal 那条路直接走 `last_hidden()`（不做整段
        读回）：16k 上整段读回归一要 87 ms，末行是 0.02 ms。
        """
        ids, length = self._ids_and_length(tokens, prefix)
        # 码表先要：没有码表的模型（h2o 是 pooled）连前向都不必跑，报错也不必等到算完。
        # 它不影响 `hidden()` / `last_hidden()` —— 那两条出口不需要码表。
        readout = self._require_readout()
        if self._pipeline is not None:
            if trace is not None or debug is not None:
                raise ValueError("Metal 流水线不提供逐层 trace 或 debug（中间量留在面上）",
                                 hint="用 metal=False 的 numpy 链做逐层对拍")
            return readout @ self.last_hidden(tokens, prefix=prefix, causal=causal)
        x = _place(self._store.read_rows(self._embed_name, ids), self._geometry.hid, self._width)
        if trace is not None:
            # 先记 embedding 输出，这样 trace[i] 是"第 i 层的输入"（i=0 即 embedding），
            # 任何一层跑偏都能定位到该层本身而不是前一层。
            trace.append(x[:, length - 1].copy())
        cos, sin = self._cos[:length], self._sin[:length]
        for index, layer in enumerate(self._layers):
            params = self._params[index]
            layer_debug = debug if (debug is not None and index == 0) else None
            if self._roles[index] is AttentionRole.FULL:
                x = self._full_layer(layer, params, x, length, cos, sin, prefix, causal)
            else:
                x = self._linear_layer(layer, params, x, length, layer_debug)
            if trace is not None:
                trace.append(x[:, length - 1].copy())
        final = rms_norm(x.T, self._final_norm)[length - 1]
        return readout @ final

    def hidden(self, tokens: Sequence[int], *, prefix: int = 0,
               causal: bool = False, trace: list[np.ndarray] | None = None) -> np.ndarray:
        """整条 prompt 走完全部层，返回 `[length, hidden]` 的最终归一化 hidden。

        这是**没有码表读出的模型**（h2o 的 `pooled`）也能用的出口，也是厂商路径对拍的抓手：
        G6 接上之前，"h2o 的链算得对不对"只能拿它比。`logits()` 就是它的最后一行过码表。

        整段读回只在逐位置对拍与 trace 时需要；只要末行请用 `last_hidden()`（出货的
        `pooled` 路径走那条）。
        """
        ids, length = self._ids_and_length(tokens, prefix)
        if self._pipeline is not None:
            if trace is not None:
                raise ValueError("Metal 流水线不提供逐层 trace；用 metal=False 的 numpy 链")
            self._pipeline.run(self._store.read_rows(self._embed_name, ids), length=length,
                               prefix=prefix, causal=causal)
            # 与 numpy 那条路同形：`[length, hidden]`（末位就是读出要的那一行）。以前这条
            # 分支只返回末位的一维向量，与自己的 docstring 不符 —— 逐位置对拍与
            # `PooledRunner` 的形状自检都会撞上。
            return rms_norm(self._pipeline.hidden_rows(length), self._final_norm)
        x = _place(self._store.read_rows(self._embed_name, ids), self._geometry.hid, self._width)
        if trace is not None:
            trace.append(x[:, length - 1].copy())
        cos, sin = self._cos[:length], self._sin[:length]
        for index, layer in enumerate(self._layers):
            params = self._params[index]
            if self._roles[index] is AttentionRole.FULL:
                x = self._full_layer(layer, params, x, length, cos, sin, prefix, causal)
            else:
                x = self._linear_layer(layer, params, x, length)
            if trace is not None:
                trace.append(x[:, length - 1].copy())
        return rms_norm(x.T, self._final_norm)[:length]

    def last_hidden(self, tokens: Sequence[int], *, prefix: int = 0,
                    causal: bool = False) -> np.ndarray:
        """最后一条 token 的最终归一化 hidden（`[hidden]` fp32）。

        `hidden(tokens)[-1]` 的那一行，但不读回整段：16k 上整段读回 + 归一 87 ms，只取末行
        0.02 ms（票据 08 §3）。`pooled` 的读出与 `logits()` 都只要这一行。
        """
        ids, length = self._ids_and_length(tokens, prefix)
        if self._pipeline is not None:
            self._pipeline.run(self._store.read_rows(self._embed_name, ids), length=length,
                               prefix=prefix, causal=causal)
            return rms_norm(self._pipeline.last_hidden(length)[None, :], self._final_norm)[0]
        # numpy 那条路不是热点：整段本来就是一小块 host 内存，直接复用 `hidden()`。
        return self.hidden(tokens, prefix=prefix, causal=causal)[-1]

    def forward(self, tokens: Sequence[int], *, options: int, prefix: int = 0,
                causal: bool = False) -> np.ndarray:
        """前 `options` 个码的 softmax 概率（选项顺序 = `codes` 顺序）。"""
        return self._probabilities(self.logits(tokens, prefix=prefix, causal=causal), options)

    def forward_shared(
        self, prefix: Sequence[int], tails: Sequence[Sequence[int]], *, options: Sequence[int],
        causal: bool = False,
    ) -> list[np.ndarray]:
        """一个请求内多题共享前缀：前缀只算一遍，每题只跑自己那条尾巴。

        与 `forward` 同一个约定（第 k 题取前 `options[k]` 个码的 softmax）。`tails[k]` 是第 k 题
        的**续写** token（不含前缀）。这条路径只有 Metal 链提供：numpy 链没有零拷贝流水线，
        前缀的 K/V 与隐藏态没有落脚点。

        一个请求里所有题的前缀都相同（`state` 只有一个），所以复用单位是"这个请求"，请求返回
        就释放 —— 不做跨请求缓存。
        """
        if len(options) != len(tails):
            raise ValueError(f"options 与尾巴条数不一致：{len(options)} vs {len(tails)}")
        logits = self.logits_shared(prefix, tails, causal=causal)
        return [self._probabilities(row, int(count)) for row, count in zip(logits, options)]

    def logits_shared(
        self, prefix: Sequence[int], tails: Sequence[Sequence[int]], *, causal: bool = False
    ) -> list[np.ndarray]:
        """共享前缀的整条前向：返回每题的 `[rows]` 原始 logits（未除温度）。"""
        if self._pipeline is None:
            raise ValueError("共享前缀只有 Metal 链提供（load 时用 metal=True）")
        ids = np.asarray(prefix, dtype=np.int64).reshape(-1)
        length = int(ids.size)
        if length == 0:
            raise ValueError("共享前缀至少要有一个 token")
        if length > self._width:
            raise ValueError(
                f"共享前缀 {length} 超过链的缓冲宽度 width={self._width}；用更大的 width 重新 load")
        rows: list[np.ndarray] = []
        for tail in tails:
            tail_ids = np.asarray(tail, dtype=np.int64).reshape(-1)
            if tail_ids.size == 0:
                raise ValueError("尾巴至少要有一个 token")
            if length + int(tail_ids.size) > self._width:
                raise ValueError(
                    f"前缀 + 尾巴 {length + int(tail_ids.size)} 超过链的缓冲宽度 "
                    f"width={self._width}；用更大的 width 重新 load")
            rows.append(tail_ids)
        hidden = self._pipeline.run_shared(
            self._store.read_rows(self._embed_name, ids),
            [self._store.read_rows(self._embed_name, tail) for tail in rows],
            prefix=length, causal=causal,
        )
        out: list[np.ndarray] = []
        for row in hidden:
            final = rms_norm(row[None, :], self._final_norm)[0]
            out.append(self._require_readout() @ final)
        return out

    def _require_readout(self) -> np.ndarray:
        """码表读出；没有就明确报错（`pooled` 读出还没接，见 `_readout`）。"""
        if self._readout is None:
            raise NotReadyError(
                "这个模型没有码表读出，`logits()` 用不了",
                readout_kind=self._readout_kind,
                hint="h2o 的 pooled 读出 + selector 还没接（票据 15 G6）；"
                     "在此之前只能用 `hidden()` / `last_hidden()` 做对拍",
            )
        return self._readout

    def _probabilities(self, logits: np.ndarray, options: int) -> np.ndarray:
        """前 `options` 个码的 softmax 概率（选项顺序 = `codes` 顺序）。

        与厂商 `D3.probabilities` 的约定一致：越界码行在 softmax 前置 -inf（exp 为 0），
        因此等价于只在前 `options` 行上归一化。
        """
        rows = 0 if self._readout is None else int(self._readout.shape[0])
        if not 1 <= options <= rows:
            raise ValueError(f"options 必须在 1..{rows}，收到 {options}")
        return softmax(logits[:options] / self._temperature)

    def _full_layer(
        self,
        layer: AneLayer,
        params: dict[str, np.ndarray],
        x: np.ndarray,
        length: int,
        cos: np.ndarray,
        sin: np.ndarray,
        prefix: int,
        causal: bool,
        debug: dict | None = None,
    ) -> np.ndarray:
        """全注意力层：q|gate 交错投影 → q/k 各做 head_dim 的 RMSNorm → 半维 RoPE → GQA → 门控。"""
        xn = _rms(x, params["input_norm"])
        projected, _ = layer.project(xn)
        p = projected.T.astype(np.float32)                 # [s, 5120]，多出的列是 padding
        heads, kv_heads, head_dim = self._geometry.q_heads, self._geometry.kv_heads, self._geometry.head_dim
        q_rows = heads * head_dim * 2
        k_rows = kv_heads * head_dim
        # q_proj 的行是每个 head 的 [q(head_dim) | gate(head_dim)] 交错排列。
        qg = p[:length, :q_rows].reshape(length, heads, 2, head_dim)
        query = rms_norm(qg[:, :, 0, :], params["q_norm"])
        gate = qg[:, :, 1, :].reshape(length, heads * head_dim)
        key = rms_norm(p[:length, q_rows:q_rows + k_rows].reshape(length, kv_heads, head_dim),
                       params["k_norm"])
        value = p[:length, q_rows + k_rows:q_rows + 2 * k_rows].reshape(length, kv_heads, head_dim)
        if debug is not None:
            # RoPE 会就地覆盖 `query` / `key`，所以旋转前的值先留一份。
            debug.update(xn=xn, p=p, query_normed=query.copy(), gate=gate,
                         key_normed=key.copy(), value=value)
        query = apply_rotary(query, cos, sin, rotary_dim=self._geometry.rotary_dim)
        key = apply_rotary(key, cos, sin, rotary_dim=self._geometry.rotary_dim)
        repeat = heads // kv_heads
        key = np.repeat(key, repeat, axis=1)
        value = np.repeat(value, repeat, axis=1)
        # k/v 已裁到真实 `length` 行，padding 列进不了注意力，所以掩码就是厂商
        # `prefix_noncausal` 的 `(q >= prefix) | (k < prefix)`，形状 `[length, length]`；
        # `causal` 换成 `k <= q`（它已经包含 `k < prefix`，因为前缀行都排在尾巴之前）。
        queries = np.arange(length)[:, None]
        keys = np.arange(length)[None, :]
        mask = keys <= queries if causal else (queries >= prefix) | (keys < prefix)
        context = attention(query, key, value, mask).reshape(length, heads * head_dim)
        if debug is not None:
            debug.update(query=query, key=key, context=context)
        y, _ = layer.output(_place(context * sigmoid(gate), self._geometry.core_out, self._s), x)
        xn2 = _rms(y, params["post_norm"])
        if debug is not None:
            debug.update(y=y, xn2=xn2)
        return layer.feed_forward(xn2, y)[0]

    def _linear_layer(
        self,
        layer: AneLayer,
        params: dict[str, np.ndarray],
        x: np.ndarray,
        length: int,
        debug: dict | None = None,
    ) -> np.ndarray:
        """线性注意力层（GDN）：depthwise 因果卷积 → q/k/v → chunked delta rule → 门控 norm。

        投影的行序是 `[qkv(conv_dim) | z(value_dim) | b(l_v_heads) | a(l_v_heads)]`（`b` 在 `a`
        之前，见 `e8_common.h:29`）。三段的宽度**都来自几何**：2B 上 `qk_dim = value_dim = 2048`
        所以 q/k/v 三段等长、`z` 也恰好 2048，按 `hidden` 写死碰巧对；4B 是 qk 2048、v 4096，
        写死就会把 v 的后 2048 行读成 z。
        """
        geom = self._geometry
        xn = _rms(x, params["input_norm"])
        projected, _ = layer.project(xn)
        p = projected.T.astype(np.float32)                 # [s, cout_gdn]
        mixed = causal_depthwise_conv(p[:length, :geom.conv_dim], params["conv"])  # 已过 silu
        qk, v_dim = geom.qk_dim, geom.value_dim
        query = mixed[:, :qk].reshape(length, geom.l_k_heads, geom.l_k_dim)
        key = mixed[:, qk:2 * qk].reshape(length, geom.l_k_heads, geom.l_k_dim)
        value = mixed[:, 2 * qk:].reshape(length, geom.l_v_heads, geom.l_v_dim)
        head = geom.l_v_heads                              # `z` 与 `b` / `a` 都按 v 头走
        z = p[:length, geom.conv_dim:geom.conv_dim + v_dim].reshape(length, head, geom.l_v_dim)
        base = geom.conv_dim + v_dim
        b = p[:length, base:base + head]
        a = p[:length, base + head:base + 2 * head]
        beta = sigmoid(b)
        g = -np.exp(params["a_log"]) * softplus(a + params["dt_bias"])
        core = delta_rule(query, key, value, g, beta,
                          repeat=head // geom.l_k_heads)   # [L, v_heads, v_dim]
        # 门控 norm 是按头做的：参考实现在 [S*heads, dim] 上归一化，不能摊平成 [S, hidden]。
        gated = rms_norm_gated(core.reshape(length * head, geom.l_v_dim),
                               z.reshape(length * head, geom.l_v_dim),
                               params["norm"]).reshape(length, v_dim)
        y, _ = layer.output(_place(gated, geom.core_out, self._s), x)
        xn2 = _rms(y, params["post_norm"])
        if debug is not None:
            debug.update(xn=xn, projected=p, mixed=mixed,
                         z=p[:length, geom.conv_dim:geom.conv_dim + v_dim],
                         a=a, b=b, beta=beta, g=g, core=core, gated=gated, y=y, xn2=xn2)
        return layer.feed_forward(xn2, y)[0]

    # ------------------------------------------------------------------ 收尾

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pipeline is not None:
            # 先解绑 GPU 侧的包装句柄，再卸载 program，最后才释放池里的面。
            self._pipeline.close()
            self._pipeline = None
        if self._own_metal and self._metal_runtime is not None:
            self._metal_runtime.close()
            self._metal_runtime = None
        for layer in self._layers:
            layer.close()
        self._layers.clear()
        for surface_set in self._pool or ():
            surface_set.free()
        self._pool = None
        if self._own_runtime:
            self._runtime.close()
        self._store.close()


# ---------------------------------------------------------------------- 产物

# 一层的产物目录里已经有了"就是这份权重、这个几何、这个桶宽、这个量化"的证明时，
# 就不再读 fp16 权重、也不再量化 —— 这两段在本机是 8.6 s，而装载本身只要 0.7 s。
def _artifact_source(
    store: WeightStore,
    *,
    prefix: str,
    layer: int,
    role: AttentionRole,
    cout: int,
    core_out: int,
    hidden: int,
    ffn: int,
    s: int,
    quantize: bool,
) -> dict:
    """产物的身份：权重容器的指纹 + 几何 + 量化开关 + `ARTIFACT_VERSION`。

    `store.fingerprint()` 是"这份权重没换过"的证据：大小与 mtime，不算内容哈希（哈希要把
    1.5 GB 读一遍，而读一遍正是缓存要省掉的那一段）。容器形态也进指纹 —— 同一份权重存成
    bf16 单文件与存成 row-int8，读出来的数值不是逐位相同的（后者多了 int8 往返），
    产物就不该互相认领。
    """
    return {
        "weights": store.fingerprint(),
        "prefix": str(prefix),
        "layer": int(layer),
        "type": str(role),
        "cout": int(cout),
        "core_out": int(core_out),
        "hidden": int(hidden),
        "ffn": int(ffn),
        "s": int(s),
        "quantize": bool(quantize),
    }


def _entry_array(entry: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """一条 ANE 权重的形状来源：未量化时是矩阵本身，量化后看码那一半（标度是一维向量）。"""
    return entry[0] if isinstance(entry, tuple) else entry


def _build_layer_program(
    store: WeightStore,
    *,
    prefix: str,
    directory: Path,
    index: int,
    role: AttentionRole,
    cout: int,
    core_out: int,
    hidden: int,
    ffn: int,
    s: int,
    quantize: bool,
    source: dict,
    split: tuple[int, int] | None = None,
) -> ProgramFiles:
    """读这一层的权重、装 blob、生成 MIL 并落盘（缓存未命中的那一路）。

    码从哪来取决于容器：row-int8 检查点自带，bf16 检查点在这里现量化。两条路都从
    `ane_layer_segments` 的同一张条目表出发，所以条目名、行拼顺序、列切位置只有一份。

    `split=(q_rows, kv_rows)` 时把整块投影**按行切成两段**（全注意力层，票据 06）：int8 是按
    输出行量化的，切行不动 scale，所以两段的 codes 与未切时逐字节相同 —— blob 与 ANE 常驻
    权重一分不多，换到的是 `qg` 可以成环。
    """
    if quantize:
        # row-int8 检查点自带 int8 码，直接用；bf16 检查点没有码，这条路上现量化。
        entries: dict = ane_layer_quantized(
            store.read, lambda name: int8_parts(store, name),
            layer=index, layer_type=role, hidden=hidden, ffn=ffn, prefix=prefix,
        )
        actual = int(_entry_array(entries["proj"]).shape[0])
    else:
        entries = ane_layer_weights(store.read, layer=index, layer_type=role,
                                    hidden=hidden, ffn=ffn, prefix=prefix)
        actual = int(entries["proj"].shape[0])
    if actual != cout:
        # `cout` 来自写死在 MSL 里的几何；权重形状与它对不上说明这个 checkpoint 不是
        # 这套几何编译的，继续下去会得到一个形状对、语义错的 program。
        raise ValueError(
            f"第 {index} 层的 proj 行数 {actual} 与几何给出的 cout={cout} 不一致"
        )
    out_shape = tuple(int(n) for n in _entry_array(entries["o"]).shape)
    if out_shape != (hidden, core_out):
        # `o` 的行是 hidden（输出），列是 core_out（输入）。列数对不上说明几何与这份
        # checkpoint 不是一套的 —— 继续下去 ANE 会按 core_out 行去读一个更窄的矩阵，
        # 得到的是"形状合法、语义错"的答案。
        raise ValueError(
            f"第 {index} 层的 o 形状 {out_shape} 与几何的 (hidden, core_out)="
            f"({hidden}, {core_out}) 不一致"
        )
    if split is not None:
        q_rows, kv_rows = split
        if q_rows + kv_rows != actual:
            raise ValueError(
                f"第 {index} 层：拆投影的两段行数 {q_rows}+{kv_rows} 与 proj 的 {actual} 不符"
            )
        proj = entries["proj"]
        rest = {key: value for key, value in entries.items() if key != "proj"}
        if quantize:
            codes, scales = proj
            entries = {**rest, "proj_q": (codes[:q_rows], scales[:q_rows]),
                       "proj_kv": (codes[q_rows:], scales[q_rows:])}
        else:
            entries = {**rest, "proj_q": proj[:q_rows], "proj_kv": proj[q_rows:]}
    if quantize:
        blob, blob_entries = build_quantized(entries)
    else:
        blob, blob_entries = build_blob(entries, quantize=False)
    mil = layer_mil(hidden=hidden, core_out=core_out, cout=cout, ffn=ffn, s=s,
                    entries=blob_entries, split=split is not None)
    return write_program(
        directory,
        mil=mil,
        blob=blob,
        s=s,
        meta={"layer": index, "type": str(role), "cout": cout, "split": split},
        source=source,
    )


# ---------------------------------------------------------------------- 权重


def _readout(root: Path, descriptor: ModelDescriptor) -> np.ndarray | None:
    """码表读出：`readout.safetensors` 的 `[rows, hidden]`，转 fp32。

    只有 `code_rows` 有这张表。`pooled`（h2o）的读出是厂商的分池头 + selector，还没接
    （票据 15 的 G6）—— 这时返回 `None`，链仍然能装、能跑逐层 hidden，`logits()` 会明确报错，
    而不是拿一张不存在的表算出一堆看似正常的数。
    """
    plan = descriptor.readout
    if plan.kind != "code_rows":
        return None
    if not plan.file or not plan.weight:
        raise ConfigError("code_rows 读出缺少 file / weight", kind=plan.kind)
    handle = SafeTensorFile(root / plan.file)
    try:
        weight = handle.read(plan.weight)
    finally:
        handle.close()
    table = np.asarray(weight, np.float32)
    if table.ndim != 2 or (plan.rows is not None and table.shape[0] != plan.rows):
        raise ConfigError(
            "读出表的形状与 descriptor 不符",
            shape=list(table.shape), rows=plan.rows, file=plan.file,
        )
    return table


def _norm(weight: np.ndarray, offset: float) -> np.ndarray:
    """norm 的真值：`w + offset`。偏移是 checkpoint 的存法，由家族给（见 `load`）。"""
    return np.asarray(weight, np.float32) + offset


def _layer_params(
    store: WeightStore, prefix: str, index: int, role: AttentionRole, norm_offset: float
) -> dict[str, np.ndarray]:
    """一层的 numpy 侧权重：各 norm、卷积、A_log / dt_bias。norm 的偏移由调用方给。"""
    base = f"{prefix}.layers.{index}"
    params = {
        "input_norm": _norm(store.read(f"{base}.input_layernorm.weight"), norm_offset),
        "post_norm": _norm(store.read(f"{base}.post_attention_layernorm.weight"), norm_offset),
    }
    if role is AttentionRole.FULL:
        attn = f"{base}.self_attn"
        params["q_norm"] = _norm(store.read(f"{attn}.q_norm.weight"), norm_offset)
        params["k_norm"] = _norm(store.read(f"{attn}.k_norm.weight"), norm_offset)
    else:
        attn = f"{base}.linear_attn"
        params["conv"] = np.asarray(store.read(f"{attn}.conv1d.weight"), np.float32).reshape(-1, 4)
        params["a_log"] = np.asarray(store.read(f"{attn}.A_log"), np.float32)
        params["dt_bias"] = np.asarray(store.read(f"{attn}.dt_bias"), np.float32)
        # 这个 norm 不跟 `norm_offset` 走：`RMSNormGated` 的初始化是全 1，这份 checkpoint 里
        # 它是按标准 RMSNorm 存的（零中心约定只覆盖上面那四个 HF norm），直接用。
        params["norm"] = np.asarray(store.read(f"{attn}.norm.weight"), np.float32)
    return params
