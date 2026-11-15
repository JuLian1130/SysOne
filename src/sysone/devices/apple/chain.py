"""mude 的 20 层前向链：ANE 跑三段程序，numpy 只跑 norm / RoPE / 注意力 / GDN。

M2.2 第 2 片先求打通与逐层对齐，不做性能优化：全注意力层与线性注意力层的"热点"仍是
numpy，第 3 片再换 Metal。精度路径与旧引擎一致（int8 权重 + fp16 累加），所以与 torch
bf16 参照不会逐位相同；链路里也不能出现 torch（`src/` 只用 numpy）。
"""

from __future__ import annotations

import json
import tempfile
from itertools import count
from pathlib import Path
from typing import Sequence

import numpy as np

from ...domain.attention import AttentionRole
from ...errors import ConfigError
from ...models.families.mude import CODE_ROWS, ane_layer_weights
from ...weights.packing.ane_blob import build as build_blob
from ...weights.store import SafeTensorFile
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


class MudeChain:
    """mude 20 层链：`forward()` 给出前 `options` 个码的概率，`logits()` 给 255 个原始 logits。

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
        store: SafeTensorFile,
        embed_name: str,
        final_norm: np.ndarray,
        readout: np.ndarray,
        temperature: float,
        cos: np.ndarray,
        sin: np.ndarray,
        hidden: int,
        s: int,
        width: int,
        heads: int,
        kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        linear_heads: int,
        linear_dim: int,
        own_runtime: bool = True,
        pipeline: object | None = None,
        pool: list[SurfaceSet] | None = None,
        own_metal: bool = False,
        metal_runtime: object | None = None,
    ) -> None:
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
        self._temperature = float(temperature)
        self._cos = cos
        self._sin = sin
        self._hidden = int(hidden)
        self._s = int(s)
        # GPU 侧缓冲的列数（= 一轮要算的 token 总数）；numpy 链把它当画布宽度用。
        self._width = int(width)
        self._heads = int(heads)
        self._kv_heads = int(kv_heads)
        self._head_dim = int(head_dim)
        self._rotary_dim = int(rotary_dim)
        self._linear_heads = int(linear_heads)
        self._linear_dim = int(linear_dim)
        self._closed = False

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str = "ane",
        norm_offset: float,
        s: int = S_NATIVE,
        width: int | None = None,
        quantize: bool = True,
        scratch_dir: str | Path | None = None,
        runtime: AneRuntime | None = None,
        metal: bool = False,
        resident: int | None = None,
    ) -> "MudeChain":
        """编译 20 层程序并从 `model_dir` 读权重。

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

        `norm_offset` 是 norm 权重的**存法偏移**：真值是 `w + offset`。mude 这份 checkpoint
        是零中心存的（`+1`；不补时第 0 层 cos 只有 0.53，见 `docs/ANE/README.md` §10.22），
        标准 RMSNorm 是 0。它由家族递给设备层（`mude.NORM_OFFSET`），设备层不认识任何具体
        checkpoint —— 所以这里**没有默认值**：默认零会让"忘了给"看起来像跑通了。
        """
        if device != "ane":
            raise ValueError(f"mude 链只有 ANE 设备，收到 {device!r}")
        root = Path(model_dir)
        text = _text_config(root)
        decision = _load_json(root / "decision_config.json")
        # `decision_config.json` 的 `attention_mode` 只是默认值（`mude.parse_config` 读它），
        # 掩码在 `forward`/`logits` 的每次调用里给：三种 `MaskMode` 这条链都能表达（M3 之前
        # causal 不行，因为核里没有三角项）。这里只取温度读出。

        hidden = int(text["hidden_size"])
        ffn = int(text["intermediate_size"])
        heads = int(text["num_attention_heads"])
        kv_heads = int(text["num_key_value_heads"])
        head_dim = int(text["head_dim"])
        rope = text.get("rope_parameters", {})
        theta = float(rope.get("rope_theta", 10_000_000.0))
        rotary_dim = int(head_dim * float(rope.get("partial_rotary_factor", 1.0)))
        linear_heads = int(text["linear_num_key_heads"])
        linear_dim = int(text["linear_key_head_dim"])
        # GDN 的 q/k 与 v 在这里同头数同头维，所以不用 repeat_interleave（参考实现里那一步恒等）。
        if int(text.get("linear_num_value_heads", linear_heads)) != linear_heads:
            raise ValueError("GDN 的 v 头数与 k 头数不同，当前实现没做 repeat_interleave")
        if int(text.get("linear_value_head_dim", linear_dim)) != linear_dim:
            raise ValueError("GDN 的 v 头维与 k 头维不同，当前实现没做 repeat_interleave")
        # 设备侧的形状只从这里取（architecture.md §13 R1）：MIL 生成、缓冲分配、kernel 网格。
        geometry = Geometry(
            hid=hidden,
            ffn=ffn,
            q_heads=heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            l_k_heads=linear_heads,
            l_k_dim=linear_dim,
            l_v_heads=linear_heads,
            l_v_dim=linear_dim,
            conv_k=int(text.get("linear_conv_kernel_dim", 4) or 4),
        )
        geometry.validate()
        roles = [
            AttentionRole.LINEAR if kind == "linear_attention" else AttentionRole.FULL
            for kind in text["layer_types"]
        ]
        if len(roles) != int(text["num_hidden_layers"]):
            raise ValueError("layer_types 数量与 num_hidden_layers 不一致")
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

        store = SafeTensorFile(root / "model.safetensors")
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
            pool = [SurfaceSet(runtime, hidden=hidden,
                               cout=max(geometry.cout_full, geometry.cout_gdn), s=s)
                    for _ in range(pool_slots(roles))]
        try:
            params = [_layer_params(store, index, role, norm_offset)
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
                    store, layer=index, role=role, cout=cout,
                    hidden=hidden, ffn=ffn, s=s, quantize=quantize,
                )
                files = cached_program(directory, source=source, s=s)
                if files is None:
                    files = _build_layer_program(
                        store, directory=directory, index=index, role=role, cout=cout,
                        hidden=hidden, ffn=ffn, s=s, quantize=quantize, source=source,
                        split=split,
                    )
                layers.append(AneLayer(runtime, files, name=f"mude{chain_id}_layer_{index}",
                                       hidden=hidden, cout=cout, ffn=ffn, s=s, pool=pool,
                                       split_rows=split))
            final_norm = _norm(store.read("language_model.norm.weight"), norm_offset)
            readout = _readout(root)
            temperature = float(decision.get("temperature", 1.0))
            cos, sin = _rope_table(theta, width, rotary_dim)
            if metal:
                # 窗口在装完之后立刻落回待命集：20 层先装再卸是"两条路只能走一条"的代价，
                # 省掉的是这一轮之外的时间里那 1.47 GB 的 ANE 权重（`residency.py`）。
                residency = Residency(layers, targets) if targets else None
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
            embed_name="language_model.embed_tokens.weight", final_norm=final_norm,
            readout=readout, temperature=temperature, cos=cos, sin=sin,
            hidden=hidden, s=s, width=width, heads=heads, kv_heads=kv_heads, head_dim=head_dim,
            rotary_dim=rotary_dim, linear_heads=linear_heads, linear_dim=linear_dim,
            own_runtime=own_runtime, pipeline=pipeline, pool=pool,
            own_metal=metal_runtime is not None, metal_runtime=metal_runtime,
        )

    # ------------------------------------------------------------------ 前向

    def logits(self, tokens: Sequence[int], *, trace: list[np.ndarray] | None = None,
               prefix: int = 0, causal: bool = False,
               debug: dict | None = None) -> np.ndarray:
        """整条 prompt 走 20 层，返回 `[CODE_ROWS]` 原始 logits（fp32，未除温度）。

        `trace` 非空时把每层输出（最后一条 token 的 `[hidden]`）按顺序追加进去，
        供探针与 torch 的逐层 hidden 对比。`prefix` 是 `prefix_noncausal` 的公共前缀长度
        （厂商 `Prepared.prefix()` 给出的那个数）：全注意力层的掩码是
        `(q >= prefix) | (k < prefix)`，额外把 padding 列当不可见。单问题、无 padding 时
        prefix=0 就退化为整段双向。

        `causal=True` 换成三角掩码 `k <= q`。它与 `prefix` 是**两个轴**：causal 下每一行
        只看自己之前（含自己），所以 `prefix` 不再改变掩码 —— 前缀行本来就是行内三角的一部
        分，尾巴行看到的 `[0, prefix)` 也在 `k <= q` 之内。
        """
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
        if self._pipeline is not None:
            if trace is not None or debug is not None:
                raise ValueError("Metal 流水线不提供逐层 trace 或 debug（中间量留在面上）",
                                 hint="用 metal=False 的 numpy 链做逐层对拍")
            embed = _place(self._store.read_rows(self._embed_name, ids), self._hidden,
                           self._width)
            self._pipeline.run(embed[:, :length].T, length=length, prefix=prefix,
                               causal=causal)
            final = rms_norm(self._pipeline.last_hidden(length)[None, :], self._final_norm)[0]
            return self._readout @ final
        x = _place(self._store.read_rows(self._embed_name, ids), self._hidden, self._width)
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
        return self._readout @ final

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
        """共享前缀的整条前向：返回每题的 `[CODE_ROWS]` 原始 logits（未除温度）。"""
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
            out.append(self._readout @ final)
        return out

    def logits_rows(self, tokens: Sequence[int], *, prefix: int = 0) -> np.ndarray:
        """与 `logits` 同语义，但 embedding 按桶逐块写进画布，不物化整宽 fp32 画布。

        探针 `.scratch/probe_place.py` 拿它量"整宽画布"那笔内存值多少；两条路径的 logits
        必须逐位相同（`logits` 仍是默认路径，没换）。
        """
        if self._pipeline is None:
            raise ValueError("按行写画布只有 Metal 链提供（load 时用 metal=True）")
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        length = int(ids.size)
        if length == 0:
            raise ValueError("prompt 至少要有一个 token")
        if length > self._width:
            raise ValueError(
                f"prompt 长度 {length} 超过链的缓冲宽度 width={self._width}；用更大的 width 重新 load")
        if not 0 <= prefix <= length:
            raise ValueError(f"prefix 必须在 0..{length}，收到 {prefix}")
        self._pipeline.run_rows(
            self._store.read_rows(self._embed_name, ids), length=length, prefix=prefix)
        final = rms_norm(self._pipeline.last_hidden(length)[None, :], self._final_norm)[0]
        return self._readout @ final

    def _probabilities(self, logits: np.ndarray, options: int) -> np.ndarray:
        """前 `options` 个码的 softmax 概率（选项顺序 = `codes` 顺序）。

        与厂商 `D3.probabilities` 的约定一致：越界码行在 softmax 前置 -inf（exp 为 0），
        因此等价于只在前 `options` 行上归一化。
        """
        if not 1 <= options <= CODE_ROWS:
            raise ValueError(f"options 必须在 1..{CODE_ROWS}，收到 {options}")
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
    ) -> np.ndarray:
        """全注意力层：q|gate 交错投影 → q/k 各做 head_dim 的 RMSNorm → 半维 RoPE → GQA → 门控。"""
        xn = _rms(x, params["input_norm"])
        projected, _ = layer.project(xn)
        p = projected.T.astype(np.float32)                 # [s, 5120]，多出的列是 padding
        heads, kv_heads, head_dim = self._heads, self._kv_heads, self._head_dim
        q_rows = heads * head_dim * 2
        k_rows = kv_heads * head_dim
        # q_proj 的行是每个 head 的 [q(head_dim) | gate(head_dim)] 交错排列。
        qg = p[:length, :q_rows].reshape(length, heads, 2, head_dim)
        query = rms_norm(qg[:, :, 0, :], params["q_norm"])
        gate = qg[:, :, 1, :].reshape(length, heads * head_dim)
        key = rms_norm(p[:length, q_rows:q_rows + k_rows].reshape(length, kv_heads, head_dim),
                       params["k_norm"])
        value = p[:length, q_rows + k_rows:q_rows + 2 * k_rows].reshape(length, kv_heads, head_dim)
        query = apply_rotary(query, cos, sin, rotary_dim=self._rotary_dim)
        key = apply_rotary(key, cos, sin, rotary_dim=self._rotary_dim)
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
        y, _ = layer.output(_place(context * sigmoid(gate), self._hidden, self._s), x)
        xn2 = _rms(y, params["post_norm"])
        return layer.feed_forward(xn2, y)[0]

    def _linear_layer(
        self,
        layer: AneLayer,
        params: dict[str, np.ndarray],
        x: np.ndarray,
        length: int,
        debug: dict | None = None,
    ) -> np.ndarray:
        """线性注意力层（GDN）：depthwise 因果卷积 → q/k/v → chunked delta rule → 门控 norm。"""
        xn = _rms(x, params["input_norm"])
        projected, _ = layer.project(xn)
        p = projected.T.astype(np.float32)                 # [s, 8224]
        heads, dim = self._linear_heads, self._linear_dim
        conv_rows = 3 * heads * dim                        # q丨k丨v，各 2048
        mixed = causal_depthwise_conv(p[:length, :conv_rows], params["conv"])   # 已过 silu
        query = mixed[:, :heads * dim].reshape(length, heads, dim)
        key = mixed[:, heads * dim:2 * heads * dim].reshape(length, heads, dim)
        value = mixed[:, 2 * heads * dim:].reshape(length, heads, dim)
        z = p[:length, conv_rows:conv_rows + self._hidden].reshape(length, heads, dim)
        # 投影行序与 Metal 核一致：`[qkv丨z丨b丨a]`（`b` 在 `a` 之前，见 `e8_common.h:29`）。
        b = p[:length, conv_rows + self._hidden:conv_rows + self._hidden + heads]
        a = p[:length, conv_rows + self._hidden + heads:conv_rows + self._hidden + 2 * heads]
        beta = sigmoid(b)
        g = -np.exp(params["a_log"]) * softplus(a + params["dt_bias"])
        core = delta_rule(query, key, value, g, beta)      # [L, heads, dim]
        # 门控 norm 是按头做的：参考实现在 [S*heads, dim] 上归一化，不能摊平成 [S, hidden]。
        gated = rms_norm_gated(core.reshape(length * heads, dim), z.reshape(length * heads, dim),
                               params["norm"]).reshape(length, heads * dim)
        y, _ = layer.output(_place(gated, self._hidden, self._s), x)
        xn2 = _rms(y, params["post_norm"])
        if debug is not None:
            debug.update(xn=xn, projected=p, mixed=mixed, z=p[:length, conv_rows:conv_rows + self._hidden],
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
    store: SafeTensorFile,
    *,
    layer: int,
    role: AttentionRole,
    cout: int,
    hidden: int,
    ffn: int,
    s: int,
    quantize: bool,
) -> dict:
    """产物的身份：权重文件的 stat + 几何 + 量化开关 + `ARTIFACT_VERSION`。

    刻意用 stat 而不是内容哈希：哈希要把 1.5 GB 读一遍，而读一遍正是要省掉的那一段。
    认不出来的情形只有"权重被原地改写且 mtime、大小都没变"，发布与替换权重都会改 mtime。
    """
    stat = store.path.stat()
    return {
        "weights": str(store.path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "layer": int(layer),
        "type": str(role),
        "cout": int(cout),
        "hidden": int(hidden),
        "ffn": int(ffn),
        "s": int(s),
        "quantize": bool(quantize),
    }


def _build_layer_program(
    store: SafeTensorFile,
    *,
    directory: Path,
    index: int,
    role: AttentionRole,
    cout: int,
    hidden: int,
    ffn: int,
    s: int,
    quantize: bool,
    source: dict,
    split: tuple[int, int] | None = None,
) -> ProgramFiles:
    """读这一层的 fp16 权重、量化、生成 MIL 并落盘（缓存未命中的那一路）。

    `split=(q_rows, kv_rows)` 时把整块投影**按行切成两段**（全注意力层，票据 06）：int8 是按
    输出行量化的，切行不动 scale，所以两段的 codes 与未切时逐字节相同 —— blob 与 ANE 常驻
    权重一分不多，换到的是 `qg` 可以成环。
    """
    entries = ane_layer_weights(store.read, layer=index, layer_type=role)
    actual = int(entries["proj"].shape[0])
    if actual != cout:
        # `cout` 来自写死在 MSL 里的几何；权重形状与它对不上说明这个 checkpoint 不是
        # 这套几何编译的，继续下去会得到一个形状对、语义错的 program。
        raise ValueError(
            f"第 {index} 层的 proj 行数 {actual} 与几何给出的 cout={cout} 不一致"
        )
    if split is not None:
        q_rows, kv_rows = split
        if q_rows + kv_rows != actual:
            raise ValueError(
                f"第 {index} 层：拆投影的两段行数 {q_rows}+{kv_rows} 与 proj 的 {actual} 不符"
            )
        proj = entries["proj"]
        rest = {key: value for key, value in entries.items() if key != "proj"}
        entries = {**rest, "proj_q": proj[:q_rows], "proj_kv": proj[q_rows:]}
    blob, blob_entries = build_blob(entries, quantize=quantize)
    mil = layer_mil(hidden=hidden, cout=cout, ffn=ffn, s=s, entries=blob_entries,
                    split=split is not None)
    return write_program(
        directory,
        mil=mil,
        blob=blob,
        s=s,
        meta={"layer": index, "type": str(role), "cout": cout, "split": split},
        source=source,
    )


# ---------------------------------------------------------------------- 权重


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _text_config(root: Path) -> dict:
    config = _load_json(root / "config.json")
    return config.get("text_config", config)


def _readout(root: Path) -> np.ndarray:
    """`readout.safetensors` 的 `[255, hidden]` 表，按配置的 readout_dtype 转 fp32。"""
    handle = SafeTensorFile(root / "readout.safetensors")
    try:
        weight = handle.read("weight")
    finally:
        handle.close()
    return np.asarray(weight, np.float32)


def _norm(weight: np.ndarray, offset: float) -> np.ndarray:
    """norm 的真值：`w + offset`。偏移是 checkpoint 的存法，由家族给（见 `load`）。"""
    return np.asarray(weight, np.float32) + offset


def _layer_params(
    store: SafeTensorFile, index: int, role: AttentionRole, norm_offset: float
) -> dict[str, np.ndarray]:
    """一层的 numpy 侧权重：各 norm、卷积、A_log / dt_bias。norm 的偏移由调用方给。"""
    base = f"language_model.layers.{index}"
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
