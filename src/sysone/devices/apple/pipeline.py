"""mude 的层内流水线：ANE 三段与 GPU 核交替，中间全部留在 IOSurface 上。

一层的数据流（与旧引擎 `e8_a1m.py` 的 `_full_attn` / `_gdn` 同形）：

```
k_norm_to(h → proj_in)             # 归一化 + 写进 ANE 的输入面，不需要额外面
ANE proj             → proj_out    # 8224 或 5120 行
full:  k_copy q → qg, k_copy k/v → kv → q_prep / kv_prep → k_attn → out_in[0:h]
linear: k_gdn(proj_out → out_in[0:h])   （先 k_copy 残差到 out_in[h:2h]）
ANE out              → out_out
k_norm_resid(out_out → ffn_in)     # 写 [rms(y) | y]
ANE ffn              → ffn_out     # 就是下一层的 h
```

为什么 GPU 要先把 q/k/v 拷进 `qg` / `kv`：`k_q_prep` / `k_kv_prep` 是**原地**核，通道基址按
"面从第 0 行开始"写死（`e8_common.h` 的 `NormArgs` 没有基址字段），所以投影的 q 与 k+v
两段必须先搬进各自的面，不能让它们在 8224 行的大面上就地做。

`k_attn` 的可见范围由 `nk` 给：这一片只做"全长可见"（`nk = length`，即 `noncausal_full`）；
`prefix_noncausal` 的前缀/tail 两遍（两次 dispatch、`nk` 不同）留给下一片。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...domain.attention import AttentionRole
from ...errors import ConfigError
from . import kernels
from .compile import AneLayer
from .metal import MetalBuffer, MetalRuntime
from .runtime import Surface

EPS = 1e-6


def _float_buffer(runtime: MetalRuntime, array: np.ndarray, dtype) -> MetalBuffer:
    array = np.ascontiguousarray(array, dtype=dtype)
    buffer = runtime.alloc(int(array.nbytes))
    buffer.array(dtype, array.shape)[:] = array
    return buffer


@dataclass
class LayerBuffers:
    """一层在 GPU 侧的句柄：六张 ANE 面 wrap 后的视图，加上这层的 norm / 卷积等权重表。"""

    ane: AneLayer
    role: AttentionRole
    surfaces: dict[str, MetalBuffer] = field(default_factory=dict)
    weights: dict[str, MetalBuffer] = field(default_factory=dict)

    def surface(self, key: str) -> MetalBuffer:
        return self.surfaces[key]

    def free(self) -> None:
        # 面和权重都要放：wrap 出来的句柄在 dylib 里按 IOSurface 指针缓存，不 unwrap 的话
        # 下一轮分配到同一地址会拿到一条指向已释放内存的句柄（跨链复用时才暴露）。
        for buffer in self.surfaces.values():
            buffer.free()
        self.surfaces.clear()
        for buffer in self.weights.values():
            buffer.free()
        self.weights.clear()


class MudePipeline:
    """20 层的流水线：持有跨层复用的 GPU 缓冲与每层权重表，`run()` 走完整个前向。

    共享缓冲（一层用完即可被下一层覆盖）：`qg`（4096×s）、`kv`（1024×s）、GDN 的
    `state` / `tail` / `scratch`、参考注意力核的 `attn_scratch`、以及承载 embedding 的 `seed`。
    """

    def __init__(
        self,
        *,
        runtime: MetalRuntime,
        layers: list[AneLayer],
        roles: list[AttentionRole],
        params: list[dict[str, np.ndarray]],
        cos: np.ndarray,
        sin: np.ndarray,
        hidden: int,
        s: int,
    ) -> None:
        self._runtime = runtime
        self._hidden = int(hidden)
        self._s = int(s)
        self._roles = list(roles)
        self.buffers: list[LayerBuffers] = []
        try:
            for index, layer in enumerate(layers):
                entry = LayerBuffers(ane=layer, role=self._roles[index])
                for key, surface in layer.surfaces.items():
                    entry.surfaces[key] = runtime.wrap(surface)
                self.buffers.append(entry)
                self._load_weights(entry, params[index])
            self._cos = _float_buffer(runtime, cos, np.float32)
            self._sin = _float_buffer(runtime, sin, np.float32)
            self._alloc_shared()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ 装载

    def _load_weights(self, entry: LayerBuffers, params: dict[str, np.ndarray]) -> None:
        # buf(1) 在这些核里声明成 `device const float *`（fp32 表），与 fp16 的 qg/kv/out
        # 不同；按 fp16 上传会让核把半个 half 当成一个 float 读，最上游 xn 就错。
        for key in ("input_norm", "post_norm"):
            entry.weights[key] = _float_buffer(self._runtime, params[key], np.float32)
        if entry.role is AttentionRole.FULL:
            entry.weights["q_norm"] = _float_buffer(self._runtime, params["q_norm"], np.float32)
            entry.weights["k_norm"] = _float_buffer(self._runtime, params["k_norm"], np.float32)
        else:
            entry.weights["conv"] = _float_buffer(self._runtime, params["conv"], np.float16)
            entry.weights["a_log"] = _float_buffer(self._runtime, params["a_log"], np.float32)
            entry.weights["dt_bias"] = _float_buffer(self._runtime, params["dt_bias"], np.float32)
            entry.weights["norm"] = _float_buffer(self._runtime, params["norm"], np.float16)

    def _alloc_shared(self) -> None:
        rt, s, hidden = self._runtime, self._s, self._hidden
        self.seed = rt.alloc(2 * hidden * s)
        self.qg = rt.alloc(2 * kernels.QK_DIM * 2 * s)
        self.kv = rt.alloc(2 * kernels.KV_PROJ * 2 * s)
        self.state = rt.alloc(4 * kernels.L_HEADS * kernels.L_KD * kernels.L_VD)
        self.tail = rt.alloc(2 * kernels.CONV_DIM * 3)
        self.gscratch = rt.alloc(2 * 8192 * s)
        self.attn_scratch = rt.alloc(4 * 8 * s * s)

    # ------------------------------------------------------------------ 执行

    def run(self, embed: np.ndarray, *, length: int, prefix: int = 0) -> None:
        """跑完 20 层。`embed` 是 `[length, hidden]` 的 embedding 输出（fp32）。"""
        if not 0 <= prefix <= length <= self._s:
            raise ConfigError("长度或 prefix 越界", length=length, prefix=prefix, s=self._s)
        if prefix not in (0, length):
            raise ConfigError(
                "流水线目前只支持全长可见（nk = length）",
                prefix=prefix, length=length,
                hint="prefix_noncausal 的前缀/tail 两遍 dispatch 在下一片落地",
            )
        view = self.seed.array(np.float16, (self._hidden, self._s))
        view[:] = 0
        view[:, :length] = np.asarray(embed, np.float32).T.astype(np.float16)
        prev: MetalBuffer | None = None
        for index, entry in enumerate(self.buffers):
            kernels.norm_to(
                self._runtime, prev if prev is not None else self.seed,
                entry.weights["input_norm"], entry.surface("proj_in"),
                stile=self._s, src_stride=self._s, eps=EPS,
            )
            # GPU 的 dispatch 只是排队；ANE 紧接着要读 proj_in，所以先把队列排空。
            self._sync()
            entry.ane.eval_proc("proj")
            if entry.role is AttentionRole.FULL:
                self._full_attention(entry, length, prev)
            else:
                self._gdn(entry, length, prev)
            self._sync()
            entry.ane.eval_proc("out")
            kernels.norm_residual(
                self._runtime, entry.surface("out_out"), entry.weights["post_norm"],
                entry.surface("ffn_in"),
                stile=self._s, src_stride=self._s, eps=EPS,
            )
            self._sync()
            entry.ane.eval_proc("ffn")
            prev = entry.surface("ffn_out")
        self._sync()

    def _sync(self) -> None:
        self._runtime.commit()
        self._runtime.wait()

    def _copy(self, src: MetalBuffer, dst: MetalBuffer, *, nchan: int,
              src_c0: int = 0, dst_c0: int = 0, length: int = 0) -> None:
        kernels.copy_columns(
            self._runtime, src, dst,
            nchan=nchan, src_c0=src_c0, dst_c0=dst_c0,
            stile=self._s, s0=0, src_stride=self._s, dst_stride=self._s,
        )

    def _full_attention(self, entry: LayerBuffers, length: int,
                        prev: MetalBuffer | None) -> None:
        self._copy(entry.surface("proj_out"), self.qg, nchan=kernels.QK_DIM * 2)
        self._copy(entry.surface("proj_out"), self.kv, nchan=kernels.KV_PROJ * 2,
                   src_c0=kernels.QK_DIM * 2)
        kernels.q_prep(self._runtime, self.qg, entry.weights["q_norm"],
                       self._cos, self._sin, stile=self._s, src_stride=self._s, eps=EPS)
        kernels.kv_prep(self._runtime, self.kv, entry.weights["k_norm"],
                        self._cos, self._sin, stile=self._s, src_stride=self._s, eps=EPS)
        self._residual(entry, prev)
        kernels.attention(
            self._runtime, self.qg, self.kv, entry.surface("out_in"), self.attn_scratch,
            # `T` 是 qg/kv 的列数（缓冲宽度），不是本批 token 数：两者在"桶宽 256、只有 8 条
            # query"时不同，S=96 的对拍里恰好相等（T == Stile == nq）才没暴露这一错位。
            T=self._s, stile=self._s, nk=length, nq=length, fast=True,
        )

    def _gdn(self, entry: LayerBuffers, length: int, prev: MetalBuffer | None) -> None:
        self._residual(entry, prev)
        # 每层的递归状态都从零开始（prefill、无 cache）：`k_gdn` 自己不清零，
        # 旧引擎也是在每层前把 w_state 填零（docs/ANE/anebench/e8_a1m.py:969-971）。
        self.state.array(np.float32, (kernels.L_HEADS, kernels.L_KD, kernels.L_VD))[:] = 0
        kernels.gdn(
            self._runtime, entry.surface("proj_out"), self.state, self.tail,
            entry.weights["conv"], entry.weights["a_log"], entry.weights["dt_bias"],
            entry.weights["norm"], entry.surface("out_in"), self.gscratch,
            stile=self._s, eps=EPS, first=True, T=length, fast=True,
        )

    def _residual(self, entry: LayerBuffers, prev: MetalBuffer | None) -> None:
        self._copy(prev if prev is not None else self.seed, entry.surface("out_in"),
                   nchan=self._hidden, dst_c0=self._hidden)

    # ------------------------------------------------------------------ 收尾

    def last_hidden(self, length: int) -> np.ndarray:
        """最后一层 `ffn_out` 的最后一条 token（`[hidden]` fp32），供 norm + 读出。"""
        entry = self.buffers[-1]
        view = entry.surface("ffn_out").array(np.float16, (self._hidden, self._s))
        return np.asarray(view[:, length - 1], np.float32)

    def close(self) -> None:
        for entry in self.buffers:
            entry.free()
        self.buffers.clear()
        for buffer in (self._cos, self._sin, getattr(self, "seed", None), getattr(self, "qg", None),
                       getattr(self, "kv", None), getattr(self, "state", None),
                       getattr(self, "tail", None), getattr(self, "gscratch", None),
                       getattr(self, "attn_scratch", None)):
            if buffer is not None:
                buffer.free()
