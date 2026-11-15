"""设备侧的模型几何：`devices/apple` 里唯一允许出现的形状来源。

规矩（[docs/sysone/architecture.md](../../../../docs/sysone/architecture.md) §13 R1）：这里生成
MIL、铺 ANE program、分配缓冲、算 grid 的时候，**只从 `Geometry` 取数**，代码里不再写死
`hidden` / 头数这类常数。`Geometry` 是 `ModelDescriptor` 的纯函数投影（读配置、不读权重、
不 import torch），所以 CI 能跑，换模型（4B）时也只是换一个实例。

两个容易混淆的点：

- **混合层的 `qk_dim` 与 `hidden` 无关**。`qk_dim = linear_key_heads × linear_key_dim`；2B 恰好
  两者都是 2048，4B 的 `qk_dim` 是 2048 而 `hidden` 是 2560，所以别拿 `hidden` 顶替它。
- **`value_dim` 未必等于 `hidden`**。它等于 `linear_value_heads × linear_value_head_dim` 的乘积；
  2B 是 2048（恰好等于 hidden），4B 是 32×128 = 4096（≠ hidden 2560）。`in_proj_z`、GDN 的
  递归状态、以及 `cout_gdn` 的拼接口径都跟着它走。

Metal 核的表里（`shaders/e8_common.h`）有一份**默认几何**（就是 mude 2B 这一组）；运行时可以
用 `Geometry.defines()` 生成的 `#define` 头覆盖它，`MetalRuntime(geometry=...)` 负责把它拼成编译源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Geometry:
    """一层的形状元数据。字段全部来自模型配置，派生量都是纯属性。"""

    hid: int
    ffn: int
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    l_k_heads: int
    l_k_dim: int
    l_v_heads: int
    l_v_dim: int
    conv_k: int = 4

    # ---- 派生量 ---------------------------------------------------------

    @property
    def qk_dim(self) -> int:
        """GDN 卷积的 q/k 通道数（每头 `l_k_dim`，共 `l_k_heads` 个头）。"""
        return self.l_k_heads * self.l_k_dim

    @property
    def value_dim(self) -> int:
        """GDN 的 v（也就是 `in_proj_z`）通道数。**不保证等于 `hid`**。"""
        return self.l_v_heads * self.l_v_dim

    @property
    def conv_dim(self) -> int:
        """因果卷积的通道数：`[q | k | v]` 拼起来，2B 是 6144。"""
        return 2 * self.qk_dim + self.value_dim

    @property
    def z_dim(self) -> int:
        return self.value_dim

    @property
    def kv_proj(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def qg_rows(self) -> int:
        """`qg` 的行数：每头 `[q(head_dim) | gate(head_dim)]`。"""
        return 2 * self.q_heads * self.head_dim

    @property
    def kv_rows(self) -> int:
        """`kv` 的行数：k 与 v 各 `kv_heads × head_dim`。"""
        return 2 * self.kv_proj

    @property
    def cout_full(self) -> int:
        """全注意力层 `in_proj` 的输出行数（q+gate+k+v），2B 是 5120。"""
        return self.qg_rows + self.kv_rows

    @property
    def cout_gdn(self) -> int:
        """线性层 `in_proj` 的输出行数（qkv+z+b+a，b 在 a 前），2B 是 8224。"""
        return self.conv_dim + self.z_dim + 2 * self.l_v_heads

    @property
    def state_floats(self) -> int:
        """一层 GDN 递归状态的 float 数（`[l_v_heads, l_k_dim, l_v_dim]`）。"""
        return self.l_v_heads * self.l_k_dim * self.l_v_dim

    @property
    def head_scale(self) -> float:
        """注意力的 `1/√head_dim`，即 `e8_common.h` 里的 `E8_HEAD_SCALE` 的来源。"""
        return float(self.head_dim) ** -0.5

    @property
    def gdn_scratch_halves(self) -> int:
        """`k_gdn` 家族每个 token 一行 scratch 的步长，单位 half。

        行内容 = `[qn | kn | v | g | beta]`，需要 `2*qk_dim + value_dim + 2*l_v_heads`；
        按 1024 half 向上取整，且不低于 8192 —— 8192 是 2B 的既有刻度，出货路径的行距
        不为省几 KB 而改（`shaders/e8_gdn.metal` 的 `E8G_S_ROW` 必须与这里算出同一个数）。
        """
        need = 2 * self.qk_dim + self.value_dim + 2 * self.l_v_heads
        return max(8192, (need + 1023) // 1024 * 1024)

    # ---- Metal 源码的宏 -------------------------------------------------

    def defines(self) -> Mapping[str, int]:
        """`#define` 头里的名字与值（与 `shaders/e8_common.h:52-67` 一一对应）。"""
        return {
            "E8_HID": self.hid,
            "E8_FFN": self.ffn,
            "E8_COUT_FULL": self.cout_full,
            "E8_COUT_GDN": self.cout_gdn,
            "E8_Q_HEADS": self.q_heads,
            "E8_KV_HEADS": self.kv_heads,
            "E8_HEAD_DIM": self.head_dim,
            "E8_ROTARY_DIM": self.rotary_dim,
            "E8_QK_DIM": self.qk_dim,
            "E8_KV_PROJ": self.kv_proj,
            "E8_L_HEADS": self.l_v_heads,
            "E8_L_K_HEADS": self.l_k_heads,
            "E8_L_KD": self.l_k_dim,
            "E8_L_VD": self.l_v_dim,
            "E8_CONV_K": self.conv_k,
            "E8_CONV_DIM": self.conv_dim,
            "E8_Z_DIM": self.z_dim,
        }

    def header(self) -> str:
        """编译时注入的 `#define` 头（第一段源码；`e8_common.h` 里的默认值被 `#ifndef` 挡住）。"""
        lines = [
            "// 由 sysone Geometry 生成；覆盖 shaders/e8_common.h 里的默认几何。",
            "// 一个进程只服务一个模型，所以一份几何对一个 Metal 库。",
            "#define E8_GEOMETRY_SET 1",
        ]
        lines.extend(f"#define {name} {value}" for name, value in self.defines().items())
        return "\n".join(lines) + "\n"

    def validate(self) -> None:
        from ...errors import ConfigError

        for name in ("hid", "ffn", "q_heads", "kv_heads", "head_dim", "rotary_dim",
                     "l_k_heads", "l_k_dim", "l_v_heads", "l_v_dim", "conv_k"):
            value = getattr(self, name)
            if int(value) < 1:
                raise ConfigError("几何字段必须为正", field=name, value=value)
        if self.rotary_dim > self.head_dim:
            raise ConfigError("rotary_dim 不能超过 head_dim",
                              rotary_dim=self.rotary_dim, head_dim=self.head_dim)
        if self.kv_heads > self.q_heads:
            raise ConfigError("kv_heads 不能超过 q_heads",
                              kv_heads=self.kv_heads, q_heads=self.q_heads)

    # ---- 构造 -----------------------------------------------------------

    @classmethod
    def from_descriptor(cls, descriptor, *, rotary_dim: int | None = None) -> "Geometry":
        """从 `ModelDescriptor` 投影。线性层的四项几何缺失时报错，不猜。"""
        from ...errors import ConfigError

        missing = [
            name
            for name in ("linear_key_heads", "linear_value_heads", "linear_key_dim", "linear_value_dim")
            if getattr(descriptor, name, None) is None
        ]
        if missing:
            raise ConfigError(
                "模型几何缺少线性层参数",
                missing=missing,
                hint="家族 parser（models/families/*）需要把 config 的 linear_num_key_heads / "
                     "linear_key_head_dim / linear_num_value_heads / linear_value_head_dim 填进 descriptor",
            )
        dim = rotary_dim
        if dim is None:
            dim = int(
                descriptor.head_dim
                * float(getattr(descriptor, "partial_rotary_factor", 0.25) or 1.0)
            )
        geometry = cls(
            hid=descriptor.hidden,
            ffn=descriptor.ffn,
            q_heads=descriptor.heads,
            kv_heads=descriptor.kv_heads,
            head_dim=descriptor.head_dim,
            rotary_dim=dim,
            l_k_heads=int(descriptor.linear_key_heads),
            l_k_dim=int(descriptor.linear_key_dim),
            l_v_heads=int(descriptor.linear_value_heads),
            l_v_dim=int(descriptor.linear_value_dim),
            conv_k=int(getattr(descriptor, "conv_kernel", 4) or 4),
        )
        geometry.validate()
        return geometry


def mude_geometry() -> Geometry:
    """mude（Qwen3.5-2B 剪到 20 层）的几何：与 `shaders/e8_common.h` 的默认宏一致。"""
    return Geometry(
        hid=2048,
        ffn=6144,
        q_heads=8,
        kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        l_k_heads=16,
        l_k_dim=128,
        l_v_heads=16,
        l_v_dim=128,
        conv_k=4,
    )
