"""双链步骤图：把一层拆成 (chain, step) 的可执行序列。

移植自旧引擎的 `gdn_steps` / `attn_steps` / `blocks`（`docs/ANE/anebench/e8_a3.py:317-399`
与 `:2235-2253`）。旧引擎的赢面有两个来源，这里都要留住：

- **NCH 条链**：一层的瓦片按 `tile % NCH` 分成两条链。两条链各自独立就绪，于是一条链等它的
  `core`（或等下一瓦片的 `h`）时，另一条还有 `proj`/`out`/`ffn` 可以喂给 ANE —— 这才是
  ANE 停摆的那半面墙。
- **A 层在前**：`blocks()` 在每个全注意力层**之前**切开，所以一个 block 里至多一个全注意力
  层、且它在第 0 位。全注意力层负责填满整个注意力窗口，后面的线性层才是填窗口的人。

一个 step 是最小可发射单位，只有三种：

- `gpu`：一次 GPU dispatch，编进某条 lane，**随后立刻提交**（一个 step 一个命令缓冲）。
  旧引擎的每个 stage 也各自提交 —— 一个 CB 里塞多个互相依赖的 dispatch 需要
  `E8M_BARRIER=1` 的 memory barrier，而我们的 dylib 走 `e8m_init_src`，那条环境变量根本
  读不到（`e8_mtl.m:121-123` 只在 `e8m_init` 里读），所以"一个 step 一个 CB"是硬约束。
- `ane`：一次 ANE eval。它读的 GPU 结果必须已经**回收**（不是"已编码"）—— ANE 从 IOSurface
  读内存，CB 没结算就没有可见性。
- `prep`：全序列的 q/k 归一 + 部分 RoPE，也是 GPU dispatch，但它是一道**序列级屏障**：
  必须等本层**所有链的所有 gather 都编码完**才能编码，且任何 `attn` 都排在它后面。

不变式（`Chain` 的游标 + 驱动的"一次一个 eval 在飞"共同守住）：

1. 一条链独占一套表面（= ANE 槽位），一次只有一个 ANE eval 在飞；
2. 瓦片 `s` 的 `h` 只有在它的 `tail` CB 回收后才算写好，下游层读它必须等这条链的 tail；
3. 共享的 `qg` / `kv` 只留一份：一个 block 里至多一个全注意力层（见 `blocks()`）；
4. **ANE 的写不排在任何一条 lane 上**：同一槽位的下一片 `proj` 写 `proj_out` 之前，必须等上一片
   `gather`（lane 0）的读结算完 —— 否则首趟冷页慢时，写会把读盖掉（`pipeline._issue` 里的
   `_wait_for(chain, "gather", tile - nch)`，票据 02）。

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ...domain.attention import AttentionRole
from .metal import LANE_LONG, LANE_SMALL

# 一层的瓦片链数。旧引擎 `NCH = 2`（`e8_a3.py:109`）：再多就把每条链排得太短，
# 等 `core` 的时候没有别的链可喂。
NCH = 2

# 一个 block 里能同时在飞的层数。旧引擎 `RING = 8 // NCH = 4`（`e8_a3.py:110`），
# 表面池 `NSETS = NCH * RING = 8` 正好够一个 block 用。
RING = 4


@dataclass(frozen=True, slots=True)
class Step:
    """一步：`kind` 是 `gpu` / `ane`，`work` 是这一步做什么，`tile` 是它负责的瓦片。

    `lane` 只对 `gpu` 有意义（ANE eval 不占 GPU 队列）。`prep` 的 `tile` 是 -1：它属于整条
    序列，不属于某个瓦片。
    """

    kind: str
    work: str
    tile: int
    lane: int = LANE_LONG


@dataclass
class Chain:
    """一层的一条瓦片链：一串有序 step，加一个游标。

    `cbs` 是这条链已经编码过的命令缓冲：`(work, tile) -> (lane, 序号)`。驱动用它来等
    "这一格的输入是哪个 CB 写的"：序号是这条 lane 上的第几个缓冲，靠 dylib 的 `wait_upto`
    回收，所以等待的粒度是一个 CB 而不是整条 lane。

    `par` 是瓦片奇偶（`NCH = 1` 时恒为 0），`sets` 是这条链可用的表面集，按瓦片号循环
    （`set_of(s) = sets[s % len(sets)]`）。旧引擎给每条链一套（`e8_a3.py:431-434`），
    一条链上的两个瓦片（`s` 与 `s + NCH`）永远不会同时在飞。
    """

    layer: int
    steps: tuple[Step, ...]
    sets: tuple[int, ...]
    slot: int = 0
    par: int = 0
    ptr: int = 0
    cbs: dict[tuple[str, int], tuple[int, int]] = field(default_factory=dict)

    def cur(self) -> Step | None:
        return None if self.done else self.steps[self.ptr]

    def advance(self) -> None:
        self.ptr += 1

    @property
    def done(self) -> bool:
        return self.ptr >= len(self.steps)

    def set_of(self, tile: int) -> int:
        return self.sets[tile % len(self.sets)]


def _gpu(work: str, tile: int, lane: int) -> Step:
    return Step("gpu", work, tile, lane)


def _ane(work: str, tile: int) -> Step:
    return Step("ane", work, tile)


def gdn_steps(n: int, par: int = 0, *, nch: int = NCH) -> tuple[Step, ...]:
    """一个 GDN 层的瓦片链：`norm → proj → core → out → norm2 → ffn → tail`。

    `n` 是瓦片数（不是 token 数），链只拿 `tile % nch == par` 的那些；`nch=1` 就是一条链跑全部
    瓦片（第 2 步的单链驱动用它，第 3 步再切回两条）。
    """
    out: list[Step] = []
    for s in range(par, n, nch):
        out += [
            _gpu("norm", s, LANE_SMALL),
            _ane("proj", s),
            _gpu("core", s, LANE_SMALL),
            _ane("out", s),
            _gpu("norm2", s, LANE_SMALL),
            _ane("ffn", s),
            _gpu("tail", s, LANE_SMALL),
        ]
    return tuple(out)


def attention_steps(
    n: int, par: int = 0, *, dead: int = 0, nch: int = NCH
) -> tuple[Step, ...]:
    """一个全注意力层的瓦片链，分两相。

    **第一相**（每个瓦片）：`norm → proj1 → gather`。`gather` 把 `proj_out` 的两段搬进全宽的
    `qg` / `kv`，走 lane 0。

    **屏障**：`prep`（全序列的 q/k 归一 + RoPE）只能在本层所有 gather 之后编码，且所有
    `attn` 都排在它之后。它写在 `par == 0` 的链上；驱动必须把另外那条链的 `attn` 卡在
    这道屏障后面，否则它会读到没 prep 过的 `qg` / `kv`。

    **第二相**（每个瓦片）：`attn → out → norm2 → ffn → tail`。

    第二相**不能**把整条链的 `attn` 排到 body 之前（试过，2026-11-13 实测两头都更慢）：
    16k / prefix 0 墙 5262 → 6075 ms，出货口径 4907 → 5678 ms。原因是 dylib 的两条 lane 在
    提交序上几乎先进先出 —— lane 0 攒着十几片注意力时，后面才编码的 `out_in` 那一次小搬运
    （把注意力输出与这一层的输入并进 `out_in`、喂给 ANE 的 `out`）就要排在下一片注意力后面，
    每瓦片白等一片（阻塞量 211 → 2710 ms）；ANE 自己的 eval 时间一点没变（2801 → 2778 ms），
    所以墙时涨的正是它的空转。交错的顺序让 lane 0 上永远只有一片在飞，反而没有这个
    "小搬运排在长 kernel 后面"的钱。

    所以保持交错：`attn` 一片接一片地排、body 紧跟其后，`out` 只等自己那一片。

    `dead`（S3）：前 `dead` 个瓦片整段落在请求的公共前缀里，而**模型最后一层**在这些瓦片上
    只欠 K/V —— 它的注意力输出与 MLP 输出没有读者（读出在 query 行上，K/V 出自 `proj1`，
    不依赖注意力输出）。所以这些瓦片保留第一相、丢掉第二相。`tail` 随之没写，那些列的 `h`
    也没人读 —— 这就是它安全的原因。
    """
    tiles = [s for s in range(n) if s % nch == par]
    out: list[Step] = []
    for s in tiles:
        out += [
            _gpu("norm", s, LANE_SMALL),
            _ane("proj1", s),
            _gpu("gather", s, LANE_LONG),
        ]
    if tiles and par == 0:
        out.append(_gpu("prep", -1, LANE_LONG))
    for s in tiles:
        if s < dead:                        # 整段落在前缀里，只欠 K/V（S3）
            continue
        out += [
            _gpu("attn", s, LANE_LONG),
            _ane("out", s),
            _gpu("norm2", s, LANE_SMALL),
            _ane("ffn", s),
            _gpu("tail", s, LANE_SMALL),
        ]
    return tuple(out)


def blocks(
    roles: Sequence[AttentionRole], *, ring: int = RING
) -> tuple[tuple[int, ...], ...]:
    """在每个全注意力层**之前**切开层表，再按 `ring` 切成长度不超过 `ring` 的块。

    mude 是 20 层、全注意力层在 3/7/11/15/19，于是切成
    `[0,1,2] [3,4,5,6] [7,8,9,10] [11,12,13,14] [15,16,17,18] [19]`：每个块至多一个全注意力层，
    且它在第 0 位（第 0 块没有全注意力层，它是 A 层的前驱）。
    """
    out: list[list[int]] = []
    cur: list[int] = []
    for index, role in enumerate(roles):
        if role is AttentionRole.FULL and cur:
            out.append(cur)
            cur = []
        cur.append(index)
    if cur:
        out.append(cur)
    if all(len(group) <= ring for group in out):
        return tuple(tuple(group) for group in out)
    return tuple(
        tuple(group[i : i + ring]) for group in out for i in range(0, len(group), ring)
    )
