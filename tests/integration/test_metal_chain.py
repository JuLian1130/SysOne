"""Metal 流水线与 numpy 链的端到端对拍（需要真 mude 权重 + 两个 dylib）。

只在设了 `SYS1_MUDE_WEIGHTS` 时跑：它会各建一条 20 层的链（一次约 40 s，同一个 module 内复用）。

`prefix` 参数覆盖 mude 出货的两种读法：0 是全长可见（`noncausal_full`），非 0 是
`prefix_noncausal`（前缀行只看前缀、尾巴行看全部）——后者在流水线里走两遍 attention dispatch。
"""

import os
from pathlib import Path

import numpy as np
import pytest

MUDE = os.environ.get("SYS1_MUDE_WEIGHTS")
pytestmark = pytest.mark.skipif(
    not MUDE or not Path(MUDE).is_dir(),
    reason="设 SYS1_MUDE_WEIGHTS 指向 mude-bench 权重目录才会跑（约 40 s）",
)

S = 256
TOKENS = [1234, 5678, 9012, 3456, 7890, 123, 456, 789]
TOL = 5e-3
PREFIXES = (0, 4)


def _chains():
    from sysone.devices.apple.chain import MudeChain
    from sysone.devices.apple.metal import available as metal_available
    from sysone.devices.apple.runtime import AneRuntime

    try:
        AneRuntime()
    except Exception as exc:  # noqa: BLE001 - 没有 dylib 就是跳过
        pytest.skip(f"ANE dylib 不可用：{exc}")
    if not metal_available():
        pytest.skip("Metal dylib 不可用")
    return MudeChain


@pytest.fixture(scope="module")
def chains():
    MudeChain = _chains()
    cpu = MudeChain.load(MUDE, s=S)
    try:
        gpu = MudeChain.load(MUDE, s=S, metal=True)
    except BaseException:
        cpu.close()
        raise
    try:
        yield cpu, gpu
    finally:
        gpu.close()
        cpu.close()


@pytest.fixture(scope="module")
def chunked():
    """桶宽只有 S/4 的链：同一个 256 列的缓冲要分四段算，用来钉住"桶宽 / 缓冲宽"这条分界。"""
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S // 4, width=S, metal=True)
    try:
        yield chain
    finally:
        chain.close()


@pytest.fixture(scope="module")
def packed():
    """批尾巴（票据 10）用的链：前缀落在桶边界（256），宽缓冲留出打包区（2 桶 = 512 列）。

    槽宽 128（尾巴 ≤ 128 token）、每桶 2 格，所以四条尾巴正好一个桶、两条桶的打包区。
    """
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S, width=S + 2 * S, metal=True)
    try:
        yield chain
    finally:
        chain.close()


@pytest.fixture(scope="module")
def windowed():
    """带待命窗口（`--W 4`）的桶 64 链：共享前缀的按块形状必须能在换窗里跑完。"""
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S // 4, width=S, metal=True, resident=4)
    try:
        yield chain
    finally:
        chain.close()


@pytest.mark.parametrize("prefix", PREFIXES)
def test_metal_chain_matches_the_numpy_chain(chains, prefix: int) -> None:
    cpu, gpu = chains
    lcpu = cpu.logits(TOKENS, prefix=prefix)
    lgpu = gpu.logits(TOKENS, prefix=prefix)
    rel = float(np.linalg.norm(lgpu - lcpu) / np.linalg.norm(lcpu))
    assert rel < TOL, f"prefix={prefix} 时 metal 与 numpy 链的 logits 偏差 {rel:.4e} 超过 {TOL}"


def test_metal_chain_keeps_the_last_layer_hidden_in_range(chains) -> None:
    cpu, gpu = chains
    trace: list = []
    cpu.logits(TOKENS, prefix=0, trace=trace)
    gpu.logits(TOKENS, prefix=0)
    last = gpu._pipeline.last_hidden(len(TOKENS))
    ref = np.asarray(trace[-1], np.float32)
    rel = float(np.linalg.norm(last - ref) / np.linalg.norm(ref))
    assert rel < 5e-2, f"末层 hidden 偏差 {rel:.4e}"


def test_chunked_bucket_matches_the_single_bucket_run(chains, chunked) -> None:
    """桶宽 S/4（四段）与桶宽 S（一段）跑同一个满宽 prompt，结果应逐位相同。

    这条用例专门钉住 `GdnArgs.Stile` 的双重身份：它既是列界也是通道步长
    （`shaders/e8_gdn.metal:332-360` 的 `proj[c*Stile + j]`），传成"本段有效列数"会让短
    prompt 全错而满桶恰好正确 —— 见 `docs/sysone/plan.md` 的 M2.2 第 4 片 B。
    """
    _, gpu = chains
    tokens = TOKENS * (S // len(TOKENS))  # 正好填满 256 列的缓冲
    assert len(tokens) == S
    once = gpu.logits(tokens, prefix=0)
    segment = chunked.logits(tokens, prefix=0)
    rel = float(np.linalg.norm(segment - once) / np.linalg.norm(once))
    assert rel < 1e-6, f"四段与单段结果不一致：{rel:.4e}"


@pytest.fixture(scope="module")
def windowed():
    """待命窗口 W=4 的链：只在块边界换常驻，其余与 `gpu` 完全一样。"""
    MudeChain = _chains()
    chain = MudeChain.load(MUDE, s=S, metal=True, resident=4)
    try:
        yield chain
    finally:
        chain.close()


@pytest.mark.parametrize("prefix", PREFIXES)
def test_resident_window_keeps_the_answer(chains, windowed, prefix) -> None:
    """窗口只换常驻、不换数值：答案与 numpy 链在同一容差内。"""
    cpu, _ = chains
    lwin = windowed.logits(TOKENS, prefix=prefix)
    lcpu = cpu.logits(TOKENS, prefix=prefix)
    rel = float(np.linalg.norm(lwin - lcpu) / np.linalg.norm(lcpu))
    assert rel < TOL, f"prefix={prefix} 时窗口链的 logits 偏差 {rel:.4e} 超过 {TOL}"


def test_resident_window_returns_to_standby(windowed) -> None:
    """一轮跑完必须落在待命窗口上：4 层常驻、峰值 2W、换窗不泄漏。

    这条就是服务契约"回到前 W 层预装载"的实测版。`programs` 是 dylib 的**进程级**计数
    （同一个进程里还有别的链），所以这里只钉"一轮换窗前后不变"，绝对层数由 `resident` 钉。
    """
    before = windowed._runtime.programs
    windowed.logits(TOKENS, prefix=0)
    stats = windowed._pipeline._residency.stats()
    assert stats["resident"] == 4
    assert stats["standby"] == [0, 1, 2, 3]
    assert stats["peak_resident"] == 8       # 块 0 的窗口 + 块 1 的预取 = 2W
    assert stats["loads"] >= 20              # 一轮要重装窗口外的层（每轮一次全量）
    assert windowed._runtime.programs == before
    assert windowed._runtime.pending is False


def test_dead_prefix_tiles_are_skipped_without_changing_the_answer(chunked) -> None:
    """S3：最后一层里整段落在公共前缀中的瓦片只做第一相，答案不变、注意力瓦片少一片。

    桶宽 S/4、满宽 prompt 正好给 4 个瓦片，`prefix = S/4` 让第 0 个瓦片整段落在前缀里。

    这条用例拿**第一次** `logits()` 比，不预热。分块链的首趟曾经与后续不同（对 numpy 链
    `relL2 1.47e-1`，pass2 起 `1.31e-3`）：投影的 ANE 写不在任何一条 lane 上，而读同一张
    `proj_out` 的 `gather` 在 lane 0 —— 首趟拷贝冷页慢，写就把上一片的读盖掉了。修法是在
    投影发射前也等到同一槽位上一片 `gather` 的 CB（见 `pipeline._issue`，票据 02/06）。

    量两件事：**出货形态（双链 + S3 开）与单链全量计算逐位相同**，以及 **S3 真的少发了一片
    注意力**（直接数 `_step_attn` 被编码的次数 —— 数 lane 0 的 `wait_upto` 次数在拆投影之后
    不再一一对应，因为死瓦片丢的是整段第二相、不止一片注意力）。
    """
    tokens = TOKENS * (S // len(TOKENS))
    prefix = S // 4
    pipeline = chunked._pipeline
    assert pipeline is not None
    original_nch = pipeline._nch
    original = pipeline._step_attn

    def run(nch: int, skip: bool, counted: dict | None = None) -> np.ndarray:
        pipeline._nch = nch
        pipeline._skip_dead = skip
        if counted is not None:
            def counted_attn(chain, entry, tile):      # type: ignore[no-untyped-def]
                counted["n"] += 1
                return original(chain, entry, tile)

            pipeline._step_attn = counted_attn         # type: ignore[method-assign]
        try:
            return chunked.logits(tokens, prefix=prefix)
        finally:
            pipeline._step_attn = original               # type: ignore[method-assign]

    try:
        shipping = run(original_nch, True)
        full_single = run(1, False)
        assert np.array_equal(shipping, full_single), (
            "出货形态（双链 + S3 开）与单链全量计算不一致"
        )
        dead_count, live_count = {"n": 0}, {"n": 0}
        with_dead = run(1, True, dead_count)
        without_dead = run(1, False, live_count)
        assert dead_count["n"] == live_count["n"] - 1, (
            f"死瓦片没有少发一片注意力：S3 开 {dead_count['n']} 次、关 {live_count['n']} 次"
        )
        assert np.array_equal(with_dead, without_dead), "S3 改变了答案（跳过的瓦片不该有读者）"
    finally:
        pipeline._nch = original_nch
        pipeline._skip_dead = True


def _ids(count: int, seed: int = 20261114) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.integers(0, 150_000, size=count))


# 共享前缀是"同一批核、同一段算术，只是把列号挪到了尾巴相"，数学上等价；实测尾巴列的差
# 在 fp16 一个 ulp 的量级（17 token 的短 prompt 上 logits 5.2e-3，长前缀的两个用例是 0），
# 所以这里按"答案必须一致 + logits 在 ulp 量级内"守，而不是逐位。
SHARED_TOL = 1e-2


def _shared_vs_single(chain, prefix: int, tails: tuple[int, ...], *, options: int = 4) -> None:
    """共享前缀的 logits 必须与"每题单独整趟"一致（答案逐题相同，数值在 ulp 量级内）。"""
    tokens = _ids(prefix + sum(tails))
    head = tokens[:prefix]
    bounds = np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    per_question = [chain.logits(head + tail, prefix=prefix) for tail in tail_ids]
    shared = chain.logits_shared(head, tail_ids)
    assert len(shared) == len(per_question)
    for index, (got, want) in enumerate(zip(shared, per_question)):
        drift = float(np.abs(got - want).max())
        assert drift <= SHARED_TOL, (
            f"共享前缀第 {index} 题与单趟差 {drift:.3e}，超过 {SHARED_TOL}")
        assert int(np.argmax(got)) == int(np.argmax(want)), f"共享前缀第 {index} 题答案变了"
    probs = chain.forward_shared(head, tail_ids, options=[options] * len(tail_ids))
    for index, row in enumerate(probs):
        assert np.allclose(
            row,
            chain.forward(head + tail_ids[index], options=options, prefix=prefix),
            rtol=0, atol=SHARED_TOL,
        ), f"共享前缀第 {index} 题的概率超过 {SHARED_TOL}"


def test_forward_shared_matches_per_question_passes(chains) -> None:
    """两题共用一条前缀：答案与逐题整趟逐位相同。"""
    _, gpu = chains
    _shared_vs_single(gpu, 200, (5, 3))


def test_forward_shared_matches_per_question_passes_on_a_single_tile(chains) -> None:
    _, gpu = chains
    _shared_vs_single(gpu, 10, (7,))


def test_forward_shared_on_a_chunked_chain(chunked) -> None:
    """桶 64 的链：前缀跨两段、尾巴从非桶对齐的位置起（`_chunks_from` 的那条路）。"""
    _shared_vs_single(chunked, 70, (3, 9, 5))


def test_batched_tails_match_per_question_passes(packed) -> None:
    """四条等长尾巴塞进一个桶：每格一份状态与前缀末状态，答案与逐题整趟相同。

    实测这几条用例逐位相同（`max|d| == 0`），这里仍按仓库既有的 `SHARED_TOL` 守，
    因为票据 07 记着一条未查到根因的 fp16 漂移。
    """
    _shared_vs_single(packed, 256, (60, 60, 60, 60))


def test_batched_tails_with_unequal_lengths(packed) -> None:
    """四条不等长尾巴：按槽位的可行数表让短的看不见邻居与自己的 padding 行。"""
    _shared_vs_single(packed, 256, (60, 63, 66, 69))


def test_batched_tails_are_bit_exact_on_a_key_block_boundary(packed) -> None:
    """前缀 192：64 对齐、但不是桶边界（256）。

    对齐时参考里没有"末尾半块前缀 key 与本题开头同块"这回事，四个槽位自己的 key 与逐题
    整趟落在同一批 64-key 块里，所以逐位（票据 10 的 1088 实测）。
    """
    prefix, tails = 192, (60, 60, 60, 60)
    tokens = _ids(prefix + sum(tails))
    head, bounds = tokens[:prefix], np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    assert packed._pipeline._slot_layout(prefix, tails) is not None
    want = [packed.logits(head + tail, prefix=prefix) for tail in tail_ids]
    got = packed.logits_shared(head, tail_ids)
    for index, (row, ref) in enumerate(zip(got, want)):
        drift = float(np.abs(row - ref).max())
        assert np.array_equal(row, ref), f"第 {index} 题（prefix=192）不是逐位相同：{drift:.3e}"


def test_batched_tails_under_an_unaligned_prefix(packed) -> None:
    """前缀 200（%64=8）：闸门对它也开（票据 10 第 3 轮），槽 0 仍逐位。

    槽 1..3 的 key 从 `prefix + j*slotw` 起，与参考的 64-key 块分组不同，差末位级；
    真实 payload 上实测 4.8e-3 logits / 5.4e-4 选项概率，argmax 不变
    （`.scratch/probe_batch_align.py`）。
    """
    prefix, tails = 200, (60, 60, 60, 60)
    tokens = _ids(prefix + sum(tails))
    head, bounds = tokens[:prefix], np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    assert packed._pipeline._slot_layout(prefix, tails) is not None
    want = [packed.logits(head + tail, prefix=prefix) for tail in tail_ids]
    got = packed.logits_shared(head, tail_ids)
    assert np.array_equal(got[0], want[0]), (
        f"槽 0（prefix=200）不再逐位：{float(np.abs(got[0] - want[0]).max()):.3e}")
    for index, (row, ref) in enumerate(zip(got, want)):
        drift = float(np.abs(row - ref).max())
        assert drift <= SHARED_TOL, f"第 {index} 题差 {drift:.3e}，超过 {SHARED_TOL}"
        assert int(np.argmax(row)) == int(np.argmax(ref)), f"第 {index} 题的答案变了"


def test_forward_shared_under_a_resident_window(windowed) -> None:
    """`--W 4`：按块交错的形状要能在换窗里跑完，且与逐题整趟逐位相同。"""
    _shared_vs_single(windowed, 150, (6, 4))
