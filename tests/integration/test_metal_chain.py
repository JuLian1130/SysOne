"""Metal 流水线与 numpy 链的端到端对拍（需要真 mude 权重 + 两个 dylib）。

只在设了 `SYS1_MUDE_WEIGHTS` 时跑：它会各建一条 20 层的链（一次约 40 s，同一个 module 内复用）。

`prefix` 参数覆盖 mude 出货的两种读法：0 是全长可见（`noncausal_full`），非 0 是
`prefix_noncausal`（前缀行只看前缀、尾巴行看全部）——后者在流水线里走两遍 attention dispatch。
"""

import os
from pathlib import Path

import numpy as np
import pytest

from sysone.models.families import mude

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
    from sysone.devices.apple.chain import SplitChain
    from sysone.devices.apple.metal import available as metal_available
    from sysone.devices.apple.runtime import AneRuntime

    try:
        AneRuntime()
    except Exception as exc:  # noqa: BLE001 - 没有 dylib 就是跳过
        pytest.skip(f"ANE dylib 不可用：{exc}")
    if not metal_available():
        pytest.skip("Metal dylib 不可用")
    return SplitChain


@pytest.fixture(scope="module")
def chains():
    SplitChain = _chains()
    weights = mude.weight_plan(Path(MUDE))
    cpu = SplitChain.load(MUDE, weights=weights, s=S)
    try:
        gpu = SplitChain.load(MUDE, weights=weights, s=S, metal=True)
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
    SplitChain = _chains()
    weights = mude.weight_plan(Path(MUDE))
    chain = SplitChain.load(MUDE, weights=weights, s=S // 4, width=S, metal=True)
    try:
        yield chain
    finally:
        chain.close()


@pytest.fixture(scope="module")
def packed():
    """批尾巴（票据 10）用的链：前缀落在桶边界（256），宽缓冲留出打包区（2 桶 = 512 列）。

    槽宽 128（尾巴 ≤ 128 token）、每桶 2 格，所以四条尾巴正好一个桶、两条桶的打包区。
    """
    SplitChain = _chains()
    weights = mude.weight_plan(Path(MUDE))
    chain = SplitChain.load(MUDE, weights=weights, s=S, width=S + 2 * S, metal=True)
    try:
        yield chain
    finally:
        chain.close()


# 装载那一刻的窗口读数。必须在 fixture 里就取：这个 fixture 是 module 级的，别的测试先跑完
# 一轮之后 `loads`/`unloads` 已经被换窗改写（而"装载不留全装"说的正是**那一刻**）。
STANDBY_AT_LOAD: dict = {}


@pytest.fixture(scope="module")
def windowed_chunked():
    """待命窗口 **加** 分桶（桶 S/4、缓冲 S）：共享前缀的按块形状必须能在换窗里跑完。

    与下面 `windowed` 的区别只在这条链**多瓦片**（桶 64 对 156 token 的 prefix+tail）——
    "共享前缀的槽位会不会被换窗打断"只有多瓦片才问得出来（一个桶里没有块间的交错）。
    """
    SplitChain = _chains()
    weights = mude.weight_plan(Path(MUDE))
    chain = SplitChain.load(MUDE, weights=weights, s=S // 4, width=S, metal=True, resident=4)
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


@pytest.mark.parametrize("causal", [False, True])
def test_last_hidden_is_the_last_row_of_hidden(chunked, causal) -> None:
    """`last_hidden()` 与 `hidden()[-1]` 是同一行：设备侧的末行**逐位**相同。

    端到端的最后一位差在 `rms_norm` 上，不在链上：`np.mean(x*x, -1)` 对 `[L, H]` 与 `[1, H]`
    的归约顺序不同（rel 2e-7，`.scratch/probe_last_hidden_gap.py` 的四层对照）。所以这里
    分别钉两件事 —— 归一之后的容差，与归一之前（读回的那一行）的逐位相同。

    用多瓦片的链（桶 S/4、prompt 跨多段），因为"只读末行"最容易在这里读错列；两种掩码都量：
    这条出口与掩码无关，causal 只改 `attn` 的参数。
    """
    rows = chunked.hidden(TOKENS, causal=causal)
    row = chunked.last_hidden(TOKENS, causal=causal)
    assert row.shape == (chunked._geometry.hid,)
    assert float(np.abs(row - rows[-1]).max()) <= 1e-5
    # 两次读回的是同一张画布（`last_hidden()` 刚跑完，中间没有别的前向）：逐位相同。
    pipeline = chunked._pipeline
    assert np.array_equal(pipeline.hidden_rows(len(TOKENS))[-1], pipeline.last_hidden(len(TOKENS)))


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
    SplitChain = _chains()
    weights = mude.weight_plan(Path(MUDE))
    chain = SplitChain.load(MUDE, weights=weights, s=S, metal=True, resident=4)
    STANDBY_AT_LOAD.update(chain._pipeline._residency.stats())   # 装载那一刻（见上面注释）
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
    assert stats["peak_resident"] == 8       # 块 0 的窗口 + 下一块的预取 = 2W
    assert stats["loads"] >= 16              # 一轮要重装窗口外的层（每轮一次全量）
    assert windowed._runtime.programs == before
    assert windowed._runtime.pending is False


def test_load_leaves_only_the_standby_window_on_the_device(windowed) -> None:
    """装载不能"20 层全落地再卸"：启动那一刻设备上只该有待命窗口那几层。

    老路径先把每层 `e8_load` 一遍、等所有层常驻之后才由窗口卸掉窗口外的 —— 那几 GB 只是
    瞬态，却正好落在启动这条最容易被别的进程撞上的时刻（票据 15 §36）。现在窗口外的层
    装一次就还回去（为的是 aned 的 lowering 与身份哈希），窗口建起来时不需要再卸任何层。
    """
    stats = STANDBY_AT_LOAD
    assert stats["resident"] == 4
    assert stats["unloads"] == 0            # 窗口没补卸一遍：链已经卸过了
    assert stats["loads"] == 0              # 也没补装
    assert stats["peak_resident"] == 4      # 启动峰值就是待命窗口，不是全装


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
            # 这条用例只钉 prefix 规则；关掉末行规则，免得 dead 从 1 变成 3。
            return chunked.logits(tokens, prefix=prefix, last_row_only=False)
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


def test_last_row_only_skips_every_tile_but_the_readout_tile(chunked) -> None:
    """末行读出：最后一层只留末瓦片，答案逐位相同，注意力少发「瓦片数 − 1」片。

    这条与上一条的区别：上一条钉 `prefix` 规则（dead=1），这条钉末行规则（dead=3）。
    两条都不能少：`prefix` 规则在 causal 下不成立（causal 不看 prefix），末行规则在
    `hidden()` 的整段出口上不能开（每一行都是读者）。
    """
    tokens = TOKENS * (S // len(TOKENS))
    pipeline = chunked._pipeline
    assert pipeline is not None
    original_nch = pipeline._nch
    original = pipeline._step_attn
    tiles = S // pipeline._bucket

    def run(nch: int, last_row_only: bool, counted: dict | None = None) -> np.ndarray:
        pipeline._nch = nch
        if counted is not None:
            def counted_attn(chain, entry, tile):      # type: ignore[no-untyped-def]
                counted["n"] += 1
                return original(chain, entry, tile)

            pipeline._step_attn = counted_attn         # type: ignore[method-assign]
        try:
            return chunked.last_hidden(tokens, causal=True, last_row_only=last_row_only)
        finally:
            pipeline._step_attn = original               # type: ignore[method-assign]

    try:
        full = run(1, False)
        skipped = run(1, True)
        assert np.array_equal(full, skipped), "末行规则改变了答案（被丢的瓦片不该有读者）"

        live_count, dead_count = {"n": 0}, {"n": 0}
        run(1, False, live_count)
        run(1, True, dead_count)
        assert dead_count["n"] == live_count["n"] - (tiles - 1), (
            f"末行规则没有少发 {tiles - 1} 片注意力："
            f"开 {dead_count['n']} 次、关 {live_count['n']} 次"
        )

        # prefix 路径同样不能因为末行规则改变答案。
        prefix = S // 4
        prefixed = chunked.logits(tokens, prefix=prefix, last_row_only=False)
        prefixed_skipped = chunked.logits(tokens, prefix=prefix, last_row_only=True)
        assert np.array_equal(prefixed, prefixed_skipped), "末行规则改变了 prefix 路径的答案"
    finally:
        pipeline._nch = original_nch
        pipeline._step_attn = original


def _ids(count: int, seed: int = 20261114) -> tuple[int, ...]:
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.integers(0, 150_000, size=count))


# 共享前缀是"同一批核、同一段算术，只是把列号挪到了尾巴相"，数学上等价；实测尾巴列的差
# 在 fp16 一个 ulp 的量级（17 token 的短 prompt 上 logits 5.2e-3，长前缀的两个用例是 0），
# 所以这里按"答案必须一致 + logits 在 ulp 量级内"守，而不是逐位。
SHARED_TOL = 1e-2


def _shared_vs_single(chain, prefix: int, tails: tuple[int, ...], *, options: int = 4,
                      causal: bool = False) -> None:
    """共享前缀的 logits 必须与"每题单独整趟"一致（答案逐题相同，数值在 ulp 量级内）。"""
    tokens = _ids(prefix + sum(tails))
    head = tokens[:prefix]
    bounds = np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    per_question = [chain.logits(head + tail, prefix=prefix, causal=causal) for tail in tail_ids]
    shared = chain.logits_shared(head, tail_ids, causal=causal)
    assert len(shared) == len(per_question)
    for index, (got, want) in enumerate(zip(shared, per_question)):
        drift = float(np.abs(got - want).max())
        assert drift <= SHARED_TOL, (
            f"共享前缀第 {index} 题与单趟差 {drift:.3e}，超过 {SHARED_TOL}")
        assert int(np.argmax(got)) == int(np.argmax(want)), f"共享前缀第 {index} 题答案变了"
    probs = chain.forward_shared(head, tail_ids, options=[options] * len(tail_ids),
                                 causal=causal)
    for index, row in enumerate(probs):
        assert np.allclose(
            row,
            chain.forward(head + tail_ids[index], options=options, prefix=prefix, causal=causal),
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


def test_last_hidden_shared_matches_per_question_passes(chains) -> None:
    """共享前缀的末行 hidden 与"每题各自整趟"的末行一致 —— pooled 的读出就靠这一行。

    `logits_shared` 是从它点出 logits 的，所以这条与上面几条不重复：这里钉的是"没有码表
    读出的模型（h2o 的 pooled）也能走上共享路径"，读出那一步在策略层做。
    """
    _, gpu = chains
    prefix, tails = 70, (5, 3, 9)
    tokens = _ids(prefix + sum(tails))
    head = tokens[:prefix]
    bounds = np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    per_question = [gpu.last_hidden(head + tail, causal=True) for tail in tail_ids]
    shared = gpu.last_hidden_shared(head, tail_ids, causal=True)
    assert len(shared) == len(per_question)
    for index, (row, want) in enumerate(zip(shared, per_question)):
        assert row.shape == want.shape
        # 实测（`.tmp/measure_hidden_shared.py`，同一份代码两跑逐位相同）：前缀落在桶边界
        # （192/256）时末行**逐位相同**；只有前缀停在瓦片中间（70）时差 1.65e-3（相对最大）。
        # 这里按 5e-3 守，形状不是"随便取一个宽界限"：它比实测值大 3 倍，比 fp16 一个 ulp 大。
        drift = float(np.abs(row - want).max() / np.abs(want).max())
        assert drift <= 5e-3, f"共享前缀第 {index} 题的末行 hidden 相对差 {drift:.3e}"


def test_a_numpy_chain_declares_no_prefix_buffer(chains) -> None:
    """共享前缀要有零拷贝流水线：numpy 形态没有那段存贮，能力位与调用都得这么说。"""
    cpu, gpu = chains
    assert cpu.can_share_prefix is False
    assert gpu.can_share_prefix is True
    with pytest.raises(ValueError):
        cpu.last_hidden_shared(TOKENS[:4], (TOKENS[4:],))


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


def test_forward_shared_under_a_resident_window(windowed_chunked) -> None:
    """`--W 4` + 桶 S/4：按块交错的形状要能在换窗里跑完，且与逐题整趟相同。

    用**多瓦片**的链：前缀 150 + 尾巴 6/4 在桶 64 下跨 3 段，块间交错与换窗撞在一起时才
    能问出"共享前缀的槽位有没有被换窗打断"。单桶链（`windowed`）问不出来 —— 这条断言把
    那个前提钉住：这条链曾经被一个同名 fixture 遮住，于是测试名说"按块交错"、跑的却是
    单桶（2026-11-16 修）。
    """
    assert windowed_chunked._s < windowed_chunked._width, "这条用例要的是多瓦片的链"
    _shared_vs_single(windowed_chunked, 150, (6, 4))


# ---- M3：causal 路径 -------------------------------------------------------------------


def test_causal_chain_matches_the_numpy_chain(chains) -> None:
    """`caus=1` 的流水线与 numpy 链的三角掩码对拍（桶 256，一整个 prompt 一段）。"""
    cpu, gpu = chains
    lcpu = cpu.logits(TOKENS, causal=True)
    lgpu = gpu.logits(TOKENS, causal=True)
    rel = float(np.linalg.norm(lgpu - lcpu) / np.linalg.norm(lcpu))
    assert rel < TOL, f"causal 下 metal 与 numpy 链的 logits 偏差 {rel:.4e} 超过 {TOL}"


def test_causal_covers_a_multi_tile_prompt(chains, chunked) -> None:
    """桶 64 的链跑同一个 prompt：四段 query、每段的上界是它自己的末列（`nchi` 那条路）。"""
    _, gpu = chains
    tokens = TOKENS * (S // len(TOKENS))       # 正好填满 256 列的缓冲
    assert len(tokens) == S
    once = gpu.logits(tokens, causal=True)
    segment = chunked.logits(tokens, causal=True)
    rel = float(np.linalg.norm(segment - once) / np.linalg.norm(once))
    assert rel < TOL, f"causal 的四段与单段差 {rel:.4e}，超过 {TOL}"


def test_causal_is_systematically_different_from_noncausal(chains) -> None:
    """计划里的验收项：causal 与双向必须**显著不同**，不能只是"看起来不同"。

    末行在两种掩码下看到的 key 集合一样（都是整条 prompt），但前几行的 hidden 不同 ⇒ 它们
    的 K/V 不同 ⇒ 末行的注意力也不同。差得少了就说明 causal 根本没生效。
    """
    cpu, _ = chains
    went = cpu.logits(TOKENS, prefix=0)
    caus = cpu.logits(TOKENS, causal=True)
    gap = float(np.abs(went - caus).max())
    assert gap > 1e-2, f"causal 与 noncausal_full 的最大 logits 差只有 {gap:.3e}，掩码像是没生效"


def test_causal_shared_prefix_matches_per_question_passes(chains) -> None:
    """causal 下共享前缀：前缀相自己成三角，尾巴行看 `[0, q]`（含整个前缀）。"""
    _, gpu = chains
    _shared_vs_single(gpu, 200, (5, 3), causal=True)


def test_causal_batched_tails_keep_the_slot_contract(packed) -> None:
    """causal 叠在批尾巴上：槽位规则之上再按行收，槽 0 仍与逐题整趟逐位相同。"""
    prefix, tails = 192, (60, 60, 60, 60)
    tokens = _ids(prefix + sum(tails))
    head, bounds = tokens[:prefix], np.cumsum((0, *tails))
    tail_ids = [tokens[prefix + int(start):prefix + int(end)]
                for start, end in zip(bounds[:-1], bounds[1:])]
    assert packed._pipeline._slot_layout(prefix, tails) is not None
    want = [packed.logits(head + tail, prefix=prefix, causal=True) for tail in tail_ids]
    got = packed.logits_shared(head, tail_ids, causal=True)
    assert np.array_equal(got[0], want[0]), (
        f"causal 槽 0 不再逐位：{float(np.abs(got[0] - want[0]).max()):.3e}")
    for index, (row, ref) in enumerate(zip(got, want)):
        drift = float(np.abs(row - ref).max())
        assert drift <= SHARED_TOL, f"causal 批尾巴第 {index} 题差 {drift:.3e}，超过 {SHARED_TOL}"
        assert int(np.argmax(row)) == int(np.argmax(ref)), f"causal 批尾巴第 {index} 题的答案变了"


def test_forward_batched_matches_per_sequence_passes(chains) -> None:
    """几条互不相干的序列并排：每条序列的读数与它单独整趟相同。

    块对角只隔离**序列之间**的可见性：每段从第 0 列起、GDN 状态各自从零起，所以这是同一条
    算术换了排布 —— 槽 0 与逐条整趟逐位相同，其余槽按票据 10 的数值契约差 ≤ `SHARED_TOL`
    logits，答案必须相同。
    """
    _, gpu = chains
    sequences = [_ids(120, seed=7), _ids(96, seed=11)]
    batched = gpu.forward_batched(sequences, options=[4, 4])
    assert batched is not None, "两条短序列该能排进一个桶"
    alone = [gpu.forward(sequence, options=4) for sequence in sequences]
    assert np.array_equal(batched[0], alone[0]), "槽 0 不再逐位"
    for index, (got, want) in enumerate(zip(batched, alone)):
        drift = float(np.abs(got - want).max())
        assert drift <= SHARED_TOL, f"第 {index} 条序列差 {drift:.3e}，超过 {SHARED_TOL}"
        assert int(np.argmax(got)) == int(np.argmax(want)), f"第 {index} 条序列的答案变了"


def test_forward_batched_returns_none_when_the_batch_does_not_pack(chains) -> None:
    """三条 100 token 的序列要两格（2 × 256 > 缓冲 256）：这批没有一次前向的解法。

    以前这条拒绝是 `ConfigError`，而策略层的预判（按 `(n-1) × 桶宽 + 最长` 估）认为能批 ——
    一次本该静默回退的请求因此变成 500。现在判据只有 `_slot_layout` 一处，装不下就是 `None`，
    由策略层退回逐条整趟。
    """
    _, gpu = chains
    sequences = [_ids(100, seed=index) for index in range(3)]
    assert gpu.forward_batched(sequences, options=[4, 4, 4]) is None
