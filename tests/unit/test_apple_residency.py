"""常驻窗口的排程契约（纯软件，不需要 ANE/Metal）。

钉住三件事：**块开跑时它的层一定都在**、**一轮跑完落在待命窗口上**、**峰值不超过 2W**
（当前窗口与正在到达的下一块窗口同时在那一刻，旧引擎同）。
"""

import pytest

from sysone.devices.apple.residency import Residency, targets_for
from sysone.devices.apple.schedule import blocks
from sysone.domain.attention import AttentionRole

ROLES = [AttentionRole.FULL if i in (3, 7, 11, 15, 19) else AttentionRole.LINEAR
         for i in range(20)]
GROUPS = blocks(ROLES)


class FakeLayer:
    """只有窗口用到的三个面：`reload` / `unload` / `resident`（断言用）。"""

    def __init__(self, index: int) -> None:
        self.index = index
        self.resident = True
        self.loads = 0
        self.unloads = 0

    def reload(self) -> None:
        assert not self.resident, f"第 {self.index} 层被重复装载"
        self.resident = True
        self.loads += 1

    def unload(self) -> None:
        assert self.resident, f"第 {self.index} 层被重复卸载"
        self.resident = False
        self.unloads += 1


def _window(w: int, n: int = 20) -> tuple[list[FakeLayer], Residency]:
    layers = [FakeLayer(i) for i in range(n)]
    return layers, Residency(layers, targets_for(GROUPS, n, w))


def test_targets_are_the_block_and_its_successors() -> None:
    targets = targets_for(GROUPS, 20, 4)
    assert targets == ((0, 1, 2, 3), (3, 4, 5, 6), (7, 8, 9, 10),
                       (11, 12, 13, 14), (15, 16, 17, 18), (19,))
    # 窗口是超集，不是重切：每个块都落在自己的窗口里
    for group, target in zip(GROUPS, targets, strict=True):
        assert set(group) <= set(target)


def test_targets_default_to_the_block_length() -> None:
    assert targets_for(GROUPS, 20) == tuple(tuple(g) for g in GROUPS)


def test_window_starts_at_standby() -> None:
    layers, residency = _window(4)
    assert [layer.resident for layer in layers] == [True] * 4 + [False] * 16
    stats = residency.stats()
    assert stats["resident"] == 4 and stats["unloads"] == 16
    assert stats["peak_resident"] == 4      # 启动那一刻不能记成 20 层


def test_every_block_is_resident_before_it_runs() -> None:
    layers, residency = _window(4)
    for index, group in enumerate(GROUPS):
        residency.begin(index)
        assert all(layers[layer].resident for layer in group), f"块 {index} 开跑前没装齐"
        assert residency.stats()["peak_resident"] <= 8       # 2W
        residency.end(index)
    residency.finish()


def test_a_pass_ends_at_standby() -> None:
    layers, residency = _window(4)
    for index in range(len(GROUPS)):
        residency.begin(index)
        residency.end(index)
    residency.finish()
    assert [i for i, layer in enumerate(layers) if layer.resident] == [0, 1, 2, 3]


def test_a_pass_reloads_the_whole_model_once() -> None:
    """窗口宽度不改变**工作量**，只改变内存和它落在哪个边界上。

    一轮要走完 20 层、待命只留 W 层，于是窗口外的层一律要补一次：待命省下的 W 层在轮末
    又要为下一轮补回来（`targets` 的滚动会把它们依次挤出去）。所以 `W ∈ [4, 19]` 时一轮
    的重装数恒为 20 —— 这是"每轮一次全量重装"的代价，不是笔误。
    """
    counts = {}
    for w in (4, 8, 12):
        layers, residency = _window(w)
        for index in range(len(GROUPS)):
            residency.begin(index)
            residency.end(index)
        residency.finish()
        counts[w] = residency.stats()["loads"]
    assert counts == {4: 20, 8: 20, 12: 20}


def test_a_pass_unloads_what_it_reloads() -> None:
    layers, residency = _window(4)
    for index in range(len(GROUPS)):
        residency.begin(index)
        residency.end(index)
    residency.finish()
    # 启动时卸掉 16 层，一轮里又卸掉 20 层（每块换窗卸掉被窗口挤出去的）
    assert sum(layer.unloads for layer in layers) == 36


def test_a_window_shorter_than_a_block_is_refused() -> None:
    """窗口装不下自己的块就是构造期错误：块里的层同时在飞，缺一层会跑到一半才炸。"""
    from sysone.errors import ConfigError

    with pytest.raises(ConfigError):
        targets_for(GROUPS, 20, 2)      # 2 < 最大块长 4


def test_unload_refuses_while_an_eval_is_in_flight() -> None:
    """窗口唯一会坏正确性而不是只坏速度的那条：本层自己的 eval 没回收就不许卸。

    判据必须是本层的在飞计数：换窗时下一块常常已经在飞，`AneRuntime.pending` 那个进程级
    单槽会是真，用它会误杀每一次换窗（集成测试里第一次跑就是这样炸的）。
    """
    from sysone.devices.apple.compile import AneLayer
    from sysone.errors import NotReadyError

    class _Quiet:
        def unload(self) -> None:
            raise AssertionError("不该走到这里")

    layer = object.__new__(AneLayer)
    layer._runtime = _Quiet()
    layer._inflight = 1
    layer.name = "fake"
    with pytest.raises(NotReadyError):
        AneLayer.unload(layer)
