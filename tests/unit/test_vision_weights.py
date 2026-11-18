"""视觉塔的权重读法：从容器里发现塔根，把 conv 权重摊成 1×1。

不需要真权重：这里验的是"名字怎么找"和"形状怎么摊"，都是纯逻辑。真权重上的数值门在
`.scratch/p2_vision_plan.py`（两个出货 checkpoint 各跑一遍）。
"""

from __future__ import annotations

import numpy as np
import pytest

from sysone.errors import MissingTensorError
from sysone.weights.vision import (
    BLOCK_TENSORS,
    MERGER_TENSORS,
    PATCH_TENSORS,
    block_entries,
    merger_entries,
    patch_entries,
    pos_embed,
    tower_root,
)


class FakeStore:
    """只实现 `names()` / `read()` 的最小容器。"""

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._tensors = tensors

    def names(self) -> frozenset[str]:
        return frozenset(self._tensors)

    def read(self, name: str) -> np.ndarray:
        try:
            return self._tensors[name]
        except KeyError:
            raise MissingTensorError("缺少张量", tensor=name) from None


def _tower(root: str, *, blocks: tuple[int, ...] = (0,)) -> dict[str, np.ndarray]:
    """一份最小的塔：只放这套读法会读到的那些名字。"""
    names = {f"{root}visual.patch_embed.proj.weight": np.zeros((1024, 3, 2, 16, 16)),
             f"{root}visual.patch_embed.proj.bias": np.zeros(1024),
             f"{root}visual.pos_embed.weight": np.zeros((2304, 1024))}
    for index in blocks:
        for key, suffix in BLOCK_TENSORS.items():
            names[f"{root}visual.blocks.{index}.{suffix}"] = np.zeros(64)
    for key, suffix in MERGER_TENSORS.items():
        names[f"{root}visual.{suffix}"] = np.zeros(64)
    return names


def test_the_root_is_discovered_from_the_container() -> None:
    """两个出货 checkpoint 只差根前缀（`visual.` / `model.visual.`），不从文本前缀拼。"""
    assert tower_root(FakeStore(_tower(""))) == ""
    assert tower_root(FakeStore(_tower("model."))) == "model."


def test_two_towers_in_one_container_is_not_guessable() -> None:
    """两份同名后缀说明"根"不是一个能推出来的东西 —— 报错，不挑先遇到的那个。"""
    tensors = {**_tower(""), **_tower("model.")}
    with pytest.raises(MissingTensorError) as excinfo:
        tower_root(FakeStore(tensors))
    assert len(excinfo.value.context["matches"]) == 2


def test_no_anchor_is_an_error() -> None:
    with pytest.raises(MissingTensorError):
        tower_root(FakeStore({"model.embed_tokens.weight": np.zeros(4)}))


def test_the_patch_weight_is_flattened_into_a_1x1_conv() -> None:
    """`nn.Conv3d` 的 `(1024, 3, 2, 16, 16)` 就是 1×1 conv 的 `[1024, 1536]`：patch 跨 2 帧、
    3 个通道，正好是 1536 列。摊平不是近似，是同一个线性映射。"""
    store = FakeStore(_tower(""))
    entries = patch_entries(store, "")
    assert entries["pe"].shape == (1024, 3 * 2 * 16 * 16)
    assert entries["pe_b"].shape == (1024,)
    assert patch_entries(FakeStore(_tower("model.")), "model.")["pe"].shape == entries["pe"].shape


def test_a_block_has_the_twelve_entries_the_mil_expects() -> None:
    """block 的 12 个条目名就是 `vision_plan.block_plan` 里用到的那些；缺一个就编不出 MIL。"""
    store = FakeStore(_tower("", blocks=(0, 3)))
    assert set(block_entries(store, "", 0)) == set(BLOCK_TENSORS)
    # 24 个 block 形状相同、名字按序号：换一个序号读到的是另一组名字。
    assert set(block_entries(store, "", 3)) == set(BLOCK_TENSORS)


def test_merger_and_pos_embed_are_read() -> None:
    store = FakeStore(_tower(""))
    assert set(merger_entries(store, "")) == set(MERGER_TENSORS)
    assert pos_embed(store, "").shape == (2304, 1024)


@pytest.mark.parametrize("root", ["", "model."])
def test_patch_names_are_the_same_under_both_roots(root: str) -> None:
    """同一个塔在两个容器里名字只差根 —— 这也正是"不按 model_type 分支"的依据。"""
    store = FakeStore(_tower(root))
    assert set(patch_entries(store, root)) == set(PATCH_TENSORS)
