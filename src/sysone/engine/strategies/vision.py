"""视觉那条路的共同一半：能力判断、契约、统计、预热，以及"请求自带图"的那段接线。

两条策略（`split` / `pooled`）在视觉上**一模一样**：塔不归它们管（归链与 `VisionSlot`），它们
只知道"有没有塔、能不能收图、把 `request.media` 交给谁"。所以这五件事只写一份 —— 两张几乎
相同的 `vision_unavailable()` 一定会漂开，而漂开的那一边表现为**同一台机器上两个模型对同一张
图给出不同的 422 理由**，是最难查的一类不一致。

**四种"收不了图"是四件不同的事**，不能压成一句"不支持多模态"：

| 原因 | 客户端该做什么 |
|---|---|
| 这个模型没有视觉塔 | 换模型 |
| 视觉被关掉了（`--vision off`） | 改启动参数 |
| 引擎还没装载完 | 等就绪再重试 |
| 链是 numpy 形态（`metal=False`） | 换装配（塔的注意力核要 Metal） |

`VisionSlot` 的三种状态（未装 / 已装 / 已卸）都在这里过一遍，`Loaded` 那条路径同时负责
`finish()`：`--vision-idle 0` 的"每请求后即卸"就是靠它。
"""

from __future__ import annotations

from typing import Any, Callable

from ...errors import UnsupportedMediaError
from ...models.vision import contract as vision_contract


class VisionRunner:
    """两条策略共用的视觉半边。子类负责 `_answers(request, mode, media)`。

    `_image_token` / `_load_tower` / `_vision` 三个字段由子类在 `__init__` 里设：占位符 id
    来自 descriptor，工厂与塔由装配阶段给（塔必须在链之后建）。
    """

    _image_token: int | None
    _load_tower: Callable[[object], object] | None
    _vision: object | None
    _vision_state: Callable[[object], dict[str, Any]] | None
    _chain: object | None

    # -- 能力 ----------------------------------------------------------------
    def vision_unavailable(self) -> str | None:
        """收不了图的原因（`None` = 收得了）；这句话会原样进 422 的 `detail`。"""
        if self._image_token is None:
            return "这个模型没有视觉塔，收不了图"
        if self._load_tower is None:
            return "视觉被关掉了：起服务时给 --vision auto（默认就是它）"
        if self._vision is None:
            return "引擎还没装载完"
        # 塔的注意力核是 Metal 的；`metal=False` 的链（numpy 形态）没有 GPU 运行时可用。
        if getattr(getattr(self, "_chain", None), "metal_runtime", None) is None:
            return "视觉塔的注意力核要 Metal 运行时，这条链是 numpy 形态"
        return None

    def image_contract(self) -> dict[str, Any]:
        """`/v1/models`、`doctor` 与引擎的前置检查共用这一份（能力只有一个来源）。"""
        return vision_contract.image_contract(unavailable=self.vision_unavailable())

    # -- 观测 ----------------------------------------------------------------
    def vision_stats(self) -> dict[str, Any]:
        """`/metrics` 的 `vision` 一栏：塔在不在、占多少字节、装过几次。"""
        stats: dict[str, Any] = {} if self._vision is None else self._vision.stats()  # type: ignore[attr-defined]
        return {"available": self.vision_unavailable() is None, **stats}

    def vision_artifacts(self) -> dict[str, Any] | None:
        """塔产物的磁盘账（`reusable` / `total`）：只读，不装塔。

        `_vision_state` 是装配层给的闭包（`chain → artifacts`），所以这里与 `doctor` 的
        `vision_devices.artifact_state` 是同一个判据；一条纯文本装配没有这个闭包，返回
        `None`，调用方不需要再判断"有没有塔"。
        """
        state = getattr(self, "_vision_state", None)
        if state is None or self._chain is None:
            return None
        return state(self._chain)

    def prepare_vision(self) -> dict[str, Any] | None:
        """首启把 26 份产物建出来（装上 → 编 → 归还），全命中时什么都不做。

        为什么值得做：冷产物上的首次带图请求要现编 26 份，实测 7.1–7.8 s、进程 RSS 涨
        ~900 MB；而编完之后把塔归还，服务仍是 0 常驻，第一次带图请求只付装载（约 0.6–2 s）。
        第二次启动 26/26 命中，连 `VisionTower` 都不构造。
        """
        state = self.vision_artifacts()
        if state is None or int(state["reusable"]) >= int(state["total"]):
            return None
        report = self.vision_warmup()
        self._park_vision()
        return report

    def vision_warmup(self) -> dict[str, Any] | None:
        """把 26 份产物建出来（**装上、不跑**）；首启预处理与显式预热共用这一条。

        建塔本身就是编译那一步（构造时 `cached_program` 未命中就现编），所以这里不需要一张
        假图。
        """
        if self._vision is None:
            return None
        self._vision.warm()                               # type: ignore[attr-defined]
        stats = self._vision.stats()                      # type: ignore[attr-defined]
        hits, total = stats["reuse"]
        return {
            "loaded": bool(stats["loaded"]),
            "bytes": int(stats["bytes"]),
            "programs": {"reusable": int(hits), "total": int(total)},
            "load_s": float(stats["load_s"]),
        }

    # -- 请求 ----------------------------------------------------------------
    def _answers_with_images(self, request: object, mode: object) -> tuple:
        """`request.media` → 塔 → `Media` → `_answers`。一次请求装一次、卸一次（按 TTL）。"""
        reason = self.vision_unavailable()
        if reason is not None:
            raise UnsupportedMediaError(reason, images=len(request.media))  # type: ignore[attr-defined]
        assert self._vision is not None
        try:
            media = self._vision.media(request.media)     # type: ignore[attr-defined]
            return self._answers(request, mode, media)    # type: ignore[attr-defined]
        finally:
            self._vision.finish()                         # type: ignore[attr-defined]

    def _park_vision(self) -> None:
        """编完产物把塔归还：`_vision` 槽位保留，下一次 `media()` 会按需重装。"""
        if self._vision is not None:
            self._vision.release()                        # type: ignore[attr-defined]

    def _release_vision(self) -> None:
        """关进程时先卸塔：塔复用链的权重容器与运行时，反过来做会让它去找已经关掉的句柄。"""
        if self._vision is not None:
            self._vision.close()                          # type: ignore[attr-defined]
            self._vision = None


__all__ = ["VisionRunner"]
