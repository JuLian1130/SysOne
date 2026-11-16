"""待命常驻窗口：一次只让一个块（加下一块预取）的层序在 ANE 里落地。

移植自旧引擎的 S4（`docs/ANE/anebench/e8_window.py`）。单位仍然是**块**，不是层：
`MudePipeline` 的步骤图让一个块里的层同时在飞（`pool_slots = NCH × 最大块长`），所以一个块
的层必须整块常驻，块边界才是唯一能换窗的地方 —— 这和旧引擎"块的层都在同一个波前里"是同一句话。

三条实测（`.scratch/m22_window_probe.py`，20 层 / 1145 MB blob）决定了这里的形态：

- 权重在 `load` 时就落地：20 层 = **1472 MB wired**，首次 eval 只再涨 14 MB；
- `e8_unload` 把它**全部**还回来（1546 MB）；`e8_suspend`（只 `unloadWithQoS:`）只还
  340 MB（23%），而且挂起后 eval 仍然成功 —— 它不给正确性设闸，所以不用它；
- aned 的 lowering 缓存**跨 unload 存活**（卸载后 `compiled_exists` 仍为 1），所以重装不是
  600-860 ms 的重新编译。票据 09 实测：`loadWithQoS` 本身亚毫秒，28.1 ms/层几乎全是那次
  **权重内容哈希**（读 56 MB + sha256 = 28.4 ms）；身份哈希记过之后 `e8_load_fast` 0.9–1 ms。

于是换窗 = 卸载（块排空后、后台线程上，~4 ms/层）+ 重装（预取线程，30 ms/层；身份哈希
记过之后 ~1 ms，见 `compile.py` 的 `ProgramFiles.load`）。旧记录"`e8_load_fast` 在本机不可用
（rc=7）"已过时：`e8_descriptor_hashes` rc=0、三个 64 字符哈希、`e8_load_fast` rc=0。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

from ...errors import ConfigError

_pc = time.perf_counter


def targets_for(
    groups: Sequence[Sequence[int]], n_layers: int, w: int | None = None
) -> tuple[tuple[int, ...], ...]:
    """每个块的窗口：层 `[块首, 块首 + w)`，裁到模型末尾。

    `w` 缺省取块长。窗口**总是包含**它自己的块（块从自己的首层开始、长度不超过块长），
    所以这是超集而不是重切：块 `[0,1,2]` 在 `w=4` 下拿到 `[0,1,2,3]`，而 3 是下一块的首层，
    于是进入下一块时只补 3 层。窗口比块还短是**构造期错误**：块里的层同时在飞，缺一层会让
    一轮跑到一半时拿到 "unknown program"，那是最难查的一类报错。
    """
    out = []
    for group in groups:
        width = len(group) if w is None else int(w)
        if width < len(group):
            raise ConfigError("常驻窗口比最大块还短：一个块里的层同时在飞，必须整块常驻",
                              width=width, block=list(group))
        out.append(tuple(range(group[0], min(group[0] + width, n_layers))))
    return tuple(out)


class Residency:
    """层序的常驻窗口。`layers[i]` 需要 `reload()` / `unload()` / `name`。

    调用点是块边界，一次一块：

        begin(k)  确保块 k 的窗口常驻（预取成功时是空操作），然后起线程预取块 k+1 的窗口
        end(k)    等预取收尾、等上一次卸载收尾，再起线程把窗口外的层卸掉
                  —— 只在这个块排空之后：在飞的请求打到已卸载的 program 上是唯一会坏
                  正确性的地方（旧引擎 S4 的原话）
        finish()  收尾两个线程，让一轮跑完**落在待命窗口上**（服务契约"回到前 W 层预装载"）

    换窗那一刻当前窗口与正在到达的下一块窗口都在，所以块**排空后立刻换窗**时常驻峰值是
    2W（旧引擎同）。这只是那种时序下的值，不是上界：长输入上块的计算比卸载线程过得慢，
    窗口外的层会多滞留几层 —— 16k 实测 `peak_resident = 10`（`W = 4` 时 2.5W，票据 15
    §26.6）。要按峰值留余量就按 2.5W 算。

    `reload`/`unload` 都是**就地**改 program（`AneLayer` 的对象不变），因为流水线按对象持有
    `AneLayer`、dylib 按名字持有 sidecar —— 换掉列表元素会让后者静默指向旧对象。
    """

    def __init__(
        self,
        layers: Sequence,
        targets: Sequence[Sequence[int]],
        *,
        resident: Sequence[int] | None = None,
    ) -> None:
        """`resident` 是**调用方已经装好**的层：缺省按"全部已装"（旧路径，装完再落回窗口）。

        出货路径改传 `targets[0]`：`SplitChain.load` 现在只让待命窗口的层留在设备上，窗口外
        的层装载一次后立刻卸掉（`unloads` 因此不再包含启动那 28 次 —— 那是链自己做的事，
        不归窗口记账）。
        """
        if not targets:
            raise ValueError("窗口至少要有一个目标（待命集）")
        n = len(layers)
        self.targets = tuple(tuple(sorted(target)) for target in targets)
        for target in self.targets:
            if not target or target[0] < 0 or target[-1] >= n:
                raise ValueError(f"窗口 {target} 越出层表 0..{n - 1}")
        self._layers = layers
        self._lock = threading.Lock()
        self._resident = set(range(n)) if resident is None else {int(i) for i in resident}
        self._busy: set[int] = set()
        self._load_thread: threading.Thread | None = None
        self._evict_thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.loads = 0
        self.unloads = 0
        self.load_ms = 0.0
        self.unload_ms = 0.0
        self.waits: list[float] = []
        # 装好但不在窗口里的层落回待命集（`resident` 已给定时这一句什么也不卸）；峰值从
        # **掉下去之后**开始算，否则启动那一刻会被记成全装。
        self.evict_except(self.targets[0])
        self.peak_resident = len(self._resident)

    # ---------------------------------------------------------------- 块边界

    def begin(self, k: int) -> int:
        """确保块 `k` 的窗口常驻，并开始预取下一块；返回这次要预取几层。"""
        self._raise_stored()
        self._ensure(self.targets[k])
        with self._lock:
            need = [i for i in self._next(k)
                    if i not in self._resident and i not in self._busy]
        if need:
            self._load_thread = threading.Thread(
                target=self._load_all, args=(need,), name="ane-prefetch", daemon=True,
            )
            self._load_thread.start()
        return len(need)

    def end(self, k: int) -> float:
        """块 `k` 已排空：预取收尾，再把窗口外的层卸掉（卸载在线程上）。返回预取还欠的时间。"""
        waited = self._join_load()
        self._raise_stored()
        self._join_evict()
        self._evict_thread = threading.Thread(
            target=self.evict_except, args=(self._next(k),), name="ane-evict", daemon=True,
        )
        self._evict_thread.start()
        return waited

    def finish(self) -> None:
        """两个线程都收尾：一轮之后真正停在待命窗口上。"""
        self._join_load()
        self._join_evict()
        self._raise_stored()

    # ---------------------------------------------------------------- 装卸

    def _next(self, k: int) -> tuple[int, ...]:
        return self.targets[k + 1] if k + 1 < len(self.targets) else self.targets[0]

    def _ensure(self, layers: Sequence[int]) -> None:
        for index in layers:
            with self._lock:
                if index in self._resident or index in self._busy:
                    continue
            self._load(index)

    def _load(self, index: int) -> None:
        with self._lock:
            if index in self._resident or index in self._busy:
                return
            self._busy.add(index)
        t0 = _pc()
        try:
            self._layers[index].reload()
        except BaseException:
            with self._lock:
                self._busy.discard(index)
            raise
        ms = (_pc() - t0) * 1e3
        with self._lock:
            self._busy.discard(index)
            self._resident.add(index)
            self.loads += 1
            self.load_ms += ms
            self.peak_resident = max(self.peak_resident, len(self._resident))

    def _load_all(self, need: Sequence[int]) -> None:
        try:
            for index in need:
                self._load(index)
        except BaseException as exc:     # noqa: BLE001 - 线程里的异常不能静默
            self._error = exc            # 下一次同步调用（begin/end/finish）把它重新抛出去

    def evict_except(self, keep: Sequence[int]) -> None:
        """把窗口外的层卸掉。卸载**在线程上**做：一次换窗 3-4 层、每层 ~4 ms 会吃掉块首的
        一段关键路径（旧引擎的 +1.5 % 全部来自这一步是内联的）。"""
        keep = set(keep)
        with self._lock:
            victims = sorted(self._resident - keep)
        for index in victims:
            t0 = _pc()
            self._layers[index].unload()
            ms = (_pc() - t0) * 1e3
            with self._lock:
                self._resident.discard(index)
                self.unloads += 1
                self.unload_ms += ms

    # ---------------------------------------------------------------- 收尾 / 读数

    def _join_load(self) -> float:
        thread, self._load_thread = self._load_thread, None
        if thread is None:
            return 0.0
        t0 = _pc()
        thread.join()
        waited = (_pc() - t0) * 1e3
        self.waits.append(waited)
        return waited

    def _join_evict(self) -> None:
        thread, self._evict_thread = self._evict_thread, None
        if thread is None:
            return
        thread.join()

    def _raise_stored(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise error

    def stats(self) -> dict:
        with self._lock:
            resident = sorted(self._resident)
        return {
            "resident": len(resident),
            "resident_layers": resident,
            "standby": list(self.targets[0]),
            "peak_resident": self.peak_resident,
            "loads": self.loads,
            "unloads": self.unloads,
            "load_ms": round(self.load_ms, 1),
            "unload_ms": round(self.unload_ms, 1),
            "wait_ms_total": round(sum(self.waits), 1),
            "wait_ms_max": round(max(self.waits), 1) if self.waits else 0.0,
        }
