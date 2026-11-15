"""4B 形状的线性层能不能装进 `MudeChain`：v 头数是 k 头数的两倍。

`tests/fixtures` 与 mude 都是 2B 的形状（`l_v_heads == l_k_heads`、`qk_dim == value_dim ==
hidden`），所以"v 头 ≠ k 头"这条路上没有任何既有覆盖。这里用合成检查点造一个**形状关系与
h2o 相同**的小模型（`value_dim = 2 × qk_dim`、`core_out == value_dim`），然后把两条链
（numpy 参考 / Metal 流水线）各跑一遍：两条路的 GDN 是**两份独立实现**（`ops.delta_rule` 的
`repeat` 展开 vs 核里 `E8G_R` 的寻址），所以它们相等才说明展开这步两边都做对了。

两条链必须建在**子进程**里：`MetalRuntime` 是进程级单例、按几何编一份库，同一个 pytest
进程里 mude 的 2B 几何已经编进去了（`test_metal_chain.py` 先跑），再要 4B 形状会撞上
"已经按另一套几何编译过"（与 `test_metal_gdn_4b.py` 同一个理由）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
TESTS = Path(__file__).resolve().parents[1]
TOL = 5e-3
S = 256
TOKENS = [7, 11, 13, 17, 19, 23]


def _skip_reason() -> str | None:
    if sys.platform != "darwin":
        return "ANE / Metal 只在 macOS 上"
    e8 = Path(os.environ.get("SYS1_E8_DIR", "/tmp/e8"))
    if not (e8 / "libe8mtl.dylib").is_file():
        return "没有 libe8mtl.dylib（先跑 tools/build_ane.sh）"
    from sysone.devices.apple import available

    if not available():
        return "ANE dylib 不可用（先跑 tools/build_ane.sh）"
    return None


pytestmark = pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")

# 子进程：造夹具、读几何、两条链各跑一次、比 logits。
CHILD = r'''
import json
import sys
from pathlib import Path

import numpy as np

from helpers import make_mude_checkpoint
from sysone.devices.apple.chain import SplitChain
from sysone.models.families import h2o, mude
from sysone.devices.apple.geometry import Geometry
from sysone.models.families import mude

S = 256
TOKENS = [7, 11, 13, 17, 19, 23]
root = Path(sys.argv[1])
model = make_mude_checkpoint(root / "model")

# 家族给的纯数据：形状（descriptor）/ 前缀 / norm 偏移 / RoPE 角基。
plan = mude.weight_plan(model)
geometry = Geometry.from_descriptor(plan.descriptor)

scratch = root / "programs"
cpu = SplitChain.load(model, weights=plan, s=S, scratch_dir=scratch)
try:
    gpu = SplitChain.load(model, weights=plan, s=S, scratch_dir=scratch, metal=True)
    try:
        numpy_logits = cpu.logits(TOKENS)
        metal_logits = gpu.logits(TOKENS)
        probabilities = cpu.forward(TOKENS, options=3)
    finally:
        gpu.close()
finally:
    cpu.close()

report = {
    "value_heads": geometry.l_v_heads,
    "key_heads": geometry.l_k_heads,
    "qk_dim": geometry.qk_dim,
    "value_dim": geometry.value_dim,
    "conv_dim": geometry.conv_dim,
    "core_out": geometry.core_out,
    "rel": float(np.linalg.norm(numpy_logits - metal_logits) / np.linalg.norm(numpy_logits)),
    "argmax_same": int(np.argmax(numpy_logits)) == int(np.argmax(metal_logits)),
    "probability_sum": float(probabilities.sum()),
    "probabilities": int(probabilities.size),
}
print(json.dumps(report))
'''


def test_a_two_to_one_value_head_ratio_agrees_across_both_chains(tmp_path: Path) -> None:
    """numpy 链与 Metal 链在 v:k = 2:1 的形状下给出同一批 logits。

    这条同时守住四件事：`load` 不再在 v 头 ≠ k 头时早退、几何从 config 读到了 v 头的
    两项、`_linear_layer` 按 `conv_dim` / `value_dim` / `l_v_heads` 切投影的行、
    `delta_rule(repeat=2)` 与核里的 `E8G_R` 寻址等价。
    """
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)])},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    report = __import__("json").loads(result.stdout.strip().splitlines()[-1])

    # 先钉住形状：几何若退化成"v 头 == k 头"，这条对拍就什么也没证明。
    assert (report["key_heads"], report["value_heads"]) == (2, 4)
    assert report["value_dim"] == 2 * report["qk_dim"]
    assert report["conv_dim"] == 2 * report["qk_dim"] + report["value_dim"]
    assert report["core_out"] == report["value_dim"]
    # 装载与两条路的对拍
    assert report["probabilities"] == 3
    assert report["probability_sum"] == pytest.approx(1.0, abs=1e-5)
    assert report["argmax_same"], "两条链的 argmax 不同"
    assert report["rel"] < TOL, f"numpy 与 Metal 的 logits 差 {report['rel']:.2e} 超过 {TOL}"
