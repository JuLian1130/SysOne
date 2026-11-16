"""无 torch 门禁：core 与参考后端只依赖 numpy；h2o 的答面连 transformers 也不需要。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
CODE = """
import sys
import sysone
import sysone.config, sysone.domain.attention, sysone.engine.engine
import sysone.devices.reference
import sysone.models.families
import sysone.models.tokenizer, sysone.models.vendors.h2o_serve
assert "torch" not in sys.modules, sorted(m for m in sys.modules if m == "torch")
assert "transformers" not in sys.modules, sorted(m for m in sys.modules if m == "transformers")
print("ok")
"""


def test_import_does_not_pull_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", CODE],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
