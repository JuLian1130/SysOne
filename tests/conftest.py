"""测试环境的 sys.path：让 `helpers` 等顶层模块可导入。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
