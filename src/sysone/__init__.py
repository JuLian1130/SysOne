"""sysone —— 本地决策推理引擎。

包根只暴露版本号，不 import 任何硬件栈（torch / ANE / Metal），
因此 `python -c "import sysone"` 在只有 numpy 的机器上也能通过（CI 的 import 门禁）。
"""

__version__ = "0.1.0"
