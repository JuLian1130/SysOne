"""Apple 后端的本地产物位置。

三份 dylib（`libe8ane` / `libe8mtl` / `libe8image`）由 `tools/build_ane.sh` 编译，落在
`SYS1_E8_DIR`，默认 `<仓库>/.tmp/e8`。默认值从仓库内的 `.tmp` 取而不是 `/tmp`：本机开机
会把 `/tmp` 整个清空（2026-11-19 实测：重启后 `/tmp` 里没有一项早于 boottime），而清空的
表现是整条 ANE 路径都以"找不到 dylib"失败，代价却只是 2 s 重编。

`SYS1_E8_DIR` 直接读环境、没有进 `env.REGISTRY`：三个加载器要的只是一个路径，不是已校验
的配置值，进注册表只会多一层类型与错误分支而没有区别。
"""

from __future__ import annotations

import os
from pathlib import Path

from ...env import scratch_root

VARIABLE = "SYS1_E8_DIR"


def dylib_dir() -> Path:
    """三份 dylib 所在的目录：`tools/build_ane.sh` 的默认输出，三个加载器的默认输入。"""
    configured = os.environ.get(VARIABLE)
    return Path(configured) if configured else scratch_root() / "e8"
