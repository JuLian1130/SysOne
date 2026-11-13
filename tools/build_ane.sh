#!/bin/bash
# 重建 Apple 后端 dylib：libe8ane.dylib（ANE 投影/FFN）与 libe8mtl.dylib（Metal 核）。
#
# 源码仍住在 docs/ANE/anebench（旧研究引擎的位置），产物落在 SYS1_E8_DIR，默认 /tmp/e8。
# /tmp 是易失目录：重启或清理后 dylib 就没了，旧引擎会直接跑不起来。所以"能从源码重建"
# 是这条链路的前提，M2 的每一步都依赖它。
#
# 有意不编译 e8_image.m（视觉解码 shim）：M2 只做 mude 的文本路径，视觉塔在 M4。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ANE_SRC:-$REPO/docs/ANE/anebench}"
OUT="${SYS1_E8_DIR:-/tmp/e8}"

if [ ! -d "$SRC" ]; then
  echo "找不到 ANE 源码目录：$SRC（可用 ANE_SRC 指定）" >&2
  exit 1
fi

mkdir -p "$OUT"

cd "$SRC"
# 这台机器只有 Command Line Tools、没有 Xcode，`xcrun metal` 不存在，所以 .metal 核由
# Metal 框架在进程启动时用运行期编译器编译（见 e8_mtl.py 的 e8m_init_src）。
clang -dynamiclib -fobjc-arc -O2 -framework Foundation -framework Metal -framework IOSurface \
      -o "$OUT/libe8mtl.dylib" e8_mtl.m

# ANE dylib 关掉 fp 收缩：`-ffp-contract=off` 保证 MIL 里的乘法不被合进 FMA，
# 与旧引擎测出的数值逐位一致（docs/ANE/e8_* 的结论都建立在这个编译选项上）。
clang -dynamiclib -fobjc-arc -O2 -ffp-contract=off -framework Foundation -framework IOSurface \
      -o "$OUT/libe8ane.dylib" e8_ane.m

ls -l "$OUT/libe8mtl.dylib" "$OUT/libe8ane.dylib"
