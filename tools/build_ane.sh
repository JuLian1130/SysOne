#!/bin/bash
# 重建 Apple 后端的三份 dylib：libe8ane.dylib（ANE 投影/FFN）、libe8mtl.dylib（Metal 核）、
# libe8image.dylib（视觉的 ImageIO 解码 shim）。
#
# 源码在本仓库 tools/native/（三份 .m 随仓库公开）；产物落在 SYS1_E8_DIR，默认 $REPO/.tmp/e8。
# ANE_SRC 可覆盖源码目录（指向实验副本时用），缺失的只是源码、不是编译产物。
# 不放 /tmp：本机开机会把它整个清空，清空后整条 ANE 路径都以"找不到 dylib"失败（测试会
# 全套 skip/失败），而代价只是重编一次。`.tmp/` 被 .gitignore 忽略，可随时 rm -rf。
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ANE_SRC:-$REPO/tools/native}"
OUT="${SYS1_E8_DIR:-$REPO/.tmp/e8}"

if [ ! -d "$SRC" ]; then
  echo "找不到 native 源码目录：$SRC（本仓库的源在 tools/native/；可用 ANE_SRC 覆盖）" >&2
  exit 1
fi

mkdir -p "$OUT"

cd "$SRC"
# 这台机器只有 Command Line Tools、没有 Xcode，`xcrun metal` 不存在，所以 .metal 核由
# Metal 框架在进程启动时用运行期编译器编译（见 e8_mtl.py 的 e8m_init_src）。
clang -dynamiclib -fobjc-arc -O2 -framework Foundation -framework Metal -framework IOSurface \
      -o "$OUT/libe8mtl.dylib" e8_mtl.m

# ANE dylib 关掉 fp 收缩：`-ffp-contract=off` 保证 MIL 里的乘法不被合进 FMA，
# 与旧引擎测出的数值逐位一致（旧引擎的结论都建立在这个编译选项上）。
clang -dynamiclib -fobjc-arc -O2 -ffp-contract=off -framework Foundation -framework IOSurface \
      -o "$OUT/libe8ane.dylib" e8_ane.m

# 视觉的图片解码（P1）：ImageIO 的系统解码器，不应用 EXIF 方向、丢 alpha 不合成。
# 与旧引擎的视觉解码 shim 同一条命令行，三者放在一起是为了"三份 dylib 一条命令重建"。
clang -dynamiclib -fobjc-arc -O2 -framework Foundation -framework ImageIO \
      -framework CoreGraphics -o "$OUT/libe8image.dylib" e8_image.m

ls -l "$OUT/libe8mtl.dylib" "$OUT/libe8ane.dylib" "$OUT/libe8image.dylib"
