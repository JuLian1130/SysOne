[English](README.md) | 简体中文

# SysOne

一个快速、本地的决策推理引擎。为边缘 AI 提供 System-1 级别的速度。

SysOne 接收一个结构化状态（JSON 对象或一段文本）加一组问题与选项，为每个选项返回一个
概率。发布的 Apple Silicon 路径跨两个设备：投影与 FFN 阶段走 ANE 程序，注意力与 GDN
状态走运行时编译的 Metal kernel。两半共用 IOSurface 零拷贝。

本 README 是从干净 clone 到在 Apple Silicon Mac 上跑起服务的完整公开路径，不依赖任何
私有仓库。英文版是主稿，本文件（`README.zh.md`）是它的中文翻译，两者必须同步修改。

## 支持范围

| 后端 | 家族 | 模型 | 已验证路径 |
|---|---|---|---|
| `pooled` | h2o | [`h2oai/h2o-lightning-4b`](https://huggingface.co/h2oai/h2o-lightning-4b) | ANE + Metal，厂商 pooled 读出 |
| `split` | mude | [`qybaihe/mude-1.0`](https://huggingface.co/qybaihe/mude-1.0) | ANE + Metal，code-row 读出 |

`--backend reference` 是测试用的 M0 语义裁判，不是生产路径，必须显式给 `--descriptor`。

两个发布模型都支持 `choice` 与 `noul` 问题；h2o 还能通过厂商的 `format_answer` 支持
`score`，mude 的 code-row 读出不能。

## 环境要求

- Apple Silicon（M1 或更新），macOS 15 或更新。
- 16 GB 或更多内存；h2o 的权重本身约 9.3 GB。
- arm64 Python 3.11 或更新。
- Xcode Command Line Tools（`xcode-select --install`），`clang` 需要它。
- 磁盘：h2o 权重约 9.3 GB、mude 约 4 GB，另加数 GiB 编译产物（h2o 约 3.33 GiB）。
  权重在首次使用时下载。

## 安装

```sh
git clone https://github.com/JuLian1130/SysOne.git sysone
cd sysone

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ./src
```

`pip install -e ./src` 是受支持的安装方式，它会带上 `huggingface_hub`——首次运行拉权重
用的也是它。目前不支持非 editable 的 wheel：Metal kernel 源码在
`src/sysone/devices/apple/shaders/`，`env.scratch_root()` 也锚在源码树上，两者都要求
仓库留在原地。

不想安装也可以，下面所有命令都可以这样跑：

```sh
PYTHONPATH=src python -m sysone.cli <command> ...
```

## 构建原生库

`serve` 需要三份小的原生库。它们由 `tools/native/` 里的 `.m` 源码构建，仓库不附带
二进制。每个 checkout 只跑一次；不跑的话第一次 `serve` 起不来：

```sh
tools/build_ane.sh
```

脚本把 `libe8ane.dylib`（ANE 私有 API 宿主）、`libe8mtl.dylib`（Metal 宿主）与
`libe8image.dylib`（只有视觉路径用到的 ImageIO 解码器）写进 `.tmp/e8`。引擎自己会找
这个目录；`SYS1_E8_DIR` 可覆盖。不要用 `/tmp`：macOS 重启会清空它。

## 启动服务

```sh
sysone serve --model h2o
```

这一条命令就是全部路径。首次启动会下载固定权重、编译缺失的 ANE 程序（含视觉塔的 26
份），然后打印 `sysone serve: ready in ...s`；之后再启动读产物缓存，编译段报
`compile 0.00s`。随时可以重跑：它是幂等的，不会重建磁盘上已有的东西。

端口在装载前就绑好，所以在 ready 行出现前，`/health` 与 `/v1/systemone` 都回 `503`。
mude 用 `--model mude`。h2o 也接受双向的 `noncausal_full` 模式，它要求整条 prompt 装进
一个 ANE 桶：长 prompt 需要 `--bucket 4096`（默认桶宽 1024），而默认的 causal 没有这个
限制。

另开一个 shell 检查：

```sh
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @examples/request-service.json
```

完整的契约——路由、认证、状态码、请求与响应的形状、各模型能力——见
[API.zh.md](API.zh.md)。

### 参数

`sysone serve` 接收共用的模型参数与 HTTP 参数。默认值就是首次运行需要的样子；所有参数
都可选。

| 参数 | 默认 / 取值 | 含义 |
|---|---|---|
| `--model {h2o,mude}` | 由 `--backend` 或 `--weights` 推断 | 模型家族 |
| `--backend {split,pooled,reference}` | 由 `--model` 推断 | 执行路径；`reference` 是只用于测试的语义裁判 |
| `--weights PATH` | 自动下载 | 显式包；只读，永不下网 |
| `--offline` | 关 | 不碰网络 |
| `--quant {int8,fp16}` | `int8` | ANE 权重存储形态；计算始终 fp16 |
| `--attention-mode {causal,noncausal_full,prefix_noncausal}` | 家族默认（h2o `causal`、mude `prefix_noncausal`） | 掩码；请求可覆盖 |
| `--bucket N` | `1024` | ANE program 列宽 |
| `--width N` | `16384` | GPU 缓冲宽度，也是最大输入 token 数 |
| `--W N` | `4` | 常驻层数；`0` = 全部常驻 |
| `--artifacts PATH` | 平台缓存 | 编译产物落点 |
| `--descriptor PATH` | — | 语义裁判的 descriptor（只配 `--backend reference`） |
| `--debug-timings` | 关 | 每个回包加分阶段耗时 |

视觉（仅带塔的模型）：

| 参数 | 默认 / 取值 | 含义 |
|---|---|---|
| `--vision {auto,off}` | `auto` | `off` 只做纯文本；带图请求回 `422` |
| `--vision-idle S` | `60` | 塔空闲多少秒后卸载；`0` 表示每个请求后立刻卸 |
| `--vision-bucket N` | 随链的桶 | 塔的瓦片宽，必须是 4 的倍数 |

HTTP：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--host HOST` | `127.0.0.1` | 监听地址 |
| `--port N` | `8000` | 监听端口；`0` 取随机空闲口 |
| `--queue N` | `8` | 准入等待位 |
| `--admit-timeout S` | `120` | 请求最多等多少秒，超过回 `429` |
| `--name NAME` | 模型 id | 回包里的 `model` 字段 |

环境变量覆盖：`SYS1_LOG_LEVEL`（默认 `info`）、`SYS1_WEIGHTS_DIR`、
`SYS1_ARTIFACTS_DIR`、`SYS1_E8_DIR`、`DECISION_API_KEY`；对应的命令行参数优先于变量。

## 故障排查

- `NotReadyError: 找不到 libe8ane.dylib`——跑 `tools/build_ane.sh`，检查
  `SYS1_E8_DIR`。缺 `clang` 就跑 `xcode-select --install`。
- `ModuleNotFoundError: tokenizers`（或 `jinja2`/`numpy`/`huggingface_hub`）——在当前
  venv 里重装：`pip install -e ./src`。
- 带 `--offline` 的 `download` 错误——去掉 `--offline` 让首次运行拉权重；或跑打印出来的
  `hf download …`，再用 `--weights <that directory>` 启动；或把下好的整包（含
  `source.json`）放到 `SYS1_WEIGHTS_DIR` 下。
- 显式 `--weights` 报 `权重文件不存在`——目录必须是发布布局：单文件 `model.safetensors`，
  或带 `int8_manifest.json` 的 row-int8 目录。不要用 `qybaihe/mude-1.0-mlx-8bit`：它的
  MLX q8 分片 ANE 链读不了。
- 回 `422` 且提示 `prompt 超过链的缓冲宽度`——加大 `--width`（必要时也加 `--bucket`），
  或缩短 prompt。
- OOM 或 wired memory 警告——调低 `--width`，并确认没有别的大模型常驻。
- 服务回 `503`——引擎还在装载（首次启动还要下载权重并编译），或者装载失败；看进程日志
  与 `/metrics`。
- `429`——加大 `--queue`/`--admit-timeout`，或按 `Retry-After` 重试。
- h2o 在 `Serve.open` 处失败——缺 `serve_config.json` 或 `h2o_lightning_shim.py`；删掉
  缓存里的包，下次启动会重下。

## 仓库布局

- `src/sysone/`——引擎（纯 Python；依赖 numpy、tokenizers、jinja2）。
- `src/sysone/devices/apple/shaders/`——随仓库携带的 Metal kernel 源码，进程启动时编译。
- `tools/native/`——三份 dylib 的 `.m` 源码；`tools/build_ane.sh` 负责构建。
- `examples/`——每种线格式各一个请求。
- `tests/`——单测与 Apple 集成测试；缺权重或缺 dylib 时集成测试自动跳过。

## 许可

模型权重带各自的许可：h2o 是 Apache-2.0；mude 的改动文件是 MIT，Qwen 基座是
Apache-2.0。它们的 `LICENSE`、`NOTICE` 与 `LICENSES/` 属于下载内容，必须保留。

本仓库的引擎源码还没有根 `LICENSE`；这是待办的发布项。
