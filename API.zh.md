[English](API.md) | 简体中文

# SysOne 服务 API

`sysone serve --model h2o` 在 `127.0.0.1:8000` 提供一个 HTTP 服务。本页是请求契约；
安装与启动见 [README.zh.md](README.zh.md)。

## 路由

| 路由 | 方法 | 认证 | 用途 |
|---|---|---|---|
| `/health` | GET | 无 | 就绪状态与模型信息 |
| `/metrics` | GET | 无 | 准入、引擎、启动、视觉、批处理计数 |
| `/v1/models` | GET | 配置时需 bearer | 模型名、最大输入 token、模态、图像契约 |
| `/v1/systemone` | POST | 配置时需 bearer | 跑一条决策请求 |

`/health` 返回 `status`、`model`、`max_input_tokens`、`modalities` 与 `authentication`。
模型带塔时，`/v1/models` 还会附上图像契约：`min_pixels`、`max_pixels`、`max_bytes`、
`max_source_pixels`、`placement`、`patch_embedding`、`decoder` 与 `image_processor`。

## 认证

设置 `DECISION_API_KEY`（不是 `SYS1_API_KEY`）即启用。此后 `/v1/models` 与
`/v1/systemone` 需要 `Authorization: Bearer $DECISION_API_KEY`；`/health` 与
`/metrics` 保持开放。

## 状态码

- 引擎装载期间所有路由都回 `503` 并带 `Retry-After`。装载失败同样回 `503`，并附失败
  原因。
- 准入队列满时，`/v1/systemone` 回 `429` 与 `Retry-After`，并且不读请求体就断开连接。
- 格式错误的请求返回与 reference 协议相同的两种 `422`：字符串 `detail`，或
  `{"invalid_questions": {id: reason}}`（一次报出所有坏问题）。
- 这个模型答不了的请求回 `400` 与一个 `detail`：它没有读出面的题型，或选项数 / `score`
  档位数超过它的读出（见下面的表）。原样重试没有用。

## 请求

`/v1/systemone` 带 `state` 与一个按问题 id 索引的 `questions` **对象**
（`examples/request-service.json`）。每个问题有 `type`、`instructions` 与 `criteria`：

```json
{
  "state": {"battery": 8},
  "questions": {
    "q1": {
      "type": "choice",
      "instructions": "What now?",
      "criteria": {"continue": "keep going", "divert": "go to a charger"}
    }
  }
}
```

`criteria` 的形状按类型定：`choice` 是选项键的对象（值是可选的描述），`score` 是有序的
等级数组，`noul` 是 `{"true": …, "false": …}`。一题最多 255 个选项。`state` 与
`instructions` 可以是字符串或任意 JSON；非字符串会序列化成 JSON 进 prompt。

`type` 是 `choice`、`noul` 或 `score`。别的（厂商那份带的是 `locate`）会回逐个问题的
`422`，并说明这条线格式带哪些题型。

`images` 接受一组 `data:image/...;base64,` data URL。设置 `--vision off` 时，带图的请求回
`422`。

`videos` 会被拒，回 `422` 与 `video inputs are not available: this engine ships no video
path`：d3 自己的发布包写了视频输入，但本引擎既没有取帧解码器，也没有时间轴。

每道题只从 state 得到答案：同一个请求里的题互不可见，所以增删改别的题不会改变某一题的
答案。这也是没有 `isolated` 或 `joint` 旋钮的原因 —— 那两种布局里题目（或 state）会互相
看见。

## 响应

```json
{
  "model": "...",
  "answers": {
    "q1": {"type": "choice", "choice": "continue", "probabilities": {"continue": 0.7, "divert": 0.3}, "confidence": 0.4}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0},
  "attention_mode": "causal",
  "attention_trained": true
}
```

- `choice` 与 `score` 的答案还带 `confidence`：退化成一点时是 `1`，和选项数一样平的时候是
  `0`。口径由模型自己定 —— h2o 报的是把最高那项的概率按那个平分布缩放（`0.7`/`0.3` 得
  `0.4`），mude 报的是 `1 - H/log n`，同一对概率得 `0.12`。上面那个例子用的是 h2o 的数。
- `noul` 的答案带 `noul`（P(true)），不带 `confidence`：两家的 yes/no 答面都没有这个字段。
- `score` 的答案在 `score` 里带期望档位（0 起算），带按序号给的 `probabilities`（键是
  `"0"`..`"n-1"`）与 `confidence`。mude 收 2 到 10 档，h2o 最多 255 档。
- 请求体里的 `attention_mode` 只对该请求覆盖 `--attention-mode` 设的进程默认值。

## 各模型能力

| | h2o | mude | d3 | torchcast | pplx12 | pplx13 |
|---|---|---|---|---|---|---|
| 问题类型 | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` | `choice`, `noul`, `score` |
| 注意力模式 | `causal` (default), `noncausal_full` | `causal`, `noncausal_full`, `prefix_noncausal`（训练过） | 只有 `noncausal_full` | 只有 `noncausal_full` | 只有 `noncausal_full` | 只有 `noncausal_full` |
| 读出 | 厂商 pooled 读出 | 255 行码表 | 255 行码表 | 26 行 `lm_head` 字母 | 255 行码表（BF16） | 255 行码表（BF16） |
| 每题最多选项 | 255（255 档） | 255（10 档） | 255（10 档） | 26（10 档） | 255（255 档） | 255（255 档） |
| 最大输入 token | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) | `--width` (default 32768) |
| 视觉 | 默认可用（`--vision auto`） | 默认可用（`--vision auto`） | 默认可用（`--vision auto`） | 默认可用（`--vision auto`） | 默认可用（`--vision auto`） | 默认可用（`--vision auto`） |

`--width` 同时是 GPU 缓冲宽度与输入上限，所以更长的 prompt 要给更大的值。带塔的家族都能
收图；mude 的塔继承自 d3-nano，按它的模型卡，那座塔没有为 mu 的决策训练或评估过 —— 它能
跑，但在那里不是一个已验证的能力。

首次启动还会把视觉塔的 26 份产物编好，然后归还塔：纯文本进程保持 0 常驻，第一次带图
请求只付装载。`--vision off` 跳过塔及其产物。第三份 dylib（`libe8image.dylib`）只在真的
服务带图请求时才需要。
