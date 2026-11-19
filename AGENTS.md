# SysOne

快速、本地的决策推理引擎。为边缘 AI 提供 System-1 级别的速度。

## Agent skills

### Issue tracker（问题追踪）

issue 与 spec 以本地 markdown 形式存放于 `.scratch/`（不走 GitHub Issues；`.scratch/` 不被主仓库跟踪，由它自己的本地 git 仓库管理）。详见 `docs/agents/issue-tracker.md`。

**提交信息**：主仓库是公开仓库，历史对读者可见，提交信息按 Conventional Commits 写，并且中英文对照、英文在前：标题行写英文；正文英文段在前、中文段在后，两段一一对应。`.scratch/`、`docs/` 仓库不受此约束。

- 格式：`<type>(<scope>): <英文短句>`，句末不加句号；正文两段之间空一行。例：`feat(navigator): Count user messages toward the step budget`，正文先写 `User messages count toward the step budget.`，再写 `用户消息计入步数预算。`。type 取 `feat` / `fix` / `docs` / `test` / `refactor` / `chore`；scope 取包名短形：`navigator`、`reasoning-pruner`、`result-clipper`、`system-one`、`system-one-channel`。只动根级文件（`docs/`、`scripts/`、`.github/`、`AGENTS.md`）时用 `docs` / `chore`，不带 scope。
- 英文标题写「实现了什么功能 / 修了什么问题」，不写票号、轮次、「复核落修」这类内部过程词。一轮复核的落修按改动性质归 `fix` / `test` / `refactor`。
- 正文两段都说清为什么这么改；中文段是英文段的译文，不另加信息。
- 票号与轮次写进正文：英文段写 `Ticket 04 round 2`，中文段写 `票据 04 第 2 轮：……`。标题是给读历史的人看的，票据日志留在 `.scratch/` 的提交里。

### Domain docs（领域文档）

单上下文（single-context）：仓库根目录的 `GLOSSARY.md`，加上 `docs/adr/`。详见 `docs/agents/domain.md`。

### Local scratch（本机临时产物）

本机临时产物放仓库根的 `.tmp/`（已被 `.gitignore` 忽略），**不要**放 `/tmp` 或 `$TMPDIR`：本机开机会清空 `/tmp`（2026-11-19 实测：重启后 `/tmp` 里没有一项早于 boottime），macOS 还会按访问时间清理 `$TMPDIR` 里的条目。两种失效都表现为"重启后一切重算"，而代价本该只是重编一次。

| 东西 | 落点 |
|---|---|
| dylib、Metal 几何头、测试依赖、探针输出 | `.tmp/`（`bash tools/build_ane.sh` 默认写 `.tmp/e8`） |
| 编译产物（GB 级，与 `serve` 共用） | 平台缓存目录（`StorageConfig.resolved_artifacts_dir()`） |

`.tmp/` 随时可以 `rm -rf`。


## 复杂度控制原则

- 优先采用能够完整满足当前明确需求的最小、惯用实现。
- 修改应限定在用户要求的范围内。不要增加无关功能，也不要重构无关代码。
- 不要为假想的未来需求引入抽象、包装层、兼容路径、回退逻辑、配置项、依赖或扩展点。
- 每个新增的层级、间接调用或条件分支，都必须解决当前已有证据支持的需求，并且其收益必须高于维护成本。
- 当任务范围内的私有实现已被完全替代时，应在确认不存在仍在使用的调用方或动态消费者后，删除被替代的代码、调用点、过时测试和失效注释。
- 除非有明确证据或得到明确授权，否则应将公共 API、持久化格式、迁移逻辑、安全控制和运行时回退机制视为仍然有效的契约。
- 保留行为层面的测试覆盖。优先测试行为契约、重要边界、回归问题和高影响故障场景，避免测试实现细节。
- 只有在存在具体故障模式、合理发生路径和实质影响时才增加防护措施。低频但高影响的风险仍然值得防护。
- 注释应解释不明显的约束或设计决策，不要重复描述代码本身。
- 如果某项简化依赖于尚未验证的外部行为假设，应明确报告该假设，不要擅自删除相关行为。