# agents

设计依据：docs/design/10 §4、§6–§9。

## 文件

```
agents/cards.py         任务卡 → 提示（TASK.md 与增量消息）
agents/prompts/         各任务类型的提示模板与固定前缀（版本化，hash 写入 AgentRun）
agents/workspace.py     工作区的创建、存档、清理；NOTES.md
agents/sessions.py      会话的开启、恢复、结束；上下文大小跟踪
agents/runner.py        领取任务、运行、监控、提交
agents/adapters/        base.py、codex.py、claude_code.py、pi.py、deepseek_harness.py、capabilities.py
agents/guard.py         权限：任务令牌、kta-agent 用户、GPU 设备权限
```

## 接口

```python
class AgentAdapter(Protocol):
    name: str
    supports_resume: bool
    def start(self, prompt: str, workspace: Path, env: dict, limits: Limits) -> tuple[str, Iterator[AgentEvent]]
    def resume(self, session: str, message: str, workspace: Path, env: dict, limits: Limits) -> Iterator[AgentEvent]
class AgentEvent:
    kind: Literal["text", "tool_call", "tool_result", "usage", "error", "exit"]
    at: float; data: dict
class Limits:
    wall_s: int; idle_s: int = 600; tokens: int | None = None   # 只有适配器能实时拿到用量时才生效

def run_task(ctx: Context, task: Task) -> AgentRun
```

## 行为

- **会话**：一个工作单元（`TaskCard.work_unit`）一个会话。同一工作单元的后续任务用 `resume`，只发增量消息。以下情况开新会话：上下文超过模型窗口的 60%；同一完成检查连续两次因相同原因失败；适配器不支持恢复；会话异常。新会话的提示要求 worker 先读工作区的 `NOTES.md`。
- **工作区**：`$KTA_DATA/work/<work_unit>/`，整个工作单元期间保留，git 管理，每个任务结束时提交一次。包含 `TASK.md`、`NOTES.md`、`inputs/`、`out/`。工作单元结束后打包为产物，再删除。
- **提示**：固定前缀（所有会话字节级相同）＋ 步骤头（一页以内：目标、完成检查原文、剩余预算、相关记录卡片最多 5 张、建议读的 skills）。不放入完整的池或历史。
- **权限**：
  - agent 进程运行在控制面机器上；GPU 在 worker 上，agent 只能经 `kta run`、`kta measure` 使用；
  - 隐藏形状种子、Contract 源文件、检查器代码、存储目录对 agent 不可写，前三者不可读。具体隔离到什么程度（只靠配置目录与工作区隔离、独立系统用户、容器）是待定事项（docs/decisions.md D09），第一阶段默认：配置目录与工作区隔离，加独立系统用户；
  - `kta` 命令用环境变量中的任务令牌鉴权，令牌只在任务运行期间有效。
  - 如果 DGX OS 上无法限制 GPU 设备权限，在交付说明中写明，并改为在 `kta` 之外调用 GPU 时由 guard 检测并终止任务（记录为决策）。
- **预算**：主要按墙钟时间（不含排队时间，硬上限为预算的 3 倍）、GPU 秒和每个工作单元的步骤数控制；token 上限只在适配器能实时拿到用量时生效。CLI 的输出和会话文件连续 `idle_s` 秒没有变化，视为卡住。
- **记录**：每次运行写 AgentRun（时间构成、退出原因、profile、CLI 与模型版本；用量有则记）；CLI 的标准输出和会话文件按工作单元持久保存并存档。
- **会话解析器**：每个适配器提供 `parse_session(path) -> list[AgentEvent]`，把保存的会话文件解析成模型段、工具调用、用量等事件，供流水图展开和事后分析使用。不要求实时。
- **适配器**：见下文"适配器"一节。

## 适配器

### 准入要求

一个 agent CLI 要接入 KTA，必须满足以下各项。不满足的，在 `capabilities.py` 中如实标注，并按右列处理：

| 要求 | 不满足时 |
| --- | --- |
| 非交互运行，从文件或参数读取提示 | 不能接入 |
| 会话持久保存，能找到并读取会话文件 | 不能接入（conventions §7.1） |
| 会话文件可解析出时间顺序的事件（或 harness 直接提供事件） | 可以接入；流水图中只显示会话起止和其中的 `kta` 调用 |
| 报告 token 用量 | 可以接入；不记录用量 |
| 恢复会话 | 可以接入；每一步开新会话，从 `NOTES.md` 接续 |
| 配置可以通过配置文件和环境变量隔离，不读取用户级配置 | 按 D09 确定的隔离程度判断；做不到时在 `capabilities.py` 中写明哪些用户级内容会被读到 |
| 认证方式 | 订阅（Codex、Claude Code）或 API（pi、DeepSeek harness），见 conventions §8 |
| 读取工作目录中的 `AGENTS.md` | 可以接入；改为把固定前缀放进首条提示 |
| 可以关闭联网检索和外部工具（MCP 等） | 不能接入 |
| 可以指定模型与 provider 的 API 地址 | 不能接入（无法连到你的其他 AI API） |

禁止使用任何不持久化会话的选项（例如临时会话模式）；适配器代码中出现这类选项，gate 视为违规。

### 适配器清单

| 适配器 | 用途 | 第一阶段状态 |
| --- | --- | --- |
| `codex` | 运行时 worker 的默认适配器 | 实现并使用。参数（`codex exec`、JSON 事件、resume、`CODEX_HOME`、沙箱与审批、关闭 web search 的方式）在 WP8 逐项核对 |
| `claude_code` | 以后可能使用 | 只保留接口；方法抛 `NotSupported` |
| `pi` | 经 pi coding agent 连接你的其他 AI API | 只保留接口；接入前按准入要求逐项核对其命令行，结果写进 `capabilities.py` |
| `deepseek_harness` | 经 DeepSeek harness 连接 DeepSeek 等 API | 同上 |

provider 的 API 地址、模型名、key 都来自 AgentProfile 和控制面机器上的机密文件，不写死在适配器中。

## 受控配置（AgentProfile）

agent 的行为由三样东西决定：给它的上下文、它的循环策略、它能用的工具。这三样全部写进一个有版本的 AgentProfile 记录（spec/schemas.md §4.9），**除此之外的任何东西都不得影响 agent**。

### 上下文来源，逐项受控

| 来源 | 控制方式 |
| --- | --- |
| CLI 自身的系统提示 | 无法修改；固定 CLI 版本，写入 profile |
| 用户级配置、用户级 AGENTS.md、用户级 skills（如 `~/.codex/`） | 把 CLI 的配置目录（Codex 为 `CODEX_HOME`）指向本工作单元的持久目录 `$KTA_DATA/<env>/agent-home/<工作单元>/`；目录首次创建时内容只来自 `profile.home_files`，之后会话文件也保存在这里（conventions §7.1） |
| 目录树上方的 AGENTS.md | 工作区是独立的 git 仓库；运行前检查从文件系统根到工作区的路径上没有其他 AGENTS.md，有则拒绝启动 |
| 工作区文件 | 只允许：`AGENTS.md`（由 prefix_template 生成）、`TASK.md`、`NOTES.md`、`inputs/`、`out/`、`.agents/skills/`（由 profile.skills 复制） |
| MCP 服务器、web search 等内置工具 | 关闭（`cli_flags` 与隔离目录中的配置） |
| 环境变量 | 白名单：任务令牌、`KTA_TRACE_PARENT`、`PATH`（只含 kta 与基础工具）、provider 的 key；其余清空 |
| 运行中获取的知识 | 只能经 `kta` 命令，每次调用都记录 |

具体的 Codex 参数（沙箱模式、审批策略、关闭 web search 的方式、`CODEX_HOME` 是否完整生效）在 WP8 中逐项核对，核对结果写进 profile 和交付说明。

### 上下文清单

运行器启动 agent 前，把上表每一项的实际内容（路径与 hash）写成上下文清单，存为产物，AgentRun 引用它。用途：

- 复现：同一个 profile、同一份清单，应得到同样的上下文；
- 实验公平：不同实验臂的清单，只能在协议声明的字段上不同（例如知识的提供方式），实验运行器逐项比对。

### 修改 profile 的流程

- profile 是记录，只能出新版本。当前使用的版本由指针 `current_profile` 指定。
- 改指针需要你批准（前端"批准"操作，写 HumanAction）。施工 agent 和运行时 worker 都不能修改 profile、提示模板或运行时 skills；这些文件不出现在 worker 工作区中。
- 改进进程可以提出新的 profile 版本，在开发集 case 上与当前版本对比（token、时间、通过率），对比结果附在批准请求里。
- 实验协议固定 profile ID，实验期间不变。

### 行为合规监测

运行器和 `kta` 根据调用日志与追踪，对每次运行统计：被拒绝的越权尝试、不在 `allowed_commands` 中的命令、工作区外的写入、`NOTES.md` 是否按要求更新。结果写入 `AgentRun.violations`。同类违规在多次运行中反复出现时，进入待处理队列。

## 运行时 skills

源文件放在 `src/kta/agents/skills/<名>/SKILL.md`，随代码版本化。运行器启动 worker 时，把任务卡列出的 skills 复制到 worker 工作区的 `.agents/skills/`，并在工作区写入 `AGENTS.md`（固定前缀）；不安装到用户级目录。skill 只写方法，不写具体硬件或 kernel 的设计结论（docs/design/10 §6）。第一批：

| skill | 用于 | 状态 |
| --- | --- | --- |
| `use-kta` | 所有任务 | WP8 编写 |
| `find-knowledge` | 所有任务 | WP8 编写 |
| `write-family` | T1、T6 | WP8 编写 |
| `decompose-kernel` | T2 | 已有初稿 |
| `write-template` | T3 | WP8 编写 |
| `diagnose-model` | T4 | WP11 编写 |
| `write-microbench` | T5 | WP11 编写 |

## 验收

- 真实运行：一次真实的 T1（用 WP7 的 GEMM 族为起点，改一个 dtype 或形状条件），从领取到被接受或被正确驳回；有完整的 AgentRun 和时间构成。
- 一次会话恢复：被驳回后重试时使用 `resume`，交付说明给出两次的墙钟时间对比；会话文件能被 `parse_session` 解析出事件。
- 权限：在 agent 工作区中尝试直接写存储目录、直接调用 GPU，均被拒绝或被检测到。
