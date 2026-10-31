# KTA 施工入口

你在构建 KTA：一个基于细化 Template 积累的 kernel 优化 agent 系统。本仓库是一个独立的 agent workspace：本文件是唯一的施工指令入口，skills 在本仓库的 `.agents/skills/`。人读版设计在 `docs/design/`，不要求通读；施工以 `spec/` 为准。

## 读什么

1. 本文件。
2. `spec/conventions.md`（必读：代码、日志、追踪、产物、环境隔离、测试、不允许的做法）。
3. `spec/workplan.md` 中你负责的工作包（WP）。
4. 该 WP 列出的 `spec/modules/*.md`，以及 `spec/schemas.md` 中相关的节。
5. 遇到"为什么这样设计"的问题时，再读 WP 指向的 `docs/design/` 小节。

本仓库位于研究仓库 `/Users/dingyi/Projects/agentic-codesign` 之中，但两者是**不同的仓库**；本仓库的 worktree 也可能位于别处，所以引用研究仓库时一律用这个绝对路径。研究仓库中的文件只能作为只读参考材料，读取时写明路径；不得修改，不得 import 其中的代码，也不受其中 `AGENTS.md` 的约束。WP 中会明确列出需要读的外部材料。

## 原则

1. **检查义务后移。** 你不需要证明分类是完整的。模式、Template 种类、硬件能力、归因类别都实现为开放的数据表；遇到描述不了的，记为 `other` 并计数，让覆盖统计能看到它。不要为了"覆盖所有情况"预先写死枚举。
2. **真实优先。** 声称接通的能力，必须以 `KTA_ENV=dev` 真实运行过（需要 GPU 的部分在真实的 GPU worker 上执行），记录列入证据清单（conventions §5）。测试替身只能放在 `tests/fakes/`，并通过接口合同测试。做不到的能力抛 `NotSupported`，不要返回假结果。任何 fixture 或简化实现，在代码中加注释 `# kta:fixture <原因>`，并在交付说明中列出。conventions §10 列出的做法一律不允许。
3. **接口以 spec 为准。** `spec/` 中的函数签名、schema 字段、命令名不得自行修改。确实需要改：如果只影响你的 WP 且可逆，按"决策"一节记录后修改，并同步改 spec；如果影响其他 WP，停下，在交付说明的"需要的变更"中写明。
4. **依赖方向。** 只能 import `spec/modules/README.md` 中层号比自己小的模块。`checks` 不得 import `knowledge`。
5. **计算资源只经 Executor。** 本仓库在本机开发，控制面（API、存储、编排器）在本机运行；编译 CUDA、运行 GPU、跑长测试，都由 Executor 交给远端 worker（spark、slai）。不要假设控制面和 worker 在同一台机器或共享文件系统（conventions §6）。施工中的手工远端操作用 skill `remote-run`（WP0 之后可用）。
6. **简单优先。** 能用标准库和已选的依赖解决的，不引入新依赖。新依赖必须写进决策日志。
7. **环境自包含。** 依赖只经锁文件安装（`uv sync --frozen`、`npm ci`），不装全局包、不装到用户级目录；新机器上用 `scripts/bootstrap`、`scripts/worker-setup` 部署（conventions §7）。
8. **调用 AI 必须可追溯。** 任何调用 AI 模型的代码都经过 agents 模块的适配器，会话持久保存、用量有记录；禁止临时会话模式和不留记录的临时环境（conventions §7.1）。

## 环境

- Python 3.12；用 uv 管理，包为 `src/kta/`。
- 质量工具：Ruff、pyright、import-linter、pytest。
- 开发与控制面：本机。数据目录由环境变量 `KTA_DATA` 指定（本机与各 worker 各自设置）。
- worker：`ssh spark`（GB10，GPU 作业）、`ssh slai`（CPU 作业、备份）。
- 语言：项目内容一律用英文，包括代码、注释、新写或修改的文档、交付说明、证据清单、决策日志新条目、review queue、commit message、给其他 session 的任务说明。已有的中文规范文档不必翻译。和用户对话可以用中文。

## skills

- **施工用的 skills** 放在本仓库的 `.agents/skills/<名>/SKILL.md`。新建的施工 skill 也一律放在这里，不安装到用户级目录（例如 `~/.codex/skills`），不使用 `CLAUDE.md`。
- **运行时 worker 用的 skills** 是产品的一部分，源文件在 `src/kta/agents/skills/<名>/SKILL.md`。运行器在启动 worker 时，把它们安装到 worker 工作区的 `.agents/skills/`（spec/modules/agents.md）。施工时不要把它们当作自己的 skill 使用，WP 明确要求时除外（例如 WP10a 使用 `decompose-kernel`）。

## 决策

- 只影响局部、可逆的决定：自己做，并在 `docs/decisions.md` 的"默认"表末尾追加一行，ID 接着编号，"文档"列写 WP 编号。
- 横切多个模块、难以逆转、涉及价值判断、或会改变验收标准的决定：先按最佳判断决定并继续推进；在 `docs/decisions.md` 中记为 `default — needs user review`，并在 `docs/review-queue.md` 追加一条（问题、选择、理由、改起来的代价）。
- 以下必须停下等用户：删除或覆盖已有数据；在远端机器上安装系统级软件或修改系统配置；需要花钱的操作；修改 main 或 push；需要用户的账号登录或凭据。

## git

- 每个 WP 一个分支 `wp/<WP 编号>`；commit message 格式 `<WP>: <摘要>`。
- 不 push，除非用户要求；不合并到 main。
- 不触发 `git gc` 或后台 maintenance。用户要求 push 时，用低优先级、单线程：`nice -n 19 git -c pack.threads=1 push`。
- 数据、数据库、日志、工作区、产物、机密一律不进 git。

## 机密

- LLM provider 的 API key 只放在控制面所在机器的 `~/.config/kta/secrets.env`。不得写进代码、记录、日志、提示、交付说明。

## Progress list (Claude Code desktop only)

If you run in the Claude Code desktop app and have the task-list tool (`mcp__ccd_turn__update_plan`), advance it as each step starts (`{"advance": true}` alongside that step's first call), not only at the end; the user watches it. Other runtimes (Codex, pi, …) do not have this tool: ignore this section.

## 交付

1. 运行 `scripts/gate <WP>`，必须全部通过。
2. 写证据清单 `deliveries/<WP>.evidence.json`（conventions §5）和交付说明 `deliveries/<WP>.md`，交付说明包含：
   - 做了什么（接通的能力，每项附真实运行记录的位置）；
   - fixture 与简化（没有就写"无"）；
   - 已知问题；
   - 新增的决策（ID 列表）；
   - 需要的变更（没有就写"无"）；
   - gate 输出（原样粘贴）。
3. 提交到分支 `wp/<WP 编号>`。

没有通过 gate 的工作不算完成，不要声称完成。
