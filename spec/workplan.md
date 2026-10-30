# 工作包

每个工作包（WP）可以由一个 agent 独立完成。依赖满足后即可开工；没有依赖关系的 WP 可以同时进行。每个 WP 的完成以 `scripts/gate <WP>` 通过、交付说明齐全为准（AGENTS.md）。

## 依赖关系

```
WP0 ─┬─ WP1 ─┬─ WP2 ─ WP3 ─┐
     │       ├─ WP4 ─ WP5 ─┴─ WP6 ─ WP7 ─┬─ WP8 ─┬─ WP10b
     │       └─ WP9a ───────────────────┘       ├─ WP9b
     │                                          └─ WP11 ─ WP12
     ├─ WP10a（只依赖 spec/schemas.md 与 WP0）
     └─ WP13（无代码依赖）
```

| 里程碑 | 包含的 WP | 意义 |
| --- | --- | --- |
| M0 探测 | WP2 的探测部分、WP3 的噪声测量、WP13 | 用实测数据修订默认值 |
| M1 不接 LLM 的完整链路 | WP0–WP7、WP9a | 基础设施本身正确 |
| M2 接入 agent | WP8、WP9b | agent 在链路中工作 |
| M3 覆盖与扩展 | WP10a、WP10b、WP11 | 拆解参考实现，覆盖可见 |
| M4 实验 | WP12 | 第一批对照实验 |

---

## WP0 仓库骨架与工具链

- **依赖**：无。
- **读**：AGENTS.md；spec/conventions.md；spec/modules/README.md；spec/gate.md。
- **产出**：
  - `.python-version`、`pyproject.toml`（uv；Python 3.12；依赖：pydantic、fastapi、uvicorn、highspy、pyyaml、numpy；开发依赖：ruff、pyright、import-linter、pytest）；
  - `src/kta/<模块>/__init__.py` 与每个模块的 `README.md`（从 spec/modules 对应规范裁剪：职责、接口、依赖）；
  - import-linter 配置：按 spec/modules/README.md 的层次，加上"checks 不得 import knowledge"；
  - `scripts/gate`（框架与通用检查，见 spec/gate.md）；`scripts/remote-run`（施工中的手工远端操作）；`scripts/bootstrap`（conventions §7）；
  - EnvManifest 的生成，以及 `kta doctor` 对照锁文件的检查；
  - `core/registry.py`、`core/errors.py`、`core/logging.py`、`core/doctor.py`（可以先于 WP1 完成，WP1 在此基础上继续）；
  - `.gitignore`（数据、日志、工作区、产物、`.env`、`secrets.env`）；
  - `tests/fakes/`、`tests/contract/` 目录及说明；证据清单的 JSON schema（`deliveries/evidence.schema.json`）；
  - `deliveries/` 目录；`tests/acceptance/LOCK`（空）；
  - 施工 skills：`.agents/skills/run-gate/SKILL.md`、`.agents/skills/write-delivery/SKILL.md`、`.agents/skills/remote-run/SKILL.md`（先写通用的 ssh 用法，WP2 补充数据目录和环境）。
- **验收**：`scripts/gate WP0` 通过；以下每一项都有一个测试，证明 gate 能拦下：违反层次的导入、`src/` 中的 `FakeX` 类、裸 `except:`、`src/` 中的 `print`、一个假的 API key、在 dev 环境注册并选用 fixture 实现。
- **允许的简化**：无。

## WP1 core 与 store

- **依赖**：WP0。
- **读**：spec/schemas.md 全文；spec/modules/core-store.md；docs/design/20。
- **产出**：`core/`、`store/` 全部内容；所有 kind 的 pydantic 模型；表达式语言；SQLite 建表与迁移。
- **验收**：spec/modules/core-store.md 的"验收"全部满足。真实运行在 spark 上完成。
- **允许的简化**：无。

## WP2 targets、execution、backends

- **依赖**：WP1。
- **读**：spec/modules/target-exec.md；docs/design/11 §2–§6。
- **产出**：三个模块；`local` 与 `ssh` 两种传输，`pull` 的协议文档与接口（抛 `NotSupported`）；worker 注册与能力标签；`scripts/worker-setup`；`targets/gb10/features.yaml` 初始清单；worker 用的 Docker 镜像定义（`docker/`）与构建脚本；补全 skill `remote-run`。
- **验收**：spec/modules/target-exec.md 的"验收"全部满足。交付说明中给出探测结果表。
- **允许的简化**：容器隔离可以先只用于外部代码的构建；KTA 自身的作业先在宿主环境运行（记为决策）。

## WP3 measure 与 checks

- **依赖**：WP2。
- **读**：spec/modules/measure-check.md；docs/design/50 §2–§5。
- **产出**：两个模块；协议记录 `measure-v1`；噪声测量结果与修订后的 `measure-v2`。
- **验收**：spec/modules/measure-check.md 的"验收"全部满足。
- **允许的简化**：CUPTI 追踪若在 GB10 上不可用，改用 nsys，并记为决策。

## WP4 knowledge

- **依赖**：WP1。
- **读**：spec/schemas.md §4；spec/modules/knowledge.md；docs/design/21–24。
- **产出**：knowledge 模块；`knowledge/seed/`（Pattern 表、Structure 表、论文 GEMM 五步的 Template）；`kta init` 导入 seed 所需的函数。
- **验收**：spec/modules/knowledge.md 的单元测试部分满足；seed 中的五个 Template 通过 `validate_template`。真实运行部分在 WP7 中完成。
- **允许的简化**：检索排序中依赖 Solution 的部分，在没有 Solution 时按保证等级和复用次数排序。

## WP5 models、formulation、solve

- **依赖**：WP4。
- **读**：spec/modules/formulation-solve-models.md；docs/design/22 §3.3；docs/design/50 §6–§7。
- **产出**：三个模块。
- **验收**：单元测试部分满足；真实运行部分在 WP7 中完成。
- **允许的简化**：MIP 导出可以先只支持线性子集中的不等式和有限菜单（one-hot），其余走枚举。

## WP6 orchestrator 与 cli

- **依赖**：WP3、WP5。
- **读**：spec/modules/orchestrator-cli.md；docs/design/10 §2–§3、11。
- **产出**：两个模块中与求解循环、扩展循环、任务队列、预算、待处理相关的部分；`kta` 命令（agent 用的命令可以先不做鉴权）。
- **验收**：单元测试部分满足；真实运行部分在 WP7 中完成。
- **允许的简化**：六个进程先只实现"求解"和"维护"两个；其余在 WP11 补齐。

## WP7 第一条真实链路：GEMM

- **依赖**：WP6。
- **读**：docs/design/22、24 §3.1；spec/modules/knowledge.md 的族目录结构。
- **产出**：
  - 一个 FP16 GEMM 的 TileLang 族（`knowledge/seed/families/gemm_fp16_tilelang/`），其链由 seed 中的 Template 组成，代码标记齐全；
  - 一个 Case：M=N=K=4096 及两个非方形形状，Contract 按 docs/design/50 §3；
  - baseline：torch.matmul（cuBLAS）在同协议下的测量；
  - 一次 `kta solve` 的完整运行记录。
- **验收**：
  - 求解循环在不接 LLM 的情况下走到 `accepted`，或走到有记录的判定（此时交付说明解释原因）；
  - 整个运行可以在全新进程中从记录重放（重放结论一致）；
  - 故障注入：分别注入越界写、错误的合法条件（允许一个会产生竞争的 assignment）、偷调 cuBLAS，三者都被检查拦下；
  - `reuse_plan` 对 dtype 改为 BF16 的新 case，正确标出 Γ 不成立的步骤；
  - 预测与实测的排序相关系数记录在交付说明中（不设阈值）。
- **允许的简化**：无。这是 M1 的验收点。

## WP8 agents 与运行时 skills

- **依赖**：WP7。
- **读**：spec/modules/agents.md；docs/design/10 §4、§6。
- **产出**：agents 模块；`kta-agent` 用户与权限设置脚本；AgentProfile 第一版（`codex-v1`）及其提示模板（固定前缀、各任务类型步骤头、增量消息，放在 `src/kta/agents/prompts/`）；上下文清单；合规统计；运行时 skills（放在 `src/kta/agents/skills/`）：`use-kta`、`find-knowledge`、`write-family`、`write-template`；完善 `decompose-kernel`。
- **验收**：spec/modules/agents.md 的"验收"全部满足；另外：
  - 在 spark 的用户级 Codex 配置目录中放一个带特殊标记的 AGENTS.md，真实运行一次 T1，确认标记没有出现在 agent 的上下文中（检查事件流）；
  - 上下文清单列出的文件与 worker 工作区实际内容一致；
  - 故意在任务中诱导一次越权写入，`AgentRun.violations` 记下它；
  - 运行结束后，会话文件在 `agent-home/<工作单元>/` 中完整存在，打包的产物可以解开查看；用该会话成功恢复一次；`parse_session` 能把会话解析成事件；会话存档中不含凭据文件。
- **允许的简化**：只实现 Codex 适配器。`claude_code`、`pi`、`deepseek_harness` 只保留接口，方法抛 `NotSupported`，不运行；对 pi 和 DeepSeek harness，如果能在本机安装，按准入要求核对一遍并把结果写进 `capabilities.py`。

## WP9a api 与只读前端

- **依赖**：WP1（开发）；WP7（真实数据验收）。
- **读**：spec/modules/api-web.md；docs/design/30。
- **产出**：api 的读接口与写入接口；web 第一版的五个只读页面。
- **验收**：spec/modules/api-web.md 的"验收"中的只读部分。开发期间可以用 origin="fixture" 的合成记录，最终验收必须用 WP7 的真实记录。

## WP9b 完整前端与操作

- **依赖**：WP8、WP9a。
- **产出**：操作接口；第二版页面。
- **验收**：spec/modules/api-web.md 的"验收"全部满足；从前端发起一次 T1、取消一次任务、驳回一个结果，记录正确。

## WP10a 手工拆解参考实现（检验知识模型）

- **依赖**：WP0；只需 spec/schemas.md，不需要代码模块。
- **读**：docs/design/24；运行时 skill `src/kta/agents/skills/decompose-kernel/SKILL.md`（施工 agent 在本 WP 中按它的方法手工拆解）；spec/schemas.md §4.1–§4.5。
- **做什么**：按 docs/design/24 §4 的首批对象，逐个拆解，产出 YAML：
  - 每个参考实现一份 Reference、一条 Chain、一份 Reconstruction（含未匹配事务）；
  - 拆解中需要的新 Pattern 和新 Template（状态 proposed，保证等级 declared）；
  - 每个未匹配事务一条 Gap。
  文件放在 `knowledge/seed/references/<名>/`。
- **验收**：
  - 每个参考实现的源码冻结到 commit，引用精确到文件和行；
  - 所有 YAML 能被 pydantic 模型解析（WP1 完成后验证；之前按 schemas.md 手工核对）；
  - 交付说明给出：每个参考实现的匹配率、未匹配事务的分类统计、新增的 Pattern 与 Template 列表，以及对知识模型本身的意见（哪里描述不了、哪里粒度不合适）。
- **允许的简化**：不要求在 GB10 上运行或复现。不能在 GB10 上运行的参考实现，标为"仅分析"。

## WP10b 自动拆解与覆盖

- **依赖**：WP8。
- **做什么**：把 WP10a 的流程交给扩展循环（T2、T3）自动执行；新增至少 3 个参考实现；覆盖页显示真实数据。
- **验收**：扩展循环对一个新参考实现从 `frozen` 走到 `covered` 或有 Gap；覆盖报告中各项都有数值。

## WP11 其余进程、预算与校准

- **依赖**：WP8。
- **产出**：学习、校准、实验、汇总进程；完整的预算层级；GB10 microbenchmark 第一版（T5 或直接编写）；skills `diagnose-model`、`write-microbench`。
- **验收**：docs/design/11 §7 描述的"普通的下午"能真实发生一次，前端可见；microbenchmark 结果写成 HwFact。

## WP12 实验

- **依赖**：WP11。
- **读**：docs/design/50 §1。
- **产出**：实验协议的冻结与运行；第一批对照（从零、文字知识、Template 池；强弱模型）。具体 case 集与预算由用户确定后开工。
- **验收**：结果表、成本表、Gap 汇总；所有运行可从记录重放。

## WP13 已有数据分析

- **依赖**：无（只读上一级研究仓库中已有的实验记录）。
- **读**（外部只读材料，在研究仓库 `/Users/dingyi/Projects/agentic-codesign` 中；worktree 可能不在 kta 目录下，所以用绝对路径）：`docs/reports/formulaic-next-batch.md`；`kernel-lab/results/formulaic-reuse-f2-20261027-01/` 及相关轨迹。
- **做什么**：
  1. F 线 Agent 构造运行的时间构成：读上下文、写代码、调试、等待编译与测量，各占多少；Template 条件比文字知识条件贵在哪里。
  2. GB10 已有模型的预测误差统计。
- **产出**：`deliveries/WP13.md`，附统计脚本（放在 `scripts/analysis/`）。
- **验收**：每个数字都能追溯到原始记录文件。
