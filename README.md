# KTA（Kernel Template Agent）

一个从 GB10 起步、基于细化 Template 积累的 kernel 优化 agent 系统。本仓库是一个独立的 agent workspace（自己的 git 仓库、AGENTS.md 和 `.agents/skills/`）。它位于研究仓库 agentic-codesign 之中，但不 import 其中的代码；上一级目录的材料只作只读参考。论文中的方法名 Formulaic 保持不变，与本系统分开。

**当前状态（2026-10-30）**：设计文档和施工规范已写好，尚未开始实现。已确认：名称 KTA、独立 workspace、运行时先接 Codex（预留 Claude Code、pi、DeepSeek harness）、加入流水图、AI 调用必须可追溯、控制面与计算资源可分离部署、环境自包含、在本机开发；其余决策见决策日志。

## 给施工 agent

从 [AGENTS.md](AGENTS.md) 开始，然后读 [spec/workplan.md](spec/workplan.md) 中你负责的工作包。

## 给人读的设计文档

| 文档 | 回答什么 | 阅读时间 |
| --- | --- | --- |
| [00 总览](docs/design/00-overview.md) | 系统要证明什么、和谁比、做哪几件事、怎样变好 | 10 分钟 |
| [01 思维模型](docs/design/01-thinking-model.md) | 人读的整体思考：要证明什么、怎样看问题、在意什么、必须存在什么、已定与待定的决策、要做的实验、范围 | 15 分钟 |
| [02 设计笔记](docs/design/02-design-notes.md) | 讨论产生的细节，按待合并的目标文档组织：推导、知识层次、Template 效应、硬件性能模型、稳态投影与 formulation、循环与桥接、验证、学习流程；完整决策台账与待定决策 | 40 分钟 |
| [变更日志](docs/changelog.md) | 每轮设计讨论改了什么、提出了哪些设计要求及其状态、施工接下来该做什么 | 5 分钟 |
| [10 Agent 系统](docs/design/10-agent-system.md) | 编排器与任务；agent 的角色、会话、上下文和实例数；工具、skills 与池知识的区分；失败处理 | 25 分钟 |
| [11 进程、资源与并发](docs/design/11-processes-resources-concurrency.md) | 几个独立进程怎样分工；时间和资源边界；并行、锁与崩溃恢复 | 15 分钟 |
| [12 Agent 控制项](docs/design/12-agent-controls.md) | 预算、失败轮数、会话与上下文、防钻牛角尖／偷懒／hack 检查、升级与可靠性：每个旋钮的默认值、它防止什么失败、在哪里设置（含拟开放的前端设置） | 10 分钟 |
| [20 存储与检索](docs/design/20-storage-retrieval.md) | 存什么、放哪里、怎样版本化、怎样查 | 15 分钟 |
| [21 知识准入与覆盖](docs/design/21-knowledge-admission-coverage.md) | 知识怎样被记录和分级使用；不知道全集时怎样估计覆盖；结构自身的覆盖 | 15 分钟 |
| [22 细化 Template、组合链与可执行族](docs/design/22-template-format-retrieval-composition.md) | 知识单元的定义（沿用论文）；组合链；可执行族；formulation 的拼装；增量复用；检索；保证等级 | 20 分钟 |
| [23 知识的结构](docs/design/23-knowledge-structure.md) | 知识对象的类型和关系图；各功能怎样在图上计算 | 10 分钟 |
| [24 拆解参考实现](docs/design/24-reference-decomposition.md) | 怎样把 SOTA 实现拆成组合链来检验覆盖；首批对象；可预见的难点 | 10 分钟 |
| [30 前端与监督](docs/design/30-frontend-and-supervision.md) | 你怎样观察系统、什么时候需要你、能做哪些介入；覆盖页；流水图 | 10 分钟 |
| [40 模块与代码质量](docs/design/40-modules-and-code-quality.md) | 模块怎样拆分、什么可以替换、软件栈 | 10 分钟 |
| [50 实验与证据](docs/design/50-experiments.md) | 实验设计、测量与检查的规则 | 10 分钟 |
| [60 施工与验收](docs/design/60-build-and-acceptance.md) | coding agent 怎样施工、怎样防止假完成、你怎样验收 | 10 分钟 |
| [70 其他 kernel agent](docs/design/70-related-systems.md) | 对照维度；已知系统；可借鉴的检查项 | 10 分钟 |
| [决策日志](docs/decisions.md) | 每个决策一行：选了什么、谁定的、状态 | 5 分钟 |

## 施工规范

| 文件 | 内容 |
| --- | --- |
| [AGENTS.md](AGENTS.md) | 施工入口：读什么、原则、环境、决策、交付 |
| [spec/conventions.md](spec/conventions.md) | 工程约定：记录、产物、日志的分工；环境与数据隔离；代码、错误、日志、测试；远端运行与证据清单；不允许的做法 |
| [spec/schemas.md](spec/schemas.md) | 全部记录与知识对象的字段；表达式语言；状态推进规则 |
| [spec/modules/](spec/modules/README.md) | 各模块的职责、接口、行为、验收；模块层次 |
| [spec/workplan.md](spec/workplan.md) | 14 个工作包：依赖、产出、验收、允许的简化 |
| [spec/gate.md](spec/gate.md) | 验收脚本的检查项 |
| [.agents/skills/](.agents/skills/README.md) | 施工 skills（workspace 级） |
| [src/kta/agents/skills/decompose-kernel](src/kta/agents/skills/decompose-kernel/SKILL.md) | 运行时 skill：拆解参考 kernel 的方法 |

## 两种 agent

- **施工 agent**：编写 KTA 本身代码的 coding agent，可以是 Claude Code、Codex 或其他模型。见 60 与 AGENTS.md。
- **运行时 agent（worker）**：KTA 运行时调用的 LLM agent，负责写族、拆解参考实现、提取 Template 等。见 10。

## 依据标记

每份设计文档末尾有"依据与待核实"一节，标出哪些内容是凭记忆或推断写的、哪些数值是先填的。检查义务后移：第一阶段靠经验和测试保证可用，形式化保证与覆盖在积累中逐步可见（22 §6、21 §2.4）。

早期长稿 [kernel-agent-design.md](../docs/discussions/kernel-agent-design.md) 仅作素材，以本目录为准。
