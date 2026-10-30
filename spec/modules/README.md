# 模块总览

包路径 `src/kta/`。每个模块对应一份规范：

| 模块 | 规范 | 职责 |
| --- | --- | --- |
| core | [core-store.md](core-store.md) | schema、ID、表达式语言、配置、上下文对象 |
| store | [core-store.md](core-store.md) | 记录、产物、运维表、状态推导、快照、失效传播 |
| targets、execution、backends | [target-exec.md](target-exec.md) | 硬件目标与探测；远端作业执行与 GPU 租约；TileLang 与 CUDA 编译为统一模块 |
| measure、checks | [measure-check.md](measure-check.md) | 测量 harness 与协议；数值检查、sanitizer、反作弊、SASS 检查、空间验证 |
| knowledge | [knowledge.md](knowledge.md) | Pattern、Template、Chain、Family 的加载与校验；Γ 求值；检索；增量复用；文字渲染 |
| formulation、solve、models | [formulation-solve-models.md](formulation-solve-models.md) | 从链拼装 formulation；枚举与 MIP 求解；下界；按单元成本模型与校准 |
| orchestrator、cli | [orchestrator-cli.md](orchestrator-cli.md) | 求解、扩展循环；任务队列；进程；预算；`kta` 命令 |
| agents | [agents.md](agents.md) | 任务卡、提示生成、工作区、会话、provider 适配、隔离与预算 |
| api、web | [api-web.md](api-web.md) | HTTP API（读与操作）；前端 |

## 依赖方向

按层排列，编号小的在下：

| 层 | 模块 |
| --- | --- |
| 0 | core |
| 1 | store |
| 2 | targets |
| 3 | execution |
| 4 | backends、knowledge、models |
| 5 | measure、formulation |
| 6 | checks、solve |
| 7 | orchestrator |
| 8 | agents |
| 9 | cli、api |

前端 `web/` 是独立的 TypeScript 项目，只通过 HTTP API 访问后端。

规则：

- 一个模块只能 import 层号比它小的模块；同层模块之间不能互相 import。
- 额外禁止：`checks` 不得 import `knowledge`（检查器不能用生成器自己的逻辑检查自己）。
- `cli` 和 `api` 是薄层：只做参数解析、调用下层、格式化输出。

由 import-linter 配置强制（WP0）。

## 共同约定

- 所有公共函数有类型标注；不使用全局状态，依赖通过 `kta.core.context.Context` 传入。
- 任何写入存储的操作都经过 `Store`；不直接写 SQLite。
- 任何远端执行都经过 `Executor`；不在业务代码里直接 `ssh`。
- 每个模块目录有一页 `README.md`：职责、公共接口、依赖。内容从本目录对应规范裁剪而来。
