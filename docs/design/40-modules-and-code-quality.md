# 40 模块、可替换性与代码质量

## 1. 什么需要可替换

| 替换轴 | 例子 | 由哪个接口隔离 |
| --- | --- | --- |
| GPU 目标 | GB10 → H100、RTX 5090 | `Target` |
| kernel 后端 | CUDA C++/CUTLASS、Triton、TileLang | `Backend` |
| 知识库 | 不同的池快照、不同的构造方式 | Store 中的快照（见 20 §5） |
| 成本模型 | 按单元瓶颈模型、学习型模型 | `CostModel` |
| 求解器 | 枚举、MIP、贝叶斯优化 | `Solver` |
| LLM provider | Claude Code、Codex、其他 | `AgentAdapter` |
| 执行位置 | 本机进程、远端 worker、容器 | `Executor` |
| 存储 | SQLite → 其他数据库 | `Store`、`Ops` |

替换某一项时，只需要新写该接口的一个实现，再在配置中选用它，上层代码不改。

## 2. 模块

```
src/kta/
  core/        记录 schema、ID 计算、公共类型。不依赖任何其他模块
  store/       Store 与 Ops 接口及 SQLite 实现；产物存储
  targets/     Target 接口；gb10/ 为第一个实现
  problems/    Problem、Contract、Case 的定义；参考实现；输入生成器
  templates/   Template 包格式、加载、校验；变量空间；合法条件求值
  backends/    Backend 接口；cuda/、triton/、tilelang/ 各一个实现
  execution/   Executor 接口；任务执行、GPU 安静租约、超时与取消
  checks/      数值检查、sanitizer、反作弊、SASS 特征比对
  measure/     测量 harness、协议、profile 适配
  models/      CostModel 接口；下界计算；按单元模型；校准
  solve/       Solver 接口；枚举、MIP、贝叶斯优化
  knowledge/   Pattern、Template、Chain、Family、Reference、Gap；Γ 求值；链一致性检查；检索；文字渲染器
  formulation/ 从组合链拼装族的 formulation（各步的变量、约束、成本块＋胶合约束）
  agents/      AgentAdapter 接口；运行器；提示生成；工作区
  orchestrator/ 三个循环的状态机；任务调度
  cli/         kta 命令
  api/         HTTP API（读接口和操作接口）
web/       前端
```

### 2.1 依赖方向

```
core ← store ← {targets, problems, templates, knowledge}
             ← {backends, execution, checks, measure, models, formulation, solve}
             ← orchestrator ← agents
             ← {cli, api} ← web
```

- 只允许依赖箭头左边的模块。
- 用 import-linter 在 CI 中强制检查。违反依赖方向的代码无法合入。
- `checks` 和 `templates` 互不依赖。检查器不能 import 生成器的代码，避免"用生成器自己的逻辑检查自己"。

### 2.2 主要接口

```python
class Target(Protocol):
    def identity(self) -> TargetIdentity: ...        # GPU、驱动、工具链、镜像
    def hw_model(self) -> HwModel: ...               # 带不确定度的参数
    def limits(self) -> ResourceLimits: ...          # smem、寄存器、线程数等上限
    def microbenchmarks(self) -> list[Microbench]: ...
    def compile_flags(self, backend: str) -> list[str]: ...

class Backend(Protocol):
    def build(self, source: Source, target: Target) -> BuildResult: ...  # 统一的可加载模块＋编译反馈
    def supports(self, target: Target) -> Support: ...

class CostModel(Protocol):
    def predict(self, candidate, case, hw: HwModel, build: BuildResult | None) -> Prediction: ...
    def fit(self, data: CalibrationSet) -> "CostModel": ...

class Solver(Protocol):
    def solve(self, space: VariableSpace, case: Case, model: CostModel, budget) -> list[Candidate]: ...

class AgentAdapter(Protocol):
    def run(self, task_md: Path, workspace: Path, env: dict, limits: Limits) -> Iterator[AgentEvent]: ...

class Executor(Protocol):
    def submit(self, job: Job) -> JobHandle: ...
    def quiet_lease(self) -> ContextManager: ...
```

所有后端都编译成同一种可加载模块（cubin 或 .so，C ABI 的 launch 入口加元数据），由同一个测量 harness 加载和计时。

## 3. 配置

- 每个部署和每次实验对应一个 TOML 配置，选择各接口的实现及其参数，例如 `target = "gb10"`、`solver = "mip"`、`agent.adapter = "claude_code"`。
- 配置的 hash 写入每条记录的 `refs`。
- 不使用全局状态。各模块通过一个上下文对象获得依赖，测试时可以替换为假实现。

## 4. 换一块 GPU 要做什么

1. 新写一个 `targets/<name>/`：身份探测、资源上限、编译参数。
2. 在新 GPU 上运行 microbenchmark，生成该 Target 的第一个 HwModel。
3. 用锚点集（已复现的参考实现）检验各后端在新 GPU 上的可用性。
4. 已有 Template 的合法条件，用新 Target 的上限重新求值。不合法的自动排除；合法的需要重新校准成本模型后才能使用。

已有记录不受影响。新 GPU 上的结果引用新的 Target，自然与旧结果区分开。

## 5. 代码质量规则

- **类型**：所有公共接口都有类型标注。用 pyright 检查，`core`、`store`、`orchestrator` 使用严格模式。
- **格式与 lint**：Ruff。
- **测试分三层**：
  1. 单元测试：纯逻辑，不依赖 GPU，几秒内跑完；
  2. 接口合同测试：每个接口一套，所有实现都必须通过。例如任何 `Backend` 实现都要能编译并运行同一个参考 kernel；
  3. 真机集成测试：作业经 Executor 在真实 worker 上运行，带标记，不在每次提交时运行，但每个里程碑必须运行。
- **模块大小**：单个文件超过 500 行就拆分。
- **测试替身的边界**：接口的假实现只能用于单元测试。任何声称"能力已接通"的说法，都必须有真机集成测试或真实运行记录作为证据（见 60）。
- **每个模块有一份 `README.md`**：一页以内，写职责、接口、依赖。它同时就是给施工 agent 读的模块规范（见 60 §2）。

## 6. 软件栈

| 用途 | 选择 | 理由 |
| --- | --- | --- |
| 语言与包管理 | Python 3.12、uv workspace | 与生态和现有经验一致 |
| schema | pydantic v2 | 记录的校验和序列化 |
| 存储 | SQLite（标准库 sqlite3，WAL 模式，FTS5） | 单机足够；不需要额外部署 |
| API | FastAPI、uvicorn、SSE | 简单，各个模型都熟悉 |
| 前端 | TypeScript、React、Vite；图表用 ECharts | 各个模型写起来都最稳定；图表需求较多 |
| MIP | HiGHS（highspy） | 已有使用经验 |
| profile 与 SASS | Nsight Compute CLI、cuobjdump、nvdisasm | 官方工具 |
| 容器 | worker 上的 Docker，镜像按 digest 固定 | 固定工具链；隔离外部代码 |
| 环境锁定 | `.python-version`＋`uv.lock`（`uv sync --frozen`）；`web/.nvmrc`＋`package-lock.json`（`npm ci`）；每台机器生成 EnvManifest | 在任何机器上可以重新部署并核对版本（spec/conventions.md §7） |
| 部署 | `scripts/bootstrap`（控制面）、`scripts/worker-setup`（worker） | 可重复执行 |
| 质量工具 | Ruff、pyright、import-linter、pytest | — |
| kernel 后端 | 待 M0 确认 sm_121 支持情况后决定 | — |

## 依据与待核实

- 各工具在 GB10（ARM、DGX OS）上的可用性需要 M0 确认，特别是 Triton、TileLang 和 Nsight Compute。
- 前端栈选 React，是基于"模型写得最稳"的经验判断，没有做过比较。
