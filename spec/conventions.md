# 工程约定

所有施工 agent 必读。本文规定代码怎样写、输出存在哪里、怎样在远端运行，以及怎样从结构上防止 mock 或 toy 实现被当成完成。gate 中的大部分检查都以本文为依据（spec/gate.md）。

## 1. 三类输出：记录、产物、日志

| | 记录 | 产物 | 日志 |
| --- | --- | --- | --- |
| 是什么 | 证据：结论依赖的一切 | 大文件：源码、二进制、原始输出、事件流 | 运行过程的调试信息 |
| 写入方式 | `Store.put` | `Store.put_blob` | `logging`（§4） |
| 存放位置 | `$KTA_DATA/<env>/kta.db` | `$KTA_DATA/<env>/objects/` | `$KTA_DATA/<env>/logs/<进程名>/<日期>.jsonl` |
| 能否作为结论依据 | 能 | 被记录引用后能 | **不能** |
| 保留 | 永久 | 永久 | 30 天 |

规则：任何结论、数字、状态变化，必须落成记录。只存在于日志、stdout 或交付说明里的数字，不算证据。

## 2. 环境与数据隔离

`KTA_ENV` 取三个值之一，决定数据目录和允许的实现：

| KTA_ENV | 用途 | 数据目录 | 允许的实现 | 允许写入的 origin |
| --- | --- | --- | --- | --- |
| `test` | pytest | 临时目录，测试结束删除 | 真实实现和 `tests/fakes/` 中的替身 | 任意 |
| `dev` | 各 WP 的真实运行（控制面在本机，作业在真实 worker 上） | `$KTA_DATA/dev/<WP>/` | 只允许 `src/` 中的真实实现 | real、fixture |
| `main` | 正式运行 | `$KTA_DATA/main/` | 只允许 `src/` 中的真实实现 | real、historical |

由代码强制：

- `Store` 在 `main` 环境拒绝 origin 为 fixture 或 simulated 的记录。
- `src/` 不得 import `tests/` 下的任何东西（import-linter 规则）。
- 接口实现通过 `kta.core.registry` 注册和选择（§3.2）。注册表只扫描 `src/kta/`。
- `kta doctor` 打印每个接口当前选用的实现、环境、数据目录。在 `dev` 和 `main` 环境，编排器启动前自动运行 doctor；任何接口选到带 `FIXTURE = True` 标志的实现，就拒绝启动。
- 只有 main 分支上的代码可以写入 `main` 环境；每条记录在 actor 中带上代码的 git commit，有未提交修改的代码只能写入 `dev`。

## 3. 代码

### 3.1 结构

- 接口用 `typing.Protocol` 定义在 `<模块>/base.py`；实现放在同一模块的子目录或文件中。
- 跨模块的数据结构用 `kta.core.schemas` 中的 pydantic 模型；模块内部可以用 `@dataclass(frozen=True)`。
- 不使用模块级可变状态、单例或全局缓存。依赖通过 `Context` 传入。
- 时间只从 `ctx.clock.now()` 取，随机数只从 `ctx.rng(name)` 取（种子写进相关记录）。这样测试可以控制时间和随机性，不需要 mock。
- 子进程只能经 `Executor` 启动（`execution` 模块自身除外）。
- 路径不写死，从 `ctx.config` 取。
- 单个文件不超过 500 行；单个函数一般不超过 60 行。
- 注释用英文，说明为什么，不复述代码做了什么。

### 3.2 实现的注册与选择

```python
# kta/core/registry.py   (revised by D77: the registry is an object carried by ctx.registry)
Registry.register(interface: str, name: str, factory: Callable[[Context], Any], fixture: bool = False)
Registry.select(interface: str, config: Config) -> Implementation   # [impl] name, else the only real one
resolve(interface: str, ctx: Context) -> Any        # = ctx.registry.resolve; fixture in dev/main -> PermissionDenied
build_registry() -> Registry   # calls register_implementations(registry) of each package under src/kta
```

- Each package under `src/kta/` that provides implementations defines `register_implementations(registry: Registry) -> None` in its `__init__.py`. There is no module-level registry (no global mutable state); tests build their own `Registry`.
- `register` refuses `fixture=True` for factories defined in `kta.*` and `fixture=False` for factories defined elsewhere.

- 每个实现在注册时声明 `fixture`。`src/` 中注册的实现一律 `fixture=False`；`tests/fakes/` 中的替身在测试启动时注册，`fixture=True`。
- 需要的能力在当前 Target 或环境上做不到时，实现必须抛 `NotSupported`（§3.3），不得返回假结果。

### 3.3 错误

`kta.core.errors`：

```
KtaError
├─ ConfigError
├─ SchemaError              记录或知识对象不合 schema
├─ NotSupported             当前 Target、后端或环境不支持这项能力
├─ ExecutionError(kind)     kind ∈ timeout | oom | gpu_hang | infra | nonzero
├─ CheckerError             检查器自身出错（与"检查不通过"不同）
├─ BudgetExceeded
└─ PermissionDenied
```

- 不允许裸 `except:`，不允许 `except Exception: pass` 或捕获后返回默认值。捕获异常后，要么处理并记录原因，要么包装成上述类型重新抛出。
- "检查不通过"是正常结果（`Check.passed=False`），不是异常。

### 3.4 单位与命名

- 字段名带单位后缀：`_s`、`_us`、`_bytes`、`_flops`。内部计算用秒，对外报告延迟用微秒。
- 时间戳一律 UTC，ISO 8601。
- 作业、任务、会话等运行时 ID 用 ULID；记录 ID 用内容 hash（schemas.md §1）。

## 4. 日志

- 用标准库 `logging`，由 `kta.core.logging.setup(ctx)` 统一配置，输出 JSON lines。
- 每行必有的字段：`ts`、`level`、`logger`、`msg`、`env`、`process`；在相应上下文中自动带上 `loop_id`、`task_id`、`job_id`、`session`（用 `contextvars` 传递）。
- 级别：DEBUG 只在 dev；INFO 记录状态转移和作业起止；WARNING 记录降级或重试；ERROR 记录失败。
- 不记录大内容：编译输出、事件流、提示全文先存为产物，日志里只写产物 ID。
- 不记录机密：日志过滤器对形如 API key、token 的字符串打码（§8）。
- `src/` 中不得使用 `print`，`cli` 模块的输出函数除外。

## 4.1 追踪（span）

系统里每一段有起止的工作都记一个 span，用于前端的流水图（docs/design/30 §1.2）和时间构成统计。

```python
# kta/core/trace.py
with span(kind, name, lane, **attrs): ...      # 进入时写开始，退出时写结束和状态
current_span_id() -> str | None
```

| 字段 | 含义 |
| --- | --- |
| `span_id`、`parent_id`、`trace_id` | ULID；`trace_id` 为循环实例、实验运行或人工命令的 ID |
| `kind` | `loop_step`、`task`、`agent_session`、`agent_model`、`tool_call`、`job_queue`、`job_run`、`lock_wait`、`lock_hold`、`compile`、`check`、`measure`、`profile`、`solve`、`human_wait`，开放表，可以新增 |
| `lane` | 泳道：`process:<名>`、`agent:<槽位>`、`worker:<主机>:<执行池>`、`gpu:<主机>`、`human` |
| `name` | 一句短说明，例如 `T1 gemm_bf16`、`nvcc gemm@3` |
| `start_ts`、`end_ts` | UTC；进行中的 span `end_ts` 为空 |
| `status` | `ok`、`fail`、`cancelled`、`running` |
| `attrs` | task_id、job_id、记录 ID、token、字节数等 |

规则：

- **系统事件由 KTA 自己维护，必须实时记录**：编排器的每次状态转移、每个任务、每个作业（排队与运行分成两个 span）、每次取锁与持锁、每次 agent 会话的起止、每次 `kta` 命令调用。`kta` 命令从环境变量 `KTA_TRACE_PARENT` 取得父 span，把自己的 span 挂在会话之下。
- **agent 循环内部的事件不要求实时记录**，要求的是会话本身被完整保存（§7.1）。需要时由该适配器的会话解析器从保存的会话文件中解析出模型段和工具段（`agent_model`、`tool_call`）；某些 harness 直接提供事件，就直接使用。解析结果可以缓存，但以会话文件为准。前端默认把一次 agent 会话显示为一个色块加上其中的 `kta` 调用，点开时再展开解析出的内部事件。
- span 存在运维表 `spans` 中（允许更新结束时间），保留 90 天。时间构成的汇总另写进 AgentRun 等记录，作为证据长期保存。
- span 只记元数据，不记大内容；需要时在 attrs 里放产物 ID。

## 5. 施工期的真实运行证据

- 施工在本机进行：控制面（API、存储、编排器）在本机以 `KTA_ENV=dev` 运行，数据在本机的 `$KTA_DATA/dev/<WP>/`。需要 GPU 或大量 CPU 的作业，经 Executor 交给远端 worker（§6）。
- worker 上的代码与控制面必须是同一个 commit：`scripts/worker-setup` 按当前 commit 部署，worker 注册时报告 commit，不一致的 worker 不接收作业。有未提交修改时，运行产生的记录标记为 dirty，不能作为证据。
- 每个 WP 交付时附证据清单 `deliveries/<WP>.evidence.json`：

```json
[{"claim": "两个后端各编译并运行一个向量加法 kernel，输出正确",
  "command": "kta ...",
  "git_commit": "…", "dirty": false, "env": "dev",
  "control_host": "…", "workers": ["spark-gpu0"],
  "record_ids": ["…"], "blob_ids": ["…"], "env_manifests": ["…"], "started_at": "…"}]
```

- gate 核对：每个 `record_ids` 存在于该 WP 的 dev 存储中，origin 为 real，代码 commit 与清单一致且不是 dirty；需要 GPU 的能力，其记录必须来自报告了 GPU 能力标签的 worker。
- 大文件、数据库、日志、工作区一律不进 git（`.gitignore` 由 WP0 写好）。

## 6. 部署拓扑与计算资源访问

控制面和计算资源可以在不同的机器、不同的集群。代码中不得假设它们在同一台机器上，也不得假设有共享文件系统。

| 角色 | 职责 | 第一阶段默认位置 |
| --- | --- | --- |
| 控制面 | API、存储、编排器、agent 运行器 | 开发时在本机；正式运行时在任意一台常驻机器，由配置决定 |
| GPU worker | 检查、测量、profile、microbenchmark；可兼做编译 | spark |
| CPU worker | 编译、CPU 侧检查、求解 | spark 或 slai |
| 前端 | 浏览器 | 本机，经 HTTP 访问控制面 |

规则：

- **访问计算资源的唯一途径是 Executor。** 除 worker 进程外，KTA 中没有任何代码直接使用 GPU，也没有任何代码直接在远端执行命令。
- **三种传输方式**，由配置为每个 worker 选择：
  - `local`：worker 与控制面在同一台机器；
  - `ssh`：控制面经 ssh 在 worker 上执行作业，适合控制面能 ssh 到 worker 的情况（例如本机 → spark）；
  - `pull`：worker 上运行守护进程 `kta worker`，主动向控制面 API 领取作业、上传结果，适合跨集群、只能由计算侧单向访问的情况。
  第一阶段实现 `local` 和 `ssh`；`pull` 的接口和协议写好，实现可以在需要时补上（未实现前抛 `NotSupported`）。
- **按能力选择 worker，不按主机名。** worker 注册时报告能力标签（由探测得出），例如 `gpu=sm_121`、`cuda=13.0`、`pool=gpu-exclusive`、`backend=tilelang`。作业声明它需要的标签，Executor 按标签匹配。
- **产物按 hash 传输。** 作业的输入和输出都是产物，传输后校验 sha256；worker 本地有产物缓存。
- **GPU 独占租约按 GPU 设备计**（worker 名＋设备号），由该 worker 执行。
- **每个作业的结果记录**：worker 名、主机、设备、环境清单 ID（§7）、代码 commit。
- **凭据**：`ssh` 方式使用用户已有的 ssh 配置别名（spark、slai）；`pull` 方式的 worker 使用控制面签发的 worker 令牌。凭据不进 git、记录和日志。
- 远端的手工操作（施工中调试用）统一用 `scripts/remote-run`，不在业务代码中直接调用 ssh。

## 7. 运行环境自包含

目标：在任何一台新机器上，只靠仓库里的文件就能重新部署控制面或 worker，并且能确认部署出来的环境与记录一致。

| 部分 | 锁定方式 | 安装方式 |
| --- | --- | --- |
| Python | `.python-version`（精确到补丁版本）、`pyproject.toml`、`uv.lock`，全部提交 | 只用 `uv sync --frozen`；不用系统 pip，不装全局包 |
| 前端 | `web/.nvmrc`（Node 精确版本）、`package.json` 的 `engines`、`package-lock.json`，全部提交 | 只用 `npm ci` |
| GPU 工具链 | 优先用固定 digest 的容器镜像；不用容器时，`scripts/worker-setup` 按锁定版本安装到 worker 的项目目录 | 不装到用户级目录 |

- **环境清单**（EnvManifest 记录，spec/schemas.md §4.6）：控制面每个进程启动时、每个 worker 注册时生成。内容：操作系统、CPU 架构、Python 与 uv 版本、`uv.lock` 的 hash、Node 与 npm 版本、`package-lock.json` 的 hash、CUDA、驱动、nvcc、ptxas、各 kernel 后端版本、容器 digest、代码 commit 与 dirty 标志。每条作业结果和 AgentRun 都引用它。
- **`kta doctor` 对照锁文件检查当前环境**：`main` 环境下不一致即拒绝启动；`dev` 环境下给出警告，差异写进环境清单。
- **部署脚本**，都必须可以重复执行：
  - `scripts/bootstrap`：在一台新机器上建立控制面（检查 uv、Node 版本，`uv sync --frozen`，`npm ci`，初始化数据目录）；
  - `scripts/worker-setup <host>`：按当前 commit 在 worker 上建立环境、探测能力、注册。
- 新增依赖必须在决策日志中写一行，说明为什么现有依赖不够。
- 不安装任何东西到用户级目录；不修改本仓库以外的配置。

## 7.1 调用 AI 模型

- 系统中所有对 AI 模型的调用，都必须经过 agents 模块的适配器（spec/modules/agents.md）。业务代码中不得直接调用模型 API。
- 每次调用都必须**有记录**：
  - 会话持久保存，事后可以查看；能恢复的应可恢复；
  - 有 AgentRun 记录和会话起止的 span。
- token 用量和费用**不要求** KTA 记录。会话文件或 harness 中带有用量的，解析时顺带记下；价格与费用由你的其他工具分析，以后需要时再补自动分析脚本。
- **禁止**使用不持久化会话的模式（例如 Codex 的临时会话选项），**禁止**在不留记录的临时环境中调用（例如运行完即删除的一次性 HOME 或容器）。
- 隔离的 CLI 配置目录按工作单元持久保存在 `$KTA_DATA/<env>/agent-home/<工作单元>/`，会话文件就在其中。工作单元结束时打包为产物；产物写入成功之前，不删除原目录。
- 施工 agent 自己的会话由你使用的工具管理，不在本约定范围内；但施工中写出的任何调用 AI 的代码，都必须遵守本节。

## 8. 机密

- 两种认证方式：
  - **订阅**（Codex、Claude Code）：CLI 的登录凭据保存在控制面机器上一个受保护的目录（`~/.config/kta/auth/<provider>/`，权限 700）。每个工作单元的隔离配置目录创建时，只把凭据文件链接或复制进去。
  - **API**（pi、DeepSeek harness）：key 放在 `~/.config/kta/secrets.env`（权限 600），由运行器在启动 agent 进程时注入环境变量。
- 凭据文件和 key 不进入上下文清单的内容、不进入会话存档的产物、不进入日志；上下文清单中只记录"凭据已提供"。
- 不得出现在 git、记录、产物、日志、提示中。gate 用正则扫描常见的 key 格式。

## 9. 测试

| 标记 | 含义 | 在哪里跑 |
| --- | --- | --- |
| （默认） | 单元测试：纯逻辑，不依赖 GPU 和网络 | 任何地方 |
| `contract` | 接口合同测试：同一个接口的每个实现都要通过，包括 `tests/fakes/` 中的替身 | 任何地方 |
| `gpu` | 需要 GPU；测试经 Executor 把作业交给带相应能力标签的 worker | 控制面所在机器发起，作业在 GPU worker（spark）上执行 |
| `acceptance` | 各 WP 的验收测试；锁定后不能修改（spec/gate.md） | 同上 |

- 替身只能放在 `tests/fakes/`，而且必须通过对应接口的合同测试。这样替身的行为与真实实现一致，不会因为替身太宽松而让测试虚假通过。
- 测试只替换外部边界（GPU、远端、LLM、时钟），不替换被测对象自身。
- 声称"能力已接通"的测试必须使用真实实现，带 `gpu` 或 `acceptance` 标记，作业在真实 worker 上执行。
- 测试中不用 `sleep`；用 `ctx.clock` 的可控时钟。

## 10. 不允许的做法

| 不允许 | 应该怎样做 |
| --- | --- |
| 返回写死的数字，假装做了测量、编译或检查 | 调用真实的模块；当前做不到就抛 `NotSupported`，并在交付说明中写明 |
| 用 Python 模拟 GPU 的计算结果来代替运行 | 经 Executor 在真实 GPU worker 上运行 |
| 捕获异常后返回默认值，让流程"继续" | 抛出，或包装成带 kind 的错误 |
| 在 `src/` 中放 `FakeX`、`StubX`、`DummyX` | 放在 `tests/fakes/`，并通过合同测试 |
| 为了让测试通过而放宽断言或容差 | 在交付说明的"需要的变更"中提出 |
| 用 LLM 的判断代替检查器的结论 | 检查器写 Check 记录 |
| "暂时"跳过某个检查或某个状态 | 不允许；做不到就 `NotSupported` |
| 把未实现的分支写成 `pass` | 抛 `NotSupported("<原因>")` |
| 交付说明里写"已完成"，但没有证据清单 | 每项能力都在证据清单中列出 |

## 11. 提交

- 分支 `wp/<WP>`；commit message 格式 `<WP>: <摘要>`，英文。
- 小步提交；每个 commit 都能通过 `scripts/gate --quick`。
- 修改 `spec/` 的 commit，必须同时修改受影响的模块 README 和决策日志。

## 12. 命令行

- 所有 `kta` 子命令支持 `--json`。
- 退出码：0 成功；1 失败（如检查不通过）；2 用法错误；3 NotSupported；4 基础设施错误；5 权限错误。
