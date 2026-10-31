# 11 独立进程、时间与资源边界、并行与锁

[10](10-agent-system.md) 讲一个任务怎样执行；本文讲多件事同时进行时，系统怎样分工、怎样分配资源、怎样避免冲突。

## 1. 独立运行的进程

系统由几个相互独立的进程组成。它们之间**不直接调用**，只通过存储中的记录和任务队列通信。一个进程停下或出错，其他进程照常运行。

| 进程 | 做什么 | 由什么驱动 | 产出 |
| --- | --- | --- | --- |
| 求解 | 运行求解循环（10 §2.1） | 新 Case；你发起的求解 | 结果、Gap |
| 学习 | 运行扩展循环（10 §2.2）：导入来源、拆解参考实现为组合链、提取 Template、扩展族 | 学习待办列表；求解产生的 Gap | 新知识、新池快照 |
| 校准 | 运行 microbenchmark，拟合和更新成本模型 | 新测量积累到一定数量；模型误差告警；Target 变更 | 新 HwModel、新模型版本 |
| 维护 | 重新验证过期记录；重跑锚点集；抽样重放（I6）；备份；完整性检查 | 过期标记；定时 | 重验结论、告警 |
| 实验 | 按冻结的协议运行实验批次 | 你批准的协议 | ExperimentRun、结果表 |
| 汇总 | 改进循环（10 §2.3）；生成报告 | 批次结束；定时；你手动触发 | Report、改进项 |

进程之间的典型配合：

- 求解进程在 `judging` 判定"差距无法解释"时，写一条 Gap，再把对应的参考实现加入学习待办。学习进程会在自己的节奏中处理它。求解循环进入 `waiting`，等这条知识被接受、新快照生成后再继续。
- 学习进程每接受一条新知识，就生成新的池快照。求解进程下一次检索时，使用最新快照。
- **实验进程始终固定在协议指定的快照上**，不受学习进程生成新快照的影响。

每个进程是控制面所在机器上的一个常驻服务，可以单独启动、停止、重启。重启后，从存储中的状态继续。

## 1.1 部署拓扑

控制面（API、存储、上述进程、agent 运行器）和计算资源（GPU worker、CPU worker）可以在不同机器、不同集群，不共享文件系统。计算资源只经 Executor 访问，worker 按能力标签选择，产物按 hash 传输。细节见 spec/conventions.md §6。

**WP23 (user, 2026-10-31; D600–D602, supersedes D300 and D592).** One compute host, two modes:

| | dev (`KTA_MODE=dev-mac`) | long-run (`KTA_MODE=spark-hk`) |
| --- | --- | --- |
| control plane (API, store, orchestrator) | Mac | spark-hk |
| agent CLIs | Mac (worker `mac`, pool agent), through the local Clash: proxy policy `required` | spark-hk (local), direct internet: proxy policy `off` for codex and claude_code in that host's system configuration |
| CPU and GPU jobs | spark-hk over ssh | spark-hk, worker `local` |
| how the Mac sees it | browser on the Mac | `ssh -N -L 7310:127.0.0.1:7310 spark-hk` (ProxyJump hk-relay), browser on the Mac |

- **spark-hk** (GB10, aarch64, 20 cores, 119 GB unified memory, CUDA 13) runs every compile, check, simulation and GPU job. slai keeps only backups. The old `spark` (spark-b6d9) is no longer the default worker; its earlier records stay valid as history.
- Mac is a daily laptop (not always on, fans under load): nothing heavy runs there in either mode.
- **The store follows the control plane.** One control plane writes a store at a time. Switching modes: `kta store backup <dir> --handover <to>` on the old side (consistent SQLite copy, blobs, system configuration, hashes; the old side then refuses to start its API and processes), copy the directory (rsync over ssh), `kta store restore <dir>` into an empty data directory on the new side. The old data directory is never overwritten. Periodic backups are the same command without `--handover`: on spark-hk to slai when spark-hk reaches it, else pulled to the Mac over ssh (D609).
- **Isolation (relaxed by the user).** Agents run under the user's own account with the CLI's own sandbox, an isolated CLI home per work unit and the workspace limits; the `kta-agent` system user is optional (`[agents] agent_user = ""`, D602).

## 2. 资源

| 资源 | 容量 | 谁使用 | 计量方式 |
| --- | --- | --- | --- |
| agent 槽位 | 同时运行的 agent 数（默认 2） | 所有 agent 任务 | 槽位数 |
| LLM 调用 | 订阅的用量限制；API 的速率限制 | agent | 运行次数、运行时间；用量在会话文件中有时才记 |
| CPU | 各 CPU worker 的核（编译、CPU 侧检查、求解） | 执行器 | 核·秒 |
| GPU 共享模式 | 快速运行、正确性检查 | 执行器 | GPU 秒 |
| GPU 独占模式 | 正式测量、profile、microbenchmark | 执行器 | GPU 秒 |
| 内存 | GB10 的 CPU 和 GPU 共用内存 | 每个作业 | 作业的内存上限 |
| 磁盘 | 工作区、产物存储 | 所有进程 | 字节 |
| **你的注意力** | 待处理队列中同时存在的事项数 | 所有会升级给你的进程 | 待处理事项数 |

"你的注意力"也作为资源管理：待处理事项超过上限（默认 10 项）时，会产生新的待处理事项的进程暂停创建新任务，直到你处理掉一部分。这样，系统不会在你没跟上的情况下继续堆积问题。

## 3. 时间与资源边界

### 3.1 预算的层级

```
全局预算（每天或每周）：agent 运行时间、GPU 小时（订阅与 API 的费用由你的其他工具统计）
 └ 进程配额：实验 50%、求解 25%、学习 15%、校准与维护 10%
    └ 循环实例预算（每个 Case、每个参考实现）：最多轮数、墙钟时间、GPU 秒
       └ 任务预算：墙钟时间、GPU 秒、步骤数（token 只在能实时拿到用量时生效）
          └ 作业限制：编译超时、运行超时、测量超时、内存上限
```

进程配额比例为默认值，可以在前端调整。实验批次未运行时，它的配额按比例分给其他进程。

### 3.2 预留与结算

- 任务开始前，从上一层**预留**它的预算。上一层的剩余不足时，任务不启动，留在队列中。这样不会出现所有任务同时开始、最后集体超支的情况。
- 任务结束后，按实际用量**结算**，未用完的部分退回上一层。
- 每一层都可以随时查看：预算、已用、已预留、剩余。

### 3.3 哪些时间计入 agent 的预算

agent 调用 `kta measure` 时，可能要在 GPU 队列中排队。排队时间是系统造成的，不应该计入 agent 的消耗。所以把时间分开计量：

| 时间 | 计入任务预算 | 说明 |
| --- | --- | --- |
| 模型时间（agent 在思考、生成） | 是 | — |
| 工具执行时间（编译、检查、测量本身） | 是 | — |
| 排队等待时间 | 否 | 单独记录，在前端显示 |

另设一个**硬上限**：总墙钟时间达到任务预算的 3 倍时，无论原因都终止任务。防止排队过长时 agent 进程无限挂起。

### 3.4 预算用尽时的行为

| 层级 | 行为 |
| --- | --- |
| 作业 | 终止作业，返回超时结果；由调用者决定下一步 |
| 任务 | 终止 agent，任务记为 `expired`，按驳回处理 |
| 循环实例 | 停在当前状态，升级到你的待处理队列 |
| 进程配额 | 该进程不再启动新任务；正在运行的任务完成 |
| 全局预算 | 所有进程停止启动新任务；用到 80% 时提醒你 |

## 4. 并行

### 4.1 作业分类与执行池

| 执行池 | 并发数 | 跑什么 |
| --- | --- | --- |
| agent | 同时运行的 agent 数（默认 2） | agent 任务 |
| cpu | 默认为 CPU 核数减 2 | 编译、CPU 侧检查、求解 |
| gpu-shared | 默认 2 | 快速运行、正确性检查 |
| gpu-exclusive | 1 | 正式测量、profile、microbenchmark |

On a unified-memory host the cpu pool is also bounded by memory: running cpu jobs may reserve at most `execution.unified_memory_cap.cpu_jobs_gb` together (a job without `mem_limit_mb` counts `job_default_gb`; default 48 GB / 4 GB, to be set from S0 data; D607).

### 4.2 调度

- 各进程按配额加权轮流获得执行机会。这样，即使求解很忙，学习进程也能持续推进。
- 同一进程内部，先按优先级，再按先来后到。你发起的任务优先级最高。
- 正式测量作业成批执行（WP23, D608）：the solve process collects measurement-ready loops and measures them in one exclusive window (`measure.batch_window`: up to `max_items`, or once the oldest waited `window_s`), so the host drains once per window, not once per candidate.

### 4.3 Stage queues with WIP limits (WP23, D610–D613)

Work flows through stages (`orchestrator/stage_queues.yaml`): intake → generate → build → check → ready-to-measure → measure (exclusive windows) → judge → analyse (agent). Each stage has a capacity (controls knob `pipeline.stage_capacity`, editable on Settings, set by X12).

- **Backpressure**: a loop is advanced only while the next stage is below its capacity; upstream stops starting new items when downstream is full. Judging (which creates agent tasks) is held while the agent stage is full.
- **Dequeue priority**: work that unblocks waiting loops first, then the more downstream stage (finishing frees capacity), then the oldest.
- **Visibility**: the Pipeline page shows per stage the depth, wait and throughput over time, GPU busy/idle and host-lock waits, all derived from spans (`GET /trace/stages`); held loops show the reason.

## 5. 锁

| 锁 | 保护什么 | 规则 |
| --- | --- | --- |
| **主机读写锁**（WP23, D603） | 统一内存主机上的整机安静 | flock files in that host's lock directory (`[workers.<w>] host_lock`, the same path in every mode). Every heavy activity takes it **shared**: Executor cpu and gpu-shared jobs; the control plane's in-process MIP solves (and L2 runs) when the control plane runs on that host (`[host] lock_dir`); agent-initiated jobs (they go through `kta` → Executor). Measurement, profiling and microbenchmarks take it **exclusive** with writer preference (a waiting writer stops new readers), a bounded wait (`measure.batch_window.exclusive_wait_s`) and the lease's 10-minute maximum hold. Works across processes and across control planes on that host |
| GPU 独占租约 | 正式测量的环境 | 取得时：停止接收新的共享作业，等正在运行的共享作业结束，并暂停编译作业；then the host lock is taken exclusive for the whole window. 单次最长持有 10 分钟，超时强制释放并记录 |
| 循环实例锁 | 同一个 Case 或参考实现 | 同一时刻只有一个任务推进它 |
| Template 线写锁 | 一条 Template 版本线 | 同一时刻只有一个任务能出新版本。另一个任务需要修改时，排队等待，或者明确开一条新的分支线（记录父版本） |
| 池快照指针 | `current_pool` | 只有学习进程生成快照，用 compare-and-set 更新指针 |
| 存储写入 | SQLite | 所有写入都经过 API 服务的一个写连接；其他进程只读。One control plane per store (handover marker, D601) |
| 任务租约 | 一个任务 | 领取时取得租约，运行中定期续约；租约过期，任务回到队列 |

**Quiet-host verification (D605).** Every timing job records host readings before warm-up, before and after the measured loop: CPU used by other processes (`/proc/stat` minus the job's own CPU), load, memory in use, GPU utilisation, memory and compute processes, SM clocks, temperature and throttle reasons (NVML). Beyond the thresholds of `measure.quiet_thresholds` (status default, set by X6) the attempt is `interfered`: kept as a Report `measurement_interfered`, requeued automatically, never written as a Measurement. X6 adds an arm "agents idle vs agents active during measurement".

**避免死锁：**

- 只能按固定顺序取锁：循环实例锁 → Template 线写锁 → GPU 独占租约 → host lock.
- agent 进程从不直接持有锁。锁由编排器和执行器代为持有。
- 持有 GPU 独占租约时，不允许再等待其他锁 (except the host lock it takes right after, in the same window).

## 6. 崩溃恢复与幂等

- 每一步都可以安全地重复执行。编译、检查、测量都按输入的 hash 缓存：同样的源码、同样的 Target，再编译一次直接返回已有的 Build 记录。
- 任何进程崩溃后重启，都从存储中的循环状态和任务表恢复。运行中的任务，其租约过期后自动回到队列。
- agent 任务被中断后不续跑，而是重新开始。上一次的工作区已存档，新的 TASK.md 会附上上一次的交付说明（如果有）。

所有这些并行与等待，都以 span 记录下来，在前端的流水图中按泳道展示（30 §1.2）。

## 7. 例子：一个普通的下午

- 实验进程正在跑第一批对照，占用了大部分 GPU 独占时间。
- 求解进程在处理你新建的两个 Case：一个在等 GPU 独占租约做测量，另一个在等学习进程处理一条 Gap。
- 学习进程用一个 agent 槽位在重建 FlashInfer decode，它的快速运行作业在 gpu-shared 中执行；正式测量排在实验作业之间。
- 校准进程发现最近 50 条测量中，decode Template 的预测普遍偏低，于是创建了一个 T4 诊断任务，排在 agent 队列中。
- 待处理队列中有 3 项：一个结果疑似异常（超过下界的 95%）、一个任务超过重试上限、实验协议第二批待批准。

## 依据与待核实

- 进程配额比例、默认并发数、注意力上限 10 项、独占租约最长 10 分钟、硬上限 3 倍：都是先填的数值。
- GB10 上 CPU 与 GPU 共用内存，并发作业的内存上限需要在 M0 实测后设定（WP23: knob `execution.unified_memory_cap`, from S0 data）。
- 编译作业是否真的会干扰测量，需要在 M0 实测。如果干扰可以忽略，就不必在测量时暂停编译（WP23: the host lock drains them by default; X6 measures agents idle vs active）。
