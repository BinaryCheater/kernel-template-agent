# orchestrator 与 cli

设计依据：docs/design/10 §2–§3、11。

## orchestrator（第 7 层）

### 文件

```
orchestrator/solve_loop.py      求解循环状态机（10 §2.1）
orchestrator/expand_loop.py     扩展循环状态机（10 §2.2）
orchestrator/improve.py         改进汇总（10 §2.3）
orchestrator/space.py           为 verify_space 枚举路径、生成、编译（见 measure-check.md）
orchestrator/tasks.py           任务卡创建、完成检查的执行、驳回与升级
orchestrator/processes.py       六个进程的主循环（11 §1）
orchestrator/budget.py          预算预留与结算（11 §3）
orchestrator/inbox.py           待处理事项的生成（30 §2）
```

### 状态机

每个循环实例是 `loops` 表中的一行：`kind`、`subject`（Case 或 Reference 的 ID）、`state`、`data_json`。状态转移函数是纯函数：

```python
def step(loop: Loop, ctx: Context) -> Transition
    # Transition：新状态、要写的记录、要创建的作业或任务、要等待的对象
```

进程的主循环反复：取出可推进的循环实例 → 加循环实例锁 → `step` → 写结果 → 释放锁。等待中的实例（等任务、等作业、等快照）在等待对象完成后被唤醒。

求解循环的状态与判定规则，严格按 docs/design/10 §2.1 实现，包括：

- `retrieving`：先调用 `knowledge.nearest_chains` 和 `reuse_plan`。有可复用的链但部分步骤不成立 → 创建 T1，输入中给出该链和不成立的步骤。完全没有 → 创建 T1，输入中给出检索到的 Template 卡片。
- `building`：编译短名单，读取 Build 反馈，执行 `check_sass`，用编译后模型重排。
- `judging`：判定规则按顺序匹配；所有判定写一条 Report 记录（kind="judgement"），包含差距分解和归因。

扩展循环在 `analyzing` 创建 T2；T2 交付的 Reconstruction 中每个未匹配事务生成一条 Gap；每条 Gap 创建一个 T3。

### 完成检查

每种任务类型的完成检查是固定的函数列表，`TaskCard.done_checks` 存函数名：

| 类型 | 完成检查 |
| --- | --- |
| T1 | `validate_family` 无错误；`verify_space` 通过；交付说明完整 |
| T2 | Reconstruction 字段完整；`matched` 的链项都引用已有 Template；`unmatched_transactions` 与 Gap 一一对应；交付说明完整 |
| T3 | `validate_template` 无错误（`possible_duplicate` 须在交付说明中解释）；交付说明完整 |
| T4 | 新模型版本在留出点上的误差低于旧版本 |
| T5 | microbenchmark 重复 5 次，变异系数低于噪声底的 2 倍 |
| T6 | 原失败用例通过；`verify_space` 通过 |

"交付说明完整"：包含做了什么、简化与 fixture（可写"无"）、已知问题。

### 预算

按 docs/design/11 §3 实现预留与结算。预算层级的键：`global`、`process:<名>`、`loop:<ID>`、`task:<ID>`。

## cli（第 9 层）

命令名 `kta`。所有命令都有 `--json` 输出。命令与 docs/design/10 §5 的表一致，另加管理命令：

```
kta init                         建库、导入 seed
kta target probe                 探测并写 Target、HwFeature 支持情况
kta case add <yaml>              新建 Problem/Contract/Case
kta solve <case>                 启动求解循环
kta reference add <yaml>         导入参考实现
kta expand <reference>           启动扩展循环
kta process start|stop <名>      控制进程
kta task list|show|cancel|retry
kta coverage [--snapshot <id>]   打印覆盖报告摘要
kta export <snapshot> <dir>      导出快照为可浏览目录
```

agent 使用的命令（`show`、`query`、`why`、`instantiate`、`build`、`sass`、`check`、`run`、`measure`、`profile`、`bound`、`solve`、`validate-template`、`validate-family`、`note`、`submit`、`src`）在 agent 环境中通过任务令牌鉴权，见 agents.md。

### 行为

- 输出默认精简：大输出存为产物，命令只打印摘要和产物 ID；`--full` 打印全文。
- 每次调用都写入调用日志（任务 ID、命令、参数、耗时、结果记录 ID）。

## 验收

- 真实运行：WP7 中，`kta solve` 在不接 LLM 的情况下，对 GEMM case 从 `new` 走到 `accepted` 或某个有记录的判定；进程中途被杀死后重启，从原状态继续。
- 单元测试：状态转移函数对每条判定规则各有一个测试；预算预留不足时任务不启动。
