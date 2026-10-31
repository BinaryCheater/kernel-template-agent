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

Implementation (WP6, D260): the state's work runs first and is handed to `step` as `loop.facts`;
`step` stays pure. `Transition` = target state (or a hold), the machines.yaml kind and rule,
loop-data updates, task requests and record drafts. The files are those listed above plus
`machines.py`, `transition.py`, `step.py`, `solve_work.py`, `expand_work.py`, `runner.py`,
`pipeline.py`, `done_checks.py` and `outputs.py`; `improve.py` belongs to the summarize process
(WP11). Open paths of the design are settled in machines.yaml `notes` (D261, D262, D264, D266, D267).

进程的主循环反复：取出可推进的循环实例 → 加循环实例锁 → `step` → 写结果 → 释放锁。等待中的实例（等任务、等作业、等快照）在等待对象完成后被唤醒。

求解循环的状态与判定规则，严格按 docs/design/10 §2.1 实现，包括：

- `retrieving`：先调用 `knowledge.nearest_chains` 和 `reuse_plan`。有可复用的链但部分步骤不成立 → 创建 T1，输入中给出该链和不成立的步骤。完全没有 → 创建 T1，输入中给出检索到的 Template 卡片。
- `building`：编译短名单，读取 Build 反馈，执行 `check_sass`，用编译后模型重排。
- `judging`：判定规则按顺序匹配；所有判定写一条 Report 记录（kind="judgement"），包含差距分解和归因。

扩展循环在 `analyzing` 创建 T2；T2 交付的 Reconstruction 中每个未匹配事务生成一条 Gap；每条 Gap 创建一个 T3。

(WP10b, D382, D384, D385 — needs user review) Implemented: each unmatched transaction of the accepted T2 is written as a Gap; one T3 is created per Template the chain names that the pool lacks (Gaps create no task). A cancelled T3 leaves its Template open; analysis-only references skip build, measurement, reproduction and ablation.

### 完成检查

每种任务类型的完成检查是固定的函数列表，`TaskCard.done_checks` 存函数名：

| 类型 | 完成检查 |
| --- | --- |
| T1 | `validate_family` 无错误；`verify_space` 通过；交付说明完整 |
| T2 | Reconstruction 字段完整；`matched` 的链项都引用已有 Template；`unmatched_transactions` 与 Gap 一一对应；交付说明完整 （计数规则与提交格式见 D380、D381）；`admission`: the chain and the pool-missing Template drafts validate without errors, every warning is explained in `warnings.yaml`, language gaps are counted (D561) |
| T3 | `validate_template` 无错误（`possible_duplicate` 须在交付说明中解释）；交付说明完整 ；扩展循环的 T3 须交付链中点名、池中尚无的 Template（`template_fills_chain`，D382、D383） |
| T4 | 新模型版本在留出点上的误差低于旧版本 |
| T5 | microbenchmark 重复 5 次，变异系数低于噪声底的 2 倍 |
| T6 | 原失败用例通过；`verify_space` 通过 |

"交付说明完整"：包含做了什么、简化与 fixture（可写"无"）、已知问题。

Function names stored in `TaskCard.done_checks` and the submission layout per type: D277, D269
(`src/kta/orchestrator/done_checks.py`). Done-check results are Reports titled `done_checks` (D276).

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
kta reference plan <id> [--target gb10]   learning-loop plan of a Reference (dry run, D569)
kta reference admit <dir>        import reference directories through admit (D561)
kta export <snapshot> <dir>      导出快照为可浏览目录
```

WP6 additions and deviations (D274): `kta loop list|show|step`, `kta inbox`, `kta budget`,
`kta card`, `kta task submit|check|recheck`; `kta solve <case>` starts the loop (the agent form
`kta solve <template> <case>` is not provided); `instantiate`, `build`, `sass`, `run`, `src` raise
NotSupported until WP8; `bound` is not provided.

agent 使用的命令（`show`、`query`、`why`、`instantiate`、`build`、`sass`、`check`、`run`、`measure`、`profile`、`bound`、`solve`、`validate-template`、`validate-family`、`note`、`submit`、`src`）在 agent 环境中通过任务令牌鉴权，见 agents.md。

### 行为

- 输出默认精简：大输出存为产物，命令只打印摘要和产物 ID；`--full` 打印全文。
- 每次调用都写入调用日志（任务 ID、命令、参数、耗时、结果记录 ID）。

## 验收

- 真实运行：WP7 中，`kta solve` 在不接 LLM 的情况下，对 GEMM case 从 `new` 走到 `accepted` 或某个有记录的判定；进程中途被杀死后重启，从原状态继续。
- 单元测试：状态转移函数对每条判定规则各有一个测试；预算预留不足时任务不启动。

## WP11 implementation notes (D400–D411)

Additive; WP6's signatures are unchanged except where stated.

- Processes: `learn` (`proc_learn.py`: expand loops, backlog, PoolSnapshots, D400), `calibrate`
  (`proc_calibrate.py`: peaks and microbenchmarks, refits, T4 alerts, T5 per hw_fact Gap, D405),
  `experiment` (`proc_experiment.py`: approved Protocols, batch environments, runs, D403),
  `summarize` (`proc_summarize.py` + `improve.py`: improvement batches, D406); shared helpers in
  `proc_common.py` (process actions are `process_action` spans, state in ops pointers, D411).
  `processes.IMPLEMENTED == ALL`; `TICKS` values return objects with a readable `str`.
- Budget hierarchy (`quotas.py`): configured and in-force shares, redistribution while no
  experiment runs (D401), periodic global budget rolled by the maintain tick (D402),
  `order_queued` for the agent pool (11 §4.2).
- `Runner.start(kind, subject, name, extra=None, distinct=False)`; tasks are booked to
  `loop.data.process` when set; the solve process skips loops of another process (D409).
- T5: `done_checks.microbench_repeats` and the T5 output are in `microbench_check.py` (D404).
- Inbox: `improvement_pending` items (system items of an improvement Report without a decision).
- The registry interface `batch_envs` (implemented by `kta.agents`) gives the experiment process
  `ensure(batch, profile)` (D410).
- CLI (`cli/process_cmds.py`): `kta learn add|backlog`, `kta improve run [--label] [--now]`,
  `kta experiment status`, `kta calibrate hw`, `kta quotas`, `kta profile write codex-v1|codex-v2`.

## WP21: controls and staged release (D482, D485, D487, D488)

### Files

```
orchestrator/stages.yaml   S0–S4 (02 §6.3): processes, task_types, agent_calls (none | gated | automatic), exit
orchestrator/release.py    stages(), stage_of(controls), process_allowed, hold_reason, admit (raises Held),
                           release_call (one attempt), held_calls
cli/controls_cmds.py       kta controls show | history | token
```

### Behaviour

- `Settings` (transition.py) is a view of a controls snapshot: `Settings.of(controls)`; it has no defaults of its own. The `[orchestrator]` configuration section is no longer read; `[budget.quotas]` and `[agents] slots` are refused with a pointer to their knobs (`budget.process_quotas`, `budget.agent_instances`). `machines.yaml` limits name their knobs (`max_rounds: budget.rounds_per_case_loop`); the API shows the value in force.
- The runner reads the controls in force at each `start` and `advance` (`kta.store.controls.ensure`), runs the state's work and `step` with that snapshot in `ctx.controls`, and stores its ID in the loop data. A new TaskCard's budget is `Settings.card_budget(type)` and its `refs.controls` the version.
- Loop budget scope: `agent_min = rounds × max task wall budget / 60`, `gpu_min = case_factor × max task GPU budget / 60` (12 §1, D488).
- The stage in force is the knob `release.stage` (default S0). A process outside the stage skips its ticks (`maintain` always runs). The agent pool (`next_task`, `step_once`) starts only a task the stage admits: its type in the stage, agent calls not `none`, and in a `gated` stage no other task leased or running and a release for exactly this attempt (`release:task:<id>` pointer + HumanAction `release_agent_call`). `kta task lease` (manual) is not gated.
- `kta controls show` stores version 1 if none exists and prints the values in force and the stage; `kta controls history` lists versions; `kta controls token` creates the operator token file (mode 600) and never prints it.

## WP20 learning-loop runner (D560–D569)

- **Admission done-check.** T2 runs `admission` last (`done_checks.t2_admission`): the chain and the Template drafts the pool lacks are validated against the pool plus those drafts and the submitted Patterns. An error or an unexplained warning rejects the T2; the queue's existing retry path (`max_retries` = 2, then `escalated`) returns the failures to the same work-unit session as a `RetryNote`, and `sessions.decide` resumes that session unless the same failure repeats. The T2 writer also writes the `admission` Report.
- **Plans** (`kta.orchestrator.learning_plan`, D569). `plan(store, reference_id, target, facts=None, core=None, ablation=None, family=None) -> Plan` lists the stages intake → decompose → admit → reproduce → ablate → cover with status `planned`, `task` or `not applicable` (by source class and tier). Intake holds Executor `Job`s (clone, checkout at the revision, license, tree hash, build in the container, run, SASS dump, profile); reproduce and ablate hold variant runs through the solve pipeline path (`generate`, `build`, `full_check`, `measure`). `dry_run` is always true: nothing is submitted (D320). The expand loop's `frozen`, `reconstructed` and `reproduced` work still raise NotSupported for runnable references and name the plan.
## WP18: gap decomposition, routing and loop disciplines (D520–D530)

### Files

```
orchestrator/gap.py          decompose, classify, accepts, case_class, task_for, selection; routing() loads routing.yaml
orchestrator/routing.yaml    versioned routing table: attribution order, class -> rule/to/task/escalate/improve, escalations, canaries
orchestrator/disciplines.py  basis, mechanisms (coupled groups count once), credit (band, attribution, null rounds, rollback)
orchestrator/canary.py       plant, classify_inputs, run (one kind per batch, rotating), missed
```

### Decomposition (02 §1 P7, §6.1)

Per judged bucket, with `P1` the solver's best prediction (`solve` report, else the smallest measured prediction) and `Popt` the model optimum:

| term | value | status when it cannot be computed |
| --- | --- | --- |
| ε | `T_meas − P1` | — |
| δ | `P1 − Popt`: 0 if model-optimal; `mip_gap × P1` for a MIP | `unbounded` (no certificate) |
| Δpre | `Popt − P_relaxed`, `P_relaxed` = model optimum with the plan's fixed variables opened | `not_assessed` (0; its share stays in L_rep) until WP19 supplies `relaxed_pred_us` |
| L_rep | `P_relaxed − T_SOL − irreducible`; bounded by `P1 − T_SOL` | — |
| irreducible | declared irreducible time (`irreducible_us`) | `not_assessed` (0) |

`T_SOL` is the Bound record (L1 SOL under minimal work). The improvement band is `max(CI half-width, rel_of_t × T_meas)` (`guard.improvement_band`). The bucket class is the first entry of `order` whose term exceeds the band, or is `unbounded` while more than the band remains; ε also counts when the model is off (`judging.model_ranking`). The case class is the most upstream bucket class.

### Judging (rules 1–6)

In order: correctness (rule 1, T6); accept (rule 3) unless the model is off — `escalation.accept`: not worse than the strongest reference within the band and `|ε| < max_unexplained_rel × T`, or `gap − irreducible` within the band; with no reference only the second clause can accept (D522); the null-round limit (rule 6, `closed`); model error above `escalation.case_to_knowledge.topk_model_error_rel` after `calibrations_first` T4s (rule 5, `closed`, escalate knowledge); model off (rule 2, T4); else the class route of routing.yaml: ε → rule 2 T4; δ → rule 5 `closed`, escalate user; Δpre → rule 4 T1; L_rep → rule 4 T2 (T1 without a reference); irreducible or none not accepted → rule 5 `closed`, escalate user. `closed` is a terminal state; `escalated` = `user` raises the inbox item, `knowledge` is counted by the summarize process. Each judgement writes a GapDecomposition before the judgement Report, which links it as `refs.decomposition`.

### Disciplines (12 §3)

- Basis of a round: the family chain's templates, the family, the model version. Mechanisms = templates added/removed, a model change, a generator-only family change; templates of one Coupling changed together count once (`guard.one_mechanism_per_round`).
- Credit: gain over the incumbent beyond the band, at least one named mechanism (`guard.mechanism_attribution`) and within the per-round limit. A round without credit is a null round (`guard.null_round_limit`).
- Hold-out and rollback: the round's fresh measurement is the hold-out of the change that produced it; a regression beyond the band keeps the incumbent, records the mechanisms in `rolled_back_mechanisms`, blocks acceptance in that round, and retrieval prefers the incumbent's family next round (`prefer_family`). The model version is not rolled back (known issue).
- Exploration and selection controls: of the `measured_points` k, `exploration_points` are the random arm, drawn uniformly (seeded reservoir) from the enumerated feasible space without the model; with `guard.selection_controls` the newest Solution of the same family on another case is the copied arm. Both are built, checked and measured like model points and recorded per bucket in `selection`; they do not route (D524).
- Duplicate proposals: a submission whose files and blob hashes equal an earlier rejected attempt of the same task is rejected without running the done checks (outcome `duplicate_proposal`, `guard.duplicate_proposal_rejection`).
- Canaries (`guard.canary_rate`): each summarize batch plants the next kind of routing.yaml `canaries` into the newest GapDecomposition's recorded inputs and re-classifies; verdict caught / missed / uninformative / no_base in a Report `canary`; a missed canary is an inbox item (`escalation.to_user.on_canary_missed`) until that kind is caught again.
- The summarize process counts GapDecompositions by `route.improve` (judgements that carry `gap_class` are not counted twice) and marks an item `escalate: knowledge` when the class recurs in `same_residual_cases` distinct cases or a round escalated to knowledge.

### Interface owed by WP19

`relaxed_pred_us` per bucket in the judging facts: the model optimum of the FormulationPlan with its fixed variables (F3) opened, on the same model and bucket. Until then Δpre is `not_assessed`.
## WP19: access discipline, task header, pull log (D544, D545)

- `orchestrator/access.py`: `situation_of(gap)` (none → `start`, ε → `eps`, wall → `wall`,
  L_rep with an idle resource → `idle_resource`, `stop`, else `diagnostic`);
  `task_header(case, case_id, bound_us, bottleneck, gap, plan_id, score, budget,
  discipline, max_lines)` → `TaskHeader` (ConfigError above `session.task_header_max_lines`,
  lines clipped at 100 columns); `record_pull`, `pull_log(store, task)`, `pull_stats(store)`,
  `plan_view(store, plan_id)`, `card_header(store, card_id)`.
- `TaskRequest.header` (optional): `TaskQueue.create` writes it into the card and refs the
  header's plan. The gap mapping is WP18's decomposition (components eps, delta, dpre,
  l_rep, irreducible in us, plus `class` and `idle`).
- The table itself is data: `core/access_discipline.yaml` (versions append-only),
  `core/access.py` (`table(version)`, `Table.classify(command, kind, situation)`,
  `render(table)`). The worker skill `use-kta` carries `render(table(v))` between markers;
  the gate checks they are identical for the version `session.access_discipline_version`
  names.


## WP23: stage queues, work units, store hand-over (D601, D610–D615)

- `orchestrator/stage_queues.yaml` + `stagequeue.py`: stages by loop state; `plan(rows, capacity, window, now, open_windows=0, kinds=("solve",), process="solve") -> Plan(advance, window, held, depth)` (pure; backpressure, unblock-first order, measurement windows). `solve_tick` advances `plan.advance`, measures `plan.window` in one `MeasureWindow` when `[measure] window_worker` is set, writes `deferred` on held loops and an ops event `stages` per tick. Capacities: knob `pipeline.stage_capacity`; windows: `measure.batch_window`.
- `orchestrator/work_units.yaml` + `workunits.py`: `roles()`, `role_of(type)`, `unit_for(loop_id, type, rule, task_key)`, `loop_of(work_unit)`. `TaskQueue.create` writes the card's `work_unit` by `session.work_unit_rule` (default `<loop>:<role>`); the ops `tasks.work_unit` column keeps the loop ID.
- `solve_work.work_solving` holds the host lock shared around the MIP solve when `[host] lock_dir` is set (long-run mode).
- CLI: `kta store backup <dir> [--handover <to>]`, `kta store restore <dir>` (`kta.store.backup`); `kta process start` and the API refuse a handed-over store.
