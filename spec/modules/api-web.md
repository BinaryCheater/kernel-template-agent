# api 与 web

设计依据：docs/design/30。

## api（第 9 层）

### 文件

```
api/app.py          FastAPI 应用；持有 Store 的唯一写连接
api/read.py         读接口
api/actions.py      操作接口（写 HumanAction）
api/events.py       SSE
api/auth.py         任务令牌（agent 的 kta 命令经 API 写入时使用）
```

### 接口

读：

```
GET /records/{id}                       记录，含当前状态与保证等级
GET /records?kind=&where=&limit=        查询
GET /search?q=&kinds=                   全文检索
GET /lineage/{id}?depth=                证据链
GET /cases/{id}/timeline                求解循环的每一轮、每个任务、每次驳回
GET /families/{id}/chain                组合链及每步的卡片
GET /coverage?snapshot=                 CoverageReport
GET /model/residuals?model=             预测对实测
GET /runs/active                        正在运行的循环、任务、作业、租约
GET /budget                             各层预算
GET /inbox                              待处理事项
GET /events?since=                      SSE：ops_events、新记录、span 的开始与结束
GET /trace?from=&to=&lanes=&trace=&kinds=   时间窗内的 span（含进行中的）；可按泳道、trace、种类过滤
GET /trace/summary?from=&to=            各泳道利用率；各种类耗时合计；排队占比
GET /trace/critical-path/{trace_id}     一个循环实例或实验运行的关键路径
```

操作（每个都写 HumanAction，`reason` 对驳回、废弃、调整预算为必填）：

```
POST /actions/pause        {scope}
POST /actions/resume       {scope}
POST /actions/cancel       {task}
POST /actions/retry        {task, note}
POST /actions/budget       {scope, limit, reason}
POST /actions/priority     {task, priority}
POST /actions/create-task  {type, goal, inputs}
POST /actions/reject       {record, reason}
POST /actions/flag         {record, reason}
POST /actions/approve      {protocol | improvement}
POST /actions/confirm      {pattern | structure}       新增的模式或结构需你确认
```

写入：`POST /write/record`（只接受带有效任务令牌或本机工具身份的请求）。

Additions made in WP9a (D170): `GET /meta` (env, record counts by origin), `GET /machines`
(the orchestrator state machines, D96), `GET /cases`, `GET /families?snapshot=`, `GET /snapshots`.
`GET /lineage/{id}` also returns `result`, the assembled evidence chain, when `id` is a Solution.
`GET /coverage?snapshot=` serves the stored coverage Report of that snapshot (501 when none exists).
Served by `python -m kta.api` (`scripts/api`) on `[api] host/port` in config/kta.toml; the host must be
a loopback address.

Additions made in WP9b (D340–D354):

- Operations: `POST /actions/deprecate {record, reason}` (reason required); `POST /actions/approve`
  also takes `{profile}` (D351, delegated to WP8); `GET /actions/capabilities` (each operation,
  what it writes, whether it is wired, what it needs); `GET /actions/log?limit=` (HumanActions,
  newest first, verb and parameters parsed). Operations that need WP6/WP8 answer 501 NotSupported
  and write nothing until wired (D341); a malformed body or a missing required reason is 400.
- A HumanAction written by the API has `action = "<verb>"` or `"<verb> <compact JSON>"`, the
  subject in refs and an actor `human` named by `[api] operator` (D342).
- `POST /write/record` needs `Authorization: Bearer <task token>` (D352).
- Reads: `GET /tasks/{id}` (task detail), `GET /blobs/{sha256}` (text blobs), `GET /agents`
  (runtime skills, AgentProfile versions, `current_profile` history, batch environments),
  `POST /agents/batches/{batch}/verify` (delegated to WP8), `GET /pool?snapshot=`,
  `GET /experiments`, `GET /improvements` (D354, conventions D345 below).

### Record conventions read by the API (D169, default — needs user review)

The read views are derived only from records, ops tables and spans. Producers (WP2, WP6, WP7, WP8,
WP10b) must write them this way for the pages to show their work:

| What | Convention |
| --- | --- |
| Loop instance | `loops` row: `kind` = `solve`/`expand`, `subject` = Case/Reference ID; `data` keys `name`, `round`, `started_at`, `state_since`, `waiting_on` (task ID), `escalated` (bool) |
| Loop state history | one `loop_step` span per state: `trace_id` = loop ID, lane `process:solve`/`process:learn`, attrs `state`, `round`, `note` |
| Task | `tasks` row; `work_unit` = loop ID; `lease_owner` = agent slot (`agent:<n>`); the agent session is an `agent_session` span with attrs `task_id` (an open one means the task is running) |
| Rejection | Check record with `actor.task` = task ID and `passed = false` |
| Job | an open `job_queue` span (queued) or `compile`/`check`/`measure`/`profile`/`job_run` span (running) on lane `worker:<host>:<pool>` or `gpu:<worker>:<device>`; attrs `job_id`, `job_kind`, `pool`, `worker`, `task_id` |
| GPU lease | lock `gpu:<worker>:<device>` held by the job ID; a `lock_hold` span with attrs `lock` |
| Worker capacity | Worker record label `pools` = `"<pool>:<capacity>,..."`; health from the `workers` table |
| Budgets | scopes `global` and `process:<name>`; resource keys `<resource>_<unit>`, e.g. `agent_min`, `gpu_min` |
| Case results | Bound, Candidate, Gap and baseline Measurement have `refs.case`; Candidate `refs.family`; Prediction and Build `refs.candidate`; Check and Measurement `refs.build`; a baseline Measurement has `refs.baseline` (a Source naming the baseline) |
| Solution | `refs.case`, `refs.family`, `refs.measurement`, `refs.check` (numeric check), `refs.bound`; the SASS check is a Check on the build whose actor name starts with `check_sass` |
| Judgement | Report `title = "judgement"`, `subject` = Case ID, `refs.case`, `refs.solution` when accepted; `data`: `kind = "judgement"`, `round`, `rule`, `units` (`unit`, `bound_us`, `predicted_us`, `measured_us`), `explained_share` |
| Coverage | Report `title = "coverage"`, `subject` = PoolSnapshot ID, `data` = the coverage page's fields (`scope`, `discovery`, `trend`, `instruction_families`, `hw_patterns`, `references`, `gap_explained`, `backlog`, `structure`) |
| Snapshot membership | a Family is in a snapshot when its ID is in the PoolSnapshot `members` |
| Experiment (D345) | Protocol `type: experiment`, `params.arms`, `params.runs_per_arm`; ExperimentRun `refs.protocol`, arm = `condition`; result Report `title: experiment`, `refs.protocol`, `data.rows`; approval HumanAction `approve_protocol`, `refs.protocol` |
| Improvement (D345) | Report `title: improvement`, `data.batch`, `data.attribution` (category → count), `data.items` (`key`, `title`, `kind`, `description`); approval HumanAction `approve_improvement`, `refs.improvement` |
| Confirmation (D345) | HumanAction `confirm_pattern` / `confirm_structure` with `refs.pattern` / `refs.structure` |
| Task detail (D354) | records written in a task have `actor.task` = task ID (AgentRun, Submission, Check, ...); its spans carry `attrs.task_id` (children via `parent_id`) |

### 行为

- 只监听 spark 的本机端口；通过 SSH 隧道访问。不做登录。
- 所有写入经过同一个写连接，保证 SQLite 单写。

## web

TypeScript、React、Vite、ECharts。独立目录 `web/`。

### 页面（按 docs/design/30 §1）

第一版（WP9a）只读：运行中、**流水图**、Case（含时间线）、结果详情（证据链）、族（组合链与每步卡片）、覆盖。

第二版（WP9b）：待处理、任务详情（实时工具调用）、池、模型、实验、改进，以及全部操作按钮。

### 要求

- 每个数字和结论都能点开对应记录。
- 所有图表可以切换池快照。
- 覆盖页顶部始终显示抽样范围（来源数量与类别）。
- 流水图（docs/design/30 §1.2）：
  - 横轴为时间，纵轴为泳道；每个 span 一个色块，颜色按种类；
  - 支持缩放、拖动、"跟随当前时间"的实时模式（经 SSE 更新）；
  - 点击任务或会话色块，展开其内部的模型段与工具调用段；
  - 悬停显示名称、起止、耗时、状态、关键 attrs；点击跳到对应记录；
  - 可按 Case、实验运行、进程、主机过滤；
  - 顶部显示时间窗内各泳道的利用率，以及 GPU 空闲比例。
- 组合链用有向图展示：节点是链项（显示 Template 名、细化的事务、保证等级），边是 `after` 依赖。

## 验收

- 真实运行：在本机通过 SSH 隧道打开前端，看到 WP7 的一次真实求解的时间线和结果证据链；截图写入交付说明。
- 每个操作接口有测试：写出的 HumanAction 字段正确；缺少必填理由时返回 400。
