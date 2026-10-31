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

---

## Round 4: design v2 (thinking model and template design, D324)

The 2026-10-30/31 design conversations (`docs/design/01-thinking-model.md`, `docs/design/02-design-notes.md`, `docs/design/12-agent-controls.md`, `docs/changelog.md`) turned into engineering gaps E1–E15 and requirements R1–R21. The user asked (2026-10-31) to update the code accordingly, still at code stage (D320): no real runs, no experiments. Round-4 WPs therefore deliver with `scripts/gate --stage code <WP>` and list every real run they owe under "Pending real runs".

Rules for every round-4 WP:

- 02 is the design source; where it is silent, decide, record a row in `docs/decisions.md`, and flag cross-cutting choices `default — needs user review` with a `docs/review-queue.md` entry. The D-open questions are answered by the defaults of D325 until the user says otherwise.
- Change `spec/` (schemas, module specs, gate) in the same branch as the code, in English. Run the `design-interrogation` skill (full pass) on every spec interface change and keep its artefacts in the delivery note.
- Records are immutable. Old record bodies stay readable: bump the schema version and upcast on read; never rewrite stored records, run IDs or hashes.
- Seed knowledge (`knowledge/seed/`) may be migrated in place, with each migration recorded; worker deliverables never go to `knowledge/seed/` (D325).
- Every new mechanism comes with an observable (record, metric or view), a failure fixture under `tests/`, and its knob in the controls registry once WP21 has merged.
- Add a `docs/changelog.md` entry ("merged work package") listing which R/E items the WP closes and which 02 sections it implemented.

```
Phase 1 (parallel):  WP14 knowledge model v2 · WP15 template effects + declarative hardware model · WP21 controls registry + staged release
Phase 2 (parallel):  WP16 transaction-level simulator (after WP15) · WP18 gap decomposition + routing (after WP15, WP21)
                     WP19 FormulationPlan + chain enumerator + access discipline + projection compiler (WP17 folded in; after WP14, WP15)
                     WP20 learning-loop runner (after WP14)
Final step:          the WP19 session adds the MIP↔L2 consistency check once WP16 has merged
```

To save tokens (D328), WP17 is not a separate session: its scope is done by WP19, which already owns the formulation module; the MIP↔L2 check is a follow-up message to the same session after WP16 merges. Sessions run on Opus 5.5 at medium effort; mechanical pieces (migration re-runs, view-only frontend additions, doc merges) may go to a Sonnet session or be done by the coordinator.

## WP14 Knowledge model v2

- **Depends**: WP4, WP10b. **Reads**: 02 §2, §8.2; 23; 21 §1; 22 §6; 01 §2 "Levels and transfer".
- **Closes**: R13 (except the HwModel split, which is WP15), R14, E5, E7, E15; D-T7, D-T8, D-T9, D-T27; D-open-1, D-open-3 (relation part).
- **Produces**:
  - `Capability` replaces `HwFeature` (declared per Target with parameters); `Template.requires_hw` becomes `requires` (capabilities); migration of seed, patterns and existing records by upcast.
  - `Realisation` split from Template: template × backend × ISA range, carrying program fragments, ISA constraints and expected SASS; Template keeps the abstract program.
  - `Finding` unifies Regime and Negative: claim, kind, scope, evidence, status, `explained_by`.
  - One status ladder for empirical knowledge (findings, model terms, chain performance) with evidence rules per step; templates keep assurance (D325).
  - `probed_on` on templates and realisations; retrieval filters by the target's capabilities; coverage counts "known, unavailable here".
  - `coupled_with` as a first-class relation with evidence and scope; consumers (formulation, hardware model) can read it.
  - Level numbering 1–7 (workload, graph, operators, tensor map, instructions, architecture, microarchitecture) with a versioned mapping from the old 0–5 values used by stored records.
  - Promotion and demotion engine (02 §2 "Promotion and demotion") on top of the existing invalidation propagation; a pool-diff view (API + web page) between two snapshots.
- **Acceptance (code stage)**: seed loads and validates under the new schema; old-format records read back; unprobed template filtered on GB10 and counted; promotion/demotion and invalidation covered by failure fixtures; pool diff page renders on seed data.

## WP15 Template effects and the declarative hardware model

- **Depends**: WP5, WP11. **Reads**: 02 §3, §4, §5.1; 22 §2; 50 §7; `knowledge/templates/`; `src/kta/models/`.
- **Closes**: R2, R3 (data and L1), R5 (schema and interference microbenchmark definitions), E1, E2; D-T2, D-T3, D-T10, D-T14, D-T15; D-open-5 (D325).
- **Produces**:
  - Template `cost` → `effects` (transactions, storage, overlap) in schema, loader, validators and every seed template; the WP5 known issue (missing contraction/operand traffic) resolved by declaring whole-kernel transactions once.
  - `HwModel` split into architecture class and chip instance, as declarative data: resources (kind, scope, peak, latency, max in-flight, ncu counter names), walls (with their cliff), operation classes (reservation tables per transaction pattern, optionally per SASS instruction class), efficiency (few-feature discount with fitting data and hold-out error), interactions (`applies_when`, evidence), composition rules. Restricted expression language plus table terms; every value points to its HwFact or source.
  - L1 pricing: the analytic steady-state evaluation of a chain's effects against a HwModel (per-unit busy, SOL per unit, predicted total); `models/bound.py` and the unit model rewritten on top of it.
  - A GB10 class + instance seeded from existing peaks and documented values, every unmeasured value marked `documented` or `default`.
  - Pairwise interference microbenchmark definitions (A alone, B alone, A+B) as jobs and the interference-matrix artefact schema; no runs.
  - Model versioning: parameter table + simulator commit + regression results, immutable; regression-set record schema (several configurations per kernel).
- **Acceptance (code stage)**: seed templates validate with `effects`; GEMM seed chain priced by L1 with per-unit SOL; changing one wall or peak changes the prediction without code changes; old `cost` records upcast; failure fixtures for unpriced operation classes (counted, not silently zero-cost: unmodelled resources default to infinite capacity and are reported).

## WP16 Transaction-level simulator, what-if and regression scoreboard

- **Depends**: WP15. **Reads**: 02 §4 (L2, L3, "Sustained saturation and walls", "Maintenance"), 02 §5.1.
- **Closes**: R3 (L2), R5, E9, E10 (adapter only), E11, E14 (scoreboard part).
- **Produces**:
  - L2: per-CTA transaction graph on HwModel resources; outputs total cycles, per-unit busy time, occupancy timeline, stall reasons and idle attribution (waiting for another unit, for data, for sync, prologue/epilogue/tail, blocked by a wall), metrics named as ncu names them; deterministic and versioned.
  - What-if runs: move a wall or a peak, re-simulate, report the shadow price (idle time removed).
  - L3 adapter interface for the external SASS-level simulator (research repo, read-only, not imported): run through the Executor as an external tool; `NotSupported` until it has been run for real (pending real run).
  - Regression set and scoreboard: every measured kernel becomes a regression case; rank correlation and error per resource per model version; API + web views (scoreboard, per-kernel predicted-vs-measured timeline).
- **Acceptance (code stage)**: GEMM seed family simulated at L2 with a timeline; L1 equals the steady-state limit of L2 on a loop-dominated fixture; what-if shows a non-zero shadow price on a walled fixture; scoreboard renders on fixture data marked FIXTURE.

## WP17 Steady-state projection compiler and MIP↔simulator check (done inside WP19, D328)

- **Depends**: WP15, WP16 (for the check), WP19. **Reads**: 02 §5.1, §5.2; 22 §3.3; 10 §4.4; `src/kta/formulation/`, `src/kta/solve/`.
- **Closes**: R1, R4, E8; D-T1, D-T4.
- **Produces**:
  - The projection compiler: from FormulationPlan + chain effects + HwModel to a MIP with ResMII, RecMII, one-hot menus, epigraph max, integer ceil, menu products, table terms; non-linearisable effects moved to outer enumeration. The agent writes none of it.
  - The MIP↔L2 consistency check without a GPU: sample the subspace, evaluate MIP and L2, require rank correlation above a threshold and MIP ≤ simulator on nearly all points; failure names the effect with the largest deviation. Thresholds in the controls registry.
  - An agent-written-formulation arm kept for comparison (D-T1): the existing hand-formulation path stays callable and is labelled as the comparison arm.
- **Acceptance (code stage)**: GEMM seed formulation compiled and solved; consistency check passes on the GEMM family and fails, naming the effect, on a fixture with a deliberately wrong reservation table.

## WP18 Gap decomposition, routing and loop disciplines

- **Depends**: WP15, WP21. **Reads**: 02 §1 (P7), §6.1, §9; 12 §3, §4; `src/kta/orchestrator/`.
- **Closes**: R6, E3, E4, E6; D-T6, D-T11, D-T12, D-T13, D-T17 (canaries).
- **Produces**:
  - Gap decomposition `T_meas − T_SOL = ε + δ + Δpre + L_rep + irreducible` as a record per round, attribution order correctness → ε → δ → Δpre → L_rep → irreducible, SOL as the bound on `L_rep`.
  - A data-driven routing table (gap class → next action / task type / escalation), versioned.
  - Loop disciplines: improvement band and mechanism attribution, null-round limit, one mechanism per round (coupled groups excepted), hold-out verification with rollback, exploration share, duplicate-proposal rejection by hash, accept and escalate rules of 12 §4.
  - Selection controls on every solve (model top-k vs copied assignment vs random k), recorded.
  - Canaries: planted-fault kinds, one per batch rotating, missed canary escalates to the user.
- **Acceptance (code stage)**: fault-injection fixtures (degraded model, deleted template, capped solver) route to the right class (O5 harness, deterministic); every threshold read from the controls registry.

## WP19 FormulationPlan, chain enumerator and access discipline

- **Depends**: WP14, WP15. **Reads**: 02 §5.3–5.5; 10 §4; 22 §5; `src/kta/agents/skills/`.
- **Closes**: R7, R8, E12; D-T18, D-T19, D-T20.
- **Produces**:
  - `FormulationPlan` schema (F1–F7: chain, bindings, variable status with the structure it invokes and the evidence it cites, menus, table terms, glue/dispatch, generator) and the deterministic default plan as the same object; plan diff and score against the default.
  - A bounded chain enumerator over the template grammar: legal chains, Γ and capability filter, SOL bound under minimal work, pruning against the incumbent; it is the default for F1 and yields coverage counts. Composition statistics as a derived view over reconstructed chains.
  - The one-screen task header and the versioned access-discipline table in the worker skill (`use-kta`/`find-knowledge`), pull log records (`kta` pulls outside the table allowed and logged), and a pull-log view.
- **Acceptance (code stage)**: default plan produced for the GEMM seed case; enumerator finds the seed chain within its bound; an agent-plan fixture is diffed and scored; header fits one screen; pull log written for a fake-adapter task.

## WP20 Learning-loop runner

- **Depends**: WP14 (and WP10b). **Reads**: 02 §8; 21; 24; 01 §7; `deliveries/WP10a.md`, `deliveries/WP10b.md`.
- **Closes**: R15 (coverage part), R16, R17, R18 (mechanics; values in the controls registry), E13, E14 (pool-growth part); D-T29 (expressibility coverage); D-open-2, D-open-4, D-open-6, D-open-7 (definitions only).
- **Produces**:
  - Pipeline per reference: intake (deterministic job definitions: clone at commit, license, hash, container build, run + SASS + profile) → decompose (T2, reference directory = the worker's deliverable) → admit (deterministic; errors back to the same session, ≤ 2 retries; warnings must be explained) → reproduce → ablate → cover (snapshot, waiting cases resume).
  - The "inexpressible" category: expression-language gaps counted and admitted at `proposed`, not refused; the seven WP10a decompositions imported through this path into the store (not into `knowledge/seed/`).
  - Cost tiers L0–L4 as data with admission rules; performance-core definition for reproduction; ablation job generator on our own families (coupled pairs only, fractional factorial when large) producing `coupled_with` evidence.
  - SOTA coverage in four levels and system-level features counted separately; per-backend expressibility coverage and intent fidelity; source-class and priority fields on references.
  - Blind-reconstruction (O12) protocol definition as the learning loop's acceptance experiment (not run).
- **Acceptance (code stage)**: the seven decompositions go through admit with counted language gaps; a fake-adapter T2 session receives rejection errors and retries; coverage page shows the four levels and the system-feature count from store data; intake/reproduce/ablate jobs defined and dry-run planned, real runs pending.

## WP21 Controls registry and staged release

- **Depends**: WP6, WP11. **Reads**: 12 (all); 02 §6.3, §11; 01 §3, §6; 30.
- **Closes**: R9 (staged release), R11, R12; D-T22, D-T23, D-T25, D-T28; D-open-6 (values).
- **Produces**:
  - One versioned controls registry holding every row of 12: value, failure prevented, where (profile / orch / protocol / UI), status (spec / default / set by X-n run), `set_by` experiment. Code reads knobs only from it; scattered constants removed. Changing a value creates a new version recorded with every run it affects.
  - A settings page (API + web) that edits only these versioned values; reading is open, writing needs the operator token; every write is a new version with an author and reason.
  - Staged release S0–S4 as orchestrator configuration: which processes and task types may run, whether agent calls are gated (S1: one call at a time, held until the user releases it after reading card, pull log and plan), exit criteria shown in the UI.
  - X1–X9 experiment protocol definitions (arms, metrics, which knob each sets); none are run.
- **Acceptance (code stage)**: every 12 row present with `set_by`; a value change is versioned and visible in a run record; S1 gate holds a fake-adapter task until released; settings page renders and writes a new version.

## WP22 System settings, agent proxy and frontend parity

- **Depends**: WP8, WP9b, WP21; reads the round-4 deliveries (WP14–WP21). **Reads**: `spec/modules/agents.md`, `spec/modules/api-web.md`, 30.
- **Why**: the Codex and Claude Code CLIs ignore the macOS system proxy, so agent runs went direct; the user requires both to go through the local Clash HTTP proxy, with no silent direct fallback, while pi and the DeepSeek harness must stay direct. The user also wants every mechanism of the round-4 design visible in the frontend.
- **Produces**:
  - A system configuration on the control plane, separate from the controls registry (infrastructure, not research knobs): proxy override, per-provider proxy policy (`required | off`), preflight hosts per provider, optional per-worker intake proxy. Written through the API with the operator token; every write is audited (author, time, reason) by a HumanAction.
  - Proxy auto-detection: `scutil --proxy` on macOS, then `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY` of the control-plane environment, then `http://127.0.0.1:7897`; the source is shown and recorded. Credentials are never stored in the configuration or records; they live in `secrets.env`.
  - Injection, not inheritance: for `required` providers the runner injects `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` (and lowercase) from the system configuration plus `NO_PROXY`/`no_proxy` that always hold loopback and the `KTA_API` host; `off` providers get no proxy variable at all.
  - Fail closed: before a `required` provider starts, a preflight on the host where the CLI runs checks the proxy port and an HTTPS CONNECT plus TLS handshake to the provider's API hosts (no model call). On failure the CLI does not start, the runner raises `ProxyUnavailable`, a Report records the failure and the Inbox shows it.
  - Traceability: AgentRun `proxy` (used, scheme, host, port, source, preflight) and the context manifest's `proxy` section; never credentials.
  - Settings page split into "Research controls" (unchanged) and "System": proxy (detected value and source, override, per-provider policy, Test), providers (CLI version, logged in yes/no), workers (address, commit vs control plane, GPU, Test job through the Executor), data directory, release stage.
  - Frontend parity audit of the round-4 mechanisms (FormulationPlan diff, L2 what-if, canaries, selection arms, learning-loop stages, X1–X12) with the missing views added.
  - Part D (added 2026-10-31 by the user): an agent workflow view per task type (T1–T6) derived from code and configuration, every node clickable to what configures it; profile parts (templates, skills, model, effort, provider) edited as a draft AgentProfile version with a diff, a live TASK.md preview and a placeholder check, approved by the operator; knobs through the controls versioning, the proxy policy through the system configuration; done checks and invariants read-only.
- **Acceptance (code stage, D320)**: contract tests for whitelist injection per policy, NO_PROXY content, fail-closed preflight (proxy down → launch refused) and no credentials in records; a real auto-detect and preflight against the local Clash on the control host (no model call); `scripts/gate --stage code WP22`.
