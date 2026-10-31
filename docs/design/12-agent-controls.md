# 12 Agent controls: the knobs that keep the agent working as intended

**Status: default values, to be tuned.** This is the one table of every setting that governs how much an agent may spend, how many times it may fail, when it is cut off, and what stops it from digging in, slacking or gaming its checks. Each row names the failure it prevents, so a knob that never prevents anything can be removed. Values marked *spec* are already in `spec/` or the numbered design documents; values marked *default* were set in design discussion (01 §13) and have no evidence yet. Every knob is meant to be adjustable without code changes: in the versioned AgentProfile (10 §4.10), in orchestrator configuration, or, where marked, in a frontend settings page (30).

How to read "where": **profile** = AgentProfile, versioned, hash recorded per run; **orch** = orchestrator configuration; **protocol** = frozen experiment protocol (50); **UI** = exposed in the frontend as a setting. **set_by** names the X experiment of 02 §11 that sets the value; "—" means no experiment is planned and the value stands as it is.

**Implemented (WP21, D480–D486).** Every row below is a knob of the controls registry (`src/kta/core/controls.yaml`; spec/modules/core-store.md "Controls registry"). Code reads the values only from the registry. The values in force are a versioned record (ControlsVersion), changed on the settings page (`/settings`, operator token, author and reason per version) and recorded with every run they affect (`refs.controls` on TaskCard, AgentRun and ExperimentRun; `controls` in loop data). Rows marked as invariants in the registry are shown on the page but are not editable. "Where" keeps the design's meaning: a knob marked *profile* is read by the agent runner from the registry at each step, not from the AgentProfile record.

## 1. Budget and rounds

| Knob | Controls | Value | Prevents | Where | Status | set_by |
| --- | --- | --- | --- | --- | --- | --- |
| rounds per case loop | iterations of the case loop before escalation to the user | 5 | endless polishing of one case | orch, UI | spec (10 §2.1) | S1/S2 logs |
| retries per task | re-queues after rejection before escalation | 2 | repeating the same failed attempt | orch, UI | spec (10 §3.2) | X5 |
| task wall-clock budget | model time + tool time per task, by type | T1 family 60 min, T2 decomposition 90 min, T3 template 30 min, T4 model diagnosis 45 min, T5 microbenchmark 30 min, T6 repair 30 min | runaway sessions | profile, UI | default | X1, X9, S1/S2 logs |
| hard wall-clock cap | total elapsed time including queueing | 3 × task budget | hanging on a long GPU queue | orch | spec (11 §3.3) | — |
| no-response timeout | CLI output and session file unchanged | 10 min | stuck process | orch | spec (10 §4.8) | — |
| lease | task lease with renewal | 30 min, renewed while running | orphaned tasks after a crash | orch | spec (10 §4.3) | — |
| GPU seconds per task / per case | executor-accounted GPU time | per type; case cap = 10 × task cap | measuring instead of thinking | profile, orch, UI | default | X1, S1/S2 logs |
| tool-call steps per task | number of `kta` invocations | 200 | thrashing | profile | default | X1, S1/S2 logs |
| budget reminder | increment message to the agent | at 80 % of any budget | surprise cut-off | profile | spec (10 §4.10) | — |
| process quotas | share of global budget | experiment 50 %, solve 25 %, learn 15 %, calibrate+maintain 10 % | one process starving the others | orch, UI | spec (11 §3.1) | — |
| agent instances | concurrent agent sessions | start 2, max 4 | GPU queueing, rate limits | orch, UI | spec (10 §4.5) | X12 |
| user attention cap | pending items at once | 10; review queue shows 5 | problems piling up unseen | orch, UI | spec (11 §2), 01 §12 | — |

## 2. Session and context

| Knob | Controls | Value | Prevents | Where | Status | set_by |
| --- | --- | --- | --- | --- | --- | --- |
| fresh-session threshold | context used before the session is replaced | 60 % of the model window | degraded reasoning in a full window | profile | spec (10 §4.2) | — |
| same-failure rule | fresh session after the same check fails for the same reason | 2 times | digging in on one line of thought | profile | spec (10 §4.2) | — |
| resume scope | sessions resumed within a work unit, never across | per unit | context pollution across unrelated work | orch | spec (10 §4.2); O11 tests resume vs fresh | X3 |
| step-header size | cards in the task header | ≤ 5 cards, ≤ 5 lines each; header ≤ 1 page | burying what matters | profile | spec (10 §4.10, 22 §5) | — |
| tool-output truncation | summary length of compile logs, ncu, checks | summary + ID; `--full` on request | token waste | profile | spec (10 §4.4) | — |
| access discipline version | which situation → fetch → decide table the worker follows | current version | ad-hoc browsing | profile | default (01 §7.1) | X4 |
| `NOTES.md` required sections | goal, done, tried-and-failed, why, next | all five | forgetting failed attempts across sessions | profile, gate | spec (10 §4.2) | — |

## 3. Against digging in, slacking and gaming

| Knob | Controls | Value | Prevents | Where | Status | set_by |
| --- | --- | --- | --- | --- | --- | --- |
| improvement band | minimum gain that counts | max(measurement CI, 5 % of T) | crediting noise | orch, UI | default (01 §9.4) | X6 |
| mechanism attribution | a credited gain must name a template step, opened variable or model term | required | unexplained tweaks presented as progress | orch | default (01 §9.4) | — |
| null-round limit | rounds that add nothing and gain below the band | 2 → close the case | polishing instead of explaining | orch, UI | default (01 §9.4) | S1/S2 logs |
| one mechanism per round | changes per iteration (coupled groups excepted) | 1 | unattributable progress | orch | default (01 §9) | — |
| exploration share | measured points chosen for information, not by the model | 2 of 8 | the model only tested where it already believes | orch, UI | default (01 §9) | X8 |
| duplicate-proposal rejection | a resubmission identical (by hash) to a rejected one | rejected without an agent run | retrying the same thing | orch | default | — |
| completion checks fixed | the agent cannot alter its task's checks | immutable task card | redefining success | orch | spec (10 §3.3) | — |
| hidden shapes and seeds | not visible to the agent; checker returns verdicts and failing samples only | hidden | overfitting to the test | orch | spec (10 §4.7) | — |
| evidence write ban | agent cannot write measurements, checks or records | enforced by user permissions | fabricated results | orch | spec (10 §4.7) | — |
| fixture declaration | any simplification must be declared; gate scans for stubs, fixed returns, skipped tests | required | silent fallback presented as done | gate | spec (60 §4) | — |
| structural dedupe | new templates deduplicated by structural signature | required | farming variants of one idea | knowledge | spec (22 §5) | — |
| evidence citation in plans | every non-default FormulationPlan decision cites a structure and evidence | required | unjustified fixing or decomposing | orch | default (01 §7.1) | — |
| selection controls | model top-k vs copied assignment vs random k recorded per solve | always | a solver that adds nothing going unnoticed | orch | default (01 §11 E4) | — |
| canary rate | planted faults the checks must catch | 1 per batch, rotating kinds | empty checks | protocol, UI | default (01 §12) | — |

## 4. Escalation and reliability

| Knob | Controls | Value | Prevents | Where | Status | set_by |
| --- | --- | --- | --- | --- | --- | --- |
| accept | not worse than strongest reference and unexplained gap < 10 % of T; or gap fully explained as irreducible | as stated | accepting unexplained results | orch, UI | default (01 §9.4) | — |
| escalate case → knowledge | no applicable template; same residual class in ≥ 3 cases; top-k model error > 30 % after one calibration | as stated | the case loop doing the knowledge loop's work badly | orch | default (01 §9.4) | S1/S2 logs |
| escalate → user | 5 rounds; budget overrun; canary missed; claim or criterion change | as stated | silent scope changes | orch | spec + default | — |
| finding ladder | observed → replicated (≥ 2 independent shapes/runs, same direction, beyond band) → explained → invalidated | as stated | one measurement becoming a rule | knowledge | default (01 §9.4) | — |
| model-term provisionality | regression cases needed before a term is trusted | 5, with hold-out improvement | an unsupported term steering selection | knowledge | default (01 §9.4) | S1/S2 logs |
| template assurance for use | declared templates may be used; results flagged | flag | unproved steps presented as proved | orch | spec (22 §6) | — |
| level choice | graph-level bound < 0.8 × sum of kernel bounds → graph moves first (phase one records only) | 0.8 | optimising a kernel whose loss is in the graph | orch | default (01 §9.4) | S1/S2 logs |

## 5. Maintenance

A knob is added with the failure it prevents and removed when it has prevented nothing for a milestone while costing effort (60 §7). Changing a value is a controls version, recorded with every run it affected. The settings page edits these same versioned values and nothing else. Adding a knob is one row in `controls.yaml` plus one typed accessor, and a row here.

## 6. How values get set

No value in this table is fixed by discussion alone. A value marked *default* stands until the experiment that sets it has run (02 §11): X1 model per task type → §1 budgets and the AgentProfile model; X2 effort level → AgentProfile; X3 resume vs fresh → §2 resume scope; X4 access discipline → §2 discipline version; X5 retries → §1; X6 noise band → §3 improvement band; X7 reproduction ε → §4; X8 measured points and exploration share → §1, §3; X9 task budgets → §1; X10 hardware-model instance values → HwModel resources and interactions (not rows here) and §7 model-version admission; X11 MIP ↔ L2 acceptance thresholds → the check's thresholds (rows added with the check, WP19); X12 concurrent agent instances → §1. Rows whose set_by cell says "S1/S2 logs" (rounds per case loop, task budgets, null-round limit, escalation thresholds, model-term provisionality, level choice) are revised from the logs of S1/S2 runs, not by a dedicated experiment (02 §11 order, step 5). When a value is set by an experiment, its status column names the experiment and the run.

## 7. Rows added from code and D325 (WP21)

Thresholds that the code already used and that govern the loops, plus the D325 defaults and the release stage. They live in the same registry, versioned the same way.

| Knob | Controls | Value | Prevents | Where | Status | set_by |
| --- | --- | --- | --- | --- | --- | --- |
| shortlist size | candidates shortlisted by the pre-compile model per bucket | 64 | compiling a search space instead of a shortlist | orch | spec (10 §2.1) | X8 |
| solver time limit | wall time of one MIP solve | 60 s | a solve blocking the loop | orch | default (D263) | — |
| model-off thresholds | judging rule 2: Spearman ρ below rank_min with ≥ rank_n points, or median relative error above rel_error_max | ρ 0.5, n 3, error 0.3 | a model that misranks steering the search unchecked | orch | default (D263) | — |
| explained-gap share | judging rule 3: share of the gap to the bound the model must explain | 0.8 | an unexplained gap passing as a result | orch | default (D263) | — |
| suspicious result | inbox item when faster than the strongest reference by this factor or closer to the bound than this | 1.3 ×; 95 % | a measurement or checking error accepted as a breakthrough | orch, UI | spec (30 §2) | — |
| repeated check failure | inbox item when the same check fails in this many tasks | 3 | a broken checker or systematic defect going unseen | orch, UI | spec (30 §2) | — |
| budget warning | inbox item when a budget scope is used to this fraction | 80 % | running out of budget without a decision | orch, UI | spec (30 §2) | — |
| infrastructure retries | retries of a loop step after an infra or timeout error before the loop blocks | 3 | one flaky worker blocking a loop; endless retry on a dead worker | orch | default (D264) | — |
| reproduction tolerance ε | relative error allowed when reproducing a reference's performance core from its chain | 10 % | a wrong decomposition passing as reproduced, or a right one refused for noise | orch, UI | default (02 §8.4, D325) | X7 |
| learning scope | workloads, target and knowledge in scope; out-of-scope features counted separately | v1 of 01 §7 | learning drifting to features we cannot use or compare | orch, UI | default (01 §7, D325) | — |
| release stage | which processes and task types may run and whether agent calls are gated (`stages.yaml`, 02 §6.3) | S0 | opening a stage before the previous one is understood | orch, UI | spec (02 §6.3) | — |
| model-version admission | hold-out regression decision of a hardware-model version (02 §4 "Maintenance"): mean Spearman ρ at least rank_min and median relative error of the total at most rel_error_max | ρ 0.7, error 0.25 | a model change that misranks the hold-out kernels being admitted | orch, UI | default (D509) | X10 |
