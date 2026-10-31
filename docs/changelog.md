# Change log

One entry per design conversation or merged work package. Each entry says what was added or changed, which design requirements came out of it with their status, and what construction should do next. Decision IDs `D-Tn` refer to `docs/design/02-design-notes.md` §12.

## 2026-10-30 → 2026-10-31: thinking-model conversations

### Files added

- `docs/design/01-thinking-model.md`: the human-readable thinking model (claims, mental model, concerns, components, decisions, experiments, scope).
- `docs/design/02-design-notes.md`: detailed material from the conversations, organised by the design document each part merges into; the full decision ledger (D-T1 … D-T30) and open decisions (D-open-1 … 7).
- `docs/design/12-agent-controls.md`: every budget, cut-off, session and anti-gaming control with the failure it prevents and where it is set.
- `.agents/skills/design-interrogation/SKILL.md`: nine questions per mechanism, an adversarial second pass at milestones, three required artefacts; narrow triggers.
- `docs/references/2610.39380-learning-to-formulate.pdf` and `docs/references/README.md`: the Formulaic paper, frozen with hash and a section map.
- this file.

### Files changed

- `README.md`: index rows for 01, 02, 12 and the change log.
- `.agents/skills/README.md`: row for `design-interrogation`.

No code, spec or seed data was changed. Nothing is committed.

### Design requirements from the conversations

Status: **decided** = the user decided; **open** = waiting for the user; **default** = assistant's choice, stands until pulled.

| # | Requirement | Where | Status |
| --- | --- | --- | --- |
| R1 | The MIP is assembled mechanically from chain + hardware model; the agent does not write MIPs per case; keep an agent-written arm for comparison | D-T1 | default |
| R2 | Template `cost` becomes `effects` (transactions, storage, overlap); hardware coefficients live only in the hardware model | D-T2, 02 §3 | default; timing open (D-open-5) |
| R3 | The hardware model is declarative data (resources, walls, operation reservation tables, efficiency, interactions, composition); simulator, MIP and reports derive from it; four fidelity levels share one parameter table | D-T3, D-T5, 02 §4 | default; timing open (D-open-5) |
| R4 | MIP = steady-state (modulo-scheduling) projection; MIP ↔ simulator consistency check without a GPU | D-T4, 02 §5 | default |
| R5 | Fine-grained resource sharing as shared resources discovered by interference microbenchmarks; completeness as measured coverage | D-T14, D-T15 | default |
| R6 | Gap decomposition routes every next action; attribution order correctness → ε → δ → Δpre → L_rep → irreducible | D-T6, D-T11 | default |
| R7 | FormulationPlan as structured object; deterministic default plan and chain enumerator; agent plans scored against the default | D-T18, D-T19 | default |
| R8 | One-screen task header + versioned access discipline; pulls logged | D-T20 | default |
| R9 | Loops with bridge objects; design loop outermost; staged release S0–S4 | D-T21, D-T22 | decided in substance |
| R10 | Independent processes, no multi-agent; session policy of 10 §4.2 kept; calibration a branch of the knowledge loop | D-T24 | decided in substance |
| R11 | One tunable controls table (12); a frontend settings page later; a knob that prevents nothing is removed | D-T25 | decided |
| R12 | System parameters set by small experiments X1–X9; defaults say so until then | D-T28 | decided |
| R13 | Capability replaces HwFeature; Realisation split from Template; Finding unifies Regime and Negative; HwModel split into class and instance; levels numbered 1–7 | D-T7…D-T10 | default |
| R14 | Unprobed mechanisms are recorded (`probed_on: []`), filtered by capability, counted as "known, unavailable here" | D-T27 | decided in substance |
| R15 | Backends: TileLang primary, CUDA C++ + PTX escape hatch, CuTe DSL not pursued for GB10, Triton for controls, MLIR not in phase one; per-backend expressibility coverage recorded | D-T29 | decided |
| R16 | Learning loop automated: intake → decompose (T2) → admit with rejection loop → reproduce → ablate → cover; reference directory = T2 deliverable format; `knowledge/seed/` = seed only | 02 §8.6 | default; D-open-2, D-open-4 |
| R17 | Coverage of SOTA reported in four levels; system-level features counted separately; a SOTA kernel yields mainly a chain + regime, rarely a template; composition statistics are a derived view | 02 §8.1–8.2 | default |
| R18 | Cost tiers L0–L4 with admission rules; "performance core" reproduction within ε = 10 %; ablation on our own families; `coupled_with` as first-class knowledge | 02 §8.4 | open (D-open-3, D-open-6) |
| R19 | Blind reconstruction (O12) as the learning loop's acceptance experiment | 01 §6 | open (D-open-7) |
| R20 | `design-interrogation` run on spec interface changes, new loops/roles/bridges, control changes, needs-review decisions, end of design conversations, milestones; not on implementation within a frozen interface | skill | default |
| R21 | 01 stays short (~250 lines); 02 is a staging area merged into numbered documents by construction | D-T30 | decided |

### What construction should do next, in order

1. Read 01, then 02, then 12. Treat 02 as proposals until the open decisions are answered.
2. Merge 02 into the numbered documents once the user confirms each part: §1 → 00; §2 → 23 and 21 §1; §3 → 22 §2; §4–5 → a new `25-hardware-model.md` plus 22 §3.3 and 10 §4.4; §6 → 10 §2 and 11 §1; §7 → 50 §4; §8 → 21 and 24. Each merge is a `design-interrogation` trigger (spec-facing).
3. Do not start schema changes (R2, R3, R13) before D-open-5 is answered.
4. Prepare S0: the GEMM golden path with no agent (WP7 scope), reporting % of SOL and the cuBLAS comparison, with every record visible.
5. Prepare L1 on the seven references: measurement under our protocol and the SASS instruction census (deterministic jobs).
6. Build the admission rejection loop and the "inexpressible" category (R16) so the seven decompositions can be imported as `proposed` (after D-open-2).
7. Add a `set_by` note to each row of 12 naming its X-experiment (12 §6), and record in `docs/decisions.md` any D-T entry the user confirms.

### What the user still has to decide

D-open-1 … D-open-7 in 02 §12.

## 2026-10-31: merged work package WP14 (knowledge model v2, code stage)

Branch `wp/WP14`; delivery note `deliveries/WP14.md`; decisions D440–D454.

### Closes

| Item | How |
| --- | --- |
| R13 (except the HwModel split, WP15) | Capability + CapabilityDecl replace HwFeature in `Template.requires`; Realisation split from Template; Finding unifies Regime and Negative; levels 1–7 |
| R14 | `probed_on` on Template and Realisation; retrieval filters by the target's capabilities; coverage counts "known, unavailable here" and "unconfirmed here" |
| E5 (population governance, knowledge part) | one ladder with evidence rules; promotion/demotion engine with stale propagation; promotion review Report |
| E7 | level numbering 1–7 with the versioned mapping `paper-0to5@1` |
| E14 (pool-diff part) | `GET /pool/diff` and the `/pool/diff` page |
| E15 | `Coupling` records (`coupled_with` with evidence and scope); unprobed mechanisms recorded and counted |
| D-T7, D-T8, D-T9, D-T27 | implemented as defaults (D441–D445) |
| D-open-1, D-open-3 (relation part) | implemented per D325 (D446, D447) |

02 sections implemented: §2 (definition, objects, promotion and demotion, unprobed mechanisms, two axes, status ladder), §8.2 (Finding and Realisation as products of a source), §8.4 (`coupled_with` relation only; ablation jobs are WP20).

### Also changed

- Schema versioning by upcast on read (D440): Pattern, Template, Gap, Reconstruction are schema 2; old records read back unchanged in storage.
- Seed migrated by `scripts/migrations/WP14_knowledge_v2.py` (ledger `knowledge/seed/migrations.yaml`); 16 Capabilities and 16 GB10 declarations added; 18 Realisations extracted from Template bodies (1 seed, 17 in references).
- Worker skills `decompose-kernel`, `write-template`, `find-knowledge` use `requires` (Capabilities), levels 1–7 and Realisations.

### Not done here

- No real run (D320): no probe confirmed any GB10 declaration; `probed_on` is empty everywhere. Listed under "Pending real runs" in the delivery note.
- Knobs (`MIN_REPLICATIONS`, `MIN_TARGETS_FOR_SCOPE`, `MIN_REPEATS_FOR_PATTERN`) are module constants until WP21's registry merges.
- 23 (knowledge structure) and 21 §1 are not rewritten; 02 §2 stays the design source until the merge of 02 into the numbered documents.
## 2026-10-31: merged work package WP15 (Template effects and the declarative hardware model)

Code stage (D320), branch `wp/WP15`; decisions D460–D474 (D325 (5) answers D-open-5 for now: R2 and R3 land before the golden path).

### Closes

| Item | How | Status |
| --- | --- | --- |
| R2, E2, D-T2 | Template `cost` → `effects` (transactions, storage, overlap) in schema, loader, validators, formulation and all 37 seed Templates (`scripts/migrations/WP15_effects.py`); schema-1 records upcast on read | code stage |
| R3 (data and L1), E1, D-T3, D-T10 | `HwArch` (class) + `HwModel` schema 2 (instance) as declarative data with per-value provenance; restricted expression language (roots `hw`, `tx`, `f`, `tbl`) and table terms; L1 steady-state pricing; `bound` and the UnitModel rebuilt on it | code stage |
| R5 (schema and job definitions), D-T14, D-T15 | interference microbenchmark definitions (A, B, A+B per pair), `InterferenceMatrix` artefact schema with coverage; unmodelled = infinite capacity, reported | code stage; runs pending |
| R8 (record schemas), R9 | `ModelVersion` and `RegressionSet` (several configurations per kernel) as immutable records | schema only |
| WP5 known issue | contraction and operand traffic declared once (instruction selection, transfer step) and priced by the model | code stage |

### 02 sections implemented

§3 (effects; the granularity statistics are not built), §4 (resources, walls with cliffs, operation classes, efficiency, interactions, composition, completeness as counted coverage; L2/L3 and maintenance views are WP16), §5.1 (L1 = the projection's terms; the MIP compiler is WP17). The interface WP16 and WP17 consume is in `spec/modules/formulation-solve-models.md` "Hardware model and L1".

### Pending real runs

`kta.measure.peaks` on the GB10 worker (measured peaks attach to `gb10` on `sm12x@1`), the interference pairs, and a first regression set with a `ModelVersion`; see `deliveries/WP15.md`.

## 2026-10-31: merged work package WP21 (controls registry and staged release)

Code stage (D320): no real runs, no experiments. Branch `wp/WP21`; decisions D480–D489.

### Closes

| Item | How |
| --- | --- |
| R9 (staged release part) | S0–S4 as orchestrator configuration (`orchestrator/stages.yaml`), stage = knob `release.stage` (default S0); S1 gates each agent call until you release it; exit criteria on the Release page |
| R11 | one versioned controls registry (`core/controls.yaml`, ControlsVersion records); code reads knobs only from it; settings page `/settings` edits only these values (operator token, author and reason per version) |
| R12 | X1–X12 defined (`core/experiments.yaml`: arms, metrics, knobs set, 02 §11 order, run or logs; X10–X12 per D328); every row of 12 names its `set_by` experiment or "S1/S2 logs" (D490) |
| D-T22, D-T23, D-T25, D-T28 | implemented as the defaults they are |
| D-open-6 (values) | scope v1 and ε = 10 % are registry values (`learning.scope`, `learning.reproduction_epsilon`), status default |

02 sections implemented: §6.3 (staged release), §11 (system-parameter experiments, definitions only); §9 thresholds now live in the registry.

### Files changed outside code

- `docs/design/12-agent-controls.md`: a `set_by` column on every row (next step 7 above), an "Implemented" paragraph, §5 wording, and a new §7 with the thresholds the code already used (shortlist, solver time limit, judging rules 2–3, inbox thresholds, infra retries), the D325 defaults and the release stage (D482).
- `spec/`: core-store.md "Controls registry" (including how to add a knob), schemas.md (ControlsVersion, `refs.controls`), orchestrator-cli.md, agents.md, api-web.md, gate.md "WP21".

### Behaviour that changes at merge

- A fresh store is in **S0**: the agent pool starts no task and the learn, experiment and summarize processes skip their ticks until you write a controls version with `release.stage` S1 or later (Release page).
- `[orchestrator]` config is no longer read; `[budget.quotas]` and `[agents] slots` are refused (use the knobs `budget.process_quotas`, `budget.agent_instances`).
- Task wall budgets follow 12 (T4 45 min, T5 30 min, T6 30 min).

### Next for construction

- WP18 and WP19 add their knobs as registry rows (spec/modules/core-store.md "Adding a knob"); WP19 fills the pull log and plan the Release page already shows as "not recorded yet".
- Owed real run: a hand-stepped S1 run with a real agent (deliveries/WP21.md "Pending real runs").

## 2026-10-31: merged work package WP20 (learning-loop runner, code stage)

Branch `wp/WP20`; delivery note `deliveries/WP20.md`; decisions D560–D569.

### Closes

| Item | How |
| --- | --- |
| R15 (coverage part) | SOTA coverage in four levels; system-level features and language gaps counted apart (`kta.knowledge.sota`, coverage page) |
| R16 | per-reference pipeline intake → decompose → admit → reproduce → ablate → cover: admit implemented (T2 done-check `admission`, `kta reference admit`), the other deterministic stages defined as dry-run plans (`kta reference plan`) |
| R17 | cost tiers L0–L4 as data with named admission rules; performance core; source class and priority on references (Reference schema 2) |
| R18 (mechanics) | ablation design generator on our own families (pairwise / fractional factorial) producing `Coupling` evidence; thresholds as table values awaiting the registry |
| E13 | language gaps admitted at `proposed` and counted; the seven WP10a decompositions imported through admit into the store |
| E14 (pool-growth part) | admission Reports per reference; coverage shows language gaps, system features and per-backend expressibility |
| D-T29 | per-backend expressibility coverage and declared intent (confirmation pending compiled SASS) |
| D-open-2, D-open-4, D-open-6, D-open-7 | definitions per D325: admit language gaps; worker deliverables into the store only; ε = 10 %; O12 protocol defined, not run |

02 sections implemented: §8.1, §8.2 (system features counted apart), §8.3 (source classes, priority), §8.4 (tiers, performance core, ablation on our families), §8.5 (per-backend coverage), §8.6 (admit, plans).

### Not done here

- No real run (D320): no intake, reproduction, ablation or O12 run; `kta reference plan` only plans.
- Thresholds read the controls registry: `budget.retries_per_task`, `learning.reproduction_epsilon`, and the new `learning.ablation_pairwise_max_runs`, `learning.ablation_max_runs`, `learning.coupling_noise_band`.

## 2026-10-31: merged work package WP16 (transaction-level simulator, what-if, regression scoreboard)

Code stage (D320): no real runs, no experiments. Branch `wp/WP16`; decisions D500–D512.

### Closes

| Item | How |
| --- | --- |
| R3 (L2) | `kta.models.l2.simulate`: per-CTA transaction graph on the declarative hardware model, waves, idle attribution; deterministic, versioned (`l2:<SIM_VERSION>:<MachineModel.version>`) |
| R5 | total, per-resource busy, occupancy timeline (waves and busy intervals), stall reasons, idle by other unit / data / sync / prologue-epilogue-tail / wall |
| E9 | what-if: move a wall or peak in memory, re-simulate, shadow price (`kta.models.whatif`) |
| E10 (adapter only) | `ExternalSassSimulator`: the research simulator as an Executor job; NotSupported until run for real |
| E11 | metrics named as ncu names them (D505) |
| E14 (scoreboard part) | regression scoring (Spearman, error per resource), admission knob, `/model/scoreboard` API + web page on FIXTURE data |

02 sections implemented: §4 L2 and L3 rows of "Four fidelity levels", "Sustained saturation and walls" (idle attribution, shadow price), "Maintenance as a software artefact" (regression cases, scoreboard, per-kernel predicted timeline, admission); §5.1 checked: L1 is the steady-state limit of L2 on a resource-bound loop, and L2 exceeds L1 by the recurrence on shallow pipelines (L1 counts latencies only).

### Files changed outside code

- `spec/modules/formulation-solve-models.md` (L2; what-if, L3, scoreboard), `spec/schemas.md` (RegressionResult fields), `spec/modules/api-web.md` (model views), `spec/gate.md` (WP16), `docs/design/12-agent-controls.md` (§7 row "model-version admission", X10 sets it).

### Next for construction

- WP19's MIP ↔ L2 consistency check can call `kta.models.l2.simulate` on `formulation.evaluate_effects(...)`; note the recurrence gap above when choosing its thresholds (X11).

### Pending real runs

The L3 simulator job on a CPU worker; the first measured RegressionSet and ModelVersion scored with L2; see `deliveries/WP16.md`.

## Merged work package: WP18 gap decomposition, routing and loop disciplines

Code stage (D320): no real runs, no experiments. Branch `wp/WP18`; decisions D520–D530.

### Closes

| Item | How |
| --- | --- |
| R6, E3 | per-round GapDecomposition record (`T_meas − T_SOL = ε + δ + Δpre + L_rep + irreducible`, attribution order D-T6, SOL bounds L_rep) and the versioned routing table `orchestrator/routing.yaml` replacing judging rules 1–4 with class routes (rules 1–6, terminal `closed`) |
| E4, D-T13 | selection controls per solve: model points, random arm (exploration share) and copied assignment measured and recorded per bucket |
| E6, D-T17 (canaries) | planted faults per summarize batch, rotating kinds; a missed canary is an inbox item |
| D-T6, D-T11, D-T12 | attribution order; SOL as stopping criterion and L_rep bound; one mechanism per round (coupled groups once), hold-out with rollback, exploration budget |

02 sections implemented: §1 P1, P6, P7; §6.1 (case-loop round: residual against SOL → attribute → one change → re-verify, else roll back); §9 (improvement band, null rounds, accept, escalation). 12 §3 and §4 knobs now consumed: improvement band, mechanism attribution, null-round limit, one mechanism per round, exploration share, duplicate-proposal rejection, selection controls, canary rate, accept, escalate case → knowledge, escalate → user (canary).

### Behaviour that changes at merge

- Judging accepts by 12 §4 (`escalation.accept`), not by `judging.explained_min`; a case without a reference accepts only near SOL (D522).
- Of the 8 measured points per bucket, 2 are random feasible points and one more may be a copied assignment (D524).
- An identical resubmission after a rejection is rejected without checks (D529).
- New terminal solve state `closed`; the frontend counts it as finished.

### Owed

- WP19: `relaxed_pred_us` per bucket (model optimum with the plan's fixed variables opened) so Δpre is assessed. Not done here: level choice (`escalation.level_choice`), the template-assurance refusal, the budget reminder message to the agent.

## 2026-10-31: merged work package WP19 (FormulationPlan, chain enumerator, access discipline; WP17 projection compiler)

Code stage (D320): no real runs, no experiments. Branch `wp/WP19`; decisions D540–D550. WP17 is folded in (D328).

### Closes

| Item | How |
| --- | --- |
| R7, D-T18, D-T19 | `FormulationPlan` record (F1–F7, citations), the deterministic default as the same object, `diff_plans` / `score_plan` (`solve/plan.py`) |
| R8, D-T20 | versioned access-discipline table (`core/access_discipline.yaml`, rendered into `use-kta`), one-screen task header (`TaskCard.header`, TASK.md starts with it), Pull records per fetch, pull-log views (`GET /pulls`, Release page) |
| E12, D-T19 | bounded chain enumerator over the Template grammar (Γ, capability, relations, SOL bound, incumbent pruning, coverage counts); composition statistics as a view (`GET /composition`) |
| R1, R4, E8, D-T1 (WP17) | projection compiler (`solve/projection.py`): ResMII, RecMII, pair bounds, one-hot menus, epigraph max, menu products, table terms, integer ceil, outer enumeration; the agent writes none of it; the hand-formulation MIP stays as the comparison arm |
| D-T4 (WP17) | pending: the MIP ↔ L2 check waits for WP16 (hook `consistency_check`, D548) |

02 sections implemented: §5.1, §5.2 (compiler; the L2 check pending), §5.3, §5.4, §5.5; §6.2 bridges "FormulationPlan, pull log" (case loop out) and "pull logs, access-discipline versions" (system loop).

### Files changed outside code

- `docs/design/12-agent-controls.md` §7: five rows (projection table size, outer enumeration limit, chain enumerator depth and budget, task header lines).
- `spec/`: schemas.md §4.8.1 and TaskCard `header`; formulation-solve-models.md "Projection compiler", "FormulationPlan and chains"; orchestrator-cli.md "WP19"; api-web.md (`/pulls`, `/plans/{id}`, `/composition`, Release fields, pull logging); agents.md (TASK.md header); gate.md WP19 row.
- Worker skill `use-kta`: the header and the access-discipline table v1.

### Behaviour that changes at merge

- Every `kta show/card/why/query` of a worker writes a Pull record.
- TaskCard is schema 2 (optional `header`); old cards read unchanged.
- `use-kta` changed, so its skill hash changes: the next approved AgentProfile carries the new version.

## 2026-10-31: WP19 follow-up — the MIP ↔ L2 consistency check (WP17 part, after WP16)

Branch `wp/WP19-check` from `integration`; decision D551 (supersedes D548). Closes D-T4 (code stage): `solve/consistency.py` compares the compiled projection with WP16's L2 on sampled feasible points (Spearman ρ, optimistic-bound share) and names the deviating effect. Knob `model.mip_l2_check` (12 §7, set by X11; X11 `sets` now lists it). On the GEMM seed: ρ 0.52–0.54, no violation; a fixture with a wrong reservation table (4× tensor per mma) fails naming `mma.mma`.


## 2026-10-31: merged work package WP22 (system settings, agent proxy, frontend parity)

Branch `wp/WP22` from `integration`; decisions D580–D592. User requirements (2026-10-31): Codex and Claude Code always through the local Clash proxy with no direct fallback; pi and the DeepSeek harness never; a settings console with a system section; every round-4 mechanism visible in the frontend.

| Item | What |
| --- | --- |
| Proxy (D580–D586) | system configuration separate from the controls registry; detection scutil → environment → `http://127.0.0.1:7897`; injection of the eight proxy variables for `required` providers only, NO_PROXY with loopback and the API host; preflight on the agent host (port, CONNECT, TLS), fail closed with a Report, a breaker and an Inbox item; `AgentRun.proxy`; credentials only in secrets.env |
| Settings → System (D587–D589) | proxy with Test, providers (CLI version, logged in), workers (commit, GPU, Test job through the Executor), control plane, release stage, change history |
| Mechanisms page (D591) | learning-loop stages, FormulationPlan diffs, selection arms, canaries, L2 what-if |
| Open (D592) | agent CLIs are configured to run on spark, which cannot reach the Mac's Clash: the user decides where they run |

### Files changed outside code

- `spec/`: workplan.md WP22; agents.md (env list, WP22 section); api-web.md WP22; schemas.md `AgentRun.proxy`; gate.md WP22.
