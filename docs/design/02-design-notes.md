# 02 Design notes from the thinking-model discussions

**Status: staging area, not adopted specification.** This file holds the detailed material produced in the design conversations of 2026-10-30/31. It is organised by the numbered design document each part should be merged into; the merge is construction work (see `docs/changelog.md`). Until merged and confirmed, `spec/` is what is built and this file is what is proposed. The human-readable account of *why* is [01 Thinking model](01-thinking-model.md); tunable values are in [12 Agent controls](12-agent-controls.md).

Each decision referenced as `D-Tn` is in the ledger at the end (§12).

---

## 1. Givens, assumptions, derivation → merge into 00

### Givens

| | Given | Basis |
| --- | --- | --- |
| G1 | Kernel performance is how compute, movement and synchronisation transactions are laid out on finite resources over time | definition |
| G2 | Every resource has a measurable peak and capacity; `T ≥ max_u W_u^min / Peak_u` and `T ≥ critical path`, computable without knowing the best kernel | physics |
| G3 | Data movement dominates cost; the memory hierarchy is a ladder of known cost order | paper §2.2 |
| G4 | One monolithic MIP over all decisions is not solvable at useful scale | paper §1 |
| G5 | One refinement step as its own formulation solves fast (93-variable MILP in 0.2 s; 1536 points per shape enumerated) | paper App. A, Table 20 |
| G6 | Current cost models are badly wrong: zero-gap solutions predict 0.333 where simulation gives 0.79–0.90; 2.92× under-prediction at length 8192; a 1.84× interaction term on Ventus | paper App. E, §5.3; 00 §10 |
| G7 | Measurement is cheap, agent time is expensive (≈52 min agent vs ≈54 s GPU per construction run) | 00 §10 |
| G8 | Agent output alone is not trustworthy, but checkers catch its faults (31 faults, 3 deadlocks caught) | paper App. F |
| G9 | Local step proofs compose to whole-workload correctness, under stated conditions | paper Theorem 1 |
| G10 | SOTA kernels decompose into refinement steps and the steps repeat: 135 steps from 7 references, 73 % matched by pre-existing templates; cumulative new templates 9, 14, 22, 27, 30, 30, 30 | `deliveries/WP10a.md` |

### Assumptions

| | Assumption | Plain words |
| --- | --- | --- |
| A1 | Finite vocabulary: the set of refinement moves saturates | experts have a finite bag of tricks |
| A2 | Local coupling: steps interact through few variables | tricks compose like bricks |
| A3 | Rankability: a resource-based model can order candidates even when its numbers are off | the model can pick what is worth measuring |
| A4 | Reliable scope: a template's Γ predicts whether a move applies in a new context | reuse does not misfire |
| A5 | Attributability: the gap to the bound can be split into model, space, solver, preconditions | we know which part to repair |
| A6 | Agent sufficiency: told what is missing, the agent writes conforming templates and generators at acceptable cost | the agent can fill the gaps |

### Derivation (each component and what it depends on)

- **P1 The target is measurable** (G2): the SOL bound exists for any case; every result is reported as % of SOL and against the strongest reference; SOL gives a stopping criterion.
- **P2 One paradigm** (G1, G3): one vocabulary, a pattern table and a resource table, both open; the undescribable goes to `other` and is counted.
- **P3 Refine in steps** (G4, G5, G9): a template is one refinement step; a chain is one family's design.
- **P4 Template shape** (P3): three readers (solver: X, C, Ĵ; verifier: Pre, π, Σ; retrieval: Γ, refines, signature) bound by K1; granularity: one object, one obligation, at least one of the three structural questions answered.
- **P5 Formulation is assembled; the hardware model is separate** (A2, P4): hardware enters as walls (constraint right-hand sides) and rates (cost coefficients); overlap structure is declared by templates. Per-template hardware coefficients would prevent shared calibration.
- **P6 Solve ranks; measurement decides** (G6, G7, A3): shortlist → measure top k; rank correlation is the model's metric; a control measures solver choice vs copied assignment.
- **P7 Gap decomposition routes the next action** (P1, P6, A5): `T_meas − T_SOL = ε + δ + Δpre + L_rep + irreducible`; SOL gives `L_rep` a computable upper bound, turning paper Table 4 into a router.
- **P8 The population is the accumulation** (A1, A4): chains/families, templates and relations, hardware model, findings, gaps and `other` counts.
- **P9 The agent appears only where creativity is needed** (G8, P7).
- **P10 Absorbing references is the cheapest extension** (G10).

Guaranteed if A1–A6 hold: correctness to the assurance level reached; monotone improvement under a fixed model; explainability. Not guaranteed: global optimality, automatic termination, amortisation (the thesis).

---

## 2. Knowledge: definition, layers, objects → merge into 23 (and 21 §1)

**Definition.** Knowledge = claim + scope + evidence + status.

**Placement rule.** A claim is attached at the most general layer at which it still holds:

| Layer | Depends on | Example | New GPU generation | NPU |
| --- | --- | --- | --- | --- |
| semantic | the computation only | reduction is associative, hence splittable | unchanged | unchanged |
| mechanism | an abstract capability | multi-buffer rotation needs a managed scratchpad plus asynchronous copy | mostly unchanged | holds where the capability exists |
| realisation | ISA or backend | TMA on Hopper, cp.async on Ampere, DataCopy in Ascend C | new realisation | all new |
| instance | one chip | DRAM peak, smem size, bank rule | re-measure | rebuild |
| finding | chip × workload | "on GB10, batch-1 decode needs split ≥ k to saturate DRAM" | demoted to prior | mostly void |

**Objects.** Pattern (semantic vocabulary) ← Template (mechanism; `requires` abstract capabilities) ← Realisation (template × backend × ISA range: fragment, ISA constraints, expected SASS); Capability (declared per Target with parameters; replaces HwFeature, D-T8); HwModel (instance, §4) with HwFact as raw measurement; Finding (unifies Regime and Negative: claim, kind, scope, evidence, status, `explained_by`, D-T9); Chain → Family → Solution.

**Promotion and demotion.** Raw evidence is immutable. A finding the hardware model can explain, surviving a hold-out check, is promoted into the model; one that holds across targets is promoted into a template's Γ or a structure's regime; repeated `other` becomes a new pattern or template. Counter-examples demote, and invalidation propagates along dependency edges (23 §4).

**Unprobed mechanisms are recorded** (D-T27). A template whose required capability no current target provides (FA3's cluster mechanisms, TMA multicast, TMEM) still enters the pool from analysis-only decomposition: assurance `declared`, `probed_on: []`, with its source and the capability it needs. Retrieval filters it out on targets lacking the capability, coverage counts it under "known, unavailable here", and it becomes searchable the day a target declares the capability. Nothing learnt on one chip is searched for again on the next.

**Two orthogonal axes.** Level (how fine; derived from the refined pattern; numbering 1–7 per the research frontend, D-T7) and dependence degree (how far it transfers). Levels do not fix decision order.

**Status ladders (open, D-open-1).** Today there are three: knowledge status (21 §1.1: proposed → reproduced → calibrated → hold-out validated), template assurance (22 §6: declared → tested → bounded → proved), and the finding ladder proposed here (observed → replicated → explained → invalidated). The user must choose one scheme; the proposal is: templates keep assurance; every empirical item (finding, model term, chain performance) uses one status ladder with evidence rules per step.

---

## 3. Template: three faces and `effects` → merge into 22 §2

A template is the formal version of one trick: a parameterised program transformation with the conditions under which it is safe, how it changes cost, and where it applies. Not a kernel, not a family, not a code template; no hardware numbers; no performance conclusion.

| Face | Answers | Fields | Reader |
| --- | --- | --- | --- |
| transformation | which coarse transaction becomes which finer ones; new object; variables; abstract program | refines, produces, O, X, P | solver, generator |
| contract | when it is safe and why | C, Pre (K1), π, Σ, Γ | verifier, retrieval |
| effect | what it does to cost, in the transaction vocabulary only | `effects`: transactions, storage, overlap | hardware model, solver |

**Why `cost` becomes `effects` (D-T2).** `multi_buffer_rotation@1` writes `{unit: dram, work: "size*iters if P == 0 else 0", overlap: serial}` to express "with P = 0 load and compute do not overlap": an overlap statement forced into a per-unit work field. With `effects` the template declares transactions added or changed, storage occupied at which level, and which operations overlap with what in-flight depth; the hardware model converts these to time. Side effect: the seed chain's missing contraction/operand-traffic cost (WP5 known issue) disappears, because whole-kernel transactions are declared once and costed by the model.

Sketch:

```yaml
effects:
  transactions:
    - {pattern: sync, kind: cta_barrier, count: "p.iterations * (1 + beta)"}
  storage:
    - {level: "p.storage_level", bytes: "S * p.size_bytes"}
  overlap:
    - {between: ["load(gen p+P)", "consume(gen p)"], when: "P >= 1", inflight: "P"}
```

Granularity checks that can be computed from chains: a template that always co-occurs with another with jointly bound variables should merge; one used once may be over-specialised (12 of 35 today).

---

## 4. Hardware performance model → new document 25 (proposed) and 50 §7

**One declarative source** (D-T3). The model is structured data; the simulator, the MIP and the profiling report are derived from it. This replaces "the agent reads a Python simulator and writes a MIP".

| Part | Answers | Source | Trust |
| --- | --- | --- | --- |
| resources | units; kind (throughput / capacity / latency+concurrency); scope (chip / SM / SMSP / CTA); peak, latency, max in-flight; ncu counters | microbenchmarks | high |
| walls | capacity and legality limits with their cliff (no launch, spill, lower occupancy) | docs + probes | high |
| operation classes | per transaction pattern (and per SASS instruction class): a reservation table of resources, amounts, cycles, latency; `serial` flags | docs, SASS, microbenchmarks | medium-high |
| efficiency | discount as a function of few features (granularity, in-flight, MMA shape, occupancy) with fitting data and hold-out error | calibration | medium |
| interactions | who interferes with whom (banks, issue ports, L2 partitions, a shared adder tree), with `applies_when` and evidence | residual-triggered, measured | in scope only |
| composition | how per-resource times combine; skeleton from templates' overlap declarations | rules | medium |

**Fine-grained sharing** (D-T15): a shared structure (e.g. one adder tree behind FP8 and FP16 MMA) is a resource that appears in every sharing operation's reservation table, possibly with different cycle counts; mutual exclusion is a capacity-1 resource; ports and queues are resources with depth. Unknown sharing is discovered by pairwise interference microbenchmarks (A alone, B alone, A+B interleaved); the interference matrix over instruction classes is a maintained artefact.

**Completeness as measured coverage** (D-T14). The SASS stream is the hardware's contract, so a SASS-level model can in principle express every timing-relevant behaviour; but the ISA alone is incomplete: memory-hierarchy state and scheduling policies must be modelled as resources and state. "Complete" means: every instruction class in a kernel's SASS has a reservation table; every pair is measured or declared independent; every memory level and policy that measurably affects time has a row. Unmodelled resources default to infinite capacity, unmodelled effects to zero: additions are monotone, versioned tightenings. Coverage statistics belong to the complete experiments; day-to-day work stays forward and "enough for now".

**Four fidelity levels, one parameter table** (D-T5):

| | Form | Input | Output | Use | Speed |
| --- | --- | --- | --- | --- | --- |
| L1 analytic | steady-state projection (§5) | template effects, variables | predicted time | inside the MIP | µs |
| L2 transaction-level simulation | per-CTA transaction graph on resources | effects expanded | per-unit occupancy timeline, stall and idle attribution | pre-compile bottleneck analysis; virtual wall moves | ms |
| L3 SASS-level simulation | the existing GB10 simulator (`kernel-lab/exports/gb10-simulator-calibration-20261020-v2`) | compiled binary | more accurate time, SM vs bus terms | post-compile reranking, diagnosis | s |
| L4 measurement | GPU timing, ncu counters, in-kernel timestamps | real run | ground truth | final decision, calibration | s |

L1 is the steady-state approximation of L2; L2 is the coarsening of L3 (one transaction's cost equals its instruction group's). Disagreement between adjacent levels is a diagnostic.

**Maintenance as a software artefact.** A model version = parameter table (each value pointing to its microbenchmark and HwFact) + simulator commit + regression results; immutable. Every measured kernel becomes a regression case; several configurations per kernel are required because what the solver relies on is ranking. Iteration: largest-error case → counters per resource → isolating microbenchmark → change labelled "adapter fix" or "machine-model change" → hold-out regression → admit or roll back. Views: regression scoreboard; per-kernel timeline (predicted over measured); mechanism coverage table; change log. Done when the coverage table stops growing and hold-out rank correlation is stable.

**Sustained saturation and walls.** A kernel is well optimised when a bottleneck unit b has `Busy_b ≈ SOL_b` (no inflated work) and `T ≈ Busy_b` (busy throughout). Idle intervals of the bottleneck are attributed: waiting for another unit (→ overlap templates), for data (→ pipeline depth), for synchronisation, prologue/epilogue/tail, or blocked by a wall. Moving a wall in L2 and reading the shortened idle time is that wall's shadow price: on a fixed chip it separates scheduling problems from hardware limits; in co-design it is what wall relaxation buys. For latency-bound workloads: no idle wait on the critical path.

**Requirements** R1 single declarative source; R2 restricted expression language plus table terms; R3 data-driven additions with conservative defaults and versions; R4 transaction and SASS granularity sharing parameters with an aggregation relation; R5 outputs total cycles, per-unit busy time, occupancy timeline, stall reasons, idle attribution; R6 metrics named as ncu names them; R7 what-if runs; R8 calibration interface and regression set; R9 determinism and versioning; R10 speed budgets.

---

## 5. Steady-state projection, MIP export, formulation → merge into 22 §3.3, 10 §4.4; new 25

### 5.1 The projection

A simulator cannot be exported to a MIP; a well-defined relaxation of it can. Modulo scheduling (Rau and Glaeser 1981, Lam 1988, Rau 1994): a loop body of operations with reservation tables and dependencies carrying latency and distance; a new iteration starts every II cycles.

```
ResMII = max_r  ceil( Σ_ops use_r(op) / cap_r )              resource bound
RecMII = max_cycles  ceil( Σ_edges lat / Σ_edges distance )   recurrence bound
MII    = max(ResMII, RecMII);   any feasible schedule has II ≥ MII
T      ≈ L_body + (trips − 1) · II   + prologue/epilogue/tail
```

Mapping to a GPU kernel: the loop is the CTA's main loop; resources are scoped (issue per SMSP, tensor and smem bandwidth per SM, DRAM and L2 per chip); concurrency multiplies capacity (`c` CTAs per SM, `n_SM` SMs), so occupancy enters the resource bound; latency hiding is the recurrence bound (`load → compute → slot reuse → load`, latency `lat_dram + lat_mma`, distance S, hence `II ≥ (lat_dram + lat_mma)/S`), which is where S and P enter; warp-level ILP is folded into operation latencies and throughputs.

Nested bounds: SOL = ResMII with minimal work; projection = `max(ResMII(W_schedule), RecMII)`; simulator adds contention and non-steady phases; hardware adds the unmodelled. Each adjacent gap has a name: work inflation; latency exposure; contention and tails; model error.

What the projection drops (= simplification error): unmodelled arbitration and contention; cache state beyond tables; scheduler policy and tails; prologue/epilogue and irregular control flow; whether ptxas achieves MII; register pressure (a wall). Tight for loop-dominated kernels, loose for latency-dominated ones (there the MIP only prunes; ranking passes to L2).

### 5.2 MIP form and guarantee

One-hot menus make single-variable functions linear; `II ≥ Σ_r use_r(x)/cap_r` is linear; `II ≥ lat/S` becomes `II ≥ Σ_s z_s · lat/s`; `max` is an epigraph; `ceil` adds an integer; products of two menu variables use one-hot products or big-M; non-algebraic effects are table terms `table(f; x1, x2)` precomputed over a few menu variables; what fits neither moves to outer enumeration. The compiler applies these rules; the agent writes none of it.

Guarantee: translation correctness is the compiler's; simplification error is checked without a GPU by sampling the subspace, evaluating MIP and L2, requiring rank correlation above a threshold and the optimistic bound (MIP ≤ simulator) to hold on nearly all points; failure names the effect with the largest deviation. (D-T1, D-T4)

### 5.3 What the agent still decides (FormulationPlan)

| Decision | Deterministic default | What a good choice needs |
| --- | --- | --- |
| F1 chain | nearest retrieved chain, or the chain enumerator's best-bound chain | which transactions dominate, which moves address them |
| F2 bindings | type- and domain-compatible merges by name; nothing fixed | which couplings matter |
| F3 open / fixed / outer | all open; non-linearisable outer | low effective dimension; findings that justify fixing |
| F4 menus | the template's declared domain | concavity knees, walls |
| F5 table terms | compiler: every non-algebraic effect | near-decomposability |
| F6 glue, dispatch | none; one bucket | the shape distribution |
| F7 generator | none | realisations, neighbouring generators |

The product is a structured **FormulationPlan** (D-T18): chain, bindings, each variable's status with the structure it invokes (paper Table 1) and the evidence it cites, menus, table terms, glue. The deterministic default is the same object, so plans are diffable and scored (D-T19).

### 5.4 Access discipline (D-T20)

A one-screen header (case; bound; last round's gap decomposition; default plan and score; budget) plus a versioned decision table in the worker skill:

| Situation | Fetch | Decide |
| --- | --- | --- |
| start | nearest chain with each step's Γ evaluated | most steps hold → start from it, fetch alternatives for failing steps; else diagnostic retrieval |
| diagnostic retrieval | bottleneck unit → patterns → templates ranked by evidence (top N cards) | moves that address the dominant transactions |
| each open variable | findings that pin it; the wall bounding it | fix with citation, or keep the menu |
| MIP too large | structure table, regimes for this workload/chip | move outward or decompose, citing the structure |
| non-algebraic effect | tabulated effects for the operations involved | use the table or request one over ≤ 2 variables |
| last round was ε | nothing from the space | hand to model diagnosis |
| last round was L_rep with an idle resource | templates reducing or overlapping that resource; references with such a step | open a dimension or add a step |
| bottleneck blocked by a wall | shadow price; templates reducing the walled resource | change the move or record a co-design finding |
| stop | — | every non-default decision cites evidence, or budget reached |

Pulls outside the table are allowed and logged; the table is revised from the logs.

### 5.5 Chain enumeration (D-T19, E12)

The population is a grammar; a bounded enumerator expands legal chains, filters by Γ and capabilities, bounds each chain by SOL under its minimal work, prunes against the incumbent. It is the deterministic baseline for F1, exposes missing rules, and yields coverage counts. Composition statistics (co-occurrence, order, conditional frequency "bottleneck X at level Y → move Z") are a *derived view* over reconstructed chains, never authored.

---

## 6. Loops, bridges, staged release, independence → merge into 10 §2, 11 §1

### 6.1 Four nested loops

| Loop | Scale | Changes | Must not change | Driven by | Monotone quantity |
| --- | --- | --- | --- | --- | --- |
| solve | s–min | the assignment | formulation, model | predicted ranking, compile feedback | best measured time |
| case | min–h | subspace, model parameters | admitted pool knowledge | gap decomposition | unexplained gap |
| knowledge | days | templates, model structure, findings | system code, checkers | recurring residuals, reference gaps | new templates per reference; hold-out error |
| system | weeks | process, checks, routing | claims, acceptance criteria | loss statistics | share per loss class |

Every round: predict → measure → residual per resource and against SOL → attribute → one change → re-verify on hold-out, else roll back. Attribution order: correctness → ε → δ → Δpre → L_rep → irreducible (D-T6). Disciplines: one mechanism per round (coupled groups excepted); hold-out with rollback; an exploration budget (D-T12).

### 6.2 Bridges

```
design loop (user + assistant)   bridges: 01, 02, ledger, claims ledger, deliveries + evidence, canaries, surprises
system loop (per batch)          in: loss statistics, pull logs, regression scoreboard, coverage tables
                                 out: checks, routing rules, access-discipline versions, model versions
knowledge loop (across cases)    in: Gap, Finding, recurring residuals, reference list
  learning / calibration /       out: Template versions, model versions, promoted findings,
  population                          invalidations → waiting cases resume
case loop (per case)             in: Case, pool snapshot, model version
                                 out: Prediction+Measurement, gap decomposition, Gap, Finding, Solution,
                                      FormulationPlan, pull log
solve loop (per formulation)     in: Formulation      out: Candidates with Prediction, MIP↔L2 check
```

Downward bridges are versions; upward bridges are evidence records. A case never changes the pool directly; a knowledge change never edits a case's history; a process change never alters a claim without the user.

### 6.3 Staged release (D-T22)

| Stage | Runs | Agent calls | Exit |
| --- | --- | --- | --- |
| S0 deterministic | golden path (GEMM on GB10), no agent | none | numbers match references; every record inspectable |
| S1 hand-stepped | one task type on one case; the user reads card, pull log and plan before each step | one, gated | the user can predict the agent from card and discipline |
| S2 one case loop | a handful of cases | automatic within the loop | router correct on injected faults; no surprises in pull logs |
| S3 knowledge loop | learning and calibration on a batch | automatic | promotions and invalidations behave as §2 says |
| S4 open | all loops, protocol | automatic | — |

### 6.4 Independence and context (D-T24)

Solving, learning (downloading libraries, decomposing, extracting) and calibrating are independent processes communicating only through records and the queue (11 §1); calibration is organisationally a branch of the knowledge loop. A case worker may offer knowledge in passing; it goes through the same admission. One worker role, 2–4 instances, no sub-agent trees. Sessions resume within a work unit, never across; `NOTES.md` is the state; fresh session on context threshold, same failure twice, unit end, error (10 §4.2). Resume vs fresh is an experiment (O11).

---

## 7. Verification → merge into 50 §4

Five legality properties with their checks: semantic correctness (template proofs once; numerical differential tests per kernel with edge shapes and adversarial inputs); memory safety (coverage / exact-once / lifetime obligations plus compute-sanitizer); termination (progress on bounded models plus timeouts); resource legality (walls, realisation ISA constraints, compile feedback); measurement integrity (immutable inputs, seeds, CUDA Graph checks). Once per template (K1, K2) and once per candidate (ms). Not covered: FP rounding, schedules beyond the bounded model, and whether an agent-written generator implements the abstract program; for the last: code markers, expected SASS, sanitizers, later IR-level extraction of the transaction and synchronisation structure. Phase one claims `tested`, not `proved`.

---

## 8. Learning: what sources become, tiers, reproduction, ablation, backends → merge into 21, 24

### 8.1 Coverage of SOTA in four levels

| Level | Meaning | Measured by | Today |
| --- | --- | --- | --- |
| 1 expressible | the SOTA kernel is a point of our space (chain + assignment) | reconstruction match rate | 73 % (WP10a) |
| 2 reproducible | generating from that chain reproduces its performance within ε | reproduction error | not done |
| 3 reachable | given only the case, the system finds that chain or an equally good one | blind reconstruction (O12) | no experiment yet |
| 4 surpassed | the blind result beats the reference under the same contract | same | — |

Level 1 does not imply level 3: with 35 templates and 20-step chains the composition space is astronomical and the solver chooses assignments, not chains. What makes SOTA reachable is composition knowledge, which is exactly what reference chains supply.

### 8.2 What a source becomes

| Product | From | Frequency |
| --- | --- | --- |
| Template (a new move) | papers (mechanism with reasoning), occasionally code | rare; saturating |
| Chain with bindings (a composition) | code | one per reference; **the bulk of accumulation** |
| Finding / Regime (why it is fast, where it holds, where it fails) | paper arguments; our ablation and measurement | several per reference |
| Realisation (how the move is written on a backend/ISA) | code | one per move per backend |
| Baseline (its number under our protocol) | running it | one per runnable reference |

Papers give mechanisms and reasons (entered as *claimed*); code gives compositions and realisations. **System-level features** (host planners, merge kernels, page tables, PDL, dispatch) are not scheduling moves: counted separately so coverage does not lie.

### 8.3 Source classes and what to do with each

| Source | Examples | Expected yield | Processing |
| --- | --- | --- | --- |
| open SOTA code | CUTLASS, FlashAttention, FlashInfer, Marlin, vLLM/SGLang kernels, DeepGEMM, ThunderKittens | chains, realisations, baselines; sometimes templates | full pipeline |
| closed libraries | cuBLAS, cuDNN | baselines; instruction families visible in SASS | measure + SASS census only |
| papers | arXiv kernel-optimisation work | candidate templates + claimed regimes | enter as claimed; queue for reproduction; no code → does not enter search |
| vendor documents | PTX ISA, programming guides, GTC material | capability table, wall values, existence of mechanisms | fill capabilities and the hardware model |

Priority of references (value = cases affected × gap closable × our uncertainty): strongest baseline for an in-scope case; SASS uses instruction families we cannot generate; chain has steps no family has reproduced; source class whose discovery curve still rises. Efficiency recorded: agent-hours per admitted template and per reproduced chain; GPU-seconds per finding; reuse count within N cases.

### 8.4 Cost tiers

| Tier | Work | Cost | Yield | Who enters |
| --- | --- | --- | --- | --- |
| L0 analysis | chain and gaps from source (WP10a) | agent hours, no GPU | chain, gaps, candidate templates | every selected reference |
| L1 measure | run under our protocol; SASS instruction census | GPU minutes, deterministic | baseline, instruction coverage | everything runnable on the target |
| L2 reproduce | our family reproduces its *performance core* | the expensive step: generator work | a Family that solves our cases | chain has a step no family reproduced, **or** it is a case's strongest baseline |
| L3 ablate | toggle steps/variables in our family | GPU seconds × toggles, automatic | effective steps, coupling groups, negatives, calibration data | after L2 |
| L4 blind | case only, full loop | expensive | reachability/surpass evidence | evaluation set only |

**Performance core.** One configuration (dtype, code path), the hot loop and its pipeline; host planners replaced by fixed inputs; system features excluded by declaration; correctness contract from the reference's own tolerance. Acceptance: within ε (start 10 %) on the reference's own configuration. Not a drop-in replacement.

**Ablation on our family, not theirs.** Coupling detection on suspected pairs only (shared family variables, related templates, co-occurring resources), fractional-factorial screening when k is large; criterion: joint effect deviates from the product of single effects beyond the noise band. Coupling is first-class knowledge: a `coupled_with` relation with evidence and scope, consumed by the formulation (open coupled variables together) and the hardware model (interaction candidates). (D-open-3)

### 8.5 Backends as realisations, chosen by measurement (D-T29)

A backend's *expressibility coverage* = steps of reconstructed chains it can realise with code markers ÷ all steps; plus *intent fidelity* = compiled SASS shows the template's expected traces. Route: TileLang primary (GB10 evidence; levels 3–4 natural; its compiler makes level-5 decisions implicitly, so intent-vs-SASS checks matter), CUDA C++ with inline PTX as the escape hatch for level-5 steps, CuTe DSL of limited value on GB10 and evaluated only if a target needs it, Triton for prototypes and controls, MLIR not in phase one. Per-backend coverage is recorded from the start so the route can change on data.

### 8.6 Automation

The learning loop is automated; WP10a's hand decomposition was the S1-style check of the knowledge model, not the mechanism. Pipeline per reference: intake (deterministic: clone at commit, license, hash, container build, run + SASS + profile) → decompose (agent, T2, `decompose-kernel`; the reference directory format is the worker's deliverable) → admit (deterministic; errors go back to the same session, ≤ 2 retries; warnings must be explained) → reproduce (deterministic; gaps → T3 extract, T1 extend) → ablate (deterministic jobs) → cover (coverage, snapshot, waiting cases resume). The 32 current import errors are what the admit step would return to an agent; the string-comparison class is an *expressibility gap of the language*, to be counted and admitted at `proposed`, not refused (D-open-2). `knowledge/seed/` holds seed only; worker deliverables live in the store (D-open-4).

---

## 9. Thresholds → in 12

See [12 Agent controls](12-agent-controls.md) for the full table. The loop-shaping rules: improvement = beyond the noise band **and** attributed to a named mechanism; two null rounds close a case; accept = not worse than the strongest reference and unexplained gap < 10 % of T, or gap fully explained as irreducible; escalation rules; knowledge reliability ladder; level choice by comparing graph-level and kernel-level bounds (ratio 0.8); go one level finer when the bottleneck is saturated or the loss sits in a transaction the current level cannot express.

---

## 10. Obligations and engineering gaps

Obligations O1–O12 are kept in [01 §6](01-thinking-model.md). Engineering gaps: E1 hardware model as first-class declarative object; E2 `cost` → `effects`; E3 gap-decomposition engine with data-driven routing; E4 selection controls; E5 population governance; E6 canaries; E7 level numbering; E8 steady-state projection compiler and MIP↔simulator check; E9 transaction-level simulator; E10 SASS simulator onto the declarative table with timeline and ncu-named outputs; E11 regression set with several configurations per kernel; E12 chain enumerator and FormulationPlan schema; E13 learning-loop runner (intake, T2/T3/T5 runners, rejection loop, reproduce and ablate jobs); E14 pool-diff and regression-scoreboard views; E15 `coupled_with` relation and unprobed-mechanism status.

---

## 11. System-parameter experiments (D-T28)

Parameters that govern the system are set by small experiments, not by fiat; until an experiment has run, a value is a default and says so. Each row of 12 names the experiment that sets it.

| ID | Question | Design | Sets |
| --- | --- | --- | --- |
| X1 | which model per task type | strong vs weak model on the same T1/T2/T3 tasks; admitted per agent-hour, rejection rate | 12 §1 budgets per type; AgentProfile model |
| X2 | reasoning effort | two effort levels on the same tasks | AgentProfile |
| X3 | resume vs fresh | O11 | 12 §2 resume scope |
| X4 | access discipline vs dump vs free browsing | O10 | 12 §2 discipline version |
| X5 | retries | 2 vs 3 on rejected tasks; marginal acceptance | 12 §1 retries |
| X6 | improvement band | measured noise of the protocol per kernel class | 12 §3 band |
| X7 | L2 reproduction ε | distribution of reproduction error over the 7 references | 12 §4 / 02 §8.4 |
| X8 | measured points k and exploration share | rank correlation and hit rate vs k, share | 12 §1, §3 |
| X9 | task wall-clock budgets | completion-time distribution per task type from S1/S2 | 12 §1 |
| X10 | hardware-model instance values | GB10 peak and pairwise interference microbenchmarks (A, B, A+B) per instruction class (D-T15) | 02 §4 resources, interactions |
| X11 | MIP ↔ L2 acceptance thresholds | sampled subspace of the GEMM family: distribution of rank correlation and of optimistic-bound violations | WP17 check thresholds |
| X12 | concurrent agent instances | 2 vs 4 instances on one batch: throughput, GPU queue wait, rate-limit hits | 12 §1 agent instances |

**Order (cheap and decisive first; added 2026-10-31).** Each step needs only what the previous ones settled.

1. *No agent, GPU minutes:* X6 noise band (every later comparison uses it) → X10 peaks and interference → S0 golden path → X8 measured points and exploration share, and O3 rank correlation from the same data.
2. *No GPU:* X11 MIP ↔ L2 thresholds on the GEMM family; O5 fault-injection routing check.
3. *References, mostly deterministic:* L1 measure + SASS census on the seven references → X7 reproduction ε from L2 attempts.
4. *Small agent spend (S1/S2), one task type at a time:* X1 model per task type (include a cheap tier: some task types may not need the strongest model) → X2 effort → X5 retries → X3/O11 resume vs fresh (also tests the 60 % context threshold) → X4/O10 access discipline vs dump vs free browsing (also header size) → X12 instances.
5. *Read from S1/S2 logs, no dedicated runs:* X9 task budgets; rounds per case, null-round limit, escalation thresholds, model-term provisionality, level-choice ratio. These are revised from logs, not experimented on.
6. *Claims:* O12 blind reconstruction, O7 amortisation (WP12); O1, O9 saturation and granularity as references accumulate.

---

## 12. Decision ledger

Every entry was chosen by the assistant unless marked confirmed. `default — needs user review` until confirmed. Confirmed entries move to `docs/decisions.md`; rejected ones are deleted with a one-line note.

| ID | Decision | Alternative(s) | Why | Change cost | Status |
| --- | --- | --- | --- | --- | --- |
| D-T1 | MIP assembled mechanically from chain + hardware model; the agent does not write MIPs per case | agent writes the formulation per case (paper's GB10 practice; collaborator's view) | checkable translation; shared coefficients; same data serves co-design | high once E8 exists; keep an agent-written arm for comparison | default |
| D-T2 | Hardware coefficients only in the hardware model; template `cost` → `effects` | keep per-unit `CostTerm` | one calibration benefits all templates | medium: schema, WP4/WP5, seeds | default |
| D-T3 | Hardware model is declarative data with a restricted expression language and table terms; simulator, MIP and reports derive from it | free Python read by the agent | structural risk control; co-design operability | high: E8–E10 | default |
| D-T4 | MIP uses the steady-state projection; non-steady kernels rank by L2 | enumerate + simulate only | theorem-backed relation to the simulator | low | default |
| D-T5 | Four fidelity levels sharing one parameter table; L2 added | L1 and L3 only | pre-compile timelines, virtual wall moves | medium: E9 | default |
| D-T6 | Attribution order with δ and Δpre added | current rules | later classes unjudgeable before earlier ones | low | default |
| D-T7 | Level numbering 1–7 (research frontend) | paper's 0–5 as in 22 | one convention across repositories | low | default |
| D-T8 | Capability replaces HwFeature; Realisation split from Template | keep HwFeature; fragments inside Template | transfer across generations and NPUs | medium | default |
| D-T9 | Regime and Negative merge into Finding with `explained_by` | two objects | one promotion mechanism | low | default |
| D-T10 | HwModel split into architecture class and chip instance | flat per target | structure reuse | low | default |
| D-T11 | SOL as stopping criterion and bound on `L_rep`; per-unit SOL, busy time, total reported separately | accept on reference only | P1, P7 | low | default |
| D-T12 | One mechanism per round; hold-out verification with rollback; exploration budget | free-form iteration | attribution needs it | low | default |
| D-T13 | Selection controls on every solve | none | paper could not isolate solver gain | low | default |
| D-T14 | Hardware-model completeness is measured coverage, never asserted | assert per architecture | principle 1 of AGENTS.md | low | default |
| D-T15 | Fine-grained sharing as shared resources in reservation tables, discovered by interference tests | ad-hoc terms | standard theory; discoverable | low | default |
| D-T16 | Documents in English | Chinese | repository rule; paper material | trivial | **confirmed 2026-10-31** |
| D-T17 | Canaries, claims ledger, golden path, contract-only review, review-queue cap | milestone acceptance only | queue of 37 + 37 unreviewable | low | default |
| D-T18 | FormulationPlan as a structured object | free prose / hand MIP | reviewable, diffable, attributable | low | default |
| D-T19 | Every agent formulation decision has a deterministic default and is scored against it | trust the plan | runs without the agent; agent value measurable | medium: E12 | default |
| D-T20 | One-screen header + versioned access discipline; pulls logged | full dump / free browsing | finite context | low | default |
| D-T21 | Design loop is the outermost loop; 01/02/ledger/deliveries/canaries are its bridges | design finished before running | nobody foresees everything | none | default |
| D-T22 | Staged release S0–S4 | open once gate passes | user must predict the system first | low | **confirmed in substance 2026-10-31** |
| D-T23 | Thresholds of 12 as first defaults | agent judgement | agents optimise for passing checks | low | default; values set by X-experiments |
| D-T24 | Session policy as 10 §4.2; calibration a branch of knowledge; one role | always fresh; separate calibration agent | tokens vs pollution; simplicity | low; O11 | default |
| D-T25 | All controls in one tunable table (12) with the failure each prevents; settings page later | scattered | one place to tune | low | **confirmed in substance** |
| D-T26 | `design-interrogation` skill with narrow triggers and three scales | rely on the user asking | defaults visible unasked | low | default |
| D-T27 | Unprobed mechanisms are recorded (`probed_on: []`), filtered by capability, counted as "known, unavailable here" | drop what cannot run here | knowledge must outlive the chip | low: one field, one filter | default (user's requirement in substance) |
| D-T28 | System parameters set by small experiments X1–X9; defaults say so until then | fix by discussion | hidden errors and inefficiency otherwise | low | **confirmed in substance** |
| D-T29 | Backends: TileLang primary, CUDA C++ + PTX escape hatch, CuTe DSL only if a target needs it, Triton for controls, MLIR not in phase one; per-backend expressibility coverage recorded | fix one backend | realisation-level choice, measurable | low | **confirmed 2026-10-31** |
| D-T30 | This file (02) is a staging area organised by merge target; merging into 10/11/21/22/23/50 and a new 25 is construction work | edit the numbered documents directly now | keeps 01 short and readable; lets the user review before documents change | low | default |

Open decisions waiting for the user (not defaults):

| ID | Question |
| --- | --- |
| D-open-1 | One status ladder for empirical knowledge (and templates keep assurance)? |
| D-open-2 | Admission of the 7 reference decompositions: build the rejection loop and count language gaps as `proposed`, rather than fixing YAML by hand? |
| D-open-3 | Ablation on our family; `coupled_with` as a first-class relation? |
| D-open-4 | `knowledge/seed/` = seed only; worker deliverables in the store? |
| D-open-5 | D-T2 and D-T3 before WP7, or after the golden path runs on the current shapes? |
| D-open-6 | Learning scope v1 (01 §7) and L2 admission with ε = 10 %? |
| D-open-7 | Blind reconstruction (O12) as the acceptance experiment of the learning loop? |
