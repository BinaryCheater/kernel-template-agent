# 01 Thinking model: why KTA is built this way

**Status: discussion record, not adopted specification.** This document collects the thinking model worked out in the design conversations of 2026-10-30 and 2026-10-31. It states the claims, the givens and assumptions, the derivation of each component, and, in §13, every decision that was taken *by default* during those conversations. Nothing here overrides `spec/` until a decision in §13 is confirmed and the corresponding spec is changed. Where this document and `spec/` disagree, `spec/` is what is built today and this document is what is proposed.

The paper this project grows out of is `docs/references/2610.39380-learning-to-formulate.pdf` (Formulaic). KTA keeps its template tuple, its refinement levels and its division of work (agent proposes, solver optimises, verifier and measurement check), and does not have to match it in every detail.

## 1. The claim

Instead of spending a strong model's tokens on every kernel from scratch, accumulate what the model learns into three kinds of knowledge that can be retrieved, composed and checked: a *space* of design moves (templates), a *model* of what each move costs on a given chip (the hardware performance model), and *findings* about where moves pay off. A new case is then mostly retrieval, solving and a few measurements; the agent is called only to extend the space or the model, and every such extension is triggered by a measured residual. The formulation that the solver sees for one case is derived mechanically from these three; nobody writes it by hand.

What this should buy: (a) *amortisation*: cost per new case falls as the pool grows, and a weak model with the pool approaches a strong model without it; (b) *coverage*: SOTA implementations can be expressed as points of the space and reproduced; (c) *diagnosability*: every shortfall is attributed to one component.

## 2. Givens and assumptions

Givens (facts or evidence):

| | Given | Basis |
| --- | --- | --- |
| G1 | Kernel performance is how compute, movement and synchronisation transactions are laid out on finite resources over time | definition |
| G2 | Every resource has a measurable peak and capacity; hence `T ≥ max_u W_u^min / Peak_u` and `T ≥ critical path`, computable without knowing the best kernel | physics |
| G3 | Data movement dominates cost and the memory hierarchy is a ladder of known cost order | paper §2.2 |
| G4 | One monolithic MIP over all decisions is not solvable at useful scale | paper §1 |
| G5 | One refinement step written as its own formulation solves fast (93-variable MILP in 0.2 s; 1536 points per shape enumerated) | paper App. A, Table 20 |
| G6 | Current cost models are badly wrong: zero-gap solutions predict 0.333 where simulation gives 0.79–0.90; 2.92× under-prediction at length 8192; a 1.84× interaction term on Ventus | paper App. E, §5.3; 00 §10 |
| G7 | Measurement is cheap, agent time is expensive (≈52 min agent vs ≈54 s GPU per construction run) | 00 §10 |
| G8 | Agent output alone is not trustworthy, but checkers catch its faults (31 faults, 3 deadlocks caught) | paper App. F |
| G9 | Local step proofs compose to whole-workload correctness, under stated conditions | paper Theorem 1 |
| G10 | SOTA kernels decompose into refinement steps and the steps repeat: 135 steps from 7 references, 73 % matched by pre-existing templates; cumulative new templates 9, 14, 22, 27, 30, 30, 30 | `deliveries/WP10a.md` |

Assumptions (the system depends on them; none is proved yet):

| | Assumption | Plain words |
| --- | --- | --- |
| A1 | Finite vocabulary: the set of refinement moves saturates | experts have a finite bag of tricks |
| A2 | Local coupling: steps interact through few variables; swapping one step changes little else | tricks compose like bricks |
| A3 | Rankability: a resource-based model can order candidates even when its absolute numbers are off | the model can pick what is worth measuring |
| A4 | Reliable scope: a template's Γ correctly predicts whether a move applies in a new context | reuse does not misfire |
| A5 | Attributability: the gap between measurement and bound can be split correctly into model error, space, solver, over-strong preconditions | we know which part to repair |
| A6 | Agent sufficiency: told what is missing, the agent writes contract-conforming templates and generators at acceptable cost | the agent can fill the gaps |

Amortisation is the thesis to prove, not an assumption.

## 3. Derivation

Each proposition names what it depends on; if an assumption falls, the components that depend on it must change.

- **P1 The target is measurable** (G2). The speed-of-light (SOL) bound exists for any case, so every result is reported as % of SOL and against the strongest reference, and SOL gives a stopping criterion. *Component: bound computation.*
- **P2 One paradigm** (G1, G3). Every kernel is transactions scheduled on resources. One vocabulary suffices: a pattern table (reduce, contract, map, store, transfer, sync, access, …) and a resource table. Both open; the undescribable goes to `other` and is counted. *Component: Pattern table, resource table.*
- **P3 Refine in steps** (G4, G5, G9). A template is one refinement step; a chain is one kernel family's full design. *Component: Template, Chain.*
- **P4 Template shape** (P3). Three readers: the solver (X, C, Ĵ), the verifier (Pre, π, Σ), retrieval (Γ, refines, signature); bound by K1 (C ⇒ Pre). Granularity: one new object, one local obligation, and at least one of the paper's three structural questions answered (what to optimise, what is independent, what to skip). *Component: Template schema, K1–K3 checks, assurance levels.*
- **P5 Formulation is assembled; the hardware model is separate** (A2, P4). Each chain item contributes variables and constraints. Hardware enters twice: as *walls* (capacity constraints' right-hand sides) and as *rates* (cost coefficients). Overlap structure (what may run concurrently) depends on scheduling decisions, so it is declared by templates; rates and walls come from the hardware model. Corollary: if each template carried its own hardware coefficients, calibration could not be shared and hardware knowledge could not amortise. *Component: a first-class hardware model (§6).*
- **P6 Solve ranks; measurement decides** (G6, G7, A3). The solver produces a shortlist, the top k are measured. The model's job is ordering, not absolute accuracy. *Component: shortlist → build → measure; rank correlation as the model's primary metric; a control that measures "solver choice vs copied assignment".*
- **P7 Gap decomposition routes the next action** (P1, P6, A5; paper Table 4).
  `T_meas − T_SOL = ε (model) + δ (solver) + Δpre (over-strong preconditions) + L_rep (space) + irreducible`. The paper's Table 4 needs `J*`, which is unknown; the SOL bound gives `L_rep` a computable upper bound, turning Table 4 from post-hoc classification into a stopping rule and a router. *Component: the judging state machine; all three flows are triggered by this table.*
- **P8 The population is the accumulation** (A1, A4). Ordered by amortisation value: chains/families; templates and their relations; the hardware model; findings (regimes, negatives); gaps and `other` counts. *Component: admission, deduplication, structural diversity, evidence status, invalidation propagation.*
- **P9 The agent appears only where creativity is needed** (G8, P7): writing templates, generators, decomposing references, adding structure to the hardware model. Everything else is deterministic code; agent output is a proposal until checked. *Component: orchestrator state machines, task cards, gates; the agent cannot write evidence.*
- **P10 Absorbing references is the cheapest extension** (G10). Decomposing a SOTA kernel yields new templates, a reproduction target and coverage evidence at once. *Component: the learning loop.*

What the derivation guarantees if A1–A6 hold: correctness of outputs (to the assurance level reached), monotone improvement per iteration under a fixed model, and explainability of every result. What it does not guarantee even then: global optimality, automatic termination (only budget), or amortisation itself.

## 4. Knowledge

**Definition.** A piece of knowledge is *claim + scope + evidence + status*: a conclusion that reduces future search or avoids a future error, with where it holds, why it is believed, and how far it is trusted.

**Placement rule.** Every claim is attached at the most general layer at which it still holds. Five degrees of dependence:

| Layer | Depends on | Example | New GPU generation | NPU |
| --- | --- | --- | --- | --- |
| semantic | the computation only | reduction is associative, hence splittable; softmax is rescalable, hence online | unchanged | unchanged |
| mechanism | an abstract *capability* | multi-buffer rotation needs a managed scratchpad plus asynchronous copy | mostly unchanged | holds where the capability exists |
| realisation | ISA or backend | TMA descriptors on Hopper, cp.async on Ampere, DataCopy in Ascend C | new realisation needed | all new |
| instance | one chip | DRAM peak, smem size, bank-conflict rule | re-measure | rebuild |
| finding | chip × workload | "on GB10, batch-1 decode needs split ≥ k to saturate DRAM" | demoted to prior | mostly void |

**Objects.** Pattern (semantic vocabulary) ← Template (mechanism; `requires` abstract capabilities) ← Realisation (template × backend × ISA range: code fragment, ISA constraints, expected SASS) ; Capability (declared per Target with parameters) ; HwModel (instance: §6) with HwFact as raw measurement ; Finding (unifies Regime and Negative: claim, kind, scope, evidence, status, `explained_by`) ; Chain (nearly hardware-free) → Family (chain + realisations + generator, bound to a backend) → Solution (assignment + measurement, bound to a Target).

**Promotion and demotion.** Raw evidence is immutable. A finding that the hardware model can *explain* (and that survives a hold-out check) is promoted into the model, and thereby holds for the whole chip; a finding that holds across targets is promoted into a template's Γ or a structure's regime; repeated `other` becomes a new pattern or template. Counter-examples demote: they tighten Pre or Γ, shrink a finding's scope, or invalidate a model term, and invalidation propagates along dependency edges (23 §4).

**Two orthogonal axes.** *Level* (how fine the refinement is) and *dependence degree* (how far it transfers) are different. Operator fusion is level 3 and semantic; warp-shuffle reduction is level 5 and needs a "lane exchange" capability; TMA multicast is level 5–6 and chip-specific. Levels are derived from the pattern a template refines; the agent does not label them by hand. Levels do not fix decision order (paper §2.3.3). Level numbering: the seven levels of the research frontend (1 workload, 2 graph, 3 operators, 4 tensor map, 5 instructions, 6 architecture, 7 microarchitecture); 22 currently uses the paper's 0–5 and must be reconciled (§13, D-T7).

## 5. Template

A template is the formal version of one *trick*: a parameterised program transformation with the conditions under which it is safe, how it changes cost, and where it applies. It is not a kernel, not a family, not a code template, carries no hardware numbers, and holds no performance conclusion. Its three faces:

| Face | Answers | Fields | Reader |
| --- | --- | --- | --- |
| transformation | which coarse transaction becomes which finer ones; new object; new variables; abstract program | refines, produces, O, X, P | solver, generator |
| contract | when it is safe and why | C, Pre (K1), π, Σ, Γ | verifier, retrieval |
| effect | what it does to cost, in the transaction vocabulary only | `effects`: transactions, storage, overlap | hardware model, solver |

Scope: one pattern (the part of its parameter space where Γ holds) × a set of capabilities. Neither shapes nor chips.

Why `cost` becomes `effects`: today `multi_buffer_rotation` writes `{unit: dram, work: "size*iters if P == 0 else 0", overlap: serial}` to express "with P = 0 load and compute do not overlap". That is an overlap statement (template's business) forced into a per-unit work field (hardware model's business). With `effects` the template says *transactions added or changed*, *storage occupied at which level*, and *which operations overlap with what in-flight depth*; the hardware model converts these into time.

## 6. Hardware performance model

**One declarative source.** The hardware model is structured data, not free Python. The simulator, the MIP and the profiling report are all *derived* from it (as LLVM derives both its scheduler and `llvm-mca` from one scheduling model). This is the structural answer to "the agent translates the simulator into a MIP and nobody can check the translation".

**Contents** (fidelity decreases downward):

| Part | Answers | Source | Trust |
| --- | --- | --- | --- |
| resources | which units exist; kind (throughput / capacity / latency+concurrency); scope (chip / SM / SMSP / CTA); peak, latency, max in-flight; the ncu counters that observe it | microbenchmarks | high |
| walls | capacity and legality limits and their cliff (no launch, spill, lower occupancy) | docs + probes | high |
| operation classes | for each transaction pattern (and, at SASS level, each instruction class): a *reservation table* — which resources it occupies, how much, for how long, with what latency; `serial` flags | docs, SASS, microbenchmarks | medium-high |
| efficiency | discount as a function of few features (access granularity, in-flight count, MMA shape, occupancy); initial values from microbenchmarks, refined from kernel measurements; each with fitting data and hold-out error | calibration | medium |
| interactions | who interferes with whom (bank conflicts, issue-port contention, L2 partitioning, a shared adder tree) with `applies_when` and evidence | residual-triggered, measured | trusted only in scope |
| composition | how per-resource times combine; skeleton from the templates' overlap declarations | rules | medium |

**Fine-grained sharing.** Structures such as "FP8 and FP16 tensor-core paths share one adder tree" are expressed in the standard way of pipelined-scheduling theory: a shared structure is a *resource*, and every operation class that uses it has that resource in its reservation table (possibly for different cycle counts); mutual exclusion is a resource of capacity one; ports and queues are resources with depth. Unknown sharing is *discovered* empirically: run class A alone, B alone, and A and B interleaved; if the combined throughput is below the sum, a shared resource exists; the pairwise interference matrix over instruction classes is a maintained artefact, and each discovered shared resource is added as a row with its microbenchmark.

**Coverage is for the complete experiments, not a prerequisite.** The experiments so far were built forward, iteratively, on the principle "enough for now"; coverage was never measured and did not need to be. Coverage statistics (below, and O1) belong to the more complete experiments that make claims about the pool; the forward, enough-for-now mode of working remains the way individual cases are done.

**Completeness as measured coverage, not assertion.** The SASS instruction stream is the hardware's contract, so a SASS-level model can in principle express every timing-relevant behaviour; but the ISA alone is not complete — memory-hierarchy state (caches, DRAM pages) and scheduling policies (warp scheduler, CTA rasterisation) are not in the ISA and must be modelled as resources and state. "Complete" therefore means: every instruction class in a kernel's SASS has a reservation table; every pair of classes is either measured for interference or declared independent; every memory level and scheduling policy that measurably affects time has a row. Coverage statistics per kernel (share of instruction classes modelled, share of pairs tested) replace any claim of completeness. Unmodelled resources default to infinite capacity and unmodelled effects to zero, so every addition is a monotone, versioned tightening with evidence.

**Four fidelity levels, one parameter table.**

| | Form | Input | Output | Use | Speed |
| --- | --- | --- | --- | --- | --- |
| L1 analytic | steady-state projection (§7) | template effects, variables | predicted time | inside the MIP; pre-compile ranking of many candidates | µs |
| L2 transaction-level simulation | per-CTA transaction dependency graph on resources | effects expanded into a transaction graph | per-unit occupancy timeline, stall and idle attribution | pre-compile bottleneck analysis; virtual wall relaxation | ms |
| L3 SASS-level simulation | the existing GB10 simulator (`kernel-lab/exports/gb10-simulator-calibration-20261020-v2`) | compiled binary | more accurate time, SM-side vs bus-side terms | post-compile reranking, diagnosis | s |
| L4 measurement | GPU timing, ncu counters, in-kernel timestamps where a timeline is needed | real run | ground truth | final decision, calibration | s |

L1 is the steady-state approximation of L2; L2 is the coarsening of L3 (one transaction's cost equals the cost of its instruction group). Disagreement between adjacent levels is itself a diagnostic.

**Maintenance as a software artefact.** A model version = parameter table (each value pointing to its microbenchmark and HwFact) + simulator commit + full regression results; versions are immutable. Every measured kernel becomes a regression case (binary, mapping, time, counters); the set must contain several configurations of the same kernel, because what the solver relies on is ranking and a single configuration per kernel cannot test it. Iteration: take the largest-error case; compare counters per resource; write a microbenchmark isolating the mechanism; change the model, labelled "adapter fix" or "machine-model change"; re-run the regression set on hold-out cases; admit only on improvement without regression, else roll back and record the dead end. Four views: regression scoreboard; per-kernel timeline (predicted over measured); mechanism coverage table; change log. "Exploration finished" means the coverage table stops growing and hold-out rank correlation is stable.

## 7. Steady-state projection

**Why a projection.** A simulator cannot be exported to a MIP as it stands: event ordering, contention and cache state are dynamic, and a time-indexed scheduling MIP over millions of events does not fit. What can be exported exactly is a well-defined *relaxation* of the simulator.

**The theory** (modulo scheduling: Rau and Glaeser 1981, Lam 1988, Rau 1994). A loop body is a set of operations; each uses resources per its reservation table; dependencies carry a latency and a *distance* (how many iterations later the consumer runs). Software pipelining starts a new iteration every II cycles. Two lower bounds:

```
ResMII = max_r  ceil( Σ_ops use_r(op) / cap_r )              resource bound
RecMII = max_cycles  ceil( Σ_edges lat / Σ_edges distance )   recurrence bound
MII    = max(ResMII, RecMII);   any feasible schedule has II ≥ MII
T      ≈ L_body + (trips − 1) · II                            plus prologue/epilogue/tail
```

The projection assumes II = MII, which is a theorem-backed lower bound; with enough buffering it is usually achievable.

**Mapping to a GPU kernel.** The loop is the CTA's main loop (the k-loop of a GEMM, the KV loop of attention). Resources are scoped: issue per SMSP, tensor and smem bandwidth per SM, DRAM and L2 per chip. Concurrency acts as a capacity multiplier: with `c` CTAs resident per SM and `n_SM` SMs, a per-SM resource sees `c` iterations' usage per II and a chip resource sees `c · n_SM`; occupancy therefore enters the resource bound directly. Latency hiding is the recurrence bound: a load issued P iterations ahead and a slot reused after S iterations give a cycle `load → compute → (slot reuse) → load` of latency `lat_dram + lat_mma` and distance S, hence `II ≥ (lat_dram + lat_mma) / S`. This is Little's law in modulo-scheduling form, and it is exactly where S and P enter the formulation. Instruction-level pipelining within a warp is folded into the operation classes' latencies and throughputs (measured by the SASS-level simulator and microbenchmarks).

**Unification with SOL.** The per-unit SOL is `ResMII` evaluated with the *minimal* work; the projection is `max(ResMII(W_schedule), RecMII(schedule))`; the simulator adds contention and non-steady phases; the hardware adds what is unmodelled. The four quantities are nested lower bounds, so each adjacent gap has a name (work inflation; latency exposure; contention and tails; model error).

**What the projection drops** (and therefore what "simplification error" means): arbitration and contention beyond the modelled interactions; cache state beyond the tabulated hit rates; scheduler policy (warp scheduling, rasterisation, tail waves — the wave count `ceil(CTAs / (n_SM · c))` is representable but coarse); prologue, epilogue and irregular control flow; whether ptxas actually achieves MII; register pressure (handled as a wall). It is tight for loop-dominated kernels (large GEMM, long prefill attention) and loose for latency-dominated ones (decode, small shapes), where the MIP is used only for coarse pruning and ranking passes to L2.

**MIP form.** Each discrete variable on a short menu is one-hot encoded, so any single-variable function is exactly linear; `II ≥ Σ_r use_r(x)/cap_r` is linear after encoding; `II ≥ lat/S` becomes `II ≥ Σ_s z_s · lat/s`; `max` is an epigraph (`T ≥` each term); `ceil` adds an integer; products of two menu variables use one-hot products or big-M; non-algebraic effects (hit rates, occupancy → efficiency, bank conflicts) are *table terms* `table(f; x1, x2)` precomputed by simulator or microbenchmark over a few menu variables and imported exactly. A term that fits neither the restricted language nor a table moves its variables to an outer enumeration. The compiler from declarative model to HiGHS applies these rules mechanically; the agent writes none of it.

**The guarantee.** Translation correctness is the compiler's; simplification error is checked automatically without a GPU: sample points of the subspace, evaluate MIP and L2 simulator, require rank correlation above a threshold and the optimistic bound to hold (MIP ≤ simulator) on nearly all points; failure names the effect with the largest deviation. The agent's remaining choices (outer-enumerated variables, which effects become tables, which dimensions to open) are scored by this check.

### 7.1 Formulation: what the agent still decides, and what it is given

With MIP export mechanical, "formulating" no longer means writing a MIP. It means choosing the subspace: the decisions below. Each has a deterministic default, so the system can always run without the agent, and the agent's choice is scored against that default.

| Decision | What it is | Deterministic default | What a good choice needs |
| --- | --- | --- | --- |
| F1 chain | which templates, applied to which transactions, in which order | nearest retrieved chain; or the chain enumerator's best-bound chain (§7.2) | knowing which transactions dominate and which moves address them |
| F2 bindings | which template variables merge into one family variable, which become constants | type- and domain-compatible merges by name; nothing fixed | knowing which couplings matter (e.g. the smem tile shared by tiling and rotation) |
| F3 open / fixed / outer | which family variables the MIP searches, which are fixed from priors, which are enumerated outside | all open; non-linearisable ones outer | low effective dimension, near-decomposability; findings that justify fixing |
| F4 menus | the discrete domain of each open variable | the template's declared domain | concavity knees, walls, discrete menus |
| F5 table terms | which non-algebraic effects are tabulated and over which few variables | compiler: every non-algebraic effect, over the variables it mentions | near-decomposability (a table over many variables is infeasible) |
| F6 glue, dispatch | family-specific constraints; shape buckets | none; one bucket | knowledge of the case's shape distribution |
| F7 generator | code that realises the chain | none (this is the gap between S(R) and S_exec) | realisations, neighbouring generators |

The agent's product for F1–F6 is a structured **FormulationPlan**, not prose: the chain, the bindings, each variable's status (open / fixed / outer) with the *structure* it invokes (paper Table 1) and the *evidence* it cites (a finding, a bound, a regime), the menus, the table terms, the glue. The deterministic default is expressed in the same object, so plans are diffable, reviewable and attributable: a wrong fix cites a finding that can be demoted; a wrong decomposition cites a structure whose regime can be marked as not holding.

**Information: an access discipline, not a dump.** The agent's context window is finite; the failure to avoid is not hidden state but *drowning*: pushing everything the system knows and burying the few items that matter. The construction therefore neither pre-assembles "the right information" (the system cannot know it per case) nor leaves the agent to browse freely. It gives three things:

1. **A short fixed header** on every formulation task card: the case (semantics, shapes, contract, target); the bound (per-unit SOL, bottleneck unit, critical-path estimate); when iterating, last round's gap decomposition (class, idle resource, blocking wall with its shadow price, MIP↔simulator inconsistency naming an effect); the deterministic default plan and its score; the budget. This is at most one screen and answers "what to optimise" and "what went wrong".
2. **An access discipline**: a decision table, kept in the worker skill and versioned, that says *in which situation to fetch what, and what to decide from it*. Everything else is fetched by ID as a card (≤ 5 lines, 22 §5) and expanded only on demand. A first version:

   | Situation | Fetch | Decide |
   | --- | --- | --- |
   | start | the nearest chain with each step's Γ evaluated in context | if most steps hold: start from it and fetch alternatives only for the failing steps; else go to diagnostic retrieval |
   | diagnostic retrieval | bottleneck unit → patterns touching it → templates refining them, ranked by evidence (top N cards) | choose the moves that address the dominant transactions; cite the bound |
   | each open variable | findings in scope that pin it; the wall it is bounded by | fix it with the finding cited, or keep the declared menu |
   | the plan's MIP is too large for the budget | the structure table and the regimes known for this workload and chip | move variables outward or decompose, citing the structure |
   | a candidate effect is non-algebraic | the hardware model's tabulated effects for the operations involved | use the table or request one over ≤ 2 variables |
   | the last round was class ε (model) | nothing from the space | do not touch the space; hand over to model diagnosis |
   | the last round was class L_rep with an idle resource | templates that reduce or overlap that resource; references whose chains contain such a step | open a dimension or add a step; one mechanism per round |
   | the bottleneck is blocked by a wall | the wall's shadow price; templates that reduce the walled resource | change the move, or record a co-design finding |
   | stop fetching | — | when every non-default decision in the plan cites evidence, or the budget is reached |

   The agent may also pull outside the table (evaluate Γ, run the L2 simulator on a point, query findings, request a microbenchmark through the executor); every pull is logged. It cannot write evidence or change checks.
3. **Scoring against the default** (D-T19), so that an under-informed or over-informed plan shows up as measured loss, not as silent degradation.

**What is and is not guaranteed.** No construction can guarantee that the agent has "enough" information in a logical sense. What is guaranteed is testable: nothing the system knows is unreachable; the discipline makes the reachable set navigable; every plan is scored; pulls are logged, so "which information helped" is measured rather than assumed (O10: ablate rows of the discipline, or compare the discipline against free browsing and against the full dump). The discipline itself is revised by the system loop (§9) from these logs: rows that are never used are dropped, pulls that recur outside the table become rows.

### 7.2 Chain enumeration as the deterministic baseline

Because the population is a grammar (templates are rewrite rules over transaction patterns, relations are composition rules), subspace selection need not depend on the agent alone. A bounded enumerator expands legal chains for a workload, filters by Γ and capabilities, assigns each chain an optimistic bound (SOL under that chain's minimal work) and prunes chains whose bound is worse than the incumbent. It gives the agent a baseline to beat, exposes missing rules (no chain's bound approaches SOL ⇒ the grammar lacks a move), and yields coverage counts (legal chains per case, best bound).

## 8. Three-level performance chain

| Pair | Metric | Disagreement means | Repair | GPU needed |
| --- | --- | --- | --- | --- |
| MIP ↔ L2 simulator | rank correlation; optimistic-bound violation rate | simplification error | table term; move variable outward | no |
| L2 ↔ L3 simulator | cycles and per-unit busy time of one kernel | transaction abstraction incomplete, or generator not implementing the intent | add transaction cost; fix generator | no |
| simulator ↔ hardware | error; per-resource counter comparison; timelines | model error | microbenchmark + calibration via the regression set | yes |

Requirements this places on the model and simulators: R1 single declarative source; R2 restricted expression language plus table terms; R3 data-driven additions with conservative defaults and versions; R4 transaction and SASS granularity sharing parameters with an aggregation relation; R5 outputs total cycles, per-unit busy time, per-unit occupancy timeline, stall reasons, idle attribution; R6 metrics named as ncu names them; R7 what-if runs (move a wall, change a parameter) for shadow prices; R8 calibration interface and regression set; R9 determinism and versioning; R10 speed budgets (L2 ms, L3 s).

**"Sustained saturation" and wall relaxation.** A kernel is well optimised when some bottleneck unit b has `Busy_b ≈ SOL_b` (no inflated work) and `T ≈ Busy_b` (busy throughout). Idle intervals of the bottleneck are attributed: waiting for another unit (phase alternation → overlap templates), for data (pipeline depth), for synchronisation, prologue/epilogue/tail, or *blocked by a wall*. Moving a wall in the L2 simulator and reading the shortened idle time gives that wall's shadow price directly from the timeline; on a fixed chip this separates "scheduling problem" from "hardware limit", and in co-design it is the price wall relaxation buys. For latency-bound workloads the criterion becomes: no idle wait on the critical path.

## 9. Iteration model

Four nested predict–measure–repair loops; each changes one kind of thing and treats the outer loop's objects as constants:

| Loop | Scale | Changes | Must not change | Driven by | Should improve monotonically |
| --- | --- | --- | --- | --- | --- |
| solve | s–min | the assignment | formulation, model | predicted ranking, compile feedback | best measured time |
| case | min–h | the subspace (open/fix dimensions, swap a template), model parameters | admitted pool knowledge | gap decomposition | unexplained gap |
| knowledge | days | new templates, model structure, findings | system code, checkers | recurring residuals across cases, reference gaps | new templates per new reference; hold-out model error |
| system | weeks | process, checks, routing rules | claims and acceptance criteria (user approval) | loss statistics over a batch | share of each loss class |

Every round: predict → measure (with counters) → residual per resource and against SOL → attribute → one change → re-verify on hold-out points, else roll back and record. Attribution order is fixed: correctness → model error ε → solver δ → preconditions Δpre → space L_rep → irreducible; a later class is not judged before the earlier ones are excluded. Three disciplines: one mechanism per round (except coupled groups, e.g. the paper's 64×64 tile that only pays when registers, tensor shape and cache move together); hold-out verification with rollback; an exploration budget (measure a few points the model is least sure of, or the model is only ever tested where it already believes it is right).

### 9.1 The complete loop and its bridges

The loops are not independent: each pair is joined by a *bridge object*, a record that one loop writes and the other reads. Nothing crosses a loop boundary except through such a record, which is what makes the whole traceable.

```
design loop (user + assistant)        bridges: this document, decision ledger, claims ledger,
  discuss → record → spec → build →            deliveries + evidence, canaries, surprises
  run (controlled, then open) → observe → discuss
    │
    ▼
system loop (per batch)               bridges in:  loss statistics by class, pull logs,
  loss stats → process change →                    regression scoreboard, coverage tables
  user approval → construction →      bridges out: new checks, routing rules, access-discipline
  gate                                             versions, model versions
    │
    ▼
knowledge loop (across cases)         bridges in:  Gap (L_rep with no template), Finding (observed),
  ├ learning: reference → chain →                  recurring residual per resource, reference list
  │   templates + reproduction        bridges out: Template versions, model versions, promoted
  ├ calibration: regression set →                  findings, invalidations → waiting cases resume
  │   model version
  └ population: admit / dedupe / promote / demote
    │
    ▼
case loop (per case)                  bridges in:  Case, pool snapshot, model version
  bound → retrieve → plan →           bridges out: Prediction+Measurement pairs, gap decomposition,
  solve → build → check → measure →                Gap, Finding, Solution, FormulationPlan,
  decompose gap → route                            pull log
    │
    ▼
solve loop (per formulation)          bridges in:  Formulation (from plan + hw model)
  MIP / enumerate → shortlist →       bridges out: Candidates with Prediction, consistency-check
  compile → rerank                                 result (MIP vs L2)
```

The bridges downward are *versions* (pool snapshot, model version, formulation); the bridges upward are *records of evidence* (measurements, gaps, findings, logs). A case never changes the pool directly; a knowledge change never edits a case's history; a process change never alters a claim without the user. The three flows of 00 §4 (solve a case, learn a reference, calibrate) are the case loop and the two branches of the knowledge loop; their coupling is through the Gap and Finding records and the regression set.

### 9.2 Staged release: controlled before autonomous

Neither humans nor models can foresee every factor before running; the design is a hypothesis that only repeated rounds can correct. The system is therefore released in stages, and a stage opens only after the user has understood and debugged the previous one:

| Stage | What runs | Agent calls | Budget | Exit condition |
| --- | --- | --- | --- | --- |
| S0 deterministic | the golden path (GEMM on GB10) with no agent: default plan, solve, build, check, measure, gap decomposition | none | GPU only | numbers match known references; every record inspectable in the frontend |
| S1 single agent task, hand-stepped | one task type at a time (T1 … T6) on one case; the user reads the task card, the pull log and the plan before the next step | one, gated by the user | small, per task | the user can predict what the agent will do from the card and the discipline |
| S2 one case loop, autonomous | the full case loop on a handful of cases, agent calls inside it | automatic within the case loop | capped per case | gap decomposition routes correctly on injected faults (O5); no surprise in the pull logs |
| S3 knowledge loop | learning and calibration branches on a batch | automatic | capped per batch | promotions and invalidations behave as §4 says |
| S4 open | all loops, experiment protocol | automatic | the experiment's budget | — |

Each stage keeps the previous stage's canaries. A stage is reopened, not skipped, when a later one shows a failure that belongs to it.

### 9.3 Independence, context and simplicity

The loops of §9.1 run as independent processes that communicate only through records and the task queue (11 §1): solving a workload, accumulating knowledge (downloading operator libraries, decomposing them, extracting moves), and calibrating the hardware model do not share a context, can run concurrently, and never call each other. Calibration is organisationally a branch of the knowledge loop (§9.1) even if it stays a separate process for scheduling. The case loop *requests* from the knowledge loop by writing a Gap (no template fits; model wrong in a way calibration did not fix) and waits; a case worker may also *offer* knowledge in passing (a proposed template or finding) which goes through the same admission as everything else (23 §6). There is one runtime role, the worker, with 2–4 instances; no sub-agent trees and no multi-agent orchestration (10 §4.1). The system is kept as simple as these loops require, not more.

**Context across steps** (10 §4.2, §4.4 already state this; summarised here because it is a thinking-model question): a session is resumed *within* a work unit (one case loop instance, one reference) and never across loops or units; state lives on disk in `NOTES.md`, so a session can be dropped at any time; a fresh session is started when context passes a threshold, when the same check fails twice for the same reason (a new viewpoint against a local optimum), when the unit ends, or on error; prompts are a fixed prefix plus a one-page step header, everything else pulled on demand (§7.1). The trade-off the user names — resume saves re-reading and keeps understanding, but can pollute context and entrench a line of thought — is not settled by argument: resume-within-unit versus fresh-from-NOTES is an experiment arm (O11), measured on tokens, latency and outcome.

### 9.4 First thresholds

Without stated thresholds an agent optimises for passing checks: it chases gains of a few per cent that reveal no mechanism, widens search instead of explaining a gap, or polishes a case that should be closed. The numbers below are defaults (D-T23) to be replaced by measured ones; what matters is that each decision has *a* rule. The complete, tunable table of every budget, cut-off and anti-gaming control, with the failure each prevents and where it is set, is [12 Agent controls](12-agent-controls.md); this section keeps only the rules that shape the loop itself.

| Decision | Default rule |
| --- | --- |
| solve scope | enumerate when the open product is < 10⁶ points, else MIP with a 60 s limit; shortlist 64, measure top 8 (spec); at most 6 variables open in one MIP, outer enumeration ≤ 32 points |
| what counts as an improvement | beyond the noise band (measurement CI, or 5 % of T, whichever is larger) **and** attributed to a named mechanism (a template step, an opened variable, a model term). Gains below the band are recorded, not credited |
| null round | a round that opens nothing, adds nothing and gains below the band; two null rounds close the case (accept with gap, or escalate) — no polishing |
| accept | not worse than the strongest reference **and** the unexplained gap (after decomposition) is below 10 % of T; or the gap is fully explained as irreducible |
| escalate case → knowledge | L_rep with no applicable template; or the same model residual class in ≥ 3 cases; or model error on the top-k > 30 % after one calibration attempt |
| escalate → user | 5 rounds without accept (10 §2.1); budget overrun; a canary missed; any change to a claim or acceptance criterion |
| knowledge reliability | templates: declared / tested / bounded / proved (22 §6). findings: observed (one measurement) → replicated (≥ 2 independent shapes or runs, same direction, beyond band) → explained (derived by the model) → invalidated. model terms: provisional until ≥ 5 regression cases support them and hold-out error improves; provisional terms flag their predictions |
| which level to work at | compute the bound at both levels; if the graph-level bound (e.g. after fusion) is < 0.8 × the sum of kernel bounds, the loss is at the graph level and graph moves come first. Phase one does not act at graph level (00 §9) but records both bounds from the start |
| when to go one level finer | the bottleneck unit is saturated or the loss is attributed to a transaction that the current level cannot express (paper §2.2: bottleneck moved) |

## 10. Verification

Verification (is the program right) is separated from validation (is the model right). Legality of a program is five properties, each with its own check: semantic correctness (template proofs once, numerical differential tests per kernel including edge shapes and adversarial inputs); memory safety (coverage / exact-once / lifetime obligations plus compute-sanitizer memcheck, racecheck, synccheck); termination (progress obligations on bounded models plus timeouts); resource legality (walls, realisation ISA constraints, compile feedback — fully automatic); measurement integrity (immutable inputs, several seeds, CUDA Graph checks). Verification happens once per template (K1, K2) and once per candidate (constraint evaluation, milliseconds). It cannot cover floating-point rounding, schedules beyond the bounded model, or — the weakest link — whether an agent-written generator implements the template's abstract program; for the last, in increasing cost: code markers, expected-SASS traces, sanitizers, and later extraction of the actual transaction and synchronisation structure from the TileLang IR for comparison with the template. Phase-one claim: `tested`, never `proved`, for anything not carrying a proof.

## 11. Obligations still open

| | To show | Evidence now | How to test |
| --- | --- | --- | --- |
| O1 | vocabulary saturates (A1) | favourable but 7 references; 13 `other`; 30 unmatched at level 5 | ~20 references; new templates per reference → 0 |
| O2 | coupling is local (A2) | none | count glue constraints per family; generator diff confined to markers when a step is swapped |
| O3 | model ranks (A3) | unfavourable (G6) | rank correlation over top-64 per case; residual per resource |
| O4 | Γ is reliable (A4) | little | rate of "Γ admits, measurement fails" |
| O5 | attribution is right (A5) | none | fault injection: degrade the model, delete a template, cap the solver; the router must name the right class |
| O6 | agent cost (A6) | unfavourable: template construction costlier than text knowledge | cost per template and family vs pool size |
| O7 | amortisation | none | learning curve; weak+pool vs strong+scratch |
| O8 | bound is tight | none | share of unexplained gaps; critical-path model for latency-bound cases |
| O9 | granularity | 12 of 35 templates used once | always-adjacent pairs (merge); single-use (over-specialised) |
| O10 | the access discipline gives the agent what it needs without drowning it (§7.1) | the paper's text-vs-template result (1.49× vs 0.84× on the smallest task) | discipline vs free browsing vs full dump; ablate rows; agent plan vs deterministic default |
| O11 | resuming a session within a work unit beats fresh sessions from `NOTES.md` (§9.3) | none | two arms on the same cases: tokens, latency, outcome, rounds stuck on one line |

O3 and O5 come first: if either fails, P6 and P7 fall and the loop does not close.

Engineering gaps: E1 hardware model as a first-class, declarative object (§6); E2 template `cost` → `effects`; E3 gap-decomposition engine with data-driven routing; E4 selection controls (model top-k, copied assignment, random k); E5 population governance; E6 fault-injection canaries; E7 level numbering; E8 steady-state projection compiler and the MIP↔simulator consistency check; E9 transaction-level simulator; E10 refactoring the existing SASS-level simulator onto the declarative parameter table with timeline and ncu-named outputs; E11 regression set with multiple configurations per kernel; E12 chain enumerator and FormulationPlan schema (§7.1, §7.2).

## 12. How the user supervises

Acceptance moves from code to a few machine-checkable facts plus one real run seen with one's own eyes: a claims ledger (O1–O9 and the two theses; every delivery states which claim it advanced); canaries (planted faults that the checkers must catch — trusting unread code rests on this); one golden path (GEMM on GB10, one command, known reference numbers); review only contracts (schemas, interfaces, acceptance tests, claims); decision triage with a cap on the review queue; alerts only on anomalies plus a one-screen daily summary; independent acceptance by a different model per milestone.

## 13. Decision ledger: defaults made visible

An assistant takes defaults at every step; only the ones the user happens to ask about become explicit decisions. The design conversations of 2026-10-30/31 were such a pass: a batch of earlier defaults was pulled out and discussed, and each became clearer for it. This ledger turns that into a standing mechanism rather than a one-off:

- every design document, spec change and delivery carries its defaults here (or in `docs/decisions.md` once a WP exists), stated as *what was chosen, what else was possible, why, and what it costs to change*;
- entries are ranked by reach: those that would change the thinking model, a claim or an acceptance criterion come first; local, reversible ones last;
- the user pulls on whichever entry they want discussed; a discussion ends with the entry confirmed, replaced, or left as default with the reasoning recorded;
- an entry nobody has pulled on stays a default, and is labelled as such wherever its consequence shows up (reports, frontend, delivery notes), so that a later surprise can be traced to it.

Every entry below was chosen by the assistant without an explicit user decision. Each is `default — needs user review` until confirmed. "Change cost" is what it takes to reverse it later.

| ID | Decision taken by default | Alternative(s) | Why this default | Change cost |
| --- | --- | --- | --- | --- |
| D-T1 | The MIP is assembled mechanically from chain + hardware model; the agent does not write MIPs per case | agent writes the formulation per case (the paper's GB10 practice; the collaborator's position: "let the agent translate") | translation becomes checkable (§7), coefficients shared and calibratable; the same structured data drives co-design | high once E8 exists; the alternative can coexist as an "agent-written formulation" arm for comparison |
| D-T2 | Hardware coefficients live only in the hardware model; template `cost` becomes `effects` (transactions, storage, overlap) | keep per-unit `CostTerm` in templates | §5; one calibration benefits all templates | medium: schema, WP4/WP5 interfaces, 5 seed templates, WP10a proposals |
| D-T3 | Hardware model is declarative data with a restricted expression language and table terms; simulator, MIP and reports derive from it | keep the simulator as free Python and let the agent read it | §6; the main structural risk control | high: E8–E10 |
| D-T4 | MIP uses the steady-state (modulo-scheduling) projection; non-steady kernels rely on L2 simulation for ranking | enumerate + simulate only (no MIP) for single kernels; MIP reserved for joint spaces | projection has a theorem-backed relation to the simulator; MIP still needed for joint/co-design spaces | low: the projection is one of several rankers |
| D-T5 | Four fidelity levels L1–L4 sharing one parameter table; a transaction-level simulator (L2) is added | only L1 and L3 | pre-compile timelines and virtual wall moves need L2 | medium: E9 |
| D-T6 | Attribution order: correctness → ε → δ → Δpre → L_rep → irreducible; δ and Δpre added to the judging rules | current rules (correctness → model → accept → extend) | later classes cannot be judged before earlier ones are excluded | low |
| D-T7 | Level numbering follows the research frontend's 1–7 | the paper's 0–5 as in 22 | one convention across the two repositories | low but touches 22, schemas, WP10a data |
| D-T8 | HwFeature → abstract Capability with per-target parameters; Realisation split from Template (template × backend × ISA range) | keep HwFeature bound to units and instructions; keep fragments and expected SASS inside Template | transfer across generations and to NPUs by capability matching | medium: schema, WP4, seed data |
| D-T9 | Regime and Negative merge into Finding with `explained_by` and promotion/demotion status | keep two objects | one promotion mechanism for all empirical knowledge | low |
| D-T10 | HwModel split into architecture class and chip instance | one flat model per target | structure reuse across generations | low |
| D-T11 | SOL bound is the stopping criterion and the upper bound on `L_rep`; per-unit SOL, busy time and total time reported separately | accept on "not worse than strongest reference" only | P1, P7 | low |
| D-T12 | Each iteration changes one mechanism, verifies on hold-out points, rolls back on failure; an exploration budget is reserved | free-form agent iteration | attribution and accumulation need it | low: orchestrator rules |
| D-T13 | Selection controls recorded on every solve (model top-k, copied assignment, random k) | none | the paper could not isolate solver gain | low |
| D-T14 | Completeness of the hardware model is a measured coverage (instruction classes, interference pairs, memory/scheduling rows), never asserted | assert completeness per architecture | principle 1 of AGENTS.md (check obligations deferred, made visible) | low |
| D-T15 | Fine-grained sharing (e.g. a shared adder tree) is modelled as a shared resource in reservation tables, discovered by pairwise interference microbenchmarks | ad-hoc interaction terms only | standard pipelined-scheduling theory; discoverable | low |
| D-T16 | This document is written in English per AGENTS.md; the conversation and the user's reading language are Chinese | write design documents in Chinese like the rest of `docs/design/` | repository rule; paper material is English | trivial — **confirmed by the user 2026-10-31** |
| D-T17 | Canaries, claims ledger, golden path, contract-only review and review-queue cap are the user's supervision mechanism | milestone acceptance as in 60 only | the current queue (37 + 37 items) is not reviewable | low, process only |
| D-T18 | The agent's formulation output is a structured FormulationPlan (chain, bindings, variable status with cited structure and evidence, menus, table terms, glue), not prose or a hand-written MIP | agent writes the formulation freely | reviewable, diffable, attributable; same object expresses the deterministic default | low: one schema |
| D-T19 | Every agent formulation decision has a deterministic default (nearest chain / chain enumerator, all variables open, declared menus) and is scored against it | no baseline; trust the agent's plan | the system runs without the agent; agent value becomes measurable (O10) | medium: chain enumerator (E12) |
| D-T20 | A formulation task receives a one-screen header plus an *access discipline* (a versioned decision table in the worker skill: situation → fetch → decide); the agent pulls the rest by ID as cards; pulls are logged; the discipline is revised from the logs | pre-assembled full context; or free browsing | finite context: the few items that matter must not be buried; guidance without full freedom | low: skill text and task-card assembly |
| D-T21 | The design loop (user and assistant: discuss → record in this document and the ledger → spec → build → run → observe → discuss) is the outermost loop; its bridge objects are this document, the ledger, deliveries, the claims ledger and canaries; this document is a hypothesis revised by runs, not a finished design | treat design as finished before running | neither humans nor models can foresee every factor before running; only repeated rounds surface them | none: it is how the project already works |
| D-T22 | Staged release S0–S4 (§9.2): controlled, user-stepped runs before any autonomous token spend; a stage opens only after the user has understood and debugged the previous one | open the full loop once the gate passes | the user must be able to predict the system before trusting it with budget | low: budget and gating configuration |
| D-T23 | The thresholds of §9.4 (improvement band, null round, accept, escalation, reliability ladder, level choice) as first defaults | leave to the agent's judgement | without rules the agent optimises for passing checks | low: configuration; replaced by measured values |
| D-T25 | All agent budgets, cut-offs and anti-gaming controls live in one tunable table (12) with the failure each prevents; a knob that prevents nothing for a milestone is removed; a frontend settings page edits these versioned values and nothing else | scatter values across spec, profile and code | the user tunes agent behaviour from one place; knobs stay justified | low: a document now, a settings page later |
| D-T26 | The way of thinking used in these conversations is fixed as a construction skill, `.agents/skills/design-interrogation`: nine questions per mechanism, an adversarial second pass in a fresh context, and three required artefacts (ledger entries, obligations, knobs); run before and after any design work | rely on the user to ask the right questions | defaults become visible without being asked for; the method is older than agents and only needed naming | low: one skill file |
| D-T24 | Session policy stays as 10 §4.2 (resume within a work unit, fresh across units and loops, `NOTES.md` as state); calibration is a branch of the knowledge loop; one worker role, no multi-agent | always fresh; or resume across units; or separate calibration agent | tokens and continuity vs context pollution; simplicity | low; O11 decides resume vs fresh |

Entries confirmed by the user move to `docs/decisions.md` with a regular ID and the corresponding spec change; entries rejected are deleted here with a one-line note of what replaces them.

## 14. Maintenance

This document is updated when the thinking model changes, not per work package. Each change states what was learnt and which of G, A, P, O, D it touches. Numbers quoted from deliveries are replaced by the delivery's own figures when they change.

**Length.** This document records the model, not the discussion. It stays within roughly 400 lines. When a section grows, its detail moves to the numbered design document it belongs to (agent and context → 10, processes → 11, templates → 22, knowledge graph → 23, experiments → 50) and a pointer remains here. Ledger entries that are confirmed move to `docs/decisions.md`; rejected ones are deleted with a one-line note. Obligations that are settled move to the claims ledger with their evidence.
