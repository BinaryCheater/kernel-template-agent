# KTA — Kernel Template Agent

**Kernel optimisation as accumulated, checkable knowledge, not one-off token spend.**

KTA is a research system for designing high-performance GPU kernels with LLM agents. Instead of asking a strong model to write each kernel from scratch, KTA makes the expert's knowledge explicit — as refined *Templates*, *Chains* of them, a declarative *hardware model*, and scoped *Findings* — and lets a deterministic pipeline search over it. The agent is called only where something is genuinely missing, and what it contributes is kept, checked and reused.

> **Status (2026-10-31): research prototype, code stage.** The planned system, including the design-v2 mechanisms, is written and passes its code-stage gates. Apart from the infrastructure checks of the first work packages, **nothing has been run for real yet**: no kernel solved end to end on the GPU, no autonomous agent run, no experiment. Every real run the code owes is listed in its work package's delivery note. Fixture data in the UI is labelled as such. See [Status and roadmap](#status-and-roadmap).

---

## The idea in one paragraph

Kernel design can be **white-boxed**. Every kernel is a set of transactions — compute, data movement, synchronisation — scheduled on finite hardware resources over time. An expert's tricks are transformations of that schedule, each with a contract saying when it is safe and what it does to cost. If those tricks are written down as executable, verifiable units and composed by a solver against a hardware model, then a search over accumulated knowledge can reach state-of-the-art kernels, and the agent is needed only to fill real gaps. The alternative — a strong model spending many tokens per kernel and keeping nothing but code — is our control arm, not our method.

## Key ideas

### One paradigm, one ruler

- **One vocabulary.** A table of transaction patterns (reduce, contract, transfer, sync, …) and a table of resources (tensor cores, shared memory, DRAM, issue slots, …). Both tables are open: what cannot be described goes to `other` and is *counted*. We never pretend the vocabulary is complete.
- **One ruler.** Every resource has a measurable peak, so for any problem the **speed-of-light (SOL) bound** is computable without knowing the best kernel. A kernel is finished when some bottleneck unit does no more than the minimal work *and* is busy from start to finish.
- **The gap decides what happens next.** The distance to the bound is decomposed into named parts —

  ```
  T_measured − T_SOL  =  ε (model error)  +  δ (solver)  +  Δpre (preconditions)  +  L_rep (representation)  +  irreducible
  ```

  — attributed in a fixed order, and that decomposition (not the agent's judgement) routes the next action: calibrate the model, open a variable, add a template, or accept. Idle time blamed on a hardware *wall* is that wall's shadow price: on a fixed chip it separates scheduling problems from hardware limits; in co-design it is what relaxing the wall would buy.

### Three kinds of knowledge

| Kind | Object | What it says |
| --- | --- | --- |
| what can be built | **Template**, **Chain** | the moves, and known-good compositions of them |
| what it costs | **Hardware model** | resources, walls, reservation tables, efficiency, interactions — as data |
| where it pays | **Finding** | an empirical claim with its scope, evidence and status |

The formulation for a case is *derived* from these three; nobody writes an optimisation model by hand. A state-of-the-art kernel mostly teaches a *sentence* (a chain with bindings and the regime where it wins) and rarely a new *letter* (a template). Composition knowledge — which moves go together, in which order — is what makes SOTA *reachable* rather than merely *expressible*, and it emerges as statistics over reconstructed chains.

### The Template: a trick with a contract

A Template is one refinement step — a parameterised program transformation that turns a coarse transaction into finer ones and introduces exactly one new object (a shared-memory slot, a pipeline stage, a split of a reduction). It has three faces, each read by a different consumer:

| Face | Answers | Read by |
| --- | --- | --- |
| transformation | which transaction becomes which; variables; abstract program | solver, generator |
| contract | when it is safe and why: constraints, preconditions, obligations, scope Γ | verifier, retrieval |
| effects | what it does to cost, *in transaction terms only* | hardware model, solver |

A template carries **no hardware numbers and no performance conclusions**: it declares the transactions it adds, the storage it occupies and what may overlap; the hardware model prices that. This keeps one calibration valid for every template, and lets knowledge outlive the chip it was learnt on.

### The hardware model is data, not code

The simulator, the optimisation model and the profiling report all derive from **one declarative source**: resources (with ncu counter names), walls (with their cliff), per-operation reservation tables, efficiency curves, and measured interactions. Fine-grained sharing (two datapaths behind one adder tree) is just another resource, discovered by pairwise interference microbenchmarks. Completeness is never asserted — it is *measured* as coverage of the instruction classes that appear in real kernels.

Four fidelity levels share one parameter table:

| Level | Form | Used for |
| --- | --- | --- |
| L1 analytic | steady-state (modulo-scheduling) projection | inside the MIP, microseconds |
| L2 transaction-level simulation | per-CTA transaction graph on resources | occupancy timelines, idle attribution, what-if wall moves |
| L3 SASS-level simulation | instruction-level simulator | post-compile re-ranking and diagnosis |
| L4 measurement | GPU timing and counters | the final word, and calibration |

Because L1 is a well-defined relaxation of L2, the MIP can be **checked against the simulator without a GPU**: rank correlation on sampled points, and the optimistic bound (MIP ≤ simulator) must hold.

### Solve ranks, measurement decides

The model's job is to *rank* candidates well enough to pick what is worth measuring, not to predict absolute time. A solve shortlists candidates, compiles them in parallel, checks their SASS against each template's expected traces, verifies correctness on the top *k*, and measures under a frozen protocol. Every solve also records selection controls (model top-*k* vs. copied assignment vs. random *k*) so that a solver that adds nothing cannot hide.

### Loops with bridges

```
design loop     (user + assistant)  ── design docs, decision ledger, deliveries, canaries
 system loop    (per batch)         ── checks, routing rules, access discipline, model versions
  knowledge loop (across cases)     ── templates, hardware-model structure, findings
   case loop     (per case)         ── subspace, model parameters  → gap decomposition
    solve loop   (per formulation)  ── the assignment              → candidates + predictions
```

Each loop changes one kind of thing and reads the outer loop's objects as constants. Loops talk only through immutable records: downward bridges are *versions*, upward bridges are *evidence*. Solving, learning and calibrating run as independent processes that never share an agent context.

### Levels and transfer

Refinement levels (1 workload · 2 graph · 3 operators · 4 tensor map · 5 instructions · 6 architecture · 7 microarchitecture) say how fine a decision is. A second, orthogonal axis says how far it *transfers*: semantic → mechanism (needs an abstract capability) → realisation (backend/ISA) → chip instance → chip × workload finding. Knowledge is attached at the most general layer where it still holds, so moving to a new chip keeps the semantics and mechanisms, replaces realisations and instance numbers, and demotes findings to priors. Mechanisms the current chip cannot run (e.g. TMA multicast on GB10) are still recorded as *unprobed* and wake up when a chip that has them arrives.

## Design principles we hold ourselves to

- **Honest verification.** Proofs cover what they cover; everything else is `tested`, never `proved`. The weakest link — whether agent-written generator code implements a template's intent — is guarded by code markers, expected-SASS checks and sanitizers.
- **Agents game checks; design for it.** A gain counts only if it exceeds the noise band *and* is attributed to a named mechanism. Two null rounds close a case. Task cards fix their checks; hidden shapes stay hidden; planted canary faults test the checkers; agents cannot write measurements or verdicts.
- **Finite context.** Agents get a one-screen task header and a versioned access discipline (situation → fetch → decide); everything else is pulled by ID, and pulls are logged.
- **Real or declared.** A capability is claimed only after a real run. Test doubles live in `tests/fakes/`; any simplification carries a `# kta:fixture` marker and is listed in the delivery note. Unsupported paths raise `NotSupported` instead of returning plausible numbers.
- **Parameters by experiment, not by fiat.** Which model per task type, effort level, resume vs. fresh sessions, retries, the noise band: each is a small experiment (X1–X12). Until it has run, the value is a labelled default in one versioned controls table.
- **Simple first.** One runtime agent role, a few instances, no sub-agent trees. Agent-hours are the scarce resource; GPU-seconds are cheap.
- **Staged release.** S0 deterministic golden path (no agent) → S1 hand-stepped agent tasks → S2 one autonomous case loop → S3 knowledge loop → S4 open. No stage opens until the previous one is understood.

## What we claim, and how it is measured

The central claim is relative and scoped — for a stated class of workloads, one architecture class and a stated correctness contract. Coverage of state-of-the-art kernels is reported in four levels; the thesis lives at level 3:

| Level | Meaning | Measured by |
| --- | --- | --- |
| 1 expressible | the SOTA kernel is a point of our space | reconstruction match rate (73 % of 135 steps over 7 references, first pass) |
| 2 reproducible | generating from that point reproduces its performance | reproduction error on the reference's own configuration |
| 3 reachable | given only the problem, the system finds that point or an equally good one | blind reconstruction |
| 4 surpassed | the blind result beats the reference under the same contract | same |

Secondary claims: **amortisation** (cost per new case falls as the pool grows; a weak model with the pool approaches a strong model without it) and **diagnosability** (every shortfall is attributed to one component). Experiments compare three knowledge conditions — from scratch, the pool rendered as text, and the executable Template pool — each with a strong and a weak model.

**Scope v1.** GEMM, quantised GEMM/GEMV, attention prefill and decode, MoE grouped GEMM, norm — on NVIDIA GB10 (sm_121). Backends: TileLang primary, CUDA C++ with inline PTX as the escape hatch, Triton for controls. Out of scope and counted separately: graph-level optimisation and system-level features (host planners, paged-KV machinery, dispatch).

## How a case is solved

```
Case ─▶ SOL bound ─▶ retrieve legal chains (Γ, capabilities) ──(none)──▶ agent task: extend the space
     ─▶ FormulationPlan (deterministic default; agent plans are scored against it)
     ─▶ projection compiler ─▶ MIP over the open variables ─▶ shortlist
     ─▶ generate + compile in parallel ─▶ SASS intent checks ─▶ correctness on top-k ─▶ measure
     ─▶ gap decomposition ─▶ accept │ calibrate the model │ extend the space │ repair a template
```

Everything except "extend", "calibrate" (diagnosis) and "repair" is deterministic code. Agent work arrives as typed tasks (T1 family, T2 decomposition, T3 template, T4 model diagnosis, T5 microbenchmark, T6 repair) with fixed checks, budgets and a persistent, resumable session per work unit.

## How the system learns

Learning from a reference implementation is an automated pipeline, not hand work:

```
intake (clone at commit, license, hash, build, run + SASS + profile)
  → decompose (agent, T2: the reference as a chain over our templates)
  → admit (deterministic; errors go back to the same session; language gaps admitted as `proposed`)
  → reproduce (our family vs. the reference's performance core, within ε)
  → ablate (toggle steps on our own family; detect coupled pairs)
  → cover (coverage update, pool snapshot, waiting cases resume)
```

Cost is tiered (L0 analysis → L1 measure → L2 reproduce → L3 ablate → L4 blind), so expensive steps run only where a new capability is at stake.

## Architecture

```
src/kta/
  core          schemas, IDs, expression language, config, context          layer 0
  store         immutable records, artefacts, snapshots, invalidation        layer 1
  targets       hardware targets and capabilities                            layer 2
  execution     remote job execution (ssh/docker workers), GPU leases        layer 3
  backends · knowledge · models                                              layer 4
  measure · formulation                                                      layer 5
  checks · solve                                                             layer 6
  orchestrator  loops, task queue, processes, budgets                        layer 7
  agents        task cards, prompts, workspaces, sessions, provider adapters layer 8
  cli · api     the `kta` command and the HTTP API                           layer 9
web/            React + Vite frontend (HTTP API only)
knowledge/      seed patterns, templates, families, references
```

A module imports only from lower layers (enforced by import-linter); checkers may not import the knowledge module, so a generator can never check itself. The control plane (API, store, orchestrator) runs locally; compilation and GPU work go through the Executor to remote workers. Every AI call goes through an adapter that persists the session and its usage; each batch gets an isolated, fully regenerable agent workspace (prompts, configuration, skills).

The frontend shows what runs now, each case's iterations, the evidence chain behind every result, coverage, the pool, the hardware model, and an inbox of things that need a human.

## Getting started

Requirements: Python 3.12 via [uv](https://docs.astral.sh/uv/) 0.11, Node (version in `web/.nvmrc`). Everything installs into the project directory.

```bash
scripts/bootstrap                 # uv sync --frozen, npm ci, data directories
export KTA_DATA=~/kta-data KTA_ENV=dev
uv run kta doctor                 # implementations in use, lock-file check
uv run kta init                   # create the store and import the seed knowledge
scripts/seed-dev                  # optional: fixture records for the UI (labelled FIXTURE)
uv run python -m kta.api          # HTTP API
(cd web && npm run dev)           # frontend at http://127.0.0.1:5173
```

GPU work needs a registered worker (`kta worker`, `scripts/worker-setup`); see `spec/modules/target-exec.md`. Quality gate for a work package:

```bash
scripts/gate --stage code WP15
```

## Status and roadmap

| | State |
| --- | --- |
| Rounds 1–3 (WP0–WP13): infrastructure, knowledge, models and solve, orchestrator, agents, API and frontend, decomposition of 7 references | merged; WP0–WP5 passed the full gate with real runs, later WPs at code stage |
| Round 4 (WP14–WP21): knowledge model v2, template effects and the declarative hardware model, transaction-level simulator, projection compiler, gap routing, FormulationPlan and access discipline, learning-loop runner, controls registry and staged release | merged at code stage (not yet run for real) |
| S0 golden path (GEMM on GB10, no agent) | next real run |
| Small parameter experiments X1–X12, then the main experiments | after S0; see `docs/design/02-design-notes.md` §11 for the order |

## Documentation

Design documents are human-readable; construction follows `spec/`. Older documents are in Chinese; new ones are in English.

| Document | Answers |
| --- | --- |
| [01 Thinking model](docs/design/01-thinking-model.md) | what we want to show, how we think about the problem, what we are careful about, decisions, experiments |
| [02 Design notes](docs/design/02-design-notes.md) | the detail behind 01: derivation, knowledge layers, template effects, hardware model, projection and formulation, loops, verification, learning; decision ledger |
| [12 Agent controls](docs/design/12-agent-controls.md) | every budget, cut-off and anti-gaming control, with the failure it prevents |
| [Change log](docs/changelog.md) | what each design conversation and work package changed |
| [00 Overview](docs/design/00-overview.md) | goals, comparisons, core concepts (zh) |
| [10 Agent system](docs/design/10-agent-system.md) · [11 Processes](docs/design/11-processes-resources-concurrency.md) | orchestration, tasks, sessions; processes, resources, concurrency (zh) |
| [20 Storage](docs/design/20-storage-retrieval.md) · [21 Admission and coverage](docs/design/21-knowledge-admission-coverage.md) · [22 Templates](docs/design/22-template-format-retrieval-composition.md) · [23 Knowledge structure](docs/design/23-knowledge-structure.md) · [24 Reference decomposition](docs/design/24-reference-decomposition.md) | the knowledge side (zh) |
| [30 Frontend](docs/design/30-frontend-and-supervision.md) · [40 Modules](docs/design/40-modules-and-code-quality.md) · [50 Experiments](docs/design/50-experiments.md) · [60 Build and acceptance](docs/design/60-build-and-acceptance.md) · [70 Related systems](docs/design/70-related-systems.md) | supervision, code structure, experiment rules, construction, related work (zh) |
| [Decisions](docs/decisions.md) · [Review queue](docs/review-queue.md) | every decision, who made it, and what awaits review |
| [AGENTS.md](AGENTS.md) · [spec/](spec/workplan.md) · [construction skills](.agents/skills/README.md) | for the coding agents that build KTA |
| [Runtime skills](src/kta/agents/skills/README.md) | skills KTA's own agents use at run time |

## Background

KTA builds on *Learning to Formulate* (Formulaic; Yi Ding and Li Shang, [arXiv:2610.39380](https://arxiv.org/abs/2610.39380); local copy in [`docs/references/`](docs/references/)), which introduced the refined-template tuple, refinement levels, walls with shadow prices and local proofs that compose. KTA keeps the template tuple, the levels and the division of labour between solver, verifier and agent, and departs from the paper where running the system argues for it — most visibly in deriving formulations mechanically from template effects and a declarative hardware model, rather than having an agent write one per case.
