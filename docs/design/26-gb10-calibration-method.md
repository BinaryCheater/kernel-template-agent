# GB10 architecture discovery and mechanistic modelling workflow

Status: proposed working method, revised 2026-11-03 from user feedback; kept as a record of the WP28-era framing. The working method now in force is [27](27-performance-model.md) (specification) and [28](28-gb10-programme.md) (programme), which supersede §§7–11 here where they differ; §5 (source intake) and §6 (investigation map) remain the reading map for the mechanism inventory. This replaces the earlier calibration-only framing. No experiments, runtime schema changes, acceptance changes or implementation completion are claimed. WP26 and the existing gates remain authoritative for implementation. This document specifies the destination and a progressive way to reach it.

## 1. Destination and KTA uses

Build a progressively white-box account of GB10, an executable mechanistic simulator, and an evidence-backed architecture reference. White-box means that relevant execution behaviour has explicit, inspectable causes and can support interventions; physical details that cannot yet be distinguished remain competing hypotheses. Architecture discovery is itself a research objective, alongside performance prediction. Total-time agreement alone does not certify the model.

KTA must use the result to:

1. Explain an implementation's performance in terms of its work, dependencies, placement and competing resources.
2. Derive labelled performance bounds and estimate achievable performance under stated constraints.
3. Compare explicit optimisation moves and configurations before expensive compilation/measurement.
4. Distinguish poor implementation, restricted Template representation, compiler lowering, an uncertain hardware mechanism and measurement interference.
5. Improve Template effects, applicability conditions and cost projections with evidence.
6. Accumulate transferable architecture knowledge, including capabilities, structure, interactions, uncertainty and unresolved alternatives.

Pipeline bubbles are one observable among many. No individual symptom defines the scope. A compact set of mechanisms may suffice for much of the workload domain; the proposed 100–200 parameters is an unproven hypothesis, not a target count. Unknown mechanisms remain visible and model growth is open-ended.

## 2. Final simulator contract

### Inputs

- Target instance and operating conditions: device identity, capabilities, clocks or clock regime, memory/cache initial state, contention assumptions and software versions.
- Kernel structure: work, dependencies, buffers, synchronisation, placement, launch dimensions, resource allocation and dynamic loop/control/address behaviour.
- L2: Template effects and schedule expanded into a common execution representation, including declared uncertainty in compiler lowering.
- L3: compiled instructions/control metadata with launch/resource metadata and sufficient dynamic execution information. Static SASS alone is insufficient for arbitrary addresses, branches or loops; use explicit analytic mappings, functional execution or captured trace according to supported scope. Record which path supplied each input.
- Versioned hardware structure and parameters, with provenance and uncertainty.

Trace-assisted L3 and predictive L3 are labelled separately. A trace of the original binary may explain it, but cannot be reused for a changed binary unless the relevant path/address invariants are checked. Hardware-observed stalls, utilisation and elapsed time are diagnostic targets, not predictors for blind prediction.

### Engine

One shared resource/dependency/state model serves L2 and L3. Retain the copied GB10 engine where useful, and audit its abstractions against this contract. Do not restart merely to match a preferred architecture. The target is event-driven, cycle-referenced execution with optional detail; cycle-referenced does not claim cycle-exact silicon reproduction.

Represent work producers/consumers, ready/completion conditions, finite resources and sharing, arbitration hypotheses, data movement, capacity constraints and placement. A coarse module has explicit input/output boundaries, state, service behaviour and assumptions. It can later split internally without destroying the surrounding dependency graph. Parameters are introduced with the behaviour they identify, not as an empty table to fill in advance.

L1 and solver projections are reduced views of the same costs and constraints. Their approximations and applicable regimes are explicit. Sampling and aggregation modes must be checked against detailed execution on representative cases and expose their own approximation error.

### Outputs

- Predicted runtime and uncertainty/domain warnings; version and input provenance.
- Execution explanation at selectable granularity: work, traffic, utilisation, queueing, readiness, dependency waits, synchronisation, residency, placement and startup/tail behaviour.
- Critical constraints and sensitivity to feasible implementation changes; distinguish service demand from contribution to wall time.
- Bound reports, achievable schedule estimates, and a gap account with unresolved residuals.
- Counterfactual predictions, assumptions changed, and whether the change is achievable in software or is a hypothetical hardware change.
- Architecture support map: observed, documented, inferred, fitted, assumed, contradicted and unsupported claims.

Every observable declares definition, units, scope and aggregation. A simulated queue timeline is labelled predicted; it is not a measured hardware trace. Counter names resembling ncu names do not establish semantic equivalence. Overlapping costs cannot be blindly added; sensitivity deltas are not generally additive and may depend on intervention order.

## 3. Bounds and explicit optimisation

Maintain distinct products:

- Resource lower bounds: necessary work divided by justified resource ceilings, with declared traffic assumptions.
- Dependency/capacity lower bounds: valid relaxations retaining selected precedence, recurrence and storage constraints.
- Restricted-design bounds: bounds for a specified Template/configuration space, not universal algorithm bounds.
- Achievable estimates: simulated or measured schedules/configurations demonstrating a concrete design.

A simulator result is not automatically a lower bound. Every bound records its domain, relaxation, formula, units, parameter uncertainty and justification. Optimistic hardware ceilings must be distinguished from fitted sustained rates; empirical rates cannot silently become proven peak limits. Counterexamples invalidate a bound's claim and trigger review.

For a frozen representation and justified model, report a bracket only when both sides are defensible. Show how broadening a Template or changing an implementation changes the bracket. Do not present the fastest tested candidate as a global speed of light. Translate uncertainty in key mechanisms into uncertainty in headroom.

An optimisation proposal contains: observed symptom; mechanism explanation; proposed code/Template change; predicted effect; correctness constraints; expected side effects; real A/B verification. Benefits inferred from a hypothetical faster memory channel are diagnostic sensitivity, not an implementable optimisation.

## 4. Current implementation starting point

As inspected on 2026-11-02, `models/l2.py` is the earlier CTA transaction graph with FIFO resource reservations and sequential resident waves. The copied `models/gb10sim/` includes SASS/SM and whole-kernel traffic/grid logic. Its `simulate_build` entry maps TileLang cp.async GEMM and explicitly rejects TMA and other unsupported mappings. `gb10.json` mixes scalars, curves and labelled parameter sources; assumptions also exist in code. Existing UnitModel fitting is a separate layer. Re-audit the checkout before implementation: these observations are not a permanent capability declaration.

WP26's shared-engine direction remains the foundation. Compare L2 vs L3 to audit translation; compare L3 vs GPU to audit inputs/mechanisms; compare L2 vs GPU for end-to-end usefulness. Agreement between two readers of one engine is not independent physical validation.

## 5. Using the SM120 reference and official sources

Reference index: https://zartbot.github.io/micro_arch/nvidia/sm_120/index.html

The site identifies the measured device as RTX PRO 5000 72GB/GB202, and says its report was AI-generated without author review. NVIDIA lists GB10 as compute capability 12.1 and the workstation GB202 family as 12.0. Common family membership motivates transfer hypotheses; it does not establish identical topology, rates or memory behaviour.

Reading status: all 20 chapter pages were retrieved and their section structure inspected. Detailed passages were inspected in architecture, memory/SM, performance model, porting, agent, cross-reference, coverage and methodology material. This is not a claim that every paragraph, diagram, referenced benchmark source or raw result was audited. Source paths named by the site are not equivalent to accessible source artifacts. Full source-level reproduction remains an explicit task.

Use this complete chapter routing map as an intake plan:

| Chapter | Reusable contribution | GB10 action |
|---|---|---|
| 01 architecture | Capability/topology investigation and resource limits | Query target, check official constraints, probe sharing only when relevant |
| 02 memory | Cache, coalescing, shared-memory, ordering and atomic experiments | Re-establish target-specific hierarchy and response curves |
| 03 SM | Resource/issue/operand/dependency hypotheses | Compare competing structures using controlled instruction mixes |
| 04 TensorCore | Formats, operand supply, independent accumulators and saturation | Check supported instructions and coupled supply/compute behaviour |
| 05 instructions | Latency/throughput and lowering distinctions | Measure instruction families encountered in workloads; expand by gaps |
| 06 async/network | Completion, synchronisation, copy and inter-CTA experiments | Check capability, then timing and interaction; retain unknown topology |
| 07 scheduler | Readiness, ILP/TLP, control flow, instruction-cache tests | Infer scheduling behaviour without assuming a unique physical scheduler |
| 08 workloads | Broad shape/regime sweeps and workload decomposition | Build development and independent grouped evaluation cases |
| 09 power | Operating-state and frequency experiments | Control/observe runtime conditions; detailed energy prediction is a separate extension |
| 10 system | Launch, placement, waves, transfer and multi-device boundaries | Prioritise kernel/host-memory boundary; keep multi-device scope explicit |
| 11 optimisation | Candidate intervention recipes | Treat recipes as hypotheses with correctness checks, not universal rules |
| 12 compiler | SASS verification, allocation, scheduling and measurement artifacts | Pin toolchain and distinguish lowering from hardware behaviour |
| 13 model | Resource accounting and composition questions | Use as baseline ideas; derive dependency/state semantics and bounds independently |
| 14 porting | Feature gating and migration checklist | Make an SM120-to-GB10 applicability matrix; no unverified value copying |
| 15 agent | Queryable evidence and experiment/optimisation workflow | Connect to KTA records, Templates and task contracts |
| 16 questions | Follow-up experiment queue | Preserve alternatives, falsifiers and unresolved conditions |
| 17 cross-reference | Claim/evidence dependency graph | Track changes and invalidate dependent claims |
| 18 coverage | Test inventory and failed-case accounting | Separate test execution success from model/architecture coverage |
| 19 parameters | Machine-readable export | Generate export from canonical evidence; never maintain parallel hand copies |
| 20 methodology | Timing, compiler controls and evidence conventions | Reproduce harness on GB10; separately assess systematic uncertainty |

Borrow experimental designs and explanatory structure aggressively. Admit numeric constants as documented constraints, locally measured values, or explicitly provisional cross-chip priors. A derived quantity must include its formula and source operands. Small replication variance does not eliminate systematic error or prove physical topology.

Potential inconsistencies found in inspected material (for provenance hygiene, not a requirement to repair the external report): index says .reuse is a no-op while chapter 03 KI-3.2 reports ALU-specific activity; chapter 13 and chapter 02 report different ldmatrix timing contexts; chapter 18 and index use different case totals. Resolve definitions/version lineage before importing these particular claims. Do not delay GB10 construction to reconcile all external prose.

Official source lanes:
- Device applicability: https://developer.nvidia.com/cuda/gpus
- Architecture: https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- Instruction semantics/target requirements: https://docs.nvidia.com/cuda/parallel-thread-execution/
- Binary inspection: https://docs.nvidia.com/cuda/cuda-binary-utilities/contents.html
- Counter/replay semantics: https://docs.nvidia.com/nsight-compute/ProfilingGuide/
- System timeline: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Library dispatch/environment: https://docs.nvidia.com/cuda/cublas/index.html

Official documentation establishes semantic/architectural constraints where stated; microbenchmarking estimates operational behaviour. Papers, patents and third-party reports propose mechanisms and discriminating experiments. A patent's existence does not prove that GB10 implements it.

## 6. Architecture investigation map

Maintain an open map of modules, connections, sharing scopes and behaviours, with an `other` category and counted unknowns. It is a question map, not a prerequisite list of all parameters.

| Investigation lane | Questions and controlled probes | Useful observations |
|---|---|---|
| Dispatch/placement | Grid sizes around residency/wave boundaries, unequal CTA durations, placement logging | Runtime scaling, active SMs, replacement/tail behaviour |
| Issue/dependencies | Independent vs dependent chains; vary instruction ratio, warp count, ILP and control code where supported | Issued/executed work, dependency latency, saturation and interference |
| Compute/operand supply | MMA/ALU/SFU/conversion alone and mixed; vary accumulator chains and operand reuse | Service demand, supply limitation, shared bottlenecks |
| Registers/residency | Allocation boundary sweeps with matched work; inspect spill instructions | Resident work, spills, discontinuities and compiler changes |
| SM memory | Bank/address patterns, vector width, shared-memory allocation, load-to-use distance | Transactions, conflicts, capacity and response curves |
| Chip memory | Working-set/reuse-distance/stride/read-write sweeps, concurrency and CPU traffic | Actual traffic, cache regimes, latency under load and bandwidth |
| Async/sync | Copy sizes and in-flight groups; consumers at controlled distances; producer/consumer imbalance | Issue vs completion, backpressure, ordering and completion gates |
| Control/instruction supply | Loop length, unroll footprint, masks/branches with matched executed work | Dynamic path, instruction pressure and reconvergence costs |
| Environment | Stable/burst execution, warm/cold state, quiet/controlled contention | Clock/temperature/frequency regime and repeatability |

Two queues coexist: workload-driven unexplained behaviour and architecture-driven unresolved questions. Selection considers KTA decision impact, architectural significance, uncertainty, discriminating power and experiment cost. Architecture research is not limited to currently failing benchmarks; report the purpose of each experiment and keep both queues visible. Allocate their effort in the run manifest, not through an invented permanent percentage.

## 7. Progressive construction

| Stage | Build and measure | New capability | Exit evidence |
|---|---|---|---|
| P0 contract and evidence | Audit current code, source applicability and available observation tools; define supported workloads and target outputs | Shared problem definition and traceability | Capability map, reading/source ledger, proposed evaluation manifest |
| P1 executable coarse model | Run representative compute-, memory-, dependency- and mixed-regime kernels; map work/dependencies onto coarse resources | Explain execution structure, issue first labelled bounds and predictions | Real runs; observable definitions; broad assumptions and limitations recorded |
| P2 targeted refinement | Select unexplained behaviour or architecture question; distinguish hypotheses; split only implicated modules | More informative architecture and counterfactual explanations | A refinement packet with real intervention and held-out development checks |
| P3 composition and breadth | Combine mechanisms, extend input adapters and libraries; validate L2 lowering against L3 | Transfer across implementations/workloads; detect interactions | Grouped workload/regime coverage, interaction regressions and mapping evidence |
| P4 KTA use | Connect costs/constraints to solver, Template selection, headroom and optimisation proposals | Explicit optimisation driven by architecture evidence | Successful and failed predicted interventions; bound checks; representation-gap examples |
| P5 independent release | Freeze model, run independent evaluation, generate reference/model card | A reproducible validated release with clear scope | Accuracy + mechanism + optimisation + coverage results and applicable gates |

These are capability milestones, not a waterfall parameter campaign. Begin P1 before P0's architecture questions are exhausted. P2 repeats throughout P3/P4. Early releases can declare partial capabilities; the final destination cannot be weakened because an early version predicts time reasonably well.

### Refinement packet: the unit of work

1. Name the KTA decision or architecture question and the current model's limitation.
2. Preserve the coarse baseline and its predictions before seeing new measurements.
3. Give competing explanations, including input-mapping and measurement artifacts.
4. Specify the smallest discriminating experiment family: controlled variables, expected different outcomes, correctness and compiler checks, valid measurement regime.
5. Collect real measurements plus selected counters and binary metadata; reject contaminated runs explicitly.
6. Choose the least detailed structure consistent with evidence; retain observationally equivalent alternatives and uncertainty where not identifiable.
7. Fit only parameters introduced or implicated by this structure. Analyse sensitivity/confounding; do not freely move all constants.
8. Re-run probes, composition cases and held-out development interventions. Show which new decision or architecture statement became possible.
9. Version the mechanism, parameters, assumptions and dependent claims. Retire overlapping correction terms. Keep a rollback parent.
10. Close as supported, falsified, unresolved, or blocked by a named observation/tool capability. Budget exhaustion is not success.

Example: a tile-transfer stage may initially expose a completion delay and a throughput limit. An unexplained response to concurrency can justify splitting issue, finite admission, service and completion. Another question may require cache state, operand supply or placement instead. No refinement type is mandatory merely because it appears in the reference.

## 8. Measurement protocol

Use KTA_ENV=dev and Executor for GPU/heavy CPU work; manual construction access follows remote-run. Evidence uses clean matching commits, environment manifests and Store records/blobs. User-authorised boundaries for system changes, credentials, spending and push remain unchanged.

Before timing: correctness; pinned input/seed; actual compiled SASS/resources; supported target; explicit timing boundary; warmup and cache/memory state; quiet-host lock; clock/thermal observations. Fixing clocks is optional and capability/authorization dependent; stable measured regimes are acceptable if recorded. Never change host configuration implicitly.

For latency, use dependent chains whose dynamic values reach a checked sink. For throughput, increase independent work until saturation and ensure completion is timed. Measure multiple chain lengths to separate intercept from slope; inspect timing instrumentation and loop costs. A fast issue rate does not prove fast completion. For capacities, combine working-set cliffs with independent observables and alternate patterns; a cliff alone does not prove associativity or queue size.

For interaction, compare A, B, A+B with matched work and vary mix/concurrency. Randomise/interleave run order where valid; repeat in independent windows and retain distributions. Inspect SASS after each change: register allocation, spills, unrolling or instruction substitution may confound a causal claim. Binary patching is an optional advanced probe with explicit correctness checks and capability support, never a prerequisite for ordinary calibration.

Keep ordinary timing separate from profiling. Record ncu version, metric availability, replay passes, cache and clock policy, serialization and metric definitions. Nsight Systems helps determine launch/sequence boundaries; Nsight Compute adds selected workload/resource observations. Missing counters are unavailable, not zero. Samples of stall states are not directly additive wall-time losses. Full hardware timelines are not promised when tools expose only aggregate measurements.

## 9. Evaluation without target reduction

Use A identification probes, B calibration/composition kernels, C development validation and D independent frozen audit. Split by implementation/shape/regime groups before fitting and deduplicate lineage. Once used to diagnose and select experiments, data is development evidence. Audit access is enforced by scorer/storage boundaries; prompt instructions alone are insufficient. Freeze submission budgets and retire disclosed diagnostic audit cases into regression data.

Maintain four jointly required scoreboards:

- Predictive: per-stratum signed/absolute errors, tail errors, short-kernel absolute error, uncertainty calibration and candidate-selection regret against the measured candidate pool.
- Mechanistic: work/traffic conservation, resource legality, dependency/ordering consistency, observable agreement and distinguishing interventions. Separate internal invariants from external validation of silicon behaviour.
- Optimisation/bounds: validity of labelled relaxations, empirical bound counterexamples, predicted vs measured move effects, and useful reductions in implementation or representation gaps. A counterexample test cannot by itself prove a bound universally.
- Architecture/coverage: supported workloads and mechanisms, strength and applicability of evidence, unresolved alternative structures and progress in answering architecture questions.

Set thresholds per capability/domain after a repeatability pilot and before fitting. Numeric runtime error alone cannot promote an unvalidated causal explanation. A bounded residual can represent an unresolved module but must preserve observable interfaces and dependencies; it cannot erase a required architectural explanation. Residual features must be available before prediction and cannot be case IDs or measured runtimes.

Library tests span representative API/algorithm regimes and unique implementations. Establish real vendor-kernel input capture first; a failed mapping stays in the unsupported denominator. An open implementation is a useful counterpart, not evidence that the vendor binary was simulated. Kernel time and multi-kernel operator/host latency are separately scored.

## 10. Durable outputs and AI handoff

Canonical records: source claim, mechanism hypothesis, experiment, run, parameter/curve, model version, prediction, bound and intervention evaluation. Reuse existing KTA schemas where possible; new cross-module interfaces require the existing spec process.

Generate from these records:

1. An architecture reference resembling the SM120 report, with module diagrams, measured behaviour, alternative explanations and citations.
2. Machine-readable structure/parameter exports consumed by the simulator.
3. Reproducible probe and workload suites, separate calibration/development/audit manifests and raw evidence.
4. Executable L1/L2/L3 model paths and input adapters, with traceability to supported mechanisms.
5. Per-kernel explanation and counterfactual reports, labelled bounds and measured comparisons.
6. A model card, coverage/gap queue, change impact graph and release evidence.

Avoid hand-maintaining conflicting prose/JSON parameter copies. A changed claim invalidates dependent parameters, model assertions and report sections for re-verification. Keep historical observations; changes supersede records rather than rewriting evidence.

AI execution instruction:

> Read this protocol, AGENTS.md, conventions and the assigned WP. Preserve the final architecture-discovery and mechanistic-analysis objectives. Inventory actual capability first, then establish an executable coarse model on real kernels. Work in refinement packets driven by KTA decisions and architectural questions. State proposed explanations and predictions before measuring. Introduce parameters with mechanisms. Reuse official constraints and applicable reference experiments, recording transfer assumptions. Preserve dependency/resource structure in approximations. Report unsupported paths and uncertainty. Validate predictions, causal interventions, bounds and architecture claims separately. Never replace the requested mechanism capability with a time-fitting surrogate. Produce evidence-backed reports and applicable full gates before claiming implementation delivery.

The first implementation planning task is P0 plus a bounded P1 pilot, with a representative compute kernel, memory kernel and mixed kernel. The existing cp.async GEMM is a useful first vertical slice, not the final scope. Budget, workload coverage and acceptance numbers must be concretised in that task; do not launch an unbounded benchmark campaign from this design conversation.

## 11. Design interrogation, defaults and obligations

| Question | Answer |
|---|---|
| What is it? | Progressive architecture discovery and a shared mechanistic simulator, supporting explicit optimisation. |
| Claim served | Inspectable mechanisms improve KTA explanations, bounds, decisions and architecture knowledge. |
| Dependencies | Existing engine and partial adapters are observed; sufficient dynamic inputs, identifiability and transferable abstractions remain unproven. |
| Scope | GB10 and recorded operating/software regimes; extensions are explicit, final capability goals preserved. |
| Failures | Time-fitting substitutes for mechanisms; parameter-first exhaustion; incorrect dynamic input; counter ambiguity; invalid bounds; leakage; report drift. |
| Detection | Four scoreboards, intervention packets, source/reader mapping, access isolation, counterexample and consistency checks. |
| Knobs | Manifest controls below; no new numeric values active in runtime. |
| Repair | Experiments refine or falsify structure, version parameters and regenerate dependent outputs; rollback failed updates. |
| Authority/cost | User fixes destination and progressive approach; implementation defaults remain reviewable. New interfaces/acceptance follow project governance. |

| Default | Alternative | Reason | Reversal cost | Status |
|---|---|---|---|---|
| Event-driven, cycle-referenced shared engine | Separate models; full cycle-stepping everywhere | Common explanations with adjustable cost | Engine/interface work | Proposed |
| Coarse modules with inspectable boundaries | Fill a full parameter catalogue first | Supports progressive identifiable refinement | Local module refactor | User-directed principle |
| Both architecture and workload question queues | Only fix benchmark errors | Preserves discovery as a goal | Planning/coverage change | User-directed principle; allocation proposed |
| Four evaluation scoreboards | Runtime-only admission | Preserves causal/bound/architecture uses | Evaluation infrastructure | Proposed |
| Evidence-generated reference and parameter exports | Separate manually maintained documents | Prevents contradictory copies | Export tooling | Proposed |
| Grouped independent audit | Random near-neighbour holdout | Tests transfer and limits leakage | New cases/access separation | Proposed |

Research obligations: dynamic input sufficiency (real mapping pilot); abstraction transfer (held-out implementations); causal attribution (discriminating interventions); bound validity (explicit relaxation argument and empirical counterexamples); practical model cost (accuracy/cost curve); architecture identifiability (competing hypotheses and unresolved outcomes); AI reproducibility (independent rerun from stored packet); source completeness (finish all-paragraph/diagram review and obtain benchmark sources when accessible). No obligation is claimed proved.

Manifest knobs: domain/regimes; minimum evidence per claim class; timing repeatability/rejection; fit bounds/residual scope; runtime and mechanism thresholds; bound assumptions/uncertainty; supported coverage floors; audit access/submission limits; simulation detail/cost; experiment budgets; effort between architecture and workload questions; stopping/escalation rules. Unset until the pilot; existing KTA controls remain in force. A stopped round records unresolved work and cannot redefine completion.
