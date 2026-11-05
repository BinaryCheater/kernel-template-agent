# Template-driven modelling, MIP search and deterministic generation

Status: first pilot run; the decision-level form below supersedes the L2 proposal. The nine-session
attention campaign remains evidence about agent-authored empirical tuning.

## Current approach: decision-level form

This section replaces the proposed L2 below as the route to a MIP. The text after it records the earlier proposal. Results: [decision-form pilot](results/decision-form-pilot.md). Code: `tools/kta_bridge/`.

The boundary for a MIP is state: sums, maxima and finite tables export exactly, while an event order that depends on queues does not. So the searchable level holds no queue state. Each mechanism L3 established becomes one of three things:

```text
T(x) = launch + stores + stage + turnover + barrier + trip
     + max(compute(x), l2_copies(x), dram(x), recurrence(x))
```

| Part | Source | How it is obtained |
|---|---|---|
| Amounts: tensor products, scalar work, bytes, blocks, trips | Decision variables and shape | Closed expression per family, from the generator |
| Regimes: occupancy, warps per tensor pipe and scheduler, register spill by share | Walls and mechanisms found by L3 | Function of decisions, one table entry each |
| Hardware rates | `gb10.json` | Copied, frozen, no kernel fit |
| Software constants: scalar cost, stage, turnover, trip, spill | Few rows of the family | Deterministic fit, against GPU or L3 |

Order of preference for a term: derive it, then calibrate it on a few rows, then leave it to an agent. Compilation supplies only register records, which replace an estimate before GPU runs. L3 remains the source of mechanisms and a possible reranker, and is no longer an input to the search.

The funnel: MIP shortlist of distinct structures → compile → rerank on records → measure a few → refit → solve again with measured points cut out.

## Objective and sequence

Demonstrate that a bounded set of Template transformations and backend
realisations can define a legal implementation space, supply its costs, and
deterministically generate implementations selected by a real MIP solver.
Assess search quality and preparation cost separately from agent coding skill.

Start with the already measured attention workload to validate the chain. Then
use an already studied operator involving quantization, fusion or irregular
boundaries to investigate transfer and knowledge benefit. No new source-study
campaign or difficult synthetic challenge is needed to validate the mechanism.

The first scope is a fixed explicit transformation chain with genuine parameter
choices. A later bounded extension introduces alternative transformations or
chains. Parameter search within a family and automatic discovery of a
composition are separate claims and receive separate results.

## Current evidence and gaps

- The current canonical file library has 312 entries. A local scan found 63 with
  nonempty transaction, storage or overlap effects. This is a coverage inventory,
  not evidence that those effects are complete or accurate.
- Abstract programs are prose by schema. Backend implementations belong to
  `Realisation`, not the abstract Template. The file-library export by itself is
  not a collection of executable, composable implementations.
- Empty effects are legitimate for a purely semantic transformation if all work
  it introduces is priced by subsequent steps. Audit whole-chain accounting:
  missing and double-counted work must both be visible.
- `formulation.assemble`, the projection compiler, `solve.export.check_export`
  and family `generate(assignment, case)` interfaces already exist. Their support
  for the new family remains to be demonstrated; source presence is not a runtime
  result.
- The convenience `solve` entry can select enumeration or fall back from a
  projection that cannot be exported. A MIP experiment must record actual method,
  exported choices, outer enumeration, status, bound and gap. An enumeration-only
  run does not become a MIP result.
- The existing TileLang cubin launcher disables TMA lowering and warp
  specialization. Use the previously exercised direct TileLang execution path
  for the pilot if necessary; do not silently narrow the implementation space
  to accommodate that launcher.

Source pointers: `src/kta/solve/{select,projection,export}.py`,
`src/kta/formulation/assemble.py`, `src/kta/orchestrator/pipeline.py::generate`,
`src/kta/backends/tilelang/gen.py`, and `spec/schemas.md` Template/Realisation.

## The L3-to-L2-to-MIP bridge is a first task, not a given

The user's follow-up identifies a missing prerequisite: recent GB10 instruction
model calibration does not automatically establish a predictive precompile L2
model or a faithful search objective. Existing L2/projection source is reusable
implementation material; a current calibrated end-to-end mapping is not shown.

The old `models/l2graph.py` averages whole-kernel effects across CTAs and trips,
then collapses work into a small fixed phase graph. The current detailed GB10
path uses SASS, launch/resource information, address/control reconstruction,
`gb10sim/machine.yaml`, `gb10.json`, and mechanism handlers. Copying calibrated
bandwidth and latency values into the old graph cannot recover lost dependencies,
heterogeneous block work, block ordering, or compiler lowering behavior.
`gb10sim/l2.py` models the hardware L2 cache; its name does not mean the abstract
model-fidelity level discussed here.

### The missing object is an implementation contract

The purpose of the detailed hardware work is useful precompile prediction and
search. Reverse engineering is a means to that end, not a separate completion
criterion. The bridge is not a general inverse compiler from concrete SASS to a
parameterized program. A concrete binary has already bound parameters and lost
the provenance of many choices; multiple source families can produce the same
observed instructions.

Bind the abstract Template chain to a restricted, parameterized implementation
contract: operations, explicit schedule/layout/storage choices, legal parameter
domains, deterministic code-generation rules, and a predicted execution
description. Use the same bindings for code and model. Initially this can be a
small generator plus structured metadata rather than a new compiler or core IR.
Generate metadata where the generator chooses a layout or pipeline; avoid two
independent handwritten descriptions of the same decision. This is a joint
authoring discipline, not proof that the descriptions match the compiled result.

An agent may author and repair that implementation and its mapping. Each search
uses a frozen version, rather than asking an agent to reinterpret the winning
parameters as new source. A repaired generator or model creates a new version
and invalidates the affected validation; it is not a hidden per-candidate patch.
Composition requires compatible storage, layouts, thread roles and synchronization
across realisations, not just compatible mathematical input/output shapes.

| Deterministic property | How it is obtained | What it does not establish |
|---|---|---|
| Assignment to source and launch | Frozen parameterized generator and tool settings | Its resource use or execution time |
| Declared execution description to L2 | Fixed service rules and scheduling semantics | Compiler conformance or hardware fidelity |
| Supported symbolic cost to MIP | Mechanical translation with direct-evaluation checks | That the cost predicts real time |
| Compiled implementation to timing prediction | Fixed L3 inputs and model | Exact agreement with hardware |

TileLang and CUDA expose useful implementation controls, but language choice is
not a conformance guarantee. Pin the compiler path and identify which decisions
are explicit and which remain inferred. In particular, register live-value
estimates and register limits are not actual allocation; register-bank behavior
and shared-memory layout conflicts are different quantities. Use compile
metadata/SASS to check register allocation, spills, instruction selection and
barriers, without requiring a new compiler research project.

Primary references: [CUDA register pressure and bank conflicts](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#registers),
[TileLang inferred-layout inspection](https://www.tilelang.com/tools/layout_visualization.html),
and [TileLang generated-code/compiler inspection](https://www.tilelang.com/tutorials/debug_tools_for_tilelang.html).

The proposed common boundary is a parameterized execution description:

```text
problem + Template transformations + backend realisation rules
    -> parameterized operations, dependencies, loops, buffers and placement
         -> compressed L2 execution
         -> solver projection / MIP
         -> deterministic generation and compilation -> detailed SASS L3
```

It should retain the distinctions relevant to the chosen search variables:
operation/resource class and amount; data and synchronization edges; producer
completion and buffer-reuse conditions; repeated-loop distances; address/layout
and reuse descriptors; thread roles and resource allocation; CTA classes with
their work and launch order; startup, steady loop, boundary and drain regions.
Semantic-only transformations modify this description rather than inventing an
extra hardware cost for every Template step.

For attention, QK, row max/exp/sum, output-state rescaling and PV are distinct
dependencies inside the repeated loop. Different query tiles have different
causal key ranges. Scheduling them in different orders must remain expressible.
The retained run-08 v14/v15 sources differ in the mapping from blocks to query
tiles and heads, while leaving the logical arithmetic unchanged. They are a
concrete translation/aggregation check; their timing difference does not by
itself identify which hardware mechanism explains it.

### Obtain L2 by controlled compression and validated lowering

L2 is a fast executable model of this smaller program, not just operation counts
divided by peak throughput. Reuse calibrated resource service rules, sharing and
completion semantics from L3 where possible. Compress repeated instruction
sequences into validated macro operations and repeated loops into summaries;
retain dependencies, capacity transitions, startup/drain and heterogeneous work.
A separate coarse scheduler is acceptable initially with explicit comparison
tests; a full simulator rewrite is not an entry requirement.

There are two input mappings to establish:

1. From existing compiled kernels, construct compressed descriptions using
   instruction/control/address information, and compare L2 with L3. This isolates
   aggregation error. Do not copy measured or simulated wall-time waits into L2
   and then treat the match as a predictive validation.
2. From Template/Realisation choices, predict those same descriptions before
   compiling a candidate. Compare with the compiled mapping on new configurations.
   Passing the compiled-input comparison does not establish this second mapping.

Maintain versioned compiler-lowering profiles per supported realisation and
configuration regime. Compile a limited set of deterministic witnesses; inspect
instruction mix, actual copy/math paths, unrolling, layouts, register/shared
allocation and spills. Derive parametric rules or small local tables for what is
stable. Keep discontinuities explicit and compile uncertain boundary candidates.
Profiles depend on the target, backend/compiler versions and generation flags.

Compiler behavior can be deterministic under fixed conditions while still being
unknown before compilation. Treat stable regions with parametric rules, observed
resource-transition regions with local profiles, and unresolved sensitive
regions with targeted compile checks or separate alternatives. A few witnesses
cannot prove absence of discontinuities throughout a large domain. Do not
replace a spill or occupancy transition with one smooth average efficiency.

The hardware mechanisms and parameter evidence have one canonical lineage.
Compiler profiles describe how software uses that hardware; they do not redefine
its rates. Reusing numerical constants alone while independently reimplementing
incompatible contention rules is not shared modelling.

### Obtain the MIP from the same description, with labelled approximation

Fixing some decision variables is partial evaluation; it does not make arbitrary
simulation code linear. A fixed event order with supported symbolic durations
can admit an exact max-plus/precedence encoding: arrival at an event is at least
each predecessor's completion plus its duration. If resource order, loop counts
or cache/control decisions change, preserving those choices can require binaries,
piecewise cases or a much larger formulation. An optimizer allowed to choose an
unrealistic resource order models a different machine even if its equations solve
exactly. Enumerating every whole-kernel configuration into a table is possible on
a finite domain, but shifts the original search cost into model construction.

Use a restricted symbolic cost representation for the parts exported to MIP;
keep numerical L2 execution for behavior outside it. Mechanical compilation is
exact relative to the declared cost fragment, not automatically relative to the
full L2 simulator. Record every projection/approximation explicitly.

Do not unroll the whole GPU event simulator into one scheduling MIP. Project
necessary resource work, critical dependency/recurrence constraints, buffer and
residency limits, and supported startup/tail/order estimates into a small search
problem. Compute simple quantities algebraically; use one-hot choices and local
tables for nonlinear or compiler-dependent terms that read a few variables.
Outer-enumerate a limited set of structural regimes when needed, recording it.
The current projection compiler already contains menu/table and outer-enumeration
machinery; its physical terms and their scope still need the bridge above.

Keep a justified optimistic bound separate from a useful time estimate. Empirical
corrections or simulated local tables can improve ranking without becoming
proven hardware lower bounds. A whole-space table of measured final kernel times
would move the tuning cost offline and does not establish predictive MIP search.

Use a funnel: MIP generates promising and structurally diverse assignments, L2
reranks them, a smaller set is compiled and assessed by L3, and GPU measurements
validate the final choices and controls. If distinct schedules tie in the coarse
objective, keep representatives rather than discard them by arbitrary tie-break.
Measure candidate recall: later reranking cannot recover a good variant already
removed by the first stage.

### Separate validation boundaries and start from existing evidence

| Comparison | What disagreement tests |
|---|---|
| Exported MIP versus direct evaluation of its projection | Algebraic translation and solving |
| Projection/MIP scores versus L2 | Search simplification, omitted dependencies/regimes, ranking and shortlist recall |
| Template-predicted L2 versus compiled-input L2/L3 | Compiler-lowering assumptions and aggregation, checked separately |
| L3 versus GPU | Detailed execution inputs, hardware mechanisms and measurement conditions |
| L2/selected candidates versus GPU | End-to-end precompile usefulness and selected candidate's loss |

Start with a retained variant group that already has source/SASS, launch metadata
and measurements. Freeze the L3/hardware version for the bridge comparison. Add
or change one representation/approximation at a time, then evaluate on a new
configuration/shape. Prioritize preserving decisions that matter: whether the
best implementations survive screening, which interventions improve performance,
selected-candidate loss, and modelling/compilation/search cost. Shared-engine
agreement is not independent validation of hardware accuracy.

A compile-then-L3 search can be a convenient initial control on a small space. It
must be labelled postcompile selection and cannot substitute for demonstrating
the Template-to-precompile-L2 mapping. This work can stay in the thin offline
pilot; it does not require frontend or full runtime orchestration.

### Accuracy follows preserved decisions, not the level number

No current comparison establishes a numerical accuracy guarantee for the proposed
precompile L2. A coarser model can approach detailed-model accuracy on a bounded
family when the omitted distinctions have little effect on its decisions. If
two implementations receive the same L2 description but have substantially
different timings, no improved coefficient on that description can distinguish
them. Refine the missing feature, narrow the supported domain, or carry the
uncertainty into candidate selection.

Track three error sources independently: implementation/compiler mapping,
execution aggregation, and solver projection. Evaluate absolute time error,
pairwise decision changes, shortlist recall and selected-candidate slowdown
against the best measured point in the declared validation pool. A low mean
prediction error can coexist with a bad chosen optimum because optimization
systematically selects optimistic mistakes. Check the proposed optimum and the
regimes surrounding it, not only representative average cases.

For a declared domain, a uniform time-error bound of epsilon microseconds would
bound the loss of an exact model minimizer by 2*epsilon microseconds. Sampled
average errors do not establish this uniform bound. If a projection first
discards candidates, its lost-candidate error must also be accounted for.

Distinguish reusable hardware service parameters, compiler-lowering profiles and
family-specific empirical performance corrections. The last can be practical and
may be fitted from designated training kernel measurements; it is an empirical
software/model correction with a declared domain, not a new measured hardware
constant or a proven lower bound. This permits a hybrid model without claiming
that every coefficient was analytically derived. Whole-kernel times for every
configuration would erase the intended predictive saving.

### A bounded construction loop with explicit cost

1. Choose one retained successful implementation and turn it into a real
   parameterized generator. Use its actual TileLang/CUDA compilation path. Fix
   the algorithm while exposing a small set of genuinely changing decisions.
   Earlier independently authored kernels are development evidence, not samples
   automatically belonging to this one controlled family.
2. Emit an execution description from the same choices. Derive work, buffer sizes,
   dependencies and address/layout rules symbolically where possible. Import
   calibrated hardware service rules; isolate unknown compiler-dependent terms.
3. Reuse relevant retained artifacts, then choose compile witnesses to cover
   structural alternatives and predicted resource transitions. Compilation can
   supply register/spill/instruction metadata without a timed GPU run. Choose GPU
   calibration points for unresolved performance terms and interactions; record
   compilation, L3 and measurement cost separately.
4. Fit only declared low-dimensional unknowns or local regimes. Identify which
   observations constrain each coefficient and whether they distinguish competing
   mechanisms. Sample count follows identifiability and uncovered interactions,
   not a claim that a fixed handful of points calibrates any family.
5. Freeze a model version, mechanically export its supported symbolic terms, and
   search. Validate selected points and structural controls. Development feedback
   may trigger an explicitly budgeted, versioned refinement loop; final held-out
   evaluation is separate and cannot be recycled as untouched validation.
6. Compare total effort and achieved selection quality with same-family random
   or empirical tuning under matched end-to-end budgets. Count authoring/tokens,
   compile/profile/GPU jobs, L3 calls, coefficient estimation, solve time and
   failed candidates. Report both cold start and reuse across new shapes/cases.

The search is useful if this work saves more downstream candidate evaluation
than it costs, or improves the chosen implementation under a comparable total
budget. A faster solver alone is insufficient. If compiler regimes or correction
tables proliferate toward whole-space coverage, stop expanding that formulation
and report the cost/coverage failure rather than call it successful prediction.
Numerical budgets, family menus and acceptance criteria remain unset here.

### Implementation-contract interrogation

| Question | Proposed answer |
|---|---|
| Definition | A bounded parameterized implementation with linked generation and execution-description rules, plus explicit compiler-sensitive unknowns. |
| Claim | Predictive search can reuse calibrated hardware knowledge without requiring full compilation/timing of every assignment or a new model-written kernel after solving. |
| Dependencies | A working retained kernel and current model are inputs; parameter-faithful generation, low-dimensional compiler regimes and useful selection remain obligations. |
| Scope | One fixed backend/toolchain and implementation family first; composition requires compatible realization contracts. |
| Failure | Hidden inferred choices, missed allocation/spill cliffs, compensating fitted constants, optimizer exploitation, or preparation as expensive as empirical tuning. |
| Detection | Compiled conformance witnesses, held-out structural contrasts, selected-point error/recall, and total-cost comparisons. |
| Controls | Declare generator/model versions, explicit versus inferred choices, calibration/holdout roles, adaptation budget and correction domains; no numerical defaults yet. |
| Repair | Diagnose mapping, aggregation or projection separately; version the affected rules, collect targeted evidence and revalidate the changed domain. |
| Choice and cost | Proposed hybrid mechanical/empirical approach over general inverse compilation, compiler reimplementation or unconstrained agent-written MIP. Cost is explicit authoring and sparse profiling; abandon the expansion if it approaches exhaustive evaluation cost. |

### Bridge design interrogation

| Question | Proposed answer |
|---|---|
| Definition and claim | A versioned mapping from Template/realisation decisions to compressed execution and a solver projection, supporting useful precompile selection. |
| Dependencies | Current detailed engine, calibration provenance and existing compiled variants are source/artifact inputs. Faithful symbolic lowering and small accurate summaries remain unverified. |
| Scope | One supported realisation family and its declared compiler/configuration regimes first; no arbitrary-SASS-to-symbolic-model claim. |
| Failure modes | Equal-work schedules collapse; old and new hardware rules drift; spills are hidden; tables consume the full search space; an approximate estimate is called a bound; a good candidate is screened out. |
| Detection | Resource/dependency/profile comparisons, decision-changing variants, ranked held-out candidates, bound counterexamples and shortlist recall. |
| Controls | Freeze hardware/lowering versions, label objective type, declare structural regimes and screening coverage; numerical settings stay in the future run manifest. |
| Repair | Correct the affected lowering, compression, projection or hardware mechanism separately; preserve the prior version and rerun its corresponding comparisons. |
| Ownership | Generator author supplies implementation rules; L2/projector computes schedules/costs; solver chooses parameters; no agent patches a selected implementation. |
| Choice and reversal | Proposed shared-description approach over independent hand-maintained models or a whole-GPU scheduling MIP. Prototype adapters first; changing the proposal does not alter current schemas. |

## Prepare a usable slice of the library

Keep the whole catalog and original provenance; finish the small dependency
closure required by the first family before trying to repair every entry.
Track independent capabilities rather than one undifferentiated ready flag:
semantic binding, model support, code generation support, and measured evidence.
Unknown or additional capabilities remain explicit records rather than forced
members of a closed classification.

For the chosen slice, record:

| Content | Required use |
|---|---|
| Transformation and interface | Explicit input/output meaning, dimensions, layout, dtype, masking and numerical policy; producer/consumer interfaces must match. |
| Decisions | Caller-fixed facts, derived quantities, numeric menus and structural alternatives are distinct. Do not bind away choices merely to simplify the solve. |
| Legality | Resource limits, storage lifetime, instruction/layout compatibility, pipeline readiness, and shape-dependent tail behavior become constraints or checked obligations. Unverified premises are not set to true by default. |
| Cost ownership | Identify which step accounts for each transaction, allocation and synchronization; price through a frozen hardware/model version. Unmodelled costs are not silently zero. |
| Realisation | Backend-specific deterministic emitters with a declared supported domain, compatibility conditions and witnesses. Unsupported combinations fail explicitly. |
| Provenance | Original contract, source/version, backend mapping, missing obligations and shared dependencies remain traceable. Deduplicate equivalent contracts, not similar names or superficially similar formulas. |

Allow an agent to author the initial modelling bindings and generation rules.
Freeze and record that work before search; count its time and tokens as one-time
preparation. No agent writes or patches each solver-selected candidate.

A restricted parameterized family generator is acceptable for the first stage,
labelled as such. It does not establish arbitrary Template composition. Its
transformation dependencies and parameter effects must be real: removing an
implementation removes that alternative, and changing a decision affects the
generated program or launch. A saved whole-kernel answer with decorative Template
labels is not an implementation of the proposed claim.

## Minimal offline chain and future runtime boundary

```text
problem + frozen Template/Realisation slice + hardware/model version
    -> explicit chain/family, bindings, choices and constraints
    -> formulation + independently inspectable cost terms
    -> compiled MIP + selected assignments / status / bounds
    -> frozen deterministic generator
    -> source files + launch description
    -> remote compilation + correctness + paired original/kernel timing
    -> retained predictions, resource feedback and measured results
```

A thin script may call the existing data types and pure modelling/solve/generate
functions directly. Reuse the working remote measurement route and persistent
records. No frontend, scheduler, service lifecycle or full control-plane repair
is needed for this pilot. Later runtime work should call the same operations and
read the same artifacts, rather than redefine the modelling semantics.

Deterministic generation means identical frozen problem, chain, realisations,
assignment and generator/backend versions produce identical source and launch
metadata, with no model call during materialization. Test this in fresh
processes. Solver reproducibility is a separate check: record solver version,
settings and stable tie-breaking. Compiled binary reproducibility across changed
toolchains is outside the initial claim.

## Controls and measurements

1. **Compiler/solver correctness:** on a genuinely small finite subspace, compare
   the actual MIP with exhaustive direct evaluation of the same frozen model.
   Check feasibility, objective values and optimum. This checks translation and
   solving, not hardware accuracy. Record structural choices outside the MIP.
2. **Deterministic generation:** repeat materialization in fresh processes; check
   source and launch equality, decision sensitivity and dependency removal.
   Keep compile/correctness rejections in the denominator.
3. **Selection quality:** compare MIP/model-selected candidates with bounded
   random selection from the same legal space, using the same generator and GPU
   evaluation count. On a small measured subspace, also report the best measured
   implementation and the time lost by the selected choice relative to it.
4. **Prediction quality:** save predictions and shortlist before GPU evaluation.
   Report rank quality, predicted/measured times and resource explanations.
   Separate reusable hardware calibration from declared software/compiler-profile
   training. Freeze each evaluated model version; version development-driven
   adaptation and keep final held-out results out of refitting. Do not use the
   evaluation answers as a hidden objective table.
5. **Knowledge benefit:** after the chain works, remove or replace a named
   transformation while holding the rest fixed and inspect lost legal variants,
   correctness and selection quality. Broader agent-with/without-library
   comparisons remain a separate question; do not pool the previous nine runs
   with new solver runs as matched treatments.

Use a fixed domain and predeclare budgets, candidate counts, shape split,
measurement rules and tie-breaks before launch. There are no numeric budget or
success-threshold defaults in this proposal. Retain solver time, generation time,
preparation time/tokens, compile/GPU evaluation count, failures, predictions and
paired timing samples. Report one-time preparation and per-use search costs
separately. A fast solve must not conceal expensive manual preparation.

No new model calibration project is prerequisite by default. Begin with the
existing model where its supported mechanisms cover the chosen space, expose
missing mechanisms, and refine only demonstrated ranking failures. Keep model
error, solver error and unsupported code generation as separate failure causes.

## Design interrogation

| Question | Answer |
|---|---|
| What is it? | A restricted offline execution of Template modelling, real MIP selection and frozen code generation. Broader chain discovery and runtime orchestration have separate evidence requirements. |
| What claim does it serve? | Knowledge can define executable legal alternatives and support efficient selection, rather than merely remind an agent of a known algorithm. |
| Dependencies? | Existing schemas/functions and recorded attention originals are verified source/artifact inputs. Model coverage, complete bindings and emitter support for this family are unverified obligations. |
| Scope? | One backend and a small declared family first, then an already studied contrast operator; transfer is checked rather than inferred from attention. |
| Failure modes? | Disguised enumeration, decorative Templates, omitted costs, impossible layouts, agent repair after selection, or a model fitted to its evaluation answers. |
| Observables? | Exported model and method, source-linked terms, dependency and decision-sensitivity checks, deterministic source/launch, all failures and saved premeasurement predictions. |
| Knobs? | Family/chain domain, menus, model version, execution backend, budgets and shortlist size live in a versioned pilot manifest; numerical settings remain unset until a run is designed. No live runtime controls change here. |
| Repair? | Diagnose a model, binding or emitter defect; change and version that component, recheck it, and begin a new declared run. Do not silently patch a selected candidate. |
| Who decides and reversal cost? | Assistant proposal under the user's modelling/search objective. Alternative is full runtime integration first, or more unconstrained agent coding. Reversal changes the thin driver/adapter; source knowledge and evaluation records remain reusable. |

Open obligations: demonstrate whole-chain accounting, nontrivial feasible choices,
actual MIP use, deterministic emission, correctness and measured selection
quality. Before an executable experiment is frozen, give its concrete domain,
budgets and failure criteria a separate review. This proposal does not change
core schemas, claim that this chain already runs, or launch additional work.
