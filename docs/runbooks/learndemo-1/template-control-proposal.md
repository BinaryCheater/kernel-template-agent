# Matched Template comparison

Status: superseded by the concrete [controlled source-reconstruction protocol](controlled-replay/plan.md) and [campaign settings](controlled-replay/campaign.json). The text below preserves the initial proposal. No new implementation sessions launched. Earlier runs remain exploratory evidence, not members of this matched comparison.

Rerun the core comparison before extending disclosure, models or workloads. Reuse the consolidated library, source studies, persistent runner, correctness/measurement machinery and retention tools after checking them against the frozen task. Start every compared implementation session fresh; do not combine old Template runs with a newly added no-Template run. Prior code and timings remain exploratory artifacts and are not deleted.

Use one studied attention kernel, one fixed model/harness and one language initially. Pin the original library revision, callable specialization, computation contract, Template mapping, GPU and direct original-kernel invocation before implementation starts. Match correctness tests and timing boundaries. The original need not have identical generated code/layout/scheduling, but it must be the named production comparator.

| Arm | Information supplied |
|---|---|
| Prescribed | Task, common language/environment instructions, frozen relevant Template subset and use instructions |
| Self-selection | Same task/common instructions, frozen full catalog/library and use instructions |
| No Template | Same task/common instructions; no Templates, catalog, Template-specific skills, source-study notes or earlier answers |

Common prompts are identical except the declared Template access/use clause. General language examples and tools are identical and contain no target-specific implementation. Each run starts in an independent persistent session/home with no earlier-run context. No arm sees the original implementation, decomposition chain or previous solutions. Record package and resolved model settings; audit reads and unexpected access. All target validation/timing uses the same evaluator and counts toward the same budget. Reading and interpreting Templates counts toward time and token cost.

The common evaluator owns one named source-kernel comparator across all arms. Record the actual backend/kernel selected, not just its Python wrapper. If that source specialization cannot run, stop preparation and record the failure; do not replace it with another backend while retaining the same experiment name. Compare the entire frozen output contract, including auxiliary outputs where required. Freeze input layout, precision, masks and preprocessing/timing boundaries; include necessary conversions and preparation consistently.

Proposed bounded replication: three fresh sessions per arm, nine runs total, fixed before observing outcomes. Shuffle launch order in each three-arm block and serialize GPU measurements under the shared lock. Retain order and waiting time. Keep all runs, failures and retries. Report per-run outcomes and ranges; do not select only the best session, stop when a preferred result appears, or treat three observations as a universal effect estimate.

Use a predeclared paired timing order applied identically in all arms; retain baseline and candidate samples together. Count compiler/runtime/numerical failures against the same declared budget. If a protocol or guide needs changing during the campaign, preserve that revision separately and identify the affected comparison; do not silently mix old and new conditions.

## Launch conditions

The coordinator must produce one retained prelaunch receipt covering the checks below before any paid implementation call. A written proposal alone is not a passed check. A separate read-only review must compare these artifacts against the original research question, not only verify that the runner works. No user approval per run is introduced.

| Required evidence | Current state |
|---|---|
| One exact studied source revision, entry/specialization, Template mapping and original invocation | Not fixed yet; launch blocked |
| Original kernel compiled and run on the target GPU, numerical checks passed, actual backend identified | Not established for the selected source; launch blocked |
| One frozen computation/output contract, shapes, tolerances and timing boundaries | Must be frozen after source selection |
| Three built public packages; task/common prompt/tools identical; difference limited to declared Template material/use clause | Not built for this campaign; inspect actual files before launch |
| Frozen model/effort, language, budgets, repetition count, launch schedule and outcome definitions | Proposal above; publish exact settings before launch |
| Fresh persistent sessions, audited Template/source/answer access policy and complete accounting | Existing mechanisms available; verify each new package/session |
| Independent review of source-to-Template-to-implementation-to-original-comparison chain | Required before launch; not done |

Report this campaign as ready only when all rows have real receipts. Retain the original-kernel run command and its source connection in the human-readable report so that the comparator can be understood without reverse-engineering logs.

Compare correctness completion, active time/tokens/cost to first correct candidate, time/tokens/cost to a predeclared production-performance target, and final correct performance under the equal budget. Retain the full progression. If no candidate reaches a target, record that outcome and the budget used rather than excluding the run. Separate model time, compilation/evaluation time and GPU waiting without double counting. No-Template parity or superiority is a valid result; the model may already know the algorithm.

| Design question | Decision or obligation |
|---|---|
| Definition | A matched information-exposure experiment on one source-linked kernel reconstruction task. |
| Claim | Determine whether supplied Templates improve completion, effort or final performance beyond the model's existing knowledge; separately observe the cost of self-selection. |
| Dependencies | Existing persistent runner and retained measurement path are available. Exact studied source invocation, compatible GPU build and final common package are not yet verified; establish them before launch. |
| Scope | One source kernel, bounded shapes, one GPU/toolchain/model. Other kernels and models require separate evidence. |
| Failure modes | Wrong comparator, unequal prompts/tools/budgets, prior-context leakage, undocumented backend fallback, untracked sweeps, post-result task changes, selected successes or missing cached usage. |
| Observables | Source invocation/backend receipt, frozen manifests/packages, native reads/settings/usage, attempt sources/checks/raw timings, randomized schedule and every terminal outcome. |
| Knobs | Proposed arm set and repetition count below; remaining existing run limits frozen equally before launch. No product control or runtime schema is introduced. |
| Repair | Coordinator records protocol defects and retains affected runs; a changed task/library/prompt becomes a separately named revision. Do not pool repaired and original conditions. |
| Ownership/cost | User proposed the no-Template control. Three-arm exposure, three repetitions and block shuffling are assistant proposals. Alternative: one pair only, cheaper but unable to reveal run variability. Reducing scope remains possible before launch; expanding after viewing results must be recorded separately. |

Experiment-local proposed knobs: `arms=[prescribed,self-selection,no-template]`, `repetitions_per_arm=3`, `launch_order=shuffled_per_block`. Freeze model/effort, language, tool access, shape suite, tolerances, active-time/evaluation budgets and performance target together in campaign/run manifests before launch. These are experiment settings, not new product controls in `docs/design/12-agent-controls.md`.

First deliverable is the named source-kernel comparator and a readable frozen three-arm task description. New-shape/W4A16 extensions and additional model comparisons follow only after this controlled case has a result.
