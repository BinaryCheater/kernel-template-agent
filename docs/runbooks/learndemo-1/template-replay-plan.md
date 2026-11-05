# Consolidated Templates and kernel replay

Status: historical generic campaign; do not use this document to launch the next experiment. Use the [controlled source-reconstruction protocol](controlled-replay/plan.md). Source-kernel reconstruction remains unverified; the [scope correction](results/scope-correction.md) supersedes the earlier completion claim. See the retained [results](results/summary.md). Order: consolidate the current
library, prepare two cases, run prescribed-Template and self-selection sessions,
then repair and replay. Further source extraction is a separate serial assignment.
No KTA frontend or runtime Store integration is required for this experiment.

## 1. Consolidate before experiments

Read seed Templates, early reference proposals and current studies together.
Exclude inputs/pool-records and other retained snapshots from candidate discovery.
For every source identity record its canonical destination, retained variant, alias
or unresolved conflict. Preserve originals and proof/source links. Equal names are
not evidence of equal semantics; different names may express the same choice.

One Template is an actionable design transformation. Keep its decision variables
inside that rule. Split choices that can be selected independently; retain coupled
choices together when separation would hide the conditions needed to implement them.
Architecture instructions and source shapes normally identify implementations,
not separate abstract rules. Do not broaden a precondition merely for deduplication.
A changed contract needs a revised proof outline and an explicit old/new mapping.

Produce a file-based canonical library under knowledge/library/:

- README.md: the agent's short usage instructions.
- catalog.json and catalog.md: generated cards for all canonical entries.
- templates/<name>/<version>/GUIDE.md: actionable explanation and a short local example.
- The same directory's template.yaml and proof.yaml: structured rule and outline.
- mapping.json: original study/seed identities and paths to canonical versions.
- unresolved.md: conflicts and incomplete rules withheld from the default menu.

GUIDE.md explains the computation before/after, binding steps, tunable versus
caller-fixed/derived variables, coupled constraints, implementation choices,
expected benefit and possible cost, related rules and proof obligations.
Distinguish structural applicability, supported implementations and observed source
instances. Mark source-backed mechanisms versus inferred extensions. No whole-kernel
answer is embedded in a card or guide. Original studies remain provenance records.

Done: every candidate has a disposition; canonical identities are unique; all
menu paths resolve; rules selected for the first cases can be instantiated from
their guides without consulting study traces. General experiments may use source
proposals, but an unresolved conflicting identity is not silently selected.
The consolidation report states counts before/after, important merges/splits,
remaining conflicts and concrete examples of improved instructions. Count is not
a success target. Canonical file publication does not imply proof or GPU assurance.

## 2. Start simple; add progressive disclosure after basic runs

First run the prescribed mode with its supplied guides, then self-selection with
one plain catalog file and normal file reads. The model receives TASK.md and
TEMPLATE_USAGE.md plus the catalog path; it can inspect headings and read entries.
Do not build a retrieval service before these runs. This first baseline tests whether
the organized library and instructions are usable at all.

Prepare concise cards during consolidation: id/version, title, when to consider
it, the choice it introduces, principal boundary, and guide path. Generate them
from canonical metadata. After the basic workflow works, inject these cards into
initial context in one separate fresh run on an existing case. If the menu is too
large, expose category descriptions first and expand cards on demand. Record exact
initial content and its token size. Categories do not restrict semantic applicability.

A subsequent lightweight helper can expose list(category), search(text), read(id)
over that same frozen library. One separate run is enough to inspect usability,
selection behavior and overhead; it is not statistical evidence of superiority.
Keep task, model, budget and library fixed when changing disclosure or tools.
No embeddings, service deployment or broad experiment matrix is needed initially.
Persist observed card/guide reads and selected identities; use session records and
a concise design note rather than requesting hidden internal reasoning.

## 3. Two cases, four independent initial sessions

Use two representative computations rather than many simple operators:

1. Dense FP16 GEMM with FP32 accumulation and FP16 output: both operands fixed
   row-major, no bias/activation. Cover square, small-M and a non-multiple tail.
   Candidate shapes (M,N,K): (1024,1024,1024), (32,4096,4096), (127,1000,769).
   This exercises tiling, instruction selection, movement, pipelining and tails.
2. Forward causal attention: FP16 Q/K/V, FP32 accumulation/statistics, FP16 output;
   scale 1/sqrt(D), no dropout, no backward, no LSE output, equal Q/KV head counts.
   Candidate (B,H,S,D): (1,8,512,64), (1,8,2048,128), (1,8,513,64).
   This exercises online normalization, fusion, mask handling and resource tradeoffs.

These are proposed bounded workloads, not claims that a particular backend is
available. Prepare and freeze exact layouts, shape suite and target after checking
the worker environment. Do not broaden to all attention/GEMM variants in this round.

For each case run a fresh session in each mode:

| Mode | Template access | Agent responsibility |
|---|---|---|
| Prescribed | A declared small set of guides, rules and outlines | Bind, compose, implement and tune; do not supply a composition chain |
| Self-selection | Plain canonical catalog and read access to the library | Select rules, bind, compose, implement and tune |

Use the same task, GPU, model settings, language policy and budget in both modes.
The prescribed set is chosen from concrete required mechanisms, not a complete
source decomposition. An agent may identify a missing rule and use its own knowledge;
record that addition instead of forcing an incorrect Template mapping. Both modes
are development exercises, not a controlled causal claim about Template benefit.

Default language: TileLang if the target's installed toolchain passes the prepared
smoke case. CUDA/CuTe DSL/Triton are alternatives only after that environment check.
Freeze one starting language for both modes of a case. Allow a switch when the agent
identifies a concrete missing operation or compiler limitation; record reason and
cost, and report it as an environment/design change, not silently comparable tuning.
Source-library language does not dictate output language. No language sweep initially.

## 4. What the experiment session receives

The isolated experiment workspace is outside the KTA repository and contains:

- TASK.md: exact math, tensor ABI/layout, supported shapes, dtype/rounding contract,
  target GPU, callable entry point, forbidden changes to semantics and run budget.
- TEMPLATE_USAGE.md: select/read/bind/apply guidance, including concrete versus
  inferred applicability and the requirement to keep a brief design note.
- library/: either the prescribed subset or full frozen canonical library/menu.
- ENVIRONMENT.md: installed language versions, working compile/run example,
  remote execution instructions and available hardware facts.
- harness/: read-only validation/benchmark interface, sample inputs and error
  feedback; candidate.py or equivalent is the editable implementation entry.

Initial prompt only asks the session to implement TASK.md using TEMPLATE_USAGE.md
and the supplied library, run the harness, and preserve its best correct candidate.
Do not compensate for unclear guides with case-specific coaching in that prompt.
No source-study reports, original production kernel source, answer chains, previous
agent solutions or writable correctness tests are exposed. Keep target source and
production benchmark code on the evaluator side. Source URLs in provenance are not
permission to fetch answer implementations. Provide language docs locally; no source
search/network browsing for target implementations during these runs. Use enforceable
workspace/tool access limits when available; otherwise disclose instruction-only
isolation and record accesses rather than claiming a blind run.

The coordinator session prepares and evaluates experiments; separate implementation
sessions run inside the prepared environment and receive only the run package.
Every implementation run MUST use a persistent, resumable harness session. Save its
session ID, transcript location, tool calls, settings and usage before marking the
run started. Fresh means separate context, never ephemeral or unrecorded. Preserve
failed and interrupted sessions and resume points. Do not use disposable no-history
modes. Verify persistence with a short smoke interaction before a paid kernel run.

Default: Codex with GPT-6.1 Sol, reasoning high. Comparison: DSH (DeepSeek Harness)
with DeepSeek V4.1 Flash, reasoning high. Resolve actual installed harness/model IDs
and save resolved identity and effort; do not silently substitute a model or effort.
Use existing configured access; never write credentials into run artifacts. Automated
model invocation goes through the traceable agents adapter, adding only the small
adapter support actually missing. Preserve harness-native transcripts as well.
Fast service mode is separate from reasoning effort and is recorded only if enabled.

## 5. Correctness and performance

Do not create an optimized or deliberately weak baseline. Supply a mathematical
reference: explicit PyTorch operations with higher-precision intermediates for the
bounded suites, plus an independently run production reference on the same GPU.
Before agent launch, verify the production reference against the mathematical one
and freeze tolerances for the requested output dtype. Test nontrivial magnitudes,
cancellation, tails and masks; reject NaNs/infs where finite outputs are required.
No post-hoc tolerance relaxation to accept a faster implementation. Input seeds and
correctness checks stay under evaluator ownership; include fresh validation inputs.

The reference implementation is an oracle, not a timed candidate. Production
latency is the performance target; do not label a chosen reference universal SOTA.
Run reference and candidate with identical timing boundaries, cache policy, warmup
and repetition rules. Include required packing/preprocessing/dispatch costs; report
amortized reuse only when TASK.md explicitly permits it. Report compilation separately.
GPU execution uses the project's remote execution/measurement path and shared GPU
scheduling; never race performance-model measurements. Preparation verifies worker
availability and toolchain before spending a model session on implementation.

Only correct candidates enter the performance table. Report all prescribed shapes,
not just the fastest example. Performance ratio = reference_latency/candidate_latency.
A provisional aim is at least 0.90 on every shape; preserve/report outcomes below it
rather than changing the task. Repeated measurements decide whether small changes
are real. This first round does not claim numerical equivalence for all inputs.

## 6. Run loop and outputs

Proposed per-session budget: 60 minutes active agent/execution time, at most 20
candidate evaluation batches, maximum 3 shapes per batch. Report GPU queue wait
separately. Stop at budget and preserve best correct candidate (or the most advanced
failure); preserve every miss. The coordinator may start a new revision after diagnosing an
actionable knowledge/instruction issue; no per-run user approval is needed within
this assignment. Record the reason and stop repeating an unchanged failure.

Read task/cards → read selected guides → brief design/bindings → implement → validate
→ measure → revise from feedback. A design note is a few concrete decisions, not a
required full chain or a lengthy plan. Save a candidate before replacing it.

Each run directory under KTA_DATA/dev/template-replay/<case>/<run>/ retains:

- frozen task, library/menu, prompt, environment and model settings;
- persistent session reference and complete input/output token/cache/cost accounting;
- candidate versions, best implementation/configuration and rerun command;
- correctness errors and timing samples for every evaluated candidate;
- a small result table and design note (rules selected, bindings, self-supplied choices);
- diagnosis and proposed library changes, with supporting failure examples.

Human report: case/mode, correct or failure reason, per-shape reference/candidate
latency and ratio, elapsed/queue time, token/cost, attempts, and one paragraph on
what the agent actually needed. Estimated costs are labeled; missing accounting is
reported missing. No synthesized success score replaces these facts.

Classify failures by evidence: retrieval, unclear rule, missing mapping/constraint,
composition conflict, implementation bug, poor parameter choice or environment.
Fix the owning guide/card/harness, not hidden hints in the next prompt. Freeze a new
library revision and retry in a fresh session, including an adjacent shape. Preserve
the original result. Full proofs and broad performance portability are future work.

## 7. Implementation order and decisions

1. Consolidate all current candidates and publish the file-based library/report.
2. Prepare the GEMM oracle, production reference, worker environment and isolated task.
3. Run prescribed GEMM; fix concrete instruction/library failures and rerun fresh.
4. Run self-selection GEMM from a fresh session, then prepare/run the two attention modes.
5. Deliver the four original runs plus clearly separated revision runs and library deltas.
6. Run one progressive-disclosure variant on an existing case after the basic loop
   works; add a small retrieval helper only when it addresses observed friction.
7. Run DSH/DeepSeek V4.1 Flash high on the same frozen case, library exposure and
   evaluation budget as a selected Codex run. Record both model and harness versions:
   this compares model+harness packages, not model quality in isolation. Preserve
   per-shape quality/performance, time, cached/uncached usage where available and
   actual or explicitly estimated cost. Do not invent token/cost fields a harness
   cannot report. Expand to a second case only if the first comparison is informative.
8. Plan new-shape/new-variant transfer after these runs; no no-Template ablation now.

Defaults and alternatives: two substantial cases rather than a broad benchmark;
one starting language rather than a language sweep; cards plus file reads rather
than a retrieval service; mathematical oracle rather than a handmade weak kernel;
file snapshots rather than frontend/backend integration. These choices are reversible
in the next experiment revision; preserve prior inputs and outcomes when changing them.

Dependencies/obligations: verify backend support and production oracle before launch;
validate canonical identity/link integrity before packaging; test input isolation;
record GPU measurement conditions; verify a claimed guide repair with a fresh session.
Failure is visible in the run artifacts, never inferred from a self-assessed report.
Knobs (menu allowance, language, shapes, tolerance, runtime/evaluation budget and
performance aim) live in the frozen case/run manifest, not unrecorded prompt edits.
The coordinator owns changes between runs; the implementation agent owns only
candidate code/configuration and its design note. No running service or permanent
product API is introduced by this plan.

## 8. Coordinator handoff and efficient completion

The owner sets a persistent goal for this plan and maintains one short STATE.md
beside this document: current phase, artifact paths, latest result, unresolved blocker
and next action. Publish library consolidation first, then the environment smoke result,
then each experiment result as it finishes. Do not wait for all experiments to give
readable results. No new broad source-library extraction is part of this goal.

Code, usage instructions, frozen case specifications and concise reports belong in
Git; run payloads/transcripts/snapshots remain under KTA_DATA with retained location
manifests. The canonical library must be sufficient for the implementation session
without access to ignored study inputs. Existing studies and downloaded sources remain.

Deliver: (1) canonical library and source mapping, (2) working task packaging and
measurement entry point, (3) the four basic runs and clearly separated repair runs,
(4) a small disclosure/tool run and a matched DSH comparison, (5) a one-page summary
of what improved Template use, what remains weak and the best implementations.
Report concrete failures when a target is not reached; do not replace runs with plans.
Do not spend the entire goal polishing every guide: cover every candidate in the map,
then prioritize actionable completeness of rules needed by the two cases. Keep
conflicts explicit until resolved. Implement the next smallest useful step.

The user authorizes the coordinator to create the necessary recorded implementation
sessions and run the experiments with existing resources. GPU use must respect the
shared worker schedule. Do not change global services, install system software, add
paid infrastructure, push, or delete worktrees/sources. Keep repository edits scoped;
no commits or branch switches without a subsequent user instruction. Use the project
skills for remote execution and measurement. Do not restart the source-study workers.
