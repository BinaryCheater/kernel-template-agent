# Source-kernel reconstruction with matched Template controls

Status: execution protocol v1. The design is concrete; implementation launch is disabled until the evidence in section 10 is complete. No implementation sessions are launched by writing this plan. This document and [campaign.json](campaign.json) supersede the earlier generic replay plan for the next campaign. Existing results remain exploratory and are not pooled with these runs.

## 1. Questions and scope

Answer two distinct questions:

1. Can an independent model use the accumulated Templates to reconstruct the computation of a named studied production kernel at competitive performance?
2. Does supplying those Templates improve completion, time, tokens/cost or final performance relative to direct implementation without Template information?

The first question requires a named original kernel as comparator. The second requires fresh matched treatment/control sessions. Neither is answered merely by a correct fast implementation. Reproduction requires the same computation/output contract and competitive performance; identical source code, layouts and schedules are not required.

Run one case, one model/harness and one implementation language. Defer GEMM, new variants, disclosure helpers and DeepSeek until this comparison is reported. Reuse library/tooling/source studies; redo the implementation sessions in all arms. Do not append only a new control to the old runs.

## 2. Exact original kernel

Use the retained **CUDA FlashAttention-2 dense causal FP16 D128 forward** from Dao-AILab/flash-attention, revision `1dc8232fe209e2c48b1d3e677f26d68e4b856212`. Its retained source record is [sources.json](../../../../knowledge/studies/flash-attention/inputs/sources.json); its study is [trace-cuda-fa2.md](../../../../knowledge/studies/flash-attention/trace-cuda-fa2.md).

- Translation unit: `csrc/flash_attn/src/flash_fwd_hdim128_fp16_causal_sm80.cu`.
- Original launcher: `run_mha_fwd_hdim128<cutlass::half_t,true>`.
- Original device body: `compute_attn_1rowblock` in `flash_fwd_kernel.h`.
- Expected GB10 non-sm8x/non-dropout branch: `Flash_fwd_kernel_traits<128,128,64,4,false,false,cutlass::half_t>` at `flash_fwd_launch_template.h:262`.
- No split-KV, cache append, paging, GQA, dropout, ALiBi, local window, softcap, backward or returned probability matrix.

The early seed decomposition used a different revision. Its step-to-rule map is a starting record, not proof that both versions are identical. Before launch, reconcile the selected current branch against that map and the current study, retaining each source location, chosen rule/version, obligation and explicit gap. Keep this map evaluator-side; the implementation model does not receive the original chain or original bindings.

Build only the unchanged original D128/FP16/causal specialization and a thin host binding for allocation, parameter setup and launch. Check parameter setup against `flash_api.cpp::mha_fwd/set_params_fprop`. Do not rewrite the device algorithm. Save the exact build recipe, source/include inventory, compiler/options, produced binary and actual runtime kernel trace. If a CUTLASS build dependency is needed, materialize the exact submodule revision, not a latest checkout; this does not resume the paused CUTLASS study.

The original must use upstream-equivalent release optimization: host/device `-O3`, NVCC `--use_fast_math`, upstream-relevant constexpr/lambda/half-operator flags, and explicit GB10 architecture code generation. No debug `-G`, `-O0` or selectively disabled optimization. Compare the recipe with retained `setup.py:357-376`; independently review any compiler/architecture compatibility deviation before sealing. An unchanged source compiled weakly is not an acceptable production comparator.

The thin binding must preserve upstream parameter normalization. Public `causal=true,window=(-1,-1)` becomes setter arguments `(-1,0)`; `set_params_fprop` derives `is_causal=true`, then stores `window_size_left=S,window_size_right=0`. Public dropout probability 0 becomes internal keep probability `p_dropout=1`, inverse keep probability 1 and integer threshold 255. Preserve source element strides, FP32 scale/log2 scale, rounded sequence lengths, d=128, head ratio 1 and LSE/output pointers. Select the original non-split launcher explicitly; no unrecorded high-level split heuristic. Record/check these parameters and compare a short causal output against a noncausal fault canary before accepting the binding.

Expose the original through an evaluator-only `original_entry(q,k,v)->(O,LSE)`. Primary timing is the complete GPU computation, independent of Python wrapper differences. Report the thin host binding openly; do not label it a full upstream Python API benchmark. No PyTorch SDPA/FlashInfer/other-kernel fallback is permitted. If this exact original cannot run, preparation stops before any model run. A different source kernel requires an explicitly revised case.

## 3. Common computation contract

All arms implement `entry(q,k,v)->(O,LSE)` in TileLang 0.1.15 on GB10 sm_121.

| Property | Fixed value |
|---|---|
| Inputs | Q/K/V contiguous `[B,S,H,D]`, FP16, same device |
| B/H/D | 1 / 8 / 128; Q and KV head counts equal |
| S suite | 512, 2048, 513; all three required |
| Scores | `QK^T/sqrt(128)`; key j visible iff j <= query i |
| Intermediate arithmetic | FP32 accumulation and normalization state; numerical refinements admitted only by the fixed checks |
| O | New contiguous FP16 `[B,S,H,D]` tensor |
| LSE | New contiguous FP32 `[B,H,S]`, natural logarithm of the causal row exponential sum |
| Required work | Both outputs and every necessary GPU conversion, packing and normalization per invocation |
| Permitted cache | Compiled executables only |
| Prohibited | Input mutation, cached data/results, host tensor arithmetic, prebuilt attention delegation, target-source/answer reads, untracked target tests/timing |

Only the submitted candidate is shipped to the evaluator; private helper files cannot hide part of the computation. Torch allocation and tensor metadata are allowed; attention arithmetic must execute in model-authored TileLang kernels. Source specialization choices and source code remain evaluator-side. Public TASK.md describes the computation and target, without giving original tile sizes or composition answers.

This changes two important omissions in the old campaign: use the source's layout and D128 specialization consistently, and compute/check LSE as well as O. Old O-only/BHSD candidates are not reused as formal results.

## 4. Arms and exact knowledge exposure

| Arm | Supplied optimization information | Role in comparison |
|---|---|---|
| P: prescribed | Unordered set of the 18 rules listed below; guides, structured rules, existing outlines where available | Tests usable supplied knowledge with selection already done |
| S: self-selection | The same 18 entries within a complete frozen catalog: 312 canonical entries plus eight labelled source proposals, total 320 | Tests practical selection and implementation from the full library |
| N: no Template | No rules, catalog, Template skills, decomposition, study notes or prior answers | Measures direct implementation using the model's own knowledge |

The 10 canonical entries are `output_tiling@2`, `reduction_partition@1`, `compute_instruction_selection@2`, `physical_storage@1`, `async_transfer@1`, `multi_buffer_rotation@2`, `operator_fusion@4`, `online_rescaled_reduction@2`, `masked_tile_pruning@2`, and `shared_resource_binding@1`.

The eight source proposals are `operand_precision_conversion@1`, `approximate_math_rewrite@1`, `cross_lane_reduction@1`, `accumulator_operand_reuse@1`, `collective_fragment_load@1`, `staged_relayout@1`, `boundary_predication@1`, and `vectorized_access@1`. They have no paired outline and were withheld from the canonical menu. Retain that status explicitly. Experimental exposure does not promote them or assert a proof. Prepare standalone usage guides from their retained rules; do not insert original composition order or tuned parameter answers. Their source-version applicability must be checked before freezing.

[template-inputs.json](template-inputs.json) records exact identities, source files and current bytes. Do not silently drop the eight proposals and still describe P as containing the available extraction. Conversely, do not claim the 18 rules completely explain the source: architecture dispatch, compiler allocation and any unresolved native scheduling/layout mappings remain counted gaps. Models may use their own knowledge; record those additions after implementation.

Freeze one campaign library after source reconciliation and guide checking, before the first model run. All P/S repetitions use the same bytes; P's files must be identical to the corresponding S files. N receives none of them. P-vs-N measures the benefit of supplied selected knowledge; S-vs-N measures practical full-library access. P-vs-S includes selection and irrelevant-library overhead and is not a pure retrieval-algorithm comparison.

This treatment is the current **pooled accumulated library**, including reusable CUTLASS/SGLang-origin rules and guidance repaired during earlier exploration. The source map justifies its applicability to this FA2 kernel. Success supports reconstruction using that accumulated knowledge; it does not establish sufficiency of an extraction made solely from this FA2 revision. Preserve provenance and prior revisions explicitly; no additional arm is introduced to answer that different question.

## 5. Common environment, prompt and access

Use the existing pinned worker base image, native Codex CLI and recorded-session adapter. Fix model `gpt-6.1-sol`, reasoning `high`, no fast service. Record actual CLI version/resolved model settings, environment image and build dependencies before launch. All runs have the same general instructions, language example, public language/API documentation, tool access, evaluator interface and feedback schema.

All model work stays in the one recorded session assigned to that slot. Disable native `multi_agent` and `multi_agent_v2` features in every arm; prohibit launching another model/CLI/API from shell tools and audit this in native transcripts. This implements the single-session budget rather than allowing uncounted child usage. Record resolved native feature settings in P6; no extra model arm or treatment difference is introduced.

Common initial prompt:

> Implement TASK.md. Use ENVIRONMENT.md and the supplied evaluation interface. Validate and optimize within the fixed budget, preserve the best correct self-contained candidate and a short DESIGN.md. Preserve failed versions. Do not read original target implementations, study traces, previous solutions or evaluator code. Run all target correctness and timing checks through the evaluator. Stop according to the shared stopping rule.

Append only P: "Read and use the supplied optimization rules in library/. Record the rules used and choices supplied from your own knowledge." Append only S: "Select relevant optimization rules from library/catalog.md, read and use them, and record the rules used and choices supplied from your own knowledge." N has no Template clause. No arm receives case-specific coaching beyond TASK.md and its declared information treatment. No generic "dummy" text is added to equalize prompt length: reading cost is part of Template utility.

Build all three packages before launch. Diff TASK.md, environment/tools/examples, evaluator interface and common prompt; require byte identity. Compare P/S shared rule files. N must have no library or project/ancestor/personal Template instructions. Start fresh external workspaces and persistent homes, excluding earlier session context and personal skills. Archive native prompts/settings/tool calls and verify session start/resume before model work.

Freeze a public language-document allowlist before launch: the same local TileLang JIT/language/API/primitive-lowering references, Torch allocation/metadata/CUDA-graph interfaces and non-target examples for every arm. No target-attention implementation examples. Record allowed paths/files and bytes in ENVIRONMENT.md; arbitrary internet documentation searches are not part of this campaign. Each slot has its own native home and verified compiler/JIT/artifact cache roots; initialize them empty and retain them afterward. Do not expose compiled artifacts or scratch paths from older/other-arm candidates. Verify the installed cache configuration in P6 instead of assuming an environment variable is honored. Fixed toolkit/standard-library binaries and the evaluator-only original binary may be reused read-only.

Native full-access read isolation is instruction-only, not an OS sandbox. Audit native tool/network/file commands in every arm. Remote compilation is allowed only in that run's retained scratch area; target evaluation goes through the common interface. No source/network search for answers, including installed attention implementation sources. Any access violation or missing transcript is retained and reported as a protocol deviation; it is not silently replaced with a successful run. Do not describe these as perfectly blind runs.

## 6. Sample size, schedule, budgets and stopping

Nine new persistent sessions: three per arm. One implementation session at a time. Use the balanced schedule generated with fixed seed 20261105:

| Block | First | Second | Third |
|---|---|---|---|
| 1 | S | P | N |
| 2 | P | N | S |
| 3 | N | S | P |

Every arm occupies each order position once. Do not reorder after seeing a result. Each run is allowed **1,800 active seconds and 20 target-evaluation submissions**, whichever is reached first. GPU queue wait is separately recorded and excluded; reading, thinking, compilation, transport/evaluation and ordinary tool work are active time. Evaluation durations overlap session wall and are not added again. Cost has no hidden group-specific allowance. All cached/uncached input, output and available reasoning usage is retained.

These limits cover model discovery, including its two unchanged-source stopping checks. Once the source is selected, the model is closed. Blind validation and three confirmation windows are fixed coordinator-only work, outside the 20 submissions/1,800 seconds, with **300 worker-active seconds per slot**, queue excluded; no more model calls, source edits or feedback. Report that extra validation/measurement time separately. If confirmation cannot finish within its cap, report unresolved confirmation and do not count target attainment.

The active-time clock is `elapsed wall minus recorded GPU-queue periods`. Queue exclusion applies only while that session is blocked on its sole evaluator request; no background target evaluations or overlapping model work. A watchdog must track open as well as completed queue intervals, stop new submissions at active-budget expiry and interrupt the native model while preserving its session. A worker execution timeout starts after lock acquisition using the remaining active time. Any native idle timeout must exclude verified queue waits too. P6 must test this behavior before launch; the old wall-clock launcher is not assumed to enforce it correctly.

The common performance target retains the original 0.90 production/candidate throughput criterion on every shape: candidate time must be at most original time divided by 0.90, approximately 11.1% longer. Human reports state microseconds and percent more/less time. "Meets target" is not automatically worded "faster" or "equal".

The model may finish early after two consecutive unchanged-source submissions pass all checks and meet the target on all shapes. Otherwise it may voluntarily finish or stop at budget; every such outcome remains. Candidate compile/runtime/numerical failures consume submissions. A documented external interference/transport failure may retry the identical source/window, never provide a free code edit: at most two additional attempts per window and two additional attempts in total per slot across discovery and confirmation (at most 18 across the campaign). Keep all retry records/durations; retries do not reset either active-time cap. If the slot's external-failure allowance is exhausted, record an unresolved slot and stop further paid slots until the common problem is diagnosed. Never extend a preferred arm because it almost succeeded.

Before final hidden tests, each session selects one source file. If it failed to select, use its latest submitted self-contained candidate and record that fallback. The coordinator does not optimize or switch candidates after seeing hidden checks or confirmation timings.

## 7. Correctness and leakage-resistant validation

Evaluator-owned FP64 explicit scores, causal mask, softmax, O and natural-log LSE provide mathematical checks. O uses elementwise `abs(error) <= 0.01 + 0.01*abs(reference)`; LSE uses `0.001 + 0.001*abs(reference)`. Check both complete output tensors, shape/dtype/device/contiguity/finiteness, no input mutation and repeat invocation with changed data. These tolerances are launch-time hypotheses: the original must pass them before launch. A failure stops preparation; do not loosen them after model outcomes.

Development checks: two private seeds per pattern, five patterns (normal, reciprocal magnitude scaling, cancellation, dominant future value, uniformly negative logits), all three shapes. Use identical evaluator-generated inputs by block/submission/pattern across arms. Generate/freeze the seed schedule before launch; keep it outside public packages. Each new submission gets fresh development inputs. Timing uses separately frozen ordinary inputs shared by all arms.

The generator and oracle must use the new BSHD layout explicitly: FP64 raw scores `einsum('bihd,bjhd->bhij',Q,K)*scale`, then `masked_scores=where(j<=i,raw_scores,-inf)`, `P=softmax(masked_scores,dim=-1)`, O `einsum('bhij,bjhd->bihd',P,V)`, and LSE `logsumexp(masked_scores,dim=-1)`. Cast the common runtime scale to FP32 before applying it in both source and oracle. Normal draws independent FP32 Gaussian tensors then converts to FP16; magnitude multiplies Q by 4 and V by 0.25; cancellation pairs `V[:,2j+1,:,:]=-V[:,2j,:,:]` leaving an odd tail random; mask sets `V[:,-1,:,:]=100`; negative logits sets Q=4 and K=-4, retaining random V. Sequence is axis 1, not the old BHSD generator's axis -2. Freeze exact generator bytes before creating validation inputs.

Final selected sources receive eight previously unseen seeds per pattern/shape (120 input cases, checking O and LSE separately). Final tests return no optimization feedback and permit no repair. Record original and candidate checks against mathematics, maximum errors and failing elements. Retain seed/generator/version identities and input checksums so every input is reproducible.

Before model launch, fault canaries must reject zero output, NaNs, wrong LSE and input mutation. Check generated timing entry does no reference delegation or CPU tensor arithmetic. Compilation failures and wrong results remain outcomes; they are not environmental exceptions.

## 8. Performance measurement and confirmation

Primary: GPU time for the complete entry computation under CUDA graph replay, including both O/LSE and all required GPU preparation. No profiler attached to timings. Source and candidate execute on the same inputs, stream, graph policy and shared exclusive worker lock. Graph capture must be validated for the original and common language example before launch; unsupported capture stops preparation rather than silently switching timing modes.

Validate the path that is timed: capture each original/candidate graph once per shape, then copy each fresh development/final validation input into its existing captured input addresses outside timing. Replay, synchronize and check O/LSE against that input's FP64 oracle and check input mutation. Also check outputs and mutation after the warmup/timed replay blocks. A correct eager entry with an incorrect/stale graph replay fails. Persistent hidden state/counters or input-dependent output caches are forbidden. The final eight-seed suite checks both ordinary entry and captured replay; no graph time is accepted solely because eager correctness passed.

After compilation/capture, warm each graph with 20 replays. Each measurement window has 30 paired blocks of 100 replays per side. Use 15 original-then-candidate and 15 candidate-then-original blocks, shuffled from the frozen measurement seed. Measure by CUDA events, divide by replay count, retain every paired block and report medians/IQR. Include every kernel launched by entry; no selective timing of the fastest subkernel. Graph memory reuse is evaluator-managed; a candidate may cache compiled handles, never input-dependent results.

The window estimator is fixed: `T_original=median(30 original per-call block times)`, `T_candidate=median(30 candidate per-call block times)`, `R=T_original/T_candidate`. It is not the median of 30 paired ratios. IQR uses inclusive quartiles. Report `100*(T_candidate/T_original-1)` as percent more/less time. For two half-window medians or three final original medians, drift is `max(values)/min(values)-1`. The original's IQR/median uses its 30 block times. Do not swap estimators near a threshold.

Secondary: ordinary eager-entry CUDA-event results, 10 warmups and 50 samples, shown separately to expose wrapper/launch effects. They do not replace the primary endpoint after results are observed. Native kernel identity is profiled separately for the original and final selected candidates.

Hold intent.lock then rw.lock exclusively. Record competing processes, load, clocks, temperature and throttle flags around measurement blocks; make no system/power/clock changes. A window is invalid if another GPU workload is present, the original's IQR/median exceeds 10%, or original medians differ by more than 5% between window halves. Retry at most twice with the same source. The criteria use comparator/environment evidence, not whether the candidate looks good.

Final confirmation: three separately launched accepted timing windows for each frozen selected source, releasing/reacquiring the host lock between windows. Revalidate numerical correctness first. Report the median of the three per-shape window medians and all window values; no best-window selection or assumption of statistical independence between adjacent windows. Original medians must satisfy the defined 5% drift limit across the three accepted windows; otherwise label performance unresolved and do not claim attainment. The final target verdict requires hidden correctness and R>=0.90 on all shapes in all three accepted windows. If confirmed performance misses a development claim, use the confirmation verdict.

## 9. Outcomes and interpretation

Primary: for each arm, how many of its three sessions end with an independently confirmed, all-shape correct candidate meeting the production target within the same budget. Keep all nine scheduled slots visible, including non-attainment, infra problems and protocol deviations. An invalid slot prevents presenting the planned complete comparison as finished; do not quietly replace it with a better run.

Secondary: first-correct and first-target times, input/cache/output tokens and API-equivalent cost; final per-shape time differences; compiler/numerical failures; selected rules and model-supplied implementation choices. Mark a target never reached as not attained, with the budget/censoring shown. Do not calculate efficiency means only among successful runs. Show each run, then medians/ranges where defined; do not fabricate a speed for an incorrect candidate.

Label earliest development-check milestones as provisional. Also report effort up to the first measured appearance of the exact final selected source; final hidden checks/confirmation determine whether that source qualifies. An early candidate that passed development inputs is not retrospectively called independently validated unless it receives separate retained confirmation. Do not use a transient timing win or an unconfirmed first-correct marker as the primary result.

Report P-vs-N and S-vs-N first. If all groups succeed similarly, the result may be that this familiar task shows little extra Template benefit. If Template arms reach correctness/production level with less effort or better final performance, state the measured magnitude on this case. Three sessions per arm describe consistency and variability, not universal superiority. No-Template parity or superiority is equally reportable. Model prior knowledge is allowed and is exactly why N exists.

No guide/prompt/task/budget changes inside this campaign. A consequential defect pauses the campaign; retain every run and make a new protocol/library version. Do not mix revisions in a causal comparison or rerun only an unfavorable arm. Additional samples/models/tasks require a separately declared follow-up, not optional stopping on this result.

## 10. Preparation order and launch evidence

| Step | Required artifact / pass condition | Current state |
|---|---|---|
| P1: establish original | Exact source/dependency inventory, release optimization flags and normalized host parameters reviewed, thin-binding build command/binary, original O/LSE checks on all shapes and actual non-split kernel trace | Source identified; build/execution not done |
| P2: validate measurement | Same original measured against itself with labels/order swapped; graph capture and stated stability criteria pass; invalid windows/retry path retained | Not done |
| P3: reconcile knowledge | Current-source decision-to-Template map with explicit gaps; 18 prescribed entries and frozen 320-entry experimental library; no answer chain in public guides | Inventory recorded; reconciliation/guides/freezing not done |
| P4: seal task/evaluator | Exact contract, seeds, O/LSE oracle, four fault canaries plus a stale captured-output canary, changed-data checks on ordinary and captured paths, timing policy and budgets; no post-launch edits | Protocol fixed here; implementation checks not done |
| P5: build and inspect packages | P/S/N common-file and prompt diff passes; N has zero Template access material; P subset bytes identical in S; frozen public language-doc allowlist; evaluator/source files excluded | Not done |
| P6: persistence/access/accounting | Non-target multi-output/graph language and native start/resume smoke, resolved model/settings, empty isolated compiler/cache roots, queue-adjusted budget/timeout test, retained native transcripts and usage; instructions/access audited | Prior machinery exists; new campaign checks not done |
| P7: independent question-to-evidence review | Reviewer checks user question -> exact source -> Template exposure -> submitted computation -> original comparison -> claimed outcome, as well as P1-P6 receipts | Design review separate; execution receipts pending |
| P8: nine-run execution | Publish a prelaunch receipt with all checks true, seal schedule/config/inputs, then execute slots exactly as specified | Launch disabled |
| P9: confirmation/report | Frozen final sources, hidden O/LSE checks, three timing windows, all nine outcomes and accounting | Not started |

Preparation has at most two hours of active engineering work before reporting any unresolved build/evaluator blocker. Source dependency setup is project/container-local. Do not spend a model run debugging preparation. No paid infrastructure, system changes, commits, pushes or deletion of retained work. No frontend/backend expansion is needed.

`campaign.json` is the experiment-local authority for settings; it is not a new runtime schema or product API. Actual packages, binaries, raw inputs/records, prompts/sessions and receipts live under a new `KTA_DATA/dev/source-attention-control-v1/` directory. Preserve all earlier data. The launcher must verify a sealed prelaunch receipt and frozen file identities; no receipt or changed files means no launch. This small check belongs in the existing experiment runner, not a new backend. It is not implemented merely by setting `launch_allowed` in the design file.

Sealing creates the executable data-root campaign manifest with populated receipts and `launch_allowed=true`, after verifying its scientific settings against this design. The repository design file remains a record of the pre-execution state. Readiness metadata may advance with receipts; changing task/treatment/measurement/budget settings requires a protocol revision. Successful passage of design review alone never sets execution readiness true.

## 11. Deliverables and design check

Deliver a source identification card readable without hashes, the three exact public packages, a complete nine-slot outcome table, per-shape original/generated microseconds and percent differences, time/token/cache/cost to milestones, chosen Template/model-supplied choices, frozen selected sources and rerun commands, every failure/deviation and the remaining gaps. Native billing stays unavailable unless a provider returns it; label API-equivalent estimates and retain rates separately from GPU cost.

The research claim, givens, unverified prerequisites, bounds, failure modes, observables, settings, repair policy and ownership are specified above. The user requested source reconstruction and the no-Template control. The concrete FA2 revision/specialization, 18-rule exposure, O/LSE tolerances, three repetitions, graph timing and 30-minute/20-submission caps are explicit assistant defaults recorded in the decision log. Alternatives rejected for this campaign: generic SDPA comparator, old-run/new-control pooling, one sample per arm, full multi-model/multi-case matrix, hidden prompt coaching, and post-result rule repair. Changing a default before launch requires one reviewed protocol revision; changing it after launch requires a separate campaign. Independent review must attack these choices before execution receipts are sealed.
