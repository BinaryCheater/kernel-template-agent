# 28 The GB10 programme: building the first instance of the performance model

Status: plan set in discussion with the user on 2026-11-03/04. The model it builds is specified in [27](27-performance-model.md). Numbers marked *proposed* are the assistant's suggestions and remain open until the user confirms them (§1). Nothing here is implemented yet; the implementation starts in a fresh workspace (§6), and the WP28 worktree is kept read-only as a source of data. This document supersedes the plan in §4 of 25 where the two differ.

### 1. Objective contract

**Goal.** Before running anything, for the kernels in our search space (and later for hardware of our own design), decide: which design is faster, where the bottleneck is, and how much a design or hardware parameter change would gain.

**Products.** The machine description for the GB10; the engine; the SASS and template front ends; the probe ladder with its records; the validation set with its error table; the analysis outputs of 27 §3.

**Acceptance evidence** (*proposed*, to be confirmed):

| Metric | Target |
| --- | --- |
| Ranking on the test set | Kendall τ ≥ 0.8 among design variants of one shape; the best variant identified |
| Bottleneck correctness | for real-GPU ablations and parameter changes, predicted direction right every time; magnitude within 20–25 % |
| Time | median error ≤ 10 % on the test set; no case above 20–25 % |
| Probe ladder | every level within its own budget |

**Looks like success but is not.** Error pushed down by a kernel-level residual fit · a case decided by a bandwidth floor counted as mechanism validation · agreement with a profiler's utilisation percentage · many probes with no improvement on the validation set · per-case hand-written adapters · tuning parameters on the test set.

**Partial success.** A kernel type may be marked unsupported with the missing primitive named. It stays in the validation set.

### 2. Governing rules

These exist to keep the work pointed at the validation set. The WP28 effort lost months by lacking them.

**R1 — Front-end check before timing.** For every kernel, work counts (instructions per pipe, global sectors, shared wavefronts, trip counts) must match the hardware counters within *proposed* 2 % before its time error is examined. A wrong execution path cannot be repaired by timing parameters.

**R2 — Error budget decides refinement.** A mechanism may be refined further only if (a) critical-path attribution shows it contributes more than *proposed* 3 % of error on a validation kernel, or (b) it is a variable of the search space or of the hardware design. Otherwise it is fixed as a constant or a fitted function, recorded, and the work moves on. Investment cap: *proposed* two discriminating experiments per open question; if still unidentified, fix the parameter, mark it *unidentified*, continue.

**R3 — Residual terms are allowed, visible and temporary.** A residual term must (1) attach to a named component or path, never to a kernel; (2) be a constant or a linear function of a named quantity; (3) be fitted on microbenchmarks only, never on the test set; (4) be entered in the ledger with its hypothesis and data; (5) appear in outputs as prediction-with and prediction-without, plus its critical-path share. Residual terms do not participate in hardware what-if: any residual sitting on a design variable must become a mechanism before that variable is explored.

**R4 — Approximate but explicit.** Unknown values take a default rule and are counted. `NotSupported` is reserved for a missing *primitive*, never for a missing value.

**R5 — Generic by construction.** Adding a kernel costs a template-parameter mapping (tens of lines), not a per-case contract.

**R6 — Light provenance.** Commit, raw-data path, measurement conditions. No hash-locked review gates.

**R7 — The agent does not grade itself.** The validation set, the budgets and the acceptance targets are the user's; the agent proposes changes to them and never applies them. "Accepted" is the user's word.

### 3. The probe ladder

Each level is *predict first, then measure*. A level's error can only be attributed to what that level adds.

| Level | Probes | Establishes |
| --- | --- | --- |
| 1 Single operation | dependent chains, independent streams | latency, throughput, issue rules |
| 2 Unit saturation | many warps on one unit | capacity and scope |
| 3 Pairwise mixes | FFMA+LOP3, LDSM+HMMA, TMA write + LDSM read, MUFU + FFMA | sharing and interference (an unexplained 23 % LOP3+FFMA slowdown already sits here) |
| 4 Memory paths | latency, bandwidth, in-flight requests, working-set sweeps, access patterns | queue depths, L2 capacity and hit curve, DRAM behaviour, CPU contention |
| 5 Async and sync | cp.async groups, TMA, mbarrier hand-offs, barriers, flag polling | completion latency, hand-off chains |
| 6 Pipeline patterns | producer/consumer at stage depths 1…n, ping-pong, multi-buffering | overlap of combined mechanisms |
| 7 Real kernels and ablations | full kernels; no-load, no-compute, stage sweeps | end-to-end, and the deltas between ablations |

Existing material enters at its level: the ~40 upstream probes of `agentic/tilelang/microbench/`, the B1–B40 suite of `calibration/suite/bench/` (research repo, read-only), and the WP28 measurements that are directly usable as parameters (tensor rates by precision, shared-allocation rule, bank results, L2-hit data). Each is re-run as *predict then measure* before it counts.

### 4. Measurement protocol

1. **Environment record** per batch: driver, CUDA, power mode, GPU idle state before and after, CPU load. GB10's LPDDR is shared with the CPU; CPU load moves bandwidth.
2. **Clock.** The model predicts cycles. Conversion to time uses the measured effective clock: either a fixed application clock (an administrative setting on spark, applied by the user) or a calibration kernel of known cycle count run before and after each batch.
3. **Cache state**, declared per case and given to the model as input: *warm* (same inputs, back-to-back, warmed up), *cold* (L2 flushed by a write larger than L2), *rotating* (several input copies in turn; decode-style streaming). *Proposed default:* warm; decode-class kernels rotating.
4. **Repetition.** N warm-ups, M timed runs; report median and IQR. A compute canary and a memory canary bracket every batch; a canary outside its band invalidates the batch.
5. **Isolation.** One measurement job at a time on spark; profiling through the WP27 profiler helper.
6. **Profiler for counts only.** A fixed metric list; replay mode, cache control and clock control declared; profiler durations are never used as timing references, and utilisation percentages are converted to absolute active cycles over the un-profiled cycle count before any comparison.
Two construction skills fix the operations this section and §3 depend on: `.agents/skills/gb10-measure` (one comparable batch on spark) and `.agents/skills/write-probe` (one mechanism per probe, prediction before measurement, registration as a regression). The loop of §7 itself is not a skill.

7. **Test tiers.** After each model change: re-run the ladder and the validation set on CPU (no GPU). GPU only for new data. One script produces the error table, which is the sole progress indicator.

### 5. Deliverables and phases

The aim is the fastest path to a usable model, because operator optimisation waits on it.

| Phase | Deliverable | Done when |
| --- | --- | --- |
| P0 Specification | this document finalised; related-work note; machine-description schema; program-representation schema; the GB10 instance written from `gb10.json` + the structure now hard-coded in `sm.py`/`ws.py` | user review of §1 numbers, §2 rules, validation-set composition |
| P1 Engine | the upstream engine made description-driven; critical-path attribution; what-if; assumption ledger; CPU replay of the upstream 22-kernel GEMM validation reproduces upstream numbers | replay matches; error table script runs |
| P2 Front end | generic SASS front end with per-lane affine value tracking (addresses, sectors, wavefronts computed; defaults counted); template front end for HGEMM; R1 check wired to NCU counts | R1 passes on the HGEMM variants |
| P3 First loop | HGEMM variants of one shape (stages, tiles, swizzle, raster order): predict → measure → attribute → fix, one full round with the protocol of §4 | ranking and bottleneck targets met on HGEMM |
| P4 Extend | FA-forward variants (adds XU/softmax, named barriers, K/V reuse in L2, uneven work); CTA scheduling layer for causal/grouped/persistent work; LDS/STS/SHFL calibrated | targets met on the test set |
| P5 Projections | L0/L1 derivation; MIP export for the HGEMM and FA templates; spot-check against L2 | MIP picks within budget of simulated values |
| P6 Portability | Ascend-style description written structurally; Ventus ISA front end and description when RTL is available; the ladder run against RTL | three descriptions in one schema; RTL-calibrated parameters |

**First measurable milestone.** P1's replay of the upstream 22-kernel GEMM set gives the baseline error table; every later phase is judged as a change to that table. P2 is the riskiest engineering piece (generic value tracking); to keep P3 from waiting on it, P3 starts on the template front end plus SASS instruction mixes, and the generic SASS front end replaces it when R1 passes.

**Test programs.** There are two kinds, and one rule between them.

- *Microbenchmarks* (also called probes): written on the spot, each isolating one mechanism; the ~40 upstream ones plus every new one. They are where parameters come from, and each one stays as a regression.
- *Test set*: real kernels, the main kernels of neural-network inference, each in a few configurations. They score the model and reveal what is missing. **Parameters are never tuned on the test set.** The 24 library operators of WP28 are kept only as a breadth check.

| Kernel | Mechanisms it exercises |
| --- | --- |
| HGEMM (tile / stage / swizzle variants) | tensor pipe, shared memory, cp.async/TMA pipelines, L2 reuse |
| FA forward (prefill) | all of the above + softmax on MUFU/FMA, intra-warp sync, K/V reuse |
| decode attention (GQA split-KV) | DRAM bandwidth, low compute, multi-kernel sequences |
| quantised GEMV (w4a16) | DRAM, unpacking on the ALU, conversions |
| RMSNorm / softmax | reductions, shuffles, LDS/STS, bandwidth floor |
| MoE grouped GEMM | unequal per-CTA work, CTA scheduling |

### 6. Workspace and what is carried over

A fresh branch from `main` in this repository. The WP28 worktree stays untouched and read-only.

Carried over: the `gb10sim` engine already in `integration` (copy of upstream `agentic/gb10`); the measured entries of `gb10.json`; the WP28 measurements listed in §3; the remote measurement tool chain (WP27 profiler helper, cubin/SASS capture, NCU capture scripts); the upstream probe sources, read from the research repository.

Not carried over: the 24 per-case contracts and family binding modules; the hash-locked evidence/review procedure; the per-lane exact-value interpreter as the only execution route.

### 7. Agent working loop and stop conditions

One research agent; a measurement script; a validation script; one state file (open questions, budget ledger, fixed parameters, next step with its expected gain on the error table). No queue, no multi-agent, no database until an observed need.

The loop: pick the largest budget-exceeding attribution on the validation set → hypothesis → the smallest discriminating probe → predict → measure → update description or engine → re-run ladder and error table → record. Each work package states up front which error-table rows it expects to improve and by how much, and closes against that.

Stop and ask the user when: a change to the validation set, a budget or an acceptance target is wanted; an administrative action on spark (clock setting, driver, packages) is needed; a kernel class needs a new primitive that changes the schema; two discriminating experiments have not identified a mechanism that sits on a design variable.

### 8. Open points for the user

1. The *proposed* numbers in §1, §2 and §4.
2. Held-out set composition (§5).
3. Clock setting on spark: have the user fix the application clock, or use the calibration-kernel method only.
4. Ascend in P6: structural description only, or a measured validation as well.
5. Whether the related-work survey precedes P0 or runs alongside it.
