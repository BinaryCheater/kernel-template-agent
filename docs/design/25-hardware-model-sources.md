# 25 Hardware model sources: inventory and proposal

Status: inventory of what exists (read-only, 2026-11-01) plus a **proposal that needs user review**. Nothing in KTA's code or data was changed by this document. Upstream files were read, not modified.

Question it answers: KTA's L1/L2 models and the GB10 instruction-level calibration were built separately. Before making the calibration the single source, what does it actually contain, and how far is KTA from it?

## 1. Sources found

Three bodies of GB10 timing knowledge exist. Paths are absolute; upstream = `/Users/dingyi/Projects/ai-infra-codesign-ventus` at commit `0eef497` (2026-10-20), a clean checkout of AI-Infra-Co-Design apart from `runs/gb10/sim_shapes.json` and an untracked `models/`.

### A. Upstream mechanistic simulator, `agentic/sim` (release tag `gb10-model-v2`, commit `468e846`, 2026-10-15)

| Part | Where | What |
|---|---|---|
| Parameter contract | `calibration/spark-6951/contract.yaml` (sha256 `3bbaabaa34bcc6ef…`) | 222 top-level keys, many of them tables or curves (rates by residency, by stride, by run length; latency distributions; interference pairs). Every value names its benchmark; nothing is fitted to kernel times |
| Structural config | `agentic/sim/config/spark-6951-gb10.yaml` | choices that are not numbers (scheduler policy, bank width, L2 policy, copy path shared or separate, input `stream` or `counts`), each with the benchmark that justified it, "phenomenological" where unconfirmed |
| Microbenchmarks | `calibration/suite/bench/` | 45 sources: B1–B40, composites C1 tensor tile, C2 TMA tile, C3 softmax rows, `bx_counters`; driver, fit and ncu cross-check scripts |
| Shared event vocabulary | `agentic/sim/events.py` | one event model per warp (IssueCopy, CopyDone, Wait, BarrierArrive/Release, Load, Store, Compute) read both by the untimed Z3 verifier (`explore.py`) and by the timed simulator |
| Timed reader | `run.py`, `warp.py`, `tensor.py`, `memory.py`, `smem.py` | discrete-event simulation: M1 warp, M3 shared memory banks, M4 async copy window, M5 L2 (LRU over measured capacity) and DRAM queue, M6 barrier, M7 compute, M8 block scheduler; warp issue model with scoreboard, four schedulers, the scheduler-to-LSU queue; tensor pipe per sub-partition; TMA engine |
| Instruction input | `isa.py`, `sass.py` | pipe table: SASS opcode → pipe, cost, warp hold, result latency, each from a benchmark; an unknown opcode is an error, never a guess. `input: stream` reads the kernel's SASS; `input: counts` builds phases from per-step instruction counts |
| Accuracy (model card) | `calibration/spark-6951/MODEL_CARD.md`, `coverage.md` | 30 GEMM designs: median 8.6 % (stream) / 8.1 % (counts), worst 34–35 %; softmax rows within 7 %; BF16 tensor tile +9/+6 %; TMA tile −21/−18 % (open) |
| Known gaps | `notes/sim_gaps.md` | 41 numbered rows; open: copy completion (cp.async and TMA), mma/MUFU issue beside FFMA, shuffles beside loads, second-wave blocks, multicast/DSMEM unmodelled; register-file bank conflicts, L2 set hash, DRAM channels "not measurable from user space so far" |

Scope stated by the card: tiled GEMM, tensor tile, softmax rows. Not covered: power, clock behaviour under load, FP64, clusters, unified-memory traffic inside a kernel.

### B. Upstream TileLang-specific model, `agentic/gb10` (commits 2026-10-20)

| Part | Where | What |
|---|---|---|
| Analytic model | `agentic/gb10/model.py` | per k-tile unit times (tensor, smem, port, latency/Little's law, bubble, spill), occupancy from driver limits, launch-order L2 simulation (LRU, scan-resistant variant, reuse-distance variant), DRAM per wave, dirty-L2 write-back |
| SASS SM simulator | `agentic/gb10/sim/sm.py`, `kernel.py`, `ws.py` | runs the real SASS under the compiler's control bits (stall, yield, six dependency counters, wait mask); warp-specialized TMA kernels with mbarrier hand-offs and register reallocation |
| Constants | `calibration/spark-6951/gb10.json` | ~35 top-level values plus 42 `sim` values, from 40 probes in `agentic/tilelang/microbench/` (clock, mma rate/types, mlp_stream, chase, tileload, l2policy, dram, instlat, mbar, cpq, storepat, sfu, interfere, …) and ablations of real kernels |
| Accuracy | `kernel-lab/exports/gb10-simulator-calibration-20261020-v2` (research repo) | the export bundle uses this simulator: gemm +82 %, w4_gemv +72 %, gqa_decode +55 %, gemm_basic +3.3 %; 10 of 20 kernels with complete binary-bound comparisons |

### C. KTA seed (`knowledge/seed/hardware/`, decision D467)

| Part | What |
|---|---|
| `arch_sm12x.yaml` (class sm12x@1) | 9 resources, 3 walls, 22 operation classes, 1 table (bank-conflict penalty). Structure written by hand in WP15 |
| `model_gb10.yaml` (gb10@1) | ~37 limits. About 20 scalars and the bank-conflict table are copied from contract.yaml (same sha256 as A) and labelled `documented`; device attributes from the CUDA guide; `sfu_ops_per_cycle_per_sm`, `mma_latency_cycles` (32, where the contract measures 29–30.1), `smem_queue_depth` are `default` |
| Data copy on spark-hk | gb10 v2 adds 7 KTA-measured values (37 documented, 7 measured, 5 default) |
| Readers | L1 (`models/l1.py`), L2 (`models/l2.py`, `l2graph.py`), projection MIP (`solve/projection.py`); L3 adapter raises NotSupported |

## 2. Mechanism inventory

One row per mechanism from upstream `coverage.md`, extended with mechanisms that appear only in `gb10.json`. "Aggregate" means KTA has a single rate for the resource and nothing about the mechanism that shapes it.

| Mechanism | A `agentic/sim` | B `agentic/gb10` | C KTA |
|---|---|---|---|
| Issue: 4 schedulers, in-order warps, SASS stream | modelled (B17, B21, B27, B29) | modelled with control bits | aggregate `issue` resource |
| Scoreboard latencies (FFMA, LDS, mma, ldmatrix, ex2, shfl) | modelled | modelled | partly (dram/l2/async latencies as op-class latency) |
| Tensor pipe, mma hold, in-flight to saturate | modelled (B20, B31–B34) | modelled | aggregate `tensor` |
| ldmatrix path and latency | modelled (B32) | modelled | folded into `smem_access` |
| Scheduler-to-LSU (MIO) queue, LSU drain rate by residency | modelled (B27, B28, B30) | modelled | absent |
| Shared-memory banks, wavefronts from addresses | modelled (B6, B26.6, B30.3) | bank ways for linear layouts | one table by word stride |
| cp.async completion under load | open gap (#23, #24, #40) | modelled by port/L2 share | latency + L2 bytes |
| TMA engine, 2-D maps, mbarrier | modelled, −21/−18 % open (#30) | modelled (ws.py) | latency + L2 bytes |
| SFU and converts | modelled (B21, B38) | modelled | `sfu` with a default rate |
| Shuffles in the LSU queue | modelled, gap beside loads (#33) | — | op class, no queue |
| L1 (carve-out, bypass by copies) | modelled (B5, B14) | — | absent |
| L2 capacity, policy (scan-resistant), reuse-distance hit curve | LRU over measured capacity | launch-order LRU, protect policy, reuse-distance curve | aggregate `l2` bandwidth only |
| Per-SM port to L2, L2 share by blocks in flight | in memory model | modelled (80.7 GB/s per SM, aggregate) | absent |
| DRAM: rate by run length, read/write mix, loaded latency, CPU contention | rates, queue | modelled with curves | one rate |
| Store path shared by the two SMs of a TPC | — | modelled (storepat) | absent |
| Atomics and reductions | contract only (B24) | modelled (red rates) | absent |
| Registers: allocation, residency, spills | modelled (B9, B19, B34, B40) | occupancy + spill | wall at 255 per thread |
| Register-file bank conflicts | not measured (needs cubin patching) | — | absent |
| Block scheduler: dispatch order, waves, refill | modelled (B9, B16) | waves, block replacement | absent |
| Launch, CUDA graph gap | contract | modelled | `host_launch` |
| Clock: sustained, by load | contract only | constant 2.52 GHz | constant 2.476 GHz |
| Interference: mma beside FFMA, TMA beside cp.async, shfl beside LDS | measured, partly modelled (#27, #31, #33) | `interfere` probe | `interactions` empty |
| L2 persistence window, clusters, DSMEM, multicast | measured, not modelled | tier-4 probes | absent |
| Unified memory | contract only (B12, B23) | — | absent |

## 3. Findings

1. **The calibration is rich; KTA uses a thin slice of it.** The contract holds 222 keys with curves and interference measurements; KTA copied about 20 scalars and one table. None of the curves (DRAM rate by run length, L2 hit by reuse distance, LSU rate by residency) or interference pairs reach KTA's L1/L2.
2. **Upstream already has the L2/L3 relation the user asked for.** `agentic/sim` runs one contract through two inputs: `counts` (per-step instruction counts, a transaction-level reading) and `stream` (the SASS). On 30 GEMM designs they agree closely (median 8.1 % against 8.6 %), and the untimed verifier reads the same event vocabulary. KTA's own `models/l2.py` is a parallel implementation with its own parameters: this is the inconsistency to remove.
3. **Upstream itself has two simulators with two constant files.** `agentic/sim` + `contract.yaml` (general, documented, better accuracy on its scope) and `agentic/gb10` + `gb10.json` (TileLang TMA warp-specialized kernels, real kernels, larger errors). Values overlap but differ in definition or measurement: clock 2.476 against 2.52 GHz; DRAM read 235 against 237 GB/s; "L2 bandwidth" 1.02 TB/s (contract `l2_bytes_per_s`) against 3.47 TB/s (gb10 `l2_aggregate_GBs`, warm L2, many loads in flight). The single-source problem exists before KTA.
4. **The 82 % figure is one export case, not the simulator's validation.** The export bundle KTA's L3 design pointed to was built on `agentic/gb10`, with per-case traffic mappings written by hand; on its own 22-kernel TileLang validation `agentic/gb10` reads median 5.3 %, worst 16 % (whole kernel). `agentic/sim` reports 8.6 % median on its own GEMM set (a different kernel family and lowering).
5. **Mechanisms the user named.** Register-file bank conflicts: upstream judged them unmeasurable from user space so far. Memory-side contention: L2 policy, DRAM loaded latency, CPU contention, TPC store sharing, TMA/cp.async mixing are measured and mostly modelled upstream; KTA has none of them.

## 4. Direction after user review (2026-11-02)

User decisions (in chat): one simulator is chosen, copied into KTA and iterated there; L1, L2 and L3 come from that one simulator, never from two simulators; a simulator holds real (measured) parameters and fitted calibration parameters, and a fitted parameter is refined into a measured mechanism as benchmarks isolate it. `agentic/sim` is the user's earlier, separate version; `agentic/gb10` is the more calibrated one.

Assistant's recommendation (not yet confirmed):

- **Base: `agentic/gb10`.** On its own 22 TileLang GEMM kernels (`runs/gb10/sim_kernel_level.json`) whole-kernel error is median 5.3 %, worst 16 %; its engine obeys the compiler's control bits; it covers TMA warp specialization, mbarrier, stores, atomics and decode shapes, which is what KTA generates. The 82 % rows of the export bundle are separate cases with hand-written traffic mappings (adapter inputs), not this validation set.
- **Brought over from `agentic/sim`:** the discipline (one parameter file; every value labelled measured / fitted / structural with its benchmark; an unknown opcode is an error, where `agentic/gb10` gives it `default_var_latency_clk`), the coverage table and gap log as the procedure for learning the hardware, and contract measurements `gb10.json` lacks (L1, LSU rate by residency, block dispatch order, interference pairs, atomics curves).
- **Before changing anything after the copy:** constants hard-coded in `agentic/gb10` code (clock 2.54e3 cycles per µs in `kernel.py`, TMA 81e9 and 1860e9 in `ws.py`, occupancy limits in `kernel.py`) move into the parameter file; then the 22-kernel validation is replayed on spark to show the copy reproduces the upstream numbers.
- **One engine, three inputs:** L3 = compiled SASS; L2 = a synthetic instruction stream generated from Template effects (each operation class expands to its instruction group, checked against compiled SASS by the instruction-mix check); L1 = closed form from the same pipe table and parameter file (per-resource busy = Σ count × cost; recurrence bound = latency chain / stages). Differences between levels then come only from the input, not from parameters.

## 4a. Original proposal (superseded by §4)

**One contract, several readers.**

1. **Source.** Upstream `contract.yaml` + structural config, identified by sha256 and commit, becomes the single parameter source. The two upstream constant files are reconciled first (one definition per quantity, the other kept as a named alias or retired). KTA stops hand-copying values: `model_gb10` is generated from the contract with each value's benchmark as provenance (`measured` upstream, not `documented`).
2. **L3 = upstream `stream` reader** on the compiled SASS.
3. **L2 = upstream `counts`/event reader** fed by events generated from Template effects (KTA's job: effects → per-warp event programs and instruction counts). KTA's own `models/l2.py` is retired or reduced to that translation.
4. **L1 = closed-form steady state of the same contract**: per-resource busy per main-loop iteration using the contract's rates and curves (looked up at the point, tabulated for the MIP), plus the recurrence bound. Mechanisms that need state (L2 reuse, waves, queues) enter L1 as tables computed by the L2 reader over a few variables.
5. **Consistency checks, each cheap:**
   - instruction-mix check: SASS instruction counts per class against the counts derived from effects;
   - L1 against L2-counts on sampled points: rank correlation and the optimistic bound (L1 ≤ L2);
   - counts against stream on compiled candidates (upstream already reports this comparison).
6. **Mechanism growth stays upstream-shaped**: a new mechanism = benchmark + contract entry + structural choice + coverage row; KTA reads it, it does not re-declare it.

**Constraint to resolve.** KTA's AGENTS forbids importing research-repo code, and AI-Infra-Co-Design is a separate repository. Options: (a) read the contract as data and run the upstream readers as an Executor job (the L3 adapter's existing design, extended to L2); (b) vendor the readers into KTA with provenance; (c) make the readers a small shared package both repositories depend on. This is cross-cutting and changes who owns the hardware model: user decision.

## 5. Questions for the user

1. Which upstream simulator is the base: `agentic/sim` (released, general) with `agentic/gb10`'s TMA warp-specialization and mbarrier mechanisms ported onto it, or the reverse?
2. Ownership: (a), (b) or (c) above?
3. Scope of the first round: GEMM and FA2-forward kernels on GB10, so that the consistency checks run on a fused kernel early?
