# 29 Mechanism inventory of the GB10 resource network

Status: written 2026-11-04 for phase P0 of [28](28-gb10-programme.md) §5. It lists what the GB10 instance of the performance model ([27](27-performance-model.md)) must eventually contain and what the current engine (`src/kta/models/gb10sim/`, a copy of upstream `agentic/gb10` at commit `0eef497`) actually contains. It answers "what is still missing" from the hardware side; it is **not** derived from kernel errors (the error table in `docs/design/perfmodel/` does that). Status was read from the engine's code, not from upstream prose.

## Legend

**Status in the current engine**

| Code | Meaning |
| --- | --- |
| M | modelled, parameter measured by a probe that isolates it |
| A | modelled, but a parameter is assumed, fitted on an ablation, or a structural choice not checked by a probe (named in the row) |
| P | partly modelled: the mechanism exists for some operation classes or configurations only |
| X | absent: nothing in the engine; an instruction that needs it gets the default rule (issue slot only, or the assumed 20-cycle result) |
| F | front-end input, not a hardware mechanism, but it decides how much of one is exercised |

**Test-set kernels** (28 §5): G HGEMM · F FA forward · D decode attention (GQA split-KV) · Q quantised GEMV w4a16 · N RMSNorm/softmax · E MoE grouped GEMM · 8 FP8 GEMM (round 3).

**Variable**: S = the search space changes how much of it is used (tile, stages, copy path, warp specialisation, store path, split, launch order, dtype); H = a hardware design variable (count, rate, capacity, latency, sharing); — = neither (a fixed rule).

**Sources**: `gb10` = upstream `notes/gb10_simulator_design.md` (origin/main) §; `cov` = upstream `calibration/spark-6951/coverage.md` row; `gap#` = upstream `notes/sim_gaps.md` row (that log belongs to the older `agentic/sim`); `25` = [25](25-hardware-model-sources.md) §2; `26:nn` = [26](26-gb10-calibration-method.md) §5 SM120-reference chapter nn; `PTX`, `TG`, `NCU` = the official PTX ISA, Blackwell tuning guide, Nsight Compute profiling guide. Probe names are upstream `agentic/tilelang/microbench/*.cu`.

## 1. Issue, scheduling and dependencies

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Warp scheduler: one instruction per cycle per sub-partition, greedy then oldest; warp i → sub-partition i mod 4 | gb10 §4.2; `sched.cu`; 25; 26:07; TG | P: M within a block (replayed: 14.91 vs 15.10 cycles per add); between two co-resident blocks (round 2, probe `mma_pair`, batch r2) the older block of a skewed pair finishes almost as if alone (317 k vs 311 k cycles alone; replay 356 k, +12 %); the replay's share comes from the tensor pipe's queue order, not the issue pick. Round 3 (no new probe, D768): with the skew loop's measured speed (23 cycles per FFMA-loop iteration, probe `launch` work points, batch r3) all three pairs fit fair sharing (s0 lockstep 530 k; s4096 first 469 k vs 464 k; s16384 first 317 k vs 311 k, span 694 k vs 688 k); oldest-block-first rejected (s0 first 319 k vs 530 k measured). M: the replay's least-loop-progress share is right; the mma_pair residuals are the skew loop (taken BRA about 5 cycles beyond its stall count, replay 17.6 vs 23 cycles per iteration), below 3 % of every valid test row | all | H (issue width, policy) |
| Yield bit, reuse bits | gb10 §4.1, §8.7 | X (parsed, ignored) | all | — |
| Stall counts from the control bits cover fixed latencies | gb10 §4.1, §4.3 | M (structural: compiler bits); `hmma_latency_clk` 29 is measured but unread | all | — |
| Six dependency counters, wait mask, `DEPBAR.LE` | gb10 §4.3; PTX | M for LDSM (34 + 4/way), MUFU/XU (18), LDG, cp.async groups, STSM, UTMACMDFLUSH; A elsewhere | all | — |
| Result latency of every other counted instruction (LDC, S2R, ATOMS, uniform ops, MEMBAR, …) | gb10 §4.11, §8.6 | A: `default_var_latency_clk` 20 (assumed), counted per opcode; LDS and SHFL measured in round 1 (rows below) | F, N, D, Q | — |
| Read-counter release of non-memory instructions | gb10 §4.3 | A: `rbar_release_clk` 8 (assumed) | all | — |
| Register-file bank conflicts, operand collector, reuse cache | gb10 §8.7; 25; gap list "not measurable from user space" | X | all | H |
| Instruction cache / fetch | gb10 §8.7 (`icache.cu`: full rate to 32 KB of loop, −10 % at 64 KB) | X (measured, not modelled) | F, E (large unrolled bodies) | S (unroll), H |
| Intra-warp divergence and reconvergence (`BSSY`/`BSYNC`, `BRA.DIV`) | gb10 §4.4, §8.7 | X (lane 0 stands for the warp; `BRA.DIV` never taken) | N (tails), E, D | — |
| Branch and loop resolution from warp-uniform values | gb10 §4.4 (`audit_values`: 46/46) | F (fallback rules: main-loop trip count, unconditional taken, else fall through) | all | — |
| Issue interference: a copy or ldmatrix beside an mma stream on one sub-partition | gb10 §8 (`interfere.cu`: 2.3 cycles per copy) | X (measured, judged negligible) | G, F | H |
| Uniform datapath (`U*` ops) | — | X (issue slot only; no unit) | all | H |

## 2. Compute pipes

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Tensor pipe per sub-partition: every `HMMA/IMMA/QMMA/OMMA` holds it 16 cycles; type sets FLOP/clk | gb10 §4.5 (`mma_rate.cu`, `mma_types.cu`); cov tensor; 26:04; probe `mma_pair` (round 2) | M; round 2 confirmed at chip scale in the HGEMM warp tile's loop (ldmatrix + 32 HMMA per k16): 310.5 k cycles at 1 and at 48 blocks (replay 316 k, +1.8 %), two blocks per SM 530 k (−1.0 %); no chip-level limit; depth: structural (a pipe held 16 cycles per instruction) | G, F, E, D | S (dtype, mma shape), H |
| mma issue beside FFMA on the same sub-partition (+43 % in B34) | gap#27; cov tensor | X | F (softmax beside mma) | H |
| mma result latency beyond stall counts; accumulators in flight to saturate | gb10 §8.7; B31–B34 | X (stall counts only) | G, F | H |
| FMA pipe: FFMA/FADD/FMUL 1 cycle; IMAD, IMUL, HFMA2, HADD2, HMUL2 2 cycles | gb10 §4.5 (`alu.cu`, `alumix.cu`) | M | all | H |
| ALU pipe: LOP3, PRMT, SHF, IADD3, ISETP, SEL, LEA, F2FP, min/max, compares, … 2 cycles | gb10 §4.5 | M | all | H |
| `HADD2.F32` on whichever of FMA/ALU is free | gb10 §4.5 (`alumix.cu`) | M | Q, N | — |
| LOP3 + FFMA mix runs 23 % below two independent pipes | gb10 §4.5; 28 §3 level 3 | X (measured, unexplained) | Q (unpack), N | H |
| SFU/XU: `MUFU` 8 cycles per instruction per sub-partition, 18-cycle result | gb10 §4.5 (`sfu.cu`); cov SFU | M | F, N | H |
| Conversions `F2I`, `I2F` (incl. `I2F.F16`), `F2F`, `FRND`, `F2IP` on the XU | gb10 §4.5 (`cvt.cu`) | M | Q, F | H |
| `I2FP.F32.S32` (int → fp32) pipe | upstream `ee6534c` (`cvt.cu`: ALU, 2 cycles) | A: this copy (upstream `0eef497`) still puts it on the XU; upstream main moved it to the ALU | Q | — |
| ex2 with FFMAs between them from one warp | gap#35; probe `ex2_ffma` (round 1, batch r1) | M (round 1): separate pipes confirmed, MUFU-bound within 3 %; the FFMA-only point is faster than the issue limit (open) | F, N | H |
| FP64, DP4A/IDP, bit-count and other untabled ALU ops | — | X (issue slot only) | — | H |

## 3. Memory-instruction slot, LSU and shared memory

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Memory-instruction slot per sub-partition (MIO): LDSM 11.7, STSM 10.8, LDGSTS 6.0 cycles; RED 2.65 per line | gb10 §4.6, §4.7, §4.9 (`instlat.cu`, `tmastore.cu`, `cpq.cu`, `stq.cu`) | M | G, F, E | H |
| MIO slot for `LDG` | engine code | P (LDG waits for the slot but never holds it) | D, Q, N | H |
| Scheduler-to-LSU queue depth, LSU acceptance rate by residency | cov LSU queue; gap#9, #13, #16; 25 | X | F, N, D | H |
| Address stage: one 128-byte line per cycle per SM | gb10 §4.7 | A ("design assumption, checked" with `cpq.cu`) | G, F, E, D, Q | H |
| Shared-memory array: 128 B/clk per SM, one queue shared by LDSM, STSM, cp.async data, TMA boxes and TMA-store reads | gb10 §4.6 (`smem_bw.cu`) | M | G, F, E | H |
| ldmatrix latency 34 + 4 per extra way | gb10 §4.6 (`instlat.cu`; derivation of 34 not recorded) | A | G, F, E | H |
| Bank conflicts | gb10 §4.6 (`smem_bw.cu`: 4 cycles × ways); cov banks | P: ldmatrix ways from the tile config when `smem_layout` is `linear`; not from addresses; none for LDS/STS | F, N | S (layout, swizzle), H |
| Plain shared loads/stores `LDS`, `STS` | gb10 §4.6, §8.6; probe `smem_ldsts` (round 1, batch r1) | M (round 1): MIO slot 5 cycles; array max(2, bytes/128 B/clk); LDS latency 24 + bytes/32; STS register read 3.25 per 128 B after the array (STS.128 at 4 warps −22 %); round 2 (2–6 warps): 12.0, 8.7, 7.0, 5.6, 4.35 cycles, neither this rule nor a per-sub-partition limit fits: `unidentified`, frozen after 2 experiments | F, N, D | H |
| Shuffles `SHFL` | gap#33; cov shuffle; probe `shfl` (round 1, batch r1) | M (round 1): MIO slot 5 cycles, 2 cycles on the shared-memory array per instruction, latency 28 | N, F, D | H |
| Shared-memory atomics `ATOMS`, CAS loops | gb10 §6.7 (GEMV driver, not copied) | X | Q (TileLang split GEMV reduction) | — |
| Shared memory vs L1 carve-out | cov L1; gap#7 | P (smem capacity enters residency; L1 capacity does not exist) | all | S, H |

## 4. Asynchronous copies, TMA and mbarriers

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| cp.async path: MIO → address stage → L2 (350) or DRAM (1560 + queue) → return port → array write → group | gb10 §4.7 | M | G, E | S (copy path, stages), H |
| cp.async completion spread under load | cov cp.async; gap#23, #24, #40 | X (fixed latency, even miss spreading) | G, E | H |
| cp.async beside TMA traffic (mixed 1.94 TB/s > either alone) | gap#31 | X | — (mixed lowerings) | S |
| TMA engine per SM: min(81 GB/s alone, 1.86 TB/s chip ÷ share) | gb10 §5 | A (both rates fitted on the TMA loads-only ablation) | G, F, E, D | S, H |
| TMA box latency: 440 cycles from L2, 1170 from DRAM, before transfer | gb10 §5 (`dramlat.cu`) | M | G, F, D | H |
| TMA box latency spread under load (a box waits for its slowest line) | gb10 §8.1 | X (decode pipeline depth gain missed) | D, G (decode shapes) | S (stages), H |
| mbarrier hand-off: arrive → waiter passes 134 cycles later (try_wait) | gb10 §5 (`mbar.cu`) | M | G, F, D, E | H |
| mbarrier arrive after a global store (217), arrive cost per lane (6.6) | gb10 §8.9 (`mbar.cu`) | X | F, D | H |
| Warp-specialised roles and waits matched by order | gb10 §5 | P (structural; TileLang WS lowering only; a TMA kernel without warp specialisation silently gets the default latency instead of NotSupported) | G, F, E | S |
| Register reallocation `USETMAXREG` hand-off | gb10 §5 | M (structural) | G, F | — |
| TMA store `UTMASTG`: pair 33.1 B/clk, chip 482 B/clk, fixed 457 | gb10 §4.9 (`tmastore.cu`) | M | G, F, E | S (store path), H |
| TMA multicast, clusters, DSMEM | gap#36, #37; upstream tier-4 probes | X (measured, not modelled) | E, F | S, H |

## 5. L1, L2 and DRAM

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| L1 hit path: 1 cycle per line (round 1; was 2), 45-cycle latency; LDG miss 342 cycles to L2 | gb10 §4.8 (`ldg_atomic.cu`); probe `l1hit` (round 1, batch r1) | M; round 3 (D765): lines and sectors per LDG from its width and its 32 lanes' addresses, L1 hits from an LRU line cache over kernel-parameter bases (128 KB less the resident blocks' shared memory, structural); ncu global-load sectors per block matched exactly (GEMV 1360 vs 1344, RMSNorm 512, softmax 256, combine 96) | D, Q, N | H |
| L1 capacity, replacement, sectors and coalescing from addresses | cov L1; gap#7; 26:02 | X (P2 front end computes sectors) | D, Q, N | H |
| Local memory (spills) `LDL`/`STL`: 4 cycles per line | gb10 §4.8 | M | any spilling kernel | S (registers) |
| Constant loads `LDC` | — | X (default latency) | all (prologue) | — |
| Per-SM L2 return port: 30.7 B/clk (16.7 cycles per 512 B) | gb10 §4.7 (`loopsteps.cu`) | M | G, E, D | H |
| Chip L2 rate for copy traffic 817.7 B/clk, shared by blocks in flight | gb10 §4.7, §6.3 (`loopsteps.cu`, `l2share.cu`); probe `l2share_copy` = P-2 (round 1) | M, rule wrong, `unidentified` (frozen after 2 experiments): round 1 per-SM delivery drops already at 16 blocks (13.6 µs vs 10.5 replayed); round 2 (probe `l2share_stages`, stages 1–4 × 1/16/48 blocks) the round trip with 1 stage is unchanged at 16 blocks, 2–4 stages lose 16–25 % at 16 blocks with the chip at 0.98 TB/s; not latency, not the chip rate; candidate: a share below the chip (TPC pair or GPC) | G, E, D, Q | H |
| L2 capacity (effective 16.78 MB) and policy: two-list LRU over k-slices of lockstep waves | gb10 §6.4, §8.2 (`l2policy.cu`) | A (capacity measured; policy chosen, over-counts panelled and multi-wave patterns by 20–35 %); round 2: for the HGEMM 2048³ raster twins the replay gives both the same DRAM traffic (16.8 MB) and ncu the same L2 hit rate (93.3 / 92.7 %, batch r1): the replay's raster gap does not come from this row (see the warp scheduler row). Round 3, probe `l2reuse` (batch r3), plain / .cg 16-byte re-reads back-to-back: miss fraction 0.02 up to 16.5 MiB, 0.08 at 18, 0.13 at 20.25, 0.62 at 24, 0.85 at 32.25 MiB (H-part; H-full 24 MiB LRU rejected); evict-first (.cs) lines are not kept beyond 1.5 MiB (H-stream for that kind); the generic front end now predicts warm misses from `l2_warm_miss_by_MiB` (D766). Under ncu every launch refills the L2 from sysmem (fill sectors = read sectors warm and cold): profiler hit rates are no evidence of the cache state (D767); depth: empirical (a curve over footprint) | G, E, D | S (swizzle, raster), H |
| L2 hit-vs-reuse-distance curve | `gb10.json` `l2_hit_vs_reuse_distance` | X in the engine (analytical model only) | G, D | H |
| Back-to-back L2 carry-over between runs (`stream_b2b.cu`) | gb10 §6.7, §8.5 | X (GEMV driver not copied) | D, Q | — |
| L2 latency under load | — | X (fixed) | all | H |
| L2 persistence window, set aside 18 MB | upstream tier-4 (`l2persist.cu`) | X | D, Q | S |
| L2 slices, set hash, NoC | gap list; 26:02 | X | — | H |
| Dirty-L2 write-back paid by a flush (`dirty_l2_extra_us`) | `gb10.json` | X in the engine (analytical model only) | measurement protocol | — |
| DRAM read rate by run length (204/233/237 GB/s), write 197, mix rule | gb10 §6.4 (`tileload.cu`, `mlp_stream.cu`, `dram.cu`) | M (wave-level) | G, D, Q, E | H |
| DRAM queue per SM: wave DRAM time spread over read misses, fair share by SM-equivalents | gb10 §6.4 | A (structural: even spreading, fair share) | G, D, Q, E | H |
| DRAM latency: 1560 cycles (copy, dependent chase), 1100 (streaming LDG; round 1), 1170 (TMA) | gb10 §4.7, §5; probe `dram_rotate` (round 1, batch r1) | M | D, Q | H |
| DRAM latency spread, channels, banks | gb10 §8.4; gap (B18) | X | D, Q | H |
| CPU contention for LPDDR5X | gb10 §6.4 (`cpuhog.c`: 237 → 214/171/138 GB/s) | F (an input, `CPU_DRAM_GBs` = 0) | D, Q | — |
| TLB misses (random chase beyond 1 MB: ~400 ns) | `gb10.json` `lat_l2_ns` source | X | D (large KV), Q | H |

## 6. Stores and atomics

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Store port shared by the two SMs of a TPC: 15 (4 B/lane) or 11 (16 B/lane; round 1, was 14) cycles per instruction per pair | gb10 §4.9 (`storepat.cu`); probe `store_tail` = P-1 (round 1, batch r1) | M; partner-busy rule confirmed at 2–16 blocks; no per-block tail | all | S (store width), H |
| Chip store limit 122 (round 1, was 137.8) / 570.9 B/clk | gb10 §4.9; probe `store_tail` (round 1) | M | all | H |
| 8-byte-per-lane stores (`STG.64`) | engine code | A (256 B at the 4-byte rates) | N, Q | — |
| Store queue: 16 STG/RED instructions before issue waits | gb10 §4.9 (`stq.cu`) | M | all | H |
| Global reductions without return `RED`: SM 25.6 B/clk, pair 2 cycles per line, chip 330 / 225 B/clk | gb10 §4.9 (`ldg_atomic.cu`, `stq.cu`) | M | G (split-K), E | S (split), H |
| Global atomics with return `ATOMG`, CAS | — | X (default latency) | E (work queues), D | H |

## 7. Synchronisation, residency, dispatch, launch, clock

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Block barrier `BAR.SYNC`: release 4 + 2 per warp after the last arrival (round 1; base was 13); named barriers with a count | gb10 §4.10; probe `namedbar` (round 1, batch r1) | M (32 warps +19 %) | all | H |
| `BAR.ARV`, `BAR.RED` | engine code; probe `namedbar` (round 1, batch r1) | M for `BAR.ARV` (round 1): counts as an arrival, does not wait (12 cycles flat); X for `BAR.RED` (treated as blocking) | F (named-barrier producer/consumer) | — |
| Memory fences `MEMBAR`, `FENCE` | — | X (default latency) | F, D, E | H |
| Flag polling on global memory | 27 §2.3 | X (primitive missing) | D (split-KV combine), E (persistent) | — |
| CTA residency: registers (256-register units), shared memory (+1 KB), threads, 24 blocks | gb10 §6.1 | M (structural driver limits); WS path forced to 1 block/SM | all | S (tile, threads), H |
| Dispatch: blocks go round the SMs in launch order; the busiest SM simulated; other SMs progress alike | gb10 §6.2, §6.3, §8.8 | A (structural) | E, F (causal), D | S (grid, raster), H |
| Block replacement when a block exits (FLOW) | gb10 §6.2 | M (structural) | all | — |
| Unequal per-CTA work, persistent / work-queue scheduling | 28 §5 (P4) | X | E, F (causal), D | S |
| Launch overhead: 2.6 µs per stream launch; 0.43 µs per graph kernel | gb10 §6.6 (`launch.cu`); probe `launch` (round 3, batch r3) | M (stream); round 3: one launch between two events and a synchronize (the test-set harness's timing) costs 2.80 µs beyond the kernel's in-kernel span (median over 8 spans 0.1–151 µs, bimodal 2.1 / 3.2); an empty kernel 2.43 µs up to 131 k threads, then rising with threads (4.64 at 524 k, 6.69 at 1 M); `launch_event_us` replaces the fitted 2.6 for KTA batch rows (D769); depth: empirical; graph value unused by the table | all | S (graphs) |
| Multi-kernel sequences (gaps between kernels with work: 3.1 µs) | gb10 §6.6 | X (one kernel per run) | D (split + combine) | S |
| Clock: cycles → time | gb10 §6.6, §12 (`clock.cu`, `sustain.cu`); protocol clock kernel (round 1) | M for KTA batches: 2.494 GHz (r1), 2.496 (r1fa), 2.485 (r2) from a dependent FFMA chain against event time; under full-chip tensor load 2.493 GHz in the kernel (`mma_pair` b48, clock64 against globaltimer), as alone (2.500); upstream rows keep 2.52 / 2.54 | all | — |
| Power and thermal states | 26:09 | X | all | H |

## 8. Counts

Rows: 84. M 34, A 10, P 5, X 33, F 2 after round 2 (round 2: the warp scheduler row M → P between co-resident blocks; chip L2 share and STS frozen `unidentified`). After round 1: M 35, A 10, P 4, X 33, F 2 (before: M 29, A 12, X 37; round 1 measured LDS/STS, SHFL, ex2 beside FFMA, BAR.ARV, the BAR.SYNC base and the clock). A row that carries two codes is counted by its first.

What this says, before any kernel error is examined:

1. **Everything the HGEMM loop needs is modelled**, most of it measured (tensor, ldmatrix, cp.async, TMA, return port, chip L2, stores). The open items there are structural choices (L2 policy, DRAM fair share, dispatch symmetry), two fitted TMA rates, and latency spreads.
2. **The other five test-set kernels lean on absent mechanisms**: plain shared loads/stores, shuffles, the LSU queue (N, F, D), shared atomics (Q), mma beside FFMA and ex2 beside FFMA (F), L1 behaviour and coalescing (D, Q, N), flag polling and multi-kernel sequences (D), uneven per-CTA work (E, F causal). Each needs a probe at ladder level 1–5 before those kernels can be scored on mechanism.
3. **Two silent defaults violate 27 §2.3** and should become named "unsupported" or be modelled: a non-warp-specialised TMA kernel, and `BAR.ARV`/`BAR.RED` treated as blocking.
