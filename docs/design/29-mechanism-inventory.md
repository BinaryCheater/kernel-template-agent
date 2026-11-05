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

**Rule depth** (LOOP §8; marked in bold in the Status cell of the rows changed in rounds 5–7): **structural** = a resource, port, pipe or queue, carried to another design by changing the description; **empirical** = a formula over the probe's points, valid over the measured range only. Round 7: 16 rows marked, 7 empirical.

**Sources**: `gb10` = upstream `notes/gb10_simulator_design.md` (origin/main) §; `cov` = upstream `calibration/spark-6951/coverage.md` row; `gap#` = upstream `notes/sim_gaps.md` row (that log belongs to the older `agentic/sim`); `25` = [25](25-hardware-model-sources.md) §2; `26:nn` = [26](26-gb10-calibration-method.md) §5 SM120-reference chapter nn; `PTX`, `TG`, `NCU` = the official PTX ISA, Blackwell tuning guide, Nsight Compute profiling guide. Probe names are upstream `agentic/tilelang/microbench/*.cu`.

## 1. Issue, scheduling and dependencies

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Warp scheduler: one instruction per cycle per sub-partition, greedy then oldest; warp i → sub-partition i mod 4 | gb10 §4.2; `sched.cu`; 25; 26:07; TG | Round 8 (probe warp_map, D797): M: warps continue round the sub-partitions across an SM's blocks **structural**; P: M within a block (replayed: 14.91 vs 15.10 cycles per add); between two co-resident blocks (round 2, probe `mma_pair`, batch r2) the older block of a skewed pair finishes almost as if alone (317 k vs 311 k cycles alone; replay 356 k, +12 %); the replay's share comes from the tensor pipe's queue order, not the issue pick. Round 3 (no new probe, D768): with the skew loop's measured speed (23 cycles per FFMA-loop iteration, probe `launch` work points, batch r3) all three pairs fit fair sharing (s0 lockstep 530 k; s4096 first 469 k vs 464 k; s16384 first 317 k vs 311 k, span 694 k vs 688 k); oldest-block-first rejected (s0 first 319 k vs 530 k measured). M: the replay's least-loop-progress share is right; the mma_pair residuals are the skew loop (taken BRA about 5 cycles beyond its stall count, replay 17.6 vs 23 cycles per iteration), below 3 % of every valid test row | all | H (issue width, policy) |
| Yield bit, reuse bits | gb10 §4.1, §8.7 | X (parsed, ignored) | all | — |
| Stall counts from the control bits cover fixed latencies | gb10 §4.1, §4.3 | M (structural: compiler bits); `hmma_latency_clk` 29 is measured but unread | all | — |
| Six dependency counters, wait mask, `DEPBAR.LE` | gb10 §4.3; PTX | M for LDSM (34 + 4/way), MUFU/XU (18), LDG, cp.async groups, STSM, UTMACMDFLUSH; A elsewhere | all | — |
| Result latency of every other counted instruction (LDC, S2R, ATOMS, uniform ops, MEMBAR, …) | gb10 §4.11, §8.6 | A: `default_var_latency_clk` 20 (assumed), counted per opcode; LDS and SHFL measured in round 1 (rows below) | F, N, D, Q | — |
| Read-counter release of non-memory instructions | gb10 §4.3 | A: `rbar_release_clk` 8 (assumed) | all | — |
| Register-file bank conflicts, operand collector, reuse cache | gb10 §8.7; 25; gap list "not measurable from user space" | P (round 6, probe `mma_pipe`, batch r6b, D783): the only visible trace, the tensor stream slows 0.05 cycles per A operand set (`unidentified`, cause not shown); X **empirical** | all | H |
| Instruction cache / fetch | gb10 §8.7 (`icache.cu`: full rate to 32 KB of loop, −10 % at 64 KB) | M (round 6, probe `branch`, batch r6, D781): 1 cycle per 128-byte line outside a buffer of 64 lines per sub-partition, a warp's wait; was X; X (measured, not modelled) **structural** | F, E (large unrolled bodies) | S (unroll), H |
| Intra-warp divergence and reconvergence (`BSSY`/`BSYNC`, `BRA.DIV`) | gb10 §4.4, §8.7 | Round 9 (D811): P: a forward branch in a BSSY region whose predicate splits the lanes runs both arms in turn, from the tracked lane values; a value from a load counts as splitting (default, counted); NVFP4 GEMM R1 6/6, instructions equal to the counter, −33 → −9 %; memory instructions in an arm stay at 32 lanes, `BRA.DIV` never taken **structural**; Round 8: now has a row, was X Round 11 (D831): lane predicates for compares, SEL, VIMNMX, guarded writes. | N (tails), E, D | — |
| Branch and loop resolution from warp-uniform values | gb10 §4.4 (`audit_values`: 46/46) | M for the cost (round 6, probe `branch`, D780): next instruction at least 11 cycles after a BRA in loops without memory instructions or barriers; ; F (fallback rules: main-loop trip count, unconditional taken, else fall through) **empirical** | all | — |
| Issue interference: a copy or ldmatrix beside an mma stream on one sub-partition | gb10 §8 (`interfere.cu`: 2.3 cycles per copy) | M (round 6: copies 2.3 cycles from upstream `interfere.cu`, ldmatrix and LDS none, probe `mma_issue`, D782); was X; X (measured, judged negligible) **empirical** | G, F | H |
| Uniform datapath (`U*` ops) | — | X (issue slot only; no unit) | all | H |

## 2. Compute pipes

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Tensor pipe per sub-partition: every `HMMA/IMMA/QMMA/OMMA` holds it 16 cycles; type sets FLOP/clk | gb10 §4.5 (`mma_rate.cu`, `mma_types.cu`); cov tensor; 26:04; probe `mma_pair` (round 2) | M (round 6, probe `mma_pipe`: 16.015 at one operand set, 2 warps per sub-partition; plus the operand-set curve, D783). Open: the host compiler's NOP-spaced stream at one warp 17.7 (`mma_issue`, `mma_types`), not on any TileLang row; M; round 2 confirmed at chip scale in the HGEMM warp tile's loop (ldmatrix + 32 HMMA per k16): 310.5 k cycles at 1 and at 48 blocks (replay 316 k, +1.8 %), two blocks per SM 530 k (−1.0 %); no chip-level limit; round 5, probe `mma_types` (batch r5): one hold for every f16, bf16, e4m3, e5m2, s8 form and accumulator (17.7 cycles per instruction in the probe's loop, chain 29.05, s8 27.1), no per-type parameter; s4 k64 has no native mma on sm_121a; depth: structural (a pipe held 16 cycles per instruction) **structural** | G, F, E, D | S (dtype, mma shape), H |
| mma issue beside FFMA on the same sub-partition (+43 % in B34) | gap#27; cov tensor | M (round 6, probe `mma_issue`, D782): 0.9 cycles per FFMA, 0.3 per ALU instruction issued while the pipe is held; was X; X **empirical** | F (softmax beside mma) | H |
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
| Bank conflicts | gb10 §4.6 (`smem_bw.cu`: 4 cycles × ways); cov banks | M for LDS/STS (round 6, probe `lds_narrow`, D784): the array by the lanes' extent, no test row strides them; P: ldmatrix ways from the tile config when `smem_layout` is `linear`; not from addresses; none for LDS/STS **structural** | F, N | S (layout, swizzle), H |
| Plain shared loads/stores `LDS`, `STS` | gb10 §4.6, §8.6; probe `smem_ldsts` (round 1, batch r1) | Round 6 (probe `lds_narrow`): `LDS.U8`, `LDS.U16` as 32-bit; extent rule D784; w4gemm's loads touch one line each; M (round 1): MIO slot 5 cycles; array max(2, bytes/128 B/clk); LDS latency 24 + bytes/32; STS register read 3.25 per 128 B after the array (STS.128 at 4 warps −22 %); round 2 (2–6 warps): 12.0, 8.7, 7.0, 5.6, 4.35 cycles, neither this rule nor a per-sub-partition limit fits: `unidentified`, frozen after 2 experiments **structural** | F, N, D | H |
| Shuffles `SHFL` | gap#33; cov shuffle; probe `shfl` (round 1, batch r1) | M (round 1): MIO slot 5 cycles, 2 cycles on the shared-memory array per instruction, latency 28 | N, F, D | H |
| Shared-memory atomics `ATOMS`, CAS loops | gb10 §6.7 (GEMV driver, not copied) | X | Q (TileLang split GEMV reduction) | — |
| Shared memory vs L1 carve-out | cov L1; gap#7 | P (smem capacity enters residency; L1 capacity does not exist) | all | S, H |

## 4. Asynchronous copies, TMA and mbarriers

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| cp.async path: MIO → address stage → L2 (350) or DRAM (1560 + queue) → return port → array write → group; round 15: an SM issues a copy every 10 clk (measured, copy_two, structural rate, D869), the fit of round 14 removed; round 14's fit on `qs*` removed | gb10 §4.7 | M | G, E | S (copy path, stages), H |
| cp.async completion spread under load | cov cp.async; gap#23, #24, #40 | Round 9 (probe first_share, D807, D808): P: misses by slice and wave of blocks, one reader takes the DRAM traffic, the others wait on its fill **structural**; a miss brings its 32-byte sector (`pieces`); spread inside a step not modelled; was X (fixed latency, even miss spreading) | G, E | H |
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
| L1 hit path: 1 cycle per line (round 1; was 2), 45-cycle latency; LDG miss 342 cycles to L2 | gb10 §4.8 (`ldg_atomic.cu`); probe `l1hit` (round 1, batch r1) | Round 9 (probe first_reader, D809): M: a miss-status entry per line: a hit on a line on its way waits for the first reader's fill, a hinge within 5 %, RMSNorm graph −22.6 → −12.8 % **structural**; M; round 3 (D765): lines and sectors per LDG from its width and its 32 lanes' addresses, L1 hits from an LRU line cache over kernel-parameter bases (128 KB less the resident blocks' shared memory, structural); ncu global-load sectors per block matched exactly (GEMV 1360 vs 1344, RMSNorm 512, softmax 256, combine 96); round 5 (probe `row_reduce`, D777): hits by sector, a second load into a line of an unfilled sector misses **structural** | D, Q, N | H |
| L1 capacity, replacement, sectors and coalescing from addresses | cov L1; gap#7; 26:02 | Round 16 (`ml_*`, D880): M: capacity 128 KiB per SM, LRU, H-rand rejected **structural**; hashed lanes not pursued. X (P2 front end computes sectors) | D, Q, N | H |
| Local memory (spills) `LDL`/`STL`: 4 cycles per line | gb10 §4.8 | Round 14 (`stl_*`, D862): index per lane past the SM's stack P, empirical; uniform reads past the L2 open. Round 12 (`stu_*`, `stk_*`, question 44): wrong with loads at capacity, −84 %, open; frame from the compiler (D842). Round 8 (probe spill_rw, D800): M: stack stores at 400 B/clk, 133 past 192 KB per SM by LRU, the DRAM write rate past the L2; loads beside them L1 only; round 7's rates had the loads folded **empirical**; M Round 7 (probe `spill`, D792): M: the stack path takes 830 B/clk at the chip's scale, 270 B/clk above the L1's 192 KB per SM; one SM alone keeps 4 cycles per line; was 4 cycles per line at any scale; 256×256×32 t256 −60 → +31 %; rates `empirical`. **structural** | any spilling kernel | S (registers) |
| Constant loads `LDC` | — | X (default latency) | all (prologue) | — |
| Per-SM L2 return port: 30.7 B/clk (16.7 cycles per 512 B) | gb10 §4.7 (`loopsteps.cu`) | M | G, E, D | H |
| Chip L2 rate for copy traffic 817.7 B/clk, shared by blocks in flight | gb10 §4.7, §6.3 (`loopsteps.cu`, `l2share.cu`); probe `l2share_copy` = P-2 (round 1) | M, rule wrong, `unidentified` (frozen after 2 experiments): round 1 per-SM delivery drops already at 16 blocks (13.6 µs vs 10.5 replayed); round 2 (probe `l2share_stages`, stages 1–4 × 1/16/48 blocks) the round trip with 1 stage is unchanged at 16 blocks, 2–4 stages lose 16–25 % at 16 blocks with the chip at 0.98 TB/s; not latency, not the chip rate; candidate: a share below the chip (TPC pair or GPC) Round 11 (probes l2_cap, l2_local, D835): M for contiguous copies 1900 B/clk, spread lanes 25 sectors/clk, LDG 14 lines/clk; tile copies stay 817 `unidentified`. Empirical. | G, E, D, Q | H |
| L2 capacity (effective 16.78 MB) and policy: two-list LRU over k-slices of lockstep waves | gb10 §6.4, §8.2 (`l2policy.cu`) | Round 16 (D878): M: capacity counts 128-byte lines, read in part or whole **structural**. A (capacity measured; policy chosen, over-counts panelled and multi-wave patterns by 20–35 %); round 2: for the HGEMM 2048³ raster twins the replay gives both the same DRAM traffic (16.8 MB) and ncu the same L2 hit rate (93.3 / 92.7 %, batch r1): the replay's raster gap does not come from this row (see the warp scheduler row). Round 3, probe `l2reuse` (batch r3), plain / .cg 16-byte re-reads back-to-back: miss fraction 0.02 up to 16.5 MiB, 0.08 at 18, 0.13 at 20.25, 0.62 at 24, 0.85 at 32.25 MiB (H-part; H-full 24 MiB LRU rejected); evict-first (.cs) lines are not kept beyond 1.5 MiB (H-stream for that kind); the generic front end now predicts warm misses from `l2_warm_miss_by_MiB` (D766). Under ncu every launch refills the L2 from sysmem (fill sectors = read sectors warm and cold): profiler hit rates are no evidence of the cache state (D767); depth: empirical (a curve over footprint); round 4 (batch r4, HGEMM warm vs rotating: +0 to +9.4 %): a warm HGEMM launch misses by the measured warm-miss curve on A + B (16 MiB: 0.02), not the launch-order simulation on A + B + C (D772); round 5: one engine rule (`l2.py`, D775): cold traffic from the addresses loaded, warm = that × the curve; MoE, RMSNorm rotating, GEMV, decode rows improve, HGEMM unchanged **empirical** Round 11 (wr_*, D832): single-use reads count written bytes in their window. | G, E, D | S (swizzle, raster), H |
| L2 hit-vs-reuse-distance curve | `gb10.json` `l2_hit_vs_reuse_distance` | X in the engine (analytical model only) | G, D | H |
| Back-to-back L2 carry-over between runs (`stream_b2b.cu`) | gb10 §6.7, §8.5 | X (GEMV driver not copied) | D, Q | — |
| L2 latency under load | — | Round 10 (`load_curve`, D816): no curve to the L2 rate; round 16 added no sweep. X (fixed) | all | H |
| L2 persistence window, set aside 18 MB | upstream tier-4 (`l2persist.cu`) | X | D, Q | S |
| L2 slices, set hash, NoC | gap list; 26:02 | X | — | H |
| Dirty-L2 write-back paid by a flush (`dirty_l2_extra_us`) | `gb10.json` | X in the engine (analytical model only) | measurement protocol | — |
| DRAM read rate by run length (204/233/237 GB/s), write 197, mix rule | gb10 §6.4 (`tileload.cu`, `mlp_stream.cu`, `dram.cu`) | Round 16 (`mr_*`, D877): M: 64-byte bursts at 237 GB/s; a sector asked alone pays the whole burst **structural**; T 32 runs 1.4 times, open. M (wave-level); round 4 (probe `dram_rw`, batch r4): the mix rule re-measured in KTA (read:write 2:0 … 0:2 within 1 % at 2:1 and 1:1, +4.6 % at 1:2; write-only 197 GB/s); rotating rows of the generic front end now pace reads and writes by it (D771); no DRAM pacing for a kernel without reads (dram_rw 0:2 −85 %) Round 7 (`dram_burst`): no burst above the plateau: 3.9–67 MB per launch within −6 to +5 % of the replay. **empirical** | G, D, Q, E | H |
| DRAM queue per SM: wave DRAM time spread over read misses, fair share by SM-equivalents | gb10 §6.4 | Round 9 (probe first_share, D807): M: per wave of blocks: the first wave to read a slice takes its misses, even spreading rejected (later waves 17.0 k, warm 17.1 k, even 33–36 k) **structural**; Round 8 (probe copy_dram, D798): M: the rate limits what has been read so far, no sum with arithmetic **structural**; A (structural: even spreading, fair share) Round 11 (wr_*, D832): M: written-back bytes share the DRAM with reads when past the dirty limit. Structural; the 1.5 bound and 7 MiB step empirical. | G, D, Q, E | H |
| DRAM latency: 1560 cycles (copy, dependent chase), 1100 (streaming LDG; round 1), 1170 (TMA) | gb10 §4.7, §5; probe `dram_rotate` (round 1, batch r1) | Round 9 (probes first_reader, first_share, D810): 1075 unloaded for copies (as LDG 1100), plus the step's burst past the chip L2, bytes / 817 B/clk: 1420 was 16 tiles of that **structural**; Round 8 (probe copy_dram, D799): 1420 for streaming copies; M | D, Q | H |
| DRAM latency spread, channels, banks | gb10 §8.4; gap (B18) | Round 16 (`md_*`, D879): not reached: strides past 256 B stayed in the L2. X | D, Q | H |
| CPU contention for LPDDR5X | gb10 §6.4 (`cpuhog.c`: 237 → 214/171/138 GB/s) | F (an input, `CPU_DRAM_GBs` = 0) | D, Q | — |
| TLB misses (random chase beyond 1 MB: ~400 ns) | `gb10.json` `lat_l2_ns` source | Round 16 (`mh_*`, D879): no cost seen to 1 GiB. X | D (large KV), Q | H |

## 6. Stores and atomics

| Mechanism | Sources | Status | Kernels | Var |
| --- | --- | --- | --- | --- |
| Store port shared by the two SMs of a TPC: 15 (4 B/lane), 11 (16 B/lane; round 1, was 14) or 8.4 (32 B/lane; round 5) cycles per instruction per pair | gb10 §4.9 (`storepat.cu`); probe `store_tail` = P-1 (round 1, batch r1); probe `st256` (round 5, batch r5b) | Round 12 (probe store_merge, D838): structural: more than 4 lines per instruction, apart or filled in part: 1.875 per line at the pair. Round 8 (D801): 1.6 cycles per instruction for 32 contiguous 4-byte lanes, the 15 is 8 rows per instruction **empirical**; M; partner-busy rule confirmed at 2–16 blocks; no per-block tail **structural** Round 11 (rw_*, D833): M: runs by lines, else 0.47 per sector; queued sectors merge. Structural; merge window 32 chosen. | all | S (store width), H |
| Chip store limit 122 (round 1, was 137.8) / 570.9 / 468.7 (32 B/lane; round 5) B/clk | gb10 §4.9; probe `store_tail` (round 1); probe `st256` (round 5, batch r5b, D777) | Round 16 (`mw_*`, D881): past the L2 stores queue on the bursts they open; in order no DRAM limit shows. Round 12 (D838, D839): 244 applies past 4 lines per instruction; stores merge at 4 lines or fewer, the dirty limit counts distinct bytes. Round 8 (probe store_spread, D801, D802): by sectors, 244 where lanes share a sector they do not fill, at any distance; 8 MiB dirty limit now in the engine **empirical**; Round 6 (probes `st256`, `row_reduce`, D785): stores at the limit wait for 0.45 of the reads' time (strided stores ≥ 1, not modelled); M at contiguous lanes; a burst up to 786 KB leaves faster (st256 g48, g96) and strided 16-byte stores slower (205 B/clk, row_reduce): both unmodelled Round 7 (probes `l2mix`, `store_lines`, D790): one limit of 420 B/clk over the 32-byte sectors the lanes touch, every width, behind a store buffer of 470 KB; the width rates stay as the ablation; a third slope above about 8 MiB written per launch (dirty L2 at the DRAM write rate) is not in the engine. Reads at the limit: share 1 less an overlap of 2200 cycles (D791), `empirical`, replacing 0.45. **structural** | all | H |
| 8-byte-per-lane stores (`STG.64`) | engine code | A (256 B at the 4-byte rates) | N, Q | — |
| Store queue: 16 STG/RED instructions before issue waits | gb10 §4.9 (`stq.cu`) | M | all | H |
| Global reductions without return `RED`: SM 25.6 B/clk, pair 2 cycles per line, chip 330 / 225 B/clk | gb10 §4.9 (`ldg_atomic.cu`, `stq.cu`) | M Round 11 (atom_*, D834): M, X → M for lanes in their own sectors: 1.44 per clock per 4 KiB region, 12 chip-wide, merge gain by bytes. Region rule structural, gain empirical. | G (split-K), E | S (split), H |
| Atomics at a uniform address: one lane carries the warp's sum (`VOTEU`, `REDUX` or `UFLO`, `POPC`) | r13c `atb_*`, D850 | Round 13: M: one request at 1.44 per clock per region; structural; few words 0.85 per clock not in the rule | H | — |
| Older block of an SM issues first, except between mma | r13c `tr_*`, `mma_pair`, D849 | Round 13: M, structural; replaces fair sharing for everything but mma | H | — |
| Global atomics with return `ATOMG`, CAS | — | Round 12 (`atr_*`, D840): M: rate of `REDG` by region, the old value a round trip later; CAS not measured; one bin −77 % | E (work queues), D | H |

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
| Launch overhead: 2.6 µs per stream launch; 0.43 µs per graph kernel | gb10 §6.6 (`launch.cu`); probe `launch` (round 3, batch r3) | Round 16 (D883): starting threads past 245 K costs 5.5 ns per 1000 **empirical**. M (round 6, probe `launch_short`, D786): spans below 0.56 µs hidden in the submission rate, floors 4.07 µs b2b and 4.6 µs event up to about 2 µs; M (stream); round 3: one launch between two events and a synchronize (the test-set harness's timing) costs 2.80 µs beyond the kernel's in-kernel span (median over 8 spans 0.1–151 µs, bimodal 2.1 / 3.2); an empty kernel 2.43 µs up to 131 k threads, then rising with threads (4.64 at 524 k, 6.69 at 1 M); `launch_event_us` replaces the fitted 2.6 for KTA batch rows (D769); depth: empirical; graph value unused by the table; round 4 (probe `launch` sequence points, batch r4): per launch in a sequence max(submit, span + gap), back-to-back gap 2.74 µs (submit 1.96), CUDA graph gap 0.60 µs (submit 0.43), not hidden behind longer kernels (D773); below a 0.5 µs span the back-to-back gap shrinks (b2b of an empty chain 1.90 µs: open) **empirical** | all | S (graphs) |
| Multi-kernel sequences (gaps between kernels with work: 3.1 µs) | gb10 §6.6 | M (round 6, batch r6, D779): program form `sequence`, L2 carried by buffer, gaps by launch mode; decode split-KV scored; X (one kernel per run); round 4: M for repeated launches of one kernel (the gap, D773); different kernels in one sequence not measured **structural** | D (split + combine) | S |
| Clock: cycles → time | gb10 §6.6, §12 (`clock.cu`, `sustain.cu`); protocol clock kernel (round 1) | M for KTA batches: 2.494 GHz (r1), 2.496 (r1fa), 2.485 (r2) from a dependent FFMA chain against event time; under full-chip tensor load 2.493 GHz in the kernel (`mma_pair` b48, clock64 against globaltimer), as alone (2.500); upstream rows keep 2.52 / 2.54 | all | — |
| Power and thermal states | 26:09 | X | all | H |

## 7a. Coverage matrix (LOOP §3a)

<!-- matrix: generated -->
Cells that can occur: 406 of 800, the rest dropped (`tools/kta_perfmodel/matrix.py`, `dropped`, says why for each).

| State | Before round 16 | Before round 17 | Now |
| --- | ---: | ---: | ---: |
| M measured with a rule | 60 | 73 | 75 |
| m measured without a rule | 18 | 20 | 18 |
| · empty | 328 | 313 | 313 |

Round 17's 54 blind rows above 10 %, placed after the rules: in M cells 25, in m cells 28, in empty cells 1. By class at first score: defect 0, known rule 34, no rule 20.

Round 16's 56 blind rows above 10 %, placed after the sweep: in M cells 27, in m cells 28, in empty cells 1. By class at first score: defect 0, known rule 35, no rule 21.

Round 15's 51 blind rows above 10 %: in M cells 9, in m cells 33, in empty cells 9 before round 16; 20, 29, 2 now. By class: defect 6, known rule 25, no rule 20.

Patterns: order (contiguous lanes walking on), stride (lanes or warps a constant apart), row (rows a pitch apart), hashed (no order in the lanes), cap (footprint across the target's capacity; for DRAM, across the TLB's reach). Sharing: w one warp, b one block, s blocks on an SM, c every SM. A cell reads `state` and, in brackets, the blind misses of round 15 in it.

**read (LDG, LDS, LDL) × shared**: decides layout, split of loads and arithmetic.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M |  | smem_ldsts, lds_narrow (D784); upstream smem_bw, shfl |
| stride | M | M | · |  | smem_ldsts, lds_narrow (D784) |
| row | · | M | m |  | lds_narrow (D784); ms_s_*: D 64, 96, 128 slower, short rows |
| hashed | · | · | · |  |  |

**read (LDG, LDS, LDL) × stack**: decides tile size at the register limit.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | M | M | spill_rw (D800) |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | M | stl_* (D862) |
| cap | · | · | · | m | stl_u_* (question 44) |

**read (LDG, LDS, LDL) × L1**: decides threads, row layout (GEMV, decode, rows).

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M | · | l1hit, first_reader (D809); ml2_* |
| stride | · | · | · | · (1) |  |
| row | · | M | · | M | row_reduce (D777) |
| hashed | · | m | · | · (1) | ml_h_*: +140 %, not pursued |
| cap | · | M | M | · | ml_*, ml2_* (D880): LRU, 128 KiB per SM |

**read (LDG, LDS, LDL) × L2**: decides threads, split, tile.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | · | M (2) | l1hit (342 clk miss); first_reader; l2_cap (D835) |
| stride | · | · | · | M (2) | l2_cap (D835) |
| row | · | · | · | m (14) | row_reduce, dl_*_warm; threads D870 |
| hashed | · | · | · | M (2) | mh_* (D878, D877) |
| cap | · | · | · | M | mk_*, md_*, mr_s* (D878): 128-byte lines |

**read (LDG, LDS, LDL) × DRAM**: decides threads, split, rotation; every rotating group.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M | M (3) | dram_spread (latency); mr_g* (D877); tw_*, gr_* (D886): no window; edges of a run 0.6 |
| stride | · | · | · | M | mr_* (D877): 64-byte bursts |
| row | · | · | · | M (5) | mr_* (D877): 64-byte bursts |
| hashed | M | · | · | M | dram_spread (chase); mr_* (D877): 64-byte bursts |
| cap | · | · | · | M | mh_*, md_* (D879): no TLB cost to 1 GiB |

**write (STG, STS, STL) × shared**: decides layout.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | m | · |  | smem_ldsts, lds_narrow; smem_ldsts: 2-6 warps unidentified |
| stride | M | m | · |  | smem_ldsts, lds_narrow; smem_ldsts: 2-6 warps unidentified |
| row | · | · | m |  | ms_s_*: D 64, 96, 128 slower, short rows |
| hashed | · | · | · |  |  |

**write (STG, STS, STL) × stack**: decides tile size at the register limit.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | M | M | spill_rw (D800) |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | M | stk_* (D852) |
| cap | · | · | · | M | spill_rw (133 B/clk past 192 KB) |

**write (STG, STS, STL) × L2**: decides store path, width, split, batch size at the dirty limit.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | M | M (2) | store_tail (partner rule); store_lines, store_spread, st256 (D838) |
| stride | · | · | · | M (1) | store_lines, store_spread, st256 (D838) |
| row | · | · | · | M | pw_*, mt_w_* (D887): 0.063 clk a sector, 0.105 a line |
| hashed | · | · | · | m (1) | mw_h_f16 −73 %: inside the L2, not pursued |
| cap | · | · | · | m (6) | wr_*, wl_* (D802, D873): 6.5 to 7.6 MiB no step |

**write (STG, STS, STL) × DRAM**: decides store path past the dirty limit.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | M | mw_o_* (D881): no limit past the store's own |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | M | mw_h_f64 (D881): a burst per lane |
| cap | · | · | · | · |  |

**read-modify-write of the same words × shared**: decides —.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · |  |  |
| stride | · | · | · |  |  |
| row | · | · | · |  |  |
| hashed | · | · | · |  |  |

**read-modify-write of the same words × stack**: decides tile size at the register limit.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | M | spill (load then store) |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | · |  |
| cap | · | · | · | m | stu_* (question 44) |

**read-modify-write of the same words × L2**: decides in-place or out-of-place rows.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | M | px_*, mx_* (D887): the same rate; `mx_*` +30 % left |
| stride | · | · | · | · |  |
| row | · | · | · | m | rmw_*, question 42 |
| hashed | · | · | · | m | bl_mixrwa |
| cap | · | · | · | · |  |

**read-modify-write of the same words × DRAM**: decides in-place or out-of-place rows.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | m | rmw_* |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | · |  |
| cap | · | · | · | · |  |

**atomic (ATOM, RED, ATOMS) × shared**: decides reduction form.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · |  |  |
| stride | · | · | · |  |  |
| row | · | · | · |  |  |
| hashed | · | · | · |  |  |

**atomic (ATOM, RED, ATOMS) × L2**: decides reduction form, bins.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | M | atom_*, atb_*, atr_* (D834, D850) |
| stride | · | · | · | · |  |
| row | · | · | · | M | atom_*, atb_*, atr_* (D834, D850) |
| hashed | · | · | · | M | atom_*, atb_*, atr_* (D834, D850) |
| cap | · | · | · | · |  |

**atomic (ATOM, RED, ATOMS) × DRAM**: decides reduction form past the L2.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | · |  |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | · |  |
| cap | · | · | · | · |  |

**copy (cp.async) × L2**: decides stages, tile, threads.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M (2) | M | copy_two, copy_depth (D869); l2_cap, loopsteps (D835) |
| stride | · | · | · | M | l2_cap, loopsteps (D835) |
| row | · | m | · | m (4) | ms_c_*: short rows, flat by D; fad_* (D871): loss by D mod 32 |
| hashed | · | · | · | · |  |
| cap | · | · | · | M | first_share (D807) |

**copy (cp.async) × DRAM**: decides stages, tile, swizzle.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | M (1) | M | first_reader (D810); copy_cold (D882): 10 clk per copy, cold as warm |
| stride | · | · | · | · |  |
| row | · | · | · | m (4) | fa rotating under bursts, no sweep of its own |
| hashed | · | · | · | · |  |
| cap | · | · | · | · |  |

**tensor load (ldmatrix from shared, TMA from global) × shared**: decides tile, layout, ways.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M |  | mma_ldsm, upstream instlat |
| stride | M | M | M |  | mma_ldsm, upstream instlat |
| row | · | · | · |  |  |
| hashed | · | · | · |  |  |

**tensor load (ldmatrix from shared, TMA from global) × L2**: decides copy path: TMA or cp.async.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | · | m | upstream dramlat (440 clk); fitted rates (TMA ablation) |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | · |  |
| cap | · | · | · | · |  |

**tensor load (ldmatrix from shared, TMA from global) × DRAM**: decides copy path: TMA or cp.async.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | M | · | · | upstream dramlat (1170 clk) |
| stride | · | · | · | · |  |
| row | · | · | · | · |  |
| hashed | · | · | · | · |  |
| cap | · | · | · | · |  |

**compute beside traffic to the target × shared**: decides warp specialisation.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M |  | mma_issue, mma_ldsm; mma_pair, tr_* (D849) |

**compute beside traffic to the target × stack**: decides tile size.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | · |  |

**compute beside traffic to the target × L1**: decides split of arithmetic.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | M | arith_beside (D809) |

**compute beside traffic to the target × L2**: decides stages, threads.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | m | wq_*: questions 32, 36 |

**compute beside traffic to the target × DRAM**: decides stages.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | · |  |

**barrier (BAR, mbarrier, fence) × shared**: decides threads, stages.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | M | M | M |  | namedbar; mma_pair |

**barrier (BAR, mbarrier, fence) × L2**: decides store path, sequence.

| Pattern | w | b | s | c | Probes |
| --- | --- | --- | --- | --- | --- |
| order | · | · | · | · |  |

<!-- matrix end -->

## 8. Counts

Round 6 moved 9 rows to M or P (see their status cells); counts below are as of round 2. Rows: 84. M 34, A 10, P 5, X 33, F 2 after round 2 (round 2: the warp scheduler row M → P between co-resident blocks; chip L2 share and STS frozen `unidentified`). After round 1: M 35, A 10, P 4, X 33, F 2 (before: M 29, A 12, X 37; round 1 measured LDS/STS, SHFL, ex2 beside FFMA, BAR.ARV, the BAR.SYNC base and the clock). A row that carries two codes is counted by its first.

What this says, before any kernel error is examined:

1. **Everything the HGEMM loop needs is modelled**, most of it measured (tensor, ldmatrix, cp.async, TMA, return port, chip L2, stores). The open items there are structural choices (L2 policy, DRAM fair share, dispatch symmetry), two fitted TMA rates, and latency spreads.
2. **The other five test-set kernels lean on absent mechanisms**: plain shared loads/stores, shuffles, the LSU queue (N, F, D), shared atomics (Q), mma beside FFMA and ex2 beside FFMA (F), L1 behaviour and coalescing (D, Q, N), flag polling and multi-kernel sequences (D), uneven per-CTA work (E, F causal). Each needs a probe at ladder level 1–5 before those kernels can be scored on mechanism.
3. **Two silent defaults violate 27 §2.3** and should become named "unsupported" or be modelled: a non-warp-specialised TMA kernel, and `BAR.ARV`/`BAR.RED` treated as blocking.
