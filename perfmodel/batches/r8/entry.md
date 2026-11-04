# Batch r8: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json`; the engine is round 7's plus the L2 miss per buffer (below). Every probe's SASS was checked on spark after a compile: the hot loop holds the intended instructions (r7b's `spill` had its loads folded, found this round).

## Selection (LOOP §3, supervisor focus)

Wrong what-ifs first (77/98 right), then the worst-inverted pair, the two rows above 25 %, one new test-set kernel.

Found before the batch, no GPU needed:

- **Event timing of a short kernel has a 0.96 µs step.** The same RMSNorm unit measured 9.54, 8.70, 8.61, 9.60 µs in four batches; samples cluster at 8.6 and 9.6. `compute removed −10 %`, `t256 −10 %`, `t64 −8 %` and softmax rows share a base sitting on the upper step. The graph rows (50 launches) resolve it: RMSNorm compute removed −19.6 %, t256 −5.2 %, t64 −1.1 %. Repeat in this batch: the what-if bases and their changes for RMSNorm, softmax, GEMV (warm); the table now also scores short kernels on graph rows, beside the `kernel` count.
- **GEMV r16, mk4 rotating (−38, −47 %, measured +3, −2 %):** an L2 replay defect. A line went to the first load that touched it, and several loads share lines, so most DRAM bytes never reached the DRAM queue. Now per buffer and over what passes the L1: 40.8, 40.5 µs, measured 37.9, 35.9. System bug, no probe.
- **`spill` (r7b) measured stores only:** the compiler folded the loads, so 7.4 and 23 cycles per stack instruction are 13 and 45 per store. Round 7's 830 and 270 B/clk are wrong by that factor as rates of "a stack instruction" (`spill_rw` below). The −43 to −70 % of the `spill` rows and the +32 % of 256×256×32 t256 are two sides of this.

## warp_map — which sub-partition a block's warps go to (question 29)

1. Mechanism: inventory §1 warp scheduler; engine: warp i of every block → sub-partition i mod 4.
2. Why now: GEMV n_part 2 warm +73 %, reduce 16 +67 %, measured −12, −1 %: 64-thread blocks, XU queue 68 % of the path; the replay puts every block's two warps on sub-partitions 0 and 1. `sim.warp_smsp_across_blocks` true: +0.2, +1.0 %.
3. Sweep: ex2 and mma streams, T 32–128, n blocks per SM. In cycles per stream of N instructions at the pipe's cost c:
   - H-local (replay): warp i of each block on sub-partition i mod 4: T 64, n 2: 2 N c; T 32, n 4: 4 N c; T 32, n 8: 8 N c.
   - H-across: warps continue round the sub-partitions across blocks: T 64, n 2: N c; T 32, n 4: N c; T 32, n 8: 2 N c.
   - H-free: each block's warps go to the least loaded sub-partition: as H-across at these points; T 96 n 2 tells them apart only in order, not in time (not separated here).

## copy_dram — cp.async from DRAM, with work between the steps (questions 10, 28)

1. Mechanism: inventory §5 DRAM latency, DRAM queue per SM.
2. Why now: HGEMM s1 rotating replay +18.6 %, measured −1.8 %; attribution DRAM latency 46 %. Question 10: 1560 (chase) vs 1100 (streaming LDG); an override of the copy path to 1100 gives +0.3 % (Kendall τ 0.73 → 0.78).
3. Sweep: rot and warm, 1, 16, 48 blocks, Z 0–1024 FFMA passes per step.
   - H-1560: one block rot − warm per step ≈ 1560 − 350 = 1210 (replay).
   - H-1100: ≈ 750.
   - H-fixed: at 48 blocks the DRAM queue is a fixed share, the step's time is latency + 48 × 20 KB at the chip rate + Z (replay; about 10.4 k + Z).
   - H-demand: at 48 blocks a block in its arithmetic leaves its share to the others: the step is max(rate term, latency + Z); hinge in Z.

## store_spread — what lanes' spread costs (question 25)

1. Mechanism: inventory §6 chip store limit: 420 B/clk of sectors up to lanes 128 B apart, 244 at TileLang's tile store, split at 4 KiB of extent (D790): two rates and a threshold between two probes.
2. Why now: HGEMM rotating Spearman 0.93 → 0.85 with D790; FP8 −8.5 %.
3. Sweep: 4 B lanes, LPR lanes per row (2–32), ROW 128–8192 B, STEP along the row.
   - H-extent (replay): 244 once the extent exceeds 4 KiB, else 420: a step between ROW 512 and 1024 at LPR 4.
   - H-partial: the rate follows the bytes filled per sector, not the distance (LPR 4 STEP 32 vs 16 differ; ROW does not).
   - H-rows: the rate follows rows (lines) touched per instruction, whatever their distance (LPR 2–32 differ at fixed ROW).

## arith_beside — arithmetic beside chip-bound streams (question 27)

1. Mechanism: no row: how compute combines with the chip's L2 and store limits.
2. Why now: RMSNorm compute removed, graph rows: −19.6 % measured, replay +6 %.
3. Sweep: L 2, S 0 or 2, A 0–512 FFMA per thread, grids 96, 512.
   - H-hide (replay): while the limits bind the arithmetic is off the path, span flat up to the knee.
   - H-add: span grows by A × (cost per warp) from A 0.

Beside it, the rows' own parts: `rmsb_m48/192_*`, `smb_*`: RMSNorm and softmax at 1 and 4 blocks per SM, stores, loads, compute removed in turn: which part carries the −20 % where the chip does not bind.

## mma_ldsm — two warps on a sub-partition (question 29)

1. Mechanism: inventory §2 tensor pipe.
2. Why now: HGEMM t256 replay −9 %, measured +1 % (rotating alike), with the 128×128×32 tile unchanged; worst thread row above the what-if tie.
3. Sweep: one tile's L ldmatrix and M mma per pass split over T = 64–512 threads, a block barrier per pass, L/M 0, 1/8, 1/2.
   - H-overlap (replay): T 256 gains: a warp's ldmatrix runs beside the other warp's mma.
   - H-none: the pass time does not change with T at fixed work.

## spill_rw — stack loads against stores (rows 256×256×32 t256 +32 / +39 %)

1. Mechanism: inventory §3 local memory.
2. Why now: largest rows; stack path 80 %. The kernel issues 257 LDL and 207 STL per pass of code; `spill` measured stores only.
3. Sweep: 256 threads, stack 64–1024 words, LD:ST 1:0 … 1:4; hot set H 32, 128 words at 1024; 1 and 48 blocks.
   - H-same (replay): one rate per stack instruction.
   - H-store: stores take the chip's store rate, loads the L1's.
   - H-capacity: above the stack's capacity per SM a cyclic walk misses everything, a hot set inside it keeps the fast rate (`H` points).

## New test-set kernel: NVFP4 GEMM (28 §5, last row not on the scorecard)

`nvfp4gemm2048_*`: e2m1 elements two per byte, one fp16 scale per 16 elements, 3 tiles, warm and rotating. sm_121a's ptxas refuses the block-scaled mma (round 5), so the expansion runs in the main loop in software, then an fp16 mma: the block-scaled mma itself stays `unsupported: mma block_scale, ptxas sm_121a`.

## Re-measurements

`hgemm2048_*` warm and rotating (20 rows): how much of τ 0.60 is the batch's noise (five warm variants lie within 2 %); `wi_hg_{b,t256,s1,s2}`.
