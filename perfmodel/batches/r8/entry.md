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

## After the run (round 8, step 4–5)

r8 valid: clock 2.491 → 2.476 GHz (−0.58 %), canaries +0.62 / +0.25 %. Invalid cases (`invalid_cases.json`): `copy_dram rot_*_g1`: at one block the start walks 28 MB and restarts per point, the reads hit what the previous point left.

**warp_map**: H-across. ex2 at T 32: 9465, 9483, 9519, 17411 at n 1, 2, 4, 8; T 64: 9465, 9484, 17371; mma alike. H-local rejected. T 96 n 2 10233, below the 12.3 k four sub-partitions would need at 8 cycles: open, no row.

**copy_dram**: 48 blocks cold 127 k, 129 k, 144 k, 417 k at Z 0, 64, 256, 1024, warm 17 k, 39 k, 110 k, 392 k: H-demand: max(rate, own path), not the sum (replay +23 to +78 %, now within 6 %). Latency, 16 blocks, large Z: cold − warm 1067 per step: 1420 from the L2's 350: neither 1560 nor 1100. Warm at 48 blocks also runs under the sum (21.2 k replayed, 17.3 k measured at Z 0).

**arith_beside**: H-hide. Grid 512: L2 S0 5036, 5030, 5022 at A 0, 32, 128, 6260 at A 512; L2 S2 11.3 k flat; grid 96 grows with A. The replay had it right (median 3.4 %). A cumulative chip L2 rule like the DRAM one was tried for warm copy_dram and withdrawn: it put arith_beside −30 % and l2mix 22 %, against the same engine without it.

**Rows' parts** (graph rows, µs, measured / replay): m48 base 1.85 / 1.69, no store 1.85 / 1.42, no compute 1.10 / 1.15, no load 1.23 / 1.30, arithmetic alone 0.82 / 0.95; m192 3.28 / 2.87, 2.38 / 1.90, 2.39 / 2.26, 2.67 / 2.57, 1.32 / 1.38. Each part alone is right; the gap needs loads and arithmetic together.

**mma_ldsm**: H-overlap: T 128 L 32 1618, T 256 L 16 1128, replay 1529, 1115; streams alone within 5 %. So HGEMM t256 is not the pipe: open.

**spill_rw**: stack stores cost alike with or without loads: 31.7 k / 31.9 k, 395 k / 396 k, 2.518 M / 2.522 M: H-store; 15.5, 48, 77 cycles per store at 64, 256, 1024 words: 396, 133, 79 B/clk, the last the DRAM write rate. H 32, H 128 at 1024 words 17–19: H-capacity survives (LRU). The compiler merged repeated loads and stores in a pass: `L4`, `S4` points equal `L1 S1`. Loads alone after a cyclic fill: 43 per load at one block, 132 at 48 blocks and 1 MB.

**store_spread**: flat in ROW 128–8192 (13.0–14.1 k): H-extent rejected; H-partial: sectors several lanes fill in part 237–286 B/clk (L 4, L 2), 8 or more lanes to a sector 440; `store_lines` one lane per sector 433.

**NVFP4 GEMM**, first score −33 to −37 %, R1 0 of 6: TileLang lowers the selects to per-lane branches (`BSSY`, `BSYNC`): intra-warp divergence, absent in the engine; lane 0 stands for the warp. The rows stay on the scorecard, not valid until R1 passes.

**Re-measurement**: event `kernel` rows move up to 3 % at 230 µs (s1 rotating 226.5 → 233.8) and 10 % at 9 µs; graph rows within 1 %. Kendall τ of the measurement against itself: `kernel` against `graph` rows 0.78 warm, 1.0 rotating; r8 against r4 0.87, 0.82: τ ≥ 0.8 on 2048³ warm sits at that ceiling (12 % spread, five variants within 2 %).

**Ablation, one change off, `--sets batches`** (final engine: HGEMM warm 2.5 % / −7.7 %, 0.73, 4; rotating 6.8 % / −8.8 %, 0.81, 3; 85 of 108). The three runs below were taken with the withdrawn L2 rule still on (then 83 of 108; HGEMM warm 3.8 % / −9.8 %, rotating 7.8 % / −11.1 %):

| Off | Moves |
| --- | --- |
| DRAM by demand (D798) | copy_dram 4 → 10 %; decode −5.1 → +5.8 %, MoE +7.7 → +10.4 %; directions 83 |
| copy latency 1560 (D799) | HGEMM warm worst −9.8 → −12.5 %; directions 83 → 82 |
| warps across blocks (D797) | GEMV n_part 2, reduce 16 back to +67 to +73 % (r7 ablation); directions 83 → 84: rows where the old map cancelled another error |
