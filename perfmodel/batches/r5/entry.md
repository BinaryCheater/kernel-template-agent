# Batch r5: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json` (engine at commit ddb7328: the one L2 model, `sim.l2_replay` true). Only new probes and new kernels are in the batch; every other row keeps its r4 measurement. New harnesses were smoke-run once on spark before this block (see "Smoke run").

## Selection (LOOP §3, supervisor guidance)

- Worst valid row at the start: MoE +231 / −63 %, DRAM queue 89 %: the generic front end ignored GEMM reuse → engine change, no probe: one L2 model (`l2.py`, below), existing measured parameters, no new fit. Replayed on r4 data before the batch: MoE +9.5 / +2.3 %.
- Next: RMSNorm warm graph −50.5 %, softmax graph −35.0 % (in-kernel span replayed fast; chip L2 share 56 % on RMSNorm, ALU and FMA stalls on softmax): question 15 → probe `row_reduce`.
- FP8 rows replay QMMA at the HMMA hold; the tensor rate per opcode and type is not measured → probe `mma_types`.
- Worst pair: sw8 vs s3_t128 is no longer an inversion above the 2 % threshold (measured +0.0 %, replay −5.1 %); its attribution diff is `queue (stg)`, 10.4 vs 6.2 % of the path: the store path, the same mechanism row_reduce tests. Worst inverted pair above threshold: sw4 vs c64x128x32_s4 (measured +2.2 %, replay −3.5 %).
- Test set: NVFP4 GEMM cannot be built: sm_121a's ptxas refuses `kind::mxf4nvf4.block_scale` ("not supported on .target 'sm_121'"), although TileLang 0.1.15 generates it for SM120. Fallback from the guidance: fused w4a16 GEMM (`w4gemm`, two tiles, warm and rotating), B's int4 tile expanded to fp16 in shared memory inside the k-loop, 64 STS/LDS per step on top of the GEMM.

## One L2 model (engine change, checked by ablation, LOOP §5)

`l2.py`: (1) `Cache`: the two-list L2 of the HGEMM front end, now shared; (2) `walk`: any kernel's global loads from its SASS, addresses from lane tracking, extended by the stride over trips and grid; a non-affine axis (MoE's `by // 2`) is walked at every index; (3) `replay_walk`: blocks in launch order, in-flight waves in lockstep, one cache → cold traffic per load instruction; (4) warm launches keep the fraction `warm_miss(read footprint)` of the cold traffic (probe l2reuse). The HGEMM front end keeps its tile accesses (panels from TileLang's raster) through the same `Cache` and warm rule. `sim.l2_replay` false restores round 4's rules.

## mma_types — tensor pipe by opcode and type

1. Mechanism: inventory §2 "Tensor pipe per sub-partition".
2. Why now: FP8 rows −4.4 / −9.0 %, tensor queue is their largest share; per-type rate never measured.
3. Sweep: 16 independent mma per iteration on one warp and on four (one per sub-partition), and one chained; f16 and bf16 k16 with f32 accumulators, f16 with f16, e4m3 k32 with f32 and f16, e5m2, s8 k32, s4 k64 (no native sm_121 form expected: the compiler expands it). Refused by sm_121a: bf16 accumulators, NVFP4 block-scaled.
   - H-same: one hold for every opcode: ≈ 16.8 cycles per instruction (replay's 16 + loop), chain 29.
   - H-elements: hold grows with the elements per instruction: QMMA.16832 and IMMA.16832 ≈ 32.
   - H-accumulator: f16 accumulators shorter than f32.
4. Engine prediction: `predicted.json` (H-same): w1 16.8, chain 29.1 (IMMA 27.1), s4 21.5 (mapped as 16-cycle mma: the replay sees the expanded code).

## row_reduce — where a short block's time goes

1. Mechanism: inventory §1 block turnover, §4 store path, §5 chip L2 share on LDG, §3 shuffles, barrier.
2. Why now: question 15.
3. Sweep: empty, ld, red, st, ld_st, ld_red, ld_red_st; grid 48 and 512 × 128 threads (the row kernels' 11 blocks per SM). Each thread 64 B: 8 KiB per block, 4.1 MB per launch at grid 512.
   - H-engine: span as replayed (below): store ≈ 2.2 × the load path at grid 512.
   - H-turnover: a gap between blocks on an SM of ≈ 100+ cycles: 11 blocks in turn, span ≈ replay + 10 gaps.
   - H-store: chip store rate binds: 4.1 MB at 122 B/clk (`chip store limit`, round 1): ≈ 34 k cycles; at 303 GB/s same.
   - H-load: LDG L2 share binds at 817 B/clk: ld_g512 ≈ 5.0 k cycles.
4. Engine prediction (cycles, span): empty 19 / 165, ld 472 / 2808, red 135 / 433, st 58 / 5459, ld_st 680 / 5910, ld_red 909 / 3241, ld_red_st 945 / 6318 (grid 48 / 512).

## Smoke run

Seen before this block was finished (one run on spark, nothing recorded; the timed batch repeats them): mma_types 17.7 cycles every f16/bf16/e4m3/e5m2 form (w1 and w4), s8 17.6, chain 29.05 (s8 27.1), s4 404; row_reduce grid 512: empty 229, ld 5181, red 461, ld_st 17138, ld_red 5553, ld_red_st 16574 cycles, blocks overlap on an SM (gap negative: several resident), empty gap +14. The hypotheses above were written from the engine and the parameter table, not tuned to these.

## After the run (round 5, step 4–5)

r5 valid: clock 2.494 → 2.480 GHz (drift −0.6 %), canaries +0.6 % / +0.1 %; 72 results. r5b (st256, entry block in `../r5b/entry.md`) valid: drift +0.1 %, canaries within 0.1 %; 19 results. Two batches, not one: r5's store points did not have the row kernels' shape.

**One L2 model** (ablation on r4 data, `sim.l2_replay` off → on, before the store change): MoE +231.1 / −63.4 → +9.5 / +2.3 %, R1 1/2 → 2/2 (part is the trip-count fix: the k-loop is called `ko`; the replay ran one trip of the 64×128 tile); RMSNorm rotating +16.0 → +7.0 %, GEMV rotating +24.5 → +19.9 %, decode partial rotating +15.2 → +12.8 %, tma 1024 full −9.5 → −7.2 %; worse: GEMV warm −10.6 → −12.8 %, FA −6.5 → −7.5 %. HGEMM warm and rotating, gemm1024, FP8 unchanged (same traffic through `Cache`, curve 0 below 16.5 MiB). Accepted. Walk cost: FA ≈ 19 s, MoE 128×128 ≈ 34 s, others ≤ 2 s per row. What the addresses showed: RMSNorm reads its weights (16 KB) from every block, GEMV its x; the replay had charged those loads as row traffic; decode partial loads 80 MB per launch from 16.8 MB of data.

**mma_types**: H-same survives: 17.7 cycles per instruction for every f16, bf16, e4m3, e5m2 form with f32 or f16 accumulators, s8 17.6, on one warp and four; chain 29.05 (s8 27.1); replay 16.8 / 29.07 / 27.07. H-elements (32) and H-accumulator rejected. So QMMA at the HMMA hold is right, FP8's −4.4 / −9.0 % is not the tensor rate, no per-type parameter is added. The −5 % on w1/w4 is 14 cycles per iteration on a 16-mma loop: the taken branch (question 4), not the pipe. s4 k64: no native mma, 400 cycles of expansion, excluded.

**row_reduce** (cycles, grid 48 / 512; replay before the changes): empty 8 / 229 (19 / 165), ld 701 / 5232 (472 / 2808), red 125 / 459 (135 / 433), st 126 / 10192 (58 / 5459), ld_st 737 / 17307 (680 / 5910), ld_red 1119 / 5554 (909 / 3241), ld_red_st 1195 / 16785 (945 / 6318). Block gap +14 empty; blocks overlap on an SM otherwise, so H-turnover is rejected; shuffles, shared-memory reduction and barrier within 8 % (the reduction is not the gap). H-load survives: 4.1 MB of sectors at 783 B/clk against the chip's 817. The replay's half came from a rule: its L1 hits by line, so the second load into a line hit; the L1 fills sectors. H-store: store time dominates, but these stores are strided 16 B per lane, not the row kernels' shape → r5b.

**st256** (cycles): st g1 205, g48 194, g96 463, g512 8949; ld 564, 677, 1208, 5034; ld_st 613, 724, 1293, 11195. H-new survives: 468.7 B/clk chip rate at 1024 B per instruction (H-1024-16B 7.3 k, H-1024-4B 34 k rejected); pair 8.4 cycles per instruction. Loads 833 B/clk, the L2 share as modelled. After the change: st g512 +15.3 %, ld g512 +6.9 %, ld_st g512 −1.8 %. At grids 48 and 96 the replay is +145 / +184 %: a burst of ≤ 786 KB leaves faster than the sustained rate (a store buffer, question 18), a few hundred cycles.

**Store width + L1 sectors** (ablation on the same data): RMSNorm warm −34.4 → −15.6 %, b2b −32.0 → −12.3 %, graph −50.5 → −27.4 %; softmax −27.1 → −18.1, −24.8 → −15.5, −35.0 → −23.3 %; worse by the burst: combine +23.3 → +25.0, b2b +57.8 → +59.9, graph +5.5 → +10.5 %, GEMV rotating +19.9 → +22.2 %. HGEMM, MoE, FP8 unchanged (they store 4 B per lane). Accepted: the rule is a defect fix (1024-byte instructions were counted as 128) with a measured rate.

**w4gemm** (test set, µs kernel; warm / rotating): 128×128 288.0 / 286.1, 64×128 432.3 / 430.8; replay −8.2 / −12.8, −11.3 / −8.4 %; R1 4/4. Predicted before the changes −10.1 / −15.9, −13.1 / −10.4 %. The 64×128 tile costs 1.5× on the GPU and in the replay: the dequantisation, not the mma, sets the rate. LDS latency 13–17 % of the path.

**sw8 vs s3_t128**: not worked on: no probe; the `queue (stg)` difference stays, 10.4 vs 6.2 % of the path (4 B/lane stores). Question 2 stays at 1 of 2.
