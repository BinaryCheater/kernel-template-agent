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
