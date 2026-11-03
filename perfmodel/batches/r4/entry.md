# Batch r4: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json` (the engine at commit 0036db6: the round-3 engine plus the new case builders and two switches left at their round-3 behaviour, `sim.dram_writes_paced` false and the `launch_seq` rule with the values known before the run). New harnesses were smoke-run once on spark before this block (see "Smoke run").

Cache state: `warm` except the `*_rotating` test kernels (rotating input sets, rot × footprint ≥ 64 MiB) and the dram_rw probe (four plane sets in turn, 480 MiB). Every test kernel is re-measured (a fourth session); every test kernel also prints `b2b` and `graph` rows (below).

## Selection (LOOP §3 with the round-3 change: worst row and worst-inverted pair)

- Worst valid row: RMSNorm rotating −41.5 % (load latency incl. DRAM queue 84 %): the generic front end paces only reads at `dram_read_GBs`; RMSNorm writes as much as it reads (open question 7, 0 of 2) → probe `dram_rw`.
- Worst-inverted pair, HGEMM 2048³ (8 inversions, Spearman 0.62): `c128x128x64_s2_t128` (measured 224.5 µs, replay 236.9) against `c128x256x32_s3_t256_sw4` (235.8, replay 223.2); measured gap −4.8 %, replay +6.2 %. Attribution diff (`scripts/perfmodel-whatif`): s2 has DRAM latency 51.5 µs + DRAM queue 35.8 µs on its path, sw4 none (its path: tensor queue 101.7 µs). Every HGEMM variant's replay reads A and B once from DRAM per warm launch (launch-order L2 simulation with `l2_effective_bytes` 16.8 MB < A + B + C 24 MiB: 16.8 MB, miss fraction 0.04–0.08); only the two-stage 128-thread variant cannot cover the 1560-cycle cp.async miss latency. The mechanism that separates the pair is therefore "DRAM misses of a warm HGEMM and their latency behind the cp.async pipeline", not the panelled prologue or the taken branch (question 4 stays below 3 %) → test-set measurement `hgemm rotating` (below).
- Launch-dominated rows (combine 91 %, RMSNorm warm 50 %, softmax 41 %; open question 5): the reference is defined explicitly (supervisor guidance): one launch between events (`kernel`, as before), `b2b` (50 launches back-to-back, per launch), `graph` (the 50 launches in one CUDA graph, per launch); scored as separate sets. Probe `launch` gains sequence work points that measure the launch-hiding mechanism.
- Test set growth (LOOP §8): MoE grouped GEMM (8 experts, balanced). FP8 rows: see the last section.

## dram_rw — DRAM reads and writes together

1. Mechanism: inventory §5 "DRAM write rate and read/write mix".
2. Why now: RMSNorm rotating −41.5 % (84 % of its path on DRAM loads); it reads 4.2 MB and writes 4.2 MB; 207 GB/s total measured, 354 GB/s of reads in the replay.
3. Sweep: NR read planes and NW written planes of 30 MiB (31.46 MB), mixes 2:0, 2:1, 1:1, 1:2, 0:2, 16-byte accesses, 96 × 512 threads, one access per plane in flight per thread.
   - H-reads (engine now): reads at 237 GB/s, writes only on the store path (chip store limit 122 B/clk ≈ 303 GB/s): 2:0 265 µs, 2:1 265, 1:1 133, 1:2 ≈ 208 (store-bound), 0:2 ≈ 208.
   - H-sum (upstream dram.cu, HGEMM front end's wave_dram): reads / 237 + writes / 197 GB/s: 2:0 265, 2:1 425, 1:1 293, 1:2 452, 0:2 319.
   - H-total: one rate for any mix (237 GB/s): 2:0 265, 2:1 398, 1:1 265, 1:2 398, 0:2 265.
   (Plus the launch term, ≈ 2.8 µs.)
4. Engine prediction: `predicted.json` (`dram_rw *`), 2:0 268.6, 2:1 268.6, 1:1 135.9, 1:2 135.9, 0:2 46.2 µs (H-reads; the replay's stores of the busiest SM are not paced by the chip store limit or DRAM: writes-only is far below every hypothesis). With `sim.dram_writes_paced` true (H-sum as the engine implements it, reads paced by reads/237 + writes/197): 2:1 428.3, 1:1 295.5, 0:2 unchanged 46.2 (no reads to pace).

RMSNorm rotating under each: H-reads (replay now) 23.7 µs; H-sum 46.9 µs (engine with `sim.dram_writes_paced` true); GEMV rotating 45.5 (writes negligible), decode partial rotating 80.2; measured r3 40.5.

## HGEMM rotating — DRAM misses of the warm HGEMM

1. Mechanism: inventory §5 "L2 capacity and replacement" applied to the HGEMM front end (launch-order L2 simulation) and §5 "DRAM latency" behind cp.async stages.
2. Why now: the worst-inverted pair above; the diff is DRAM latency + queue, 87 µs of s2's path.
3. Sweep: the ten variants with rotating operands (three input sets of 24 MiB), beside their warm rows.
   - H-warm-misses (engine): a warm launch already reads A + B once from DRAM; rotating adds little (C write-back, which the warm replay also pays): rotating ≈ warm within a few % for every variant on the GPU; s2's warm penalty on the GPU is absent because the GPU hides the 1560-cycle latency (then the latency rule behind cp.async is wrong).
   - H-warm-hits: a warm launch hits the L2 for nearly all of A and B (the GPU's reuse, not the cyclic-scan miss curve); rotating launches are slower, by most for the variants with the fewest bytes in flight (s2 t128), and the engine's warm DRAM traffic is the error.
   - Engine prediction: rotating = warm for every variant (201.98, 190.84, 238.93, 231.83, 220.97 µs for the five 128-thread / c128x128x64 t256 variants; 220.69, 216.22, 208.15, 222.70, 229.57 for the others, within 1 % of warm): the replay's warm launch already reads A and B once from DRAM.

## launch — launches in a sequence

1. Mechanism: inventory §1 "Launch" (launch hidden behind short kernels, open question 5, 1 of 2).
2. Why now: combine 91 %, RMSNorm warm 50 %, softmax 41 % of the path are the launch term; inference runs these kernels back-to-back or in graphs, not one launch between two events.
3. Sweep: the round-3 FFMA work points (48 × 128, I = 0–16384, spans 0.1–151 µs) as 50 launches back-to-back between two events and as one CUDA graph of 50 launches, per launch, with the median in-kernel span; 100 empty launches in one graph at grids 1, 512, 4096 × 128.
   - H-add: per launch = span + a constant gap (b2b 3.1 µs, upstream launch.cu with 21 µs of work), at least the submission rate (1.96 µs b2b, 0.43 graph).
   - H-hide: the host submits the next launch while the previous kernel runs; per launch = span + the device-side gap only (≈ 0.5–1 µs, the same for b2b and graph once span > submission).
   - Engine prediction (H-add with b2b gap 3.1, graph gap 0.43 assumed): b2b work i0 3.21, i64 3.74, i256 5.10, i1024 10.54, i4096 32.31, i16384 119.39 µs; graph 0.54, 1.07, 2.43, 7.87, 29.64, 116.72; graph empty g1 / g512 / g4096 0.43 / 0.44 / 0.50 (the replay's spans: the FFMA loop runs 17.6 cycles per iteration in the replay against 23.0 on the GPU, question 4, so long spans are fast in every mode).
4. The test-set kernels' b2b and graph rows are scored with the same rule (span = the replay).

## MoE grouped GEMM (test set growth)

8 experts × 256 tokens (2048 tokens sorted by expert, balanced routing), hidden K 2048, expert width N 1024, fp16, weights 8 × 2048 × 1024 (32 MiB), tiles 128×128×32 s3 t128 and 64×128×32 s4 t128; block row → expert by division (gen.py `moe_gemm`). The HGEMM front end has one B; the expert weights go through the generic front end (busiest SM, footprint miss curve: 44 MiB → 0.85 of every load, including the A and W re-reads of the GEMM's reuse). Engine prediction 581.6 µs (128×128 s3) and 62.1 µs (64×128 s4). Expected to be wrong on DRAM traffic: the row exists to make the GEMM-family reuse in the generic front end visible on the scorecard.

## FP8 rows: is the tensor-pipe rate per data type exercised?

No. `machine.yaml` gives HMMA, IMMA, QMMA and OMMA the same hold (`sim.hmma_16816_pipe_clk` 16 cycles per instruction); the FP8 kernels' own SASS (QMMA.16832.F32.E4M3.E4M3, half the instructions per k of the HGEMM's HMMA.16816) is replayed at that hold. So the FP8 rows check the instruction count and the byte-equivalent front end, and a per-type rate only to the extent QMMA.16832 = HMMA.16816 = 16 cycles (upstream mma_types.cu). A per-type parameter does not exist; the round-2 mma_pair probe measured HMMA only.

## Smoke run

Seen before this block was finished (one run each on spark, nothing recorded; the timed batch repeats them): dram_rw 266.0, 429.2, 289.9, 435.2, 319.7 µs (2:0 … 0:2); launch b2b work points span + 1.8–3.2 µs, graph span + 0.60–0.65 µs; graph empty 0.44 / 0.60 / 2.52 µs; RMSNorm rotating kernel / b2b / graph 42.3 / 38.9 / 35.6 µs; combine 3.6 / 2.0 / 0.83; MoE 128×128 171.5 µs; HGEMM s2 t128 rotating 245.8 µs (warm r3 224.5). The hypotheses above were written from the engine and upstream numbers, not tuned to these.

## After the run (round 4, step 4–5)

Batch valid: clock 2.4942 → 2.4757 GHz (drift −0.7 %), canaries +0.4 % / +0.1 %; 155 results.

**dram_rw** (µs; total GB/s): 2:0 267.5 (235), 2:1 428.3 (220), 1:1 289.2 (218), 1:2 432.2 (218), 0:2 319.8 (197). H-sum 265 / 425 / 293 / 452 / 319: within 1 % except 1:2 (−4.4 %); H-total (265 / 398 / 265 / 398 / 265) and H-reads rejected. → `sim.dram_writes_paced` true (D771). Replay after: 274.5, 437.7, 302.0, 465.1 µs (+2.2 to +7.6 %); 0:2 47.2 (−85 %: no pacing without reads, question 17). RMSNorm rotating 41.22 measured: 23.7 → 47.8 µs (−42.4 → +16.0 %).

**HGEMM rotating** (µs, rotating / warm): s3_t128 230.8 / 222.6 (+3.7 %), sw8 222.8 / 222.7 (0.0), s4 267.6 / 249.0 (+7.5), c128x128x64 s2 246.0 / 225.7 (+9.0), c128x128x64 s3 t256 248.0 / 226.7 (+9.4), c128x256x32 239.6 / 237.0 (+1.1), sw4 237.8 / 236.9 (+0.4), c128x256x64 s2 230.0 / 222.7 (+3.3), c256x128 239.9 / 233.8 (+2.6), c64x128 251.1 / 231.8 (+8.3). H-warm-hits survives: warm launches pay little DRAM; panels gain only when cold (sw8 −3.5 % against s3_t128 rotating). → warm HGEMM DRAM reads by the warm-miss curve on A + B (D772). Ablation (r4 data, D772 off → on): HGEMM median 4.5 → 3.9 %, worst −12.4 → −13.4 %, Spearman 0.59 → 0.77, inversions 8 → 4; the worst-inverted pair now s2 −1.4 % / sw4 −6.2 % (measured gap −4.7 %, replay +0.2 %, inside the 2 % noise threshold). Rotating set (engine unchanged by D772): median 7.3 %, worst −12.5 %, Spearman 0.93, 2 inversions.

**launch sequences** (per launch minus span, µs): b2b 1.80 (span 0.1), 2.96, 2.35, 2.74, 3.12, 2.70 (spans 0.74–151); graph 0.61, 0.77, 0.69, 0.59, 0.60, 0.59; graph empty g1 / g512 / g4096 0.44 / 0.60 / 2.52. H-add survives with gaps 2.74 (b2b) and 0.60 (graph), H-hide rejected; below 0.5 µs span b2b runs at the submission rate (question 14). → `launch_b2b_gap_us` 2.74, `launch_graph_gap_us` 0.60 measured (D773). Ablation (pre-run 3.1 / 0.43 → measured): combine b2b +76.6 → +57.8 %, graph −14.9 → +5.5 %; RMSNorm warm b2b −27.6 → −32.0 %, softmax b2b −21.0 → −24.8 % (the kernels' own spans are fast: question 15).

**Test kernels, three ways** (kernel / b2b / graph, µs measured): combine 2.50 / 1.91 / 0.83; RMSNorm warm 8.61 / 8.22 / 6.97, rotating 41.22 / 40.06 / 36.91; softmax 9.60 / 9.24 / 7.38; GEMV warm 16.74 / 16.08 / 13.35, rotating 37.34 / 36.87 / 34.02; decode partial warm 25.89 / 24.79 / 23.42, rotating 71.04 / 70.81 / 69.36; FA 253.5 / 255.2 / 255.7; HGEMM within ±3 % of `kernel` in every mode. With launch 0.6 µs in a graph, RMSNorm warm's own span is ≈ 6.4 µs (replay 2.8) and softmax's ≈ 6.8 (replay 4.2).

**MoE grouped GEMM**: 179.6 / 173.1 µs (128×128 s3 / 64×128 s4); replay 594 / 63 (+231 / −63 %), R1 1/2; generic front end (D774), question 16.

**FP8**: 123.2 and 137.7 µs (r3 123.1, 137.6); −4.4 / −9.0 %. QMMA shares the HMMA hold; per-type tensor rate not exercised.

