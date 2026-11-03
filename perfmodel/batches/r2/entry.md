# Batch r2: entry blocks

Written before the run (write-probe). The engine's replay of every case's own SASS is in `predicted.json` (the engine at the batch's build commit; microsecond rows at the description's kernel clock 2.54 GHz until the batch measures its own). Sources: `perfmodel/probes/<name>/probe.cu`, test kernels `perfmodel/testset/gen.py`. SASS checked per probe (below).

Cache state: `warm` everywhere except the test kernels named `*_rotating` (`rot` input sets in turn, rot × footprint ≥ 64 MB > the 24 MB L2; `launch.json` gives rot and footprint). From this batch the profiler pass runs with `--cache-control none`: its counters see the harness's cache state (round 1's were taken after the profiler's own L2 flush). Extra counters (`dram__bytes_read/write`, L2 lookup hit/miss sectors) run in a second profiler pass so that an unknown metric name cannot cost the main counts.

The ten HGEMMs and the FA forward of r1 ride along unchanged (same generator and SASS): a re-measurement in a second session, now with warm-L2 counters.

## Selection (LOOP §3)

Worst valid row: `hgemm2048 c128x128x32_s3_t128_sw8` −12.4 % (mechanisms only −12.7 %). Top of its path: tensor queue (mma) 39 %, stall after mma 24 %, a measured mechanism (inventory §2 "Tensor pipe per sub-partition") → a probe that discriminates the rule. The same row carries the raster question (open question 3): with r1's data and no GPU, the engine gives both raster twins the same DRAM traffic (16.8 MB, miss fraction 0.062) and ncu shows the same L2 hit rate (93.3 % vs 92.7 %); the engine's 5.5 % gain for the panelled launch is not an L2 effect. In the replay the swizzle prologue (integer division, a fallback branch) starts the two co-resident blocks of an SM out of step, and the two then run desynchronised (first block done at 118 k cycles, second at 185 k; without the panel both at 173–176 k), so loads of one overlap the other's mma. On the GPU the twins measure the same. The mechanism is how the SM shares its tensor pipes and issue slots between two co-resident blocks (inventory §1 "Warp scheduler: greedy then oldest", `M` for single-block issue only). One probe, `mma_pair`, carries both questions.

## mma_pair — tensor pipe in the GEMM loop shape; two blocks per SM; clock under full-chip load

1. Mechanism: inventory §2 "Tensor pipe per sub-partition: every HMMA holds it 16 cycles"; §1 "Warp scheduler … greedy then oldest" between co-resident blocks; §7 "Clock" under full-chip tensor load.
2. Why now: tensor queue 39 % + stall after mma 24 % of the worst valid row's path; the raster twins (two inversions' worth of the HGEMM ranking) differ in the replay only by how two blocks share an SM. Raster order and blocks per SM are design variables.
3. Sweep and hypotheses. Each warp: 512 k16 steps of 4 ldmatrix.x4 + 4 ldmatrix.x4.trans + 32 independent HMMA.16816.F32 (the 64×64 warp tile of the test kernel), shared memory only; one warp per sub-partition; W = 512 × 32 × 16 = 262 144 cycles of tensor work per warp. Points b1, b48, b96_s0 / s4096 / s16384 (blocks 48–95 first run S dependent FFMA iterations, about 4.5–9 cycles each).
   - H-a (current rule): the tensor pipe is the limit, 16 cycles per HMMA, nothing chip-wide: b1 span ≈ b48 span ≈ W/0.83–1.0 = 262–315 k; b96 span ≈ 2 W; GHz at b48 as the clock kernel (2.49).
   - H-b clock under full-chip tensor load: cycles as H-a, GHz at b48/b96 below 2.45 (upstream sustain.cu: 2.476 after 13 s at 121.5 TFLOP/s).
   - H-c a chip-level tensor or power limit: b48 and b96 cycles per HMMA above b1's by > 3 %.
   - Arbitration between the two blocks of an SM (the `_first` rows; skew S cycles of block 2 before its loop, block 1 alone meanwhile): H-fair (round-robin over warps): first ≈ 2 W − S; H-oldest (oldest block first): first ≈ W + S·0 ≈ 1.0–1.2 W whatever S; the engine (least loop progress first, which equalises): its numbers in `predicted.json`.
4. Engine prediction (cycles): b1 span 316 k, b48 316 k (no chip effect); b96 s0 span 525 k, first 524 k (the two blocks in lockstep); s4096 span 551 k, first 482 k; s16384 span 632 k, first 356 k. So the engine is near H-fair at small skew and moves towards H-oldest at large skew.

## l2share_stages — P-2, second experiment: cp.async delivery by stages × blocks

1. Mechanism: inventory §5 "Chip L2 rate for copy traffic 817.7 B/clk, shared by blocks in flight" (marked a wrong rule in round 1) and the cp.async L2 latency 350 cycles.
2. Why now: P-2 is 1 of 2 experiments; round 1 showed per-SM delivery falling already at 16 blocks (13.6 µs vs 10.5) where the chip limit does not bind. Stages are a design variable of every test HGEMM.
3. Sweep: TileLang's pipelined copy loop, loads only, 128×256×32 tile (24 KB per stage), 32 k-tiles; stages 1, 2, 3, 4 × blocks 1, 16, 48 (one per SM). Stage 1 waits for each tile before the next: a latency meter under load.
   - H-a flat chip share (engine): stage-independent from 2 stages; b16 = b1 (port-bound, ≈ 768 KB / 30.7 B/clk ≈ 25 k cycles ≈ 10 µs); b48 slower by the chip share.
   - H-b load-dependent L2 latency, delivery = bytes in flight / latency (Little): s1 grows with blocks (its time is the latency: 32 × (latency + transfer)); s2 grows at b16 and b48 and s3, s4 recover towards b1.
   - H-c a lower chip rate (≈ 1.3 TB/s for this pattern): b16 and b48 slow, independent of stages ≥ 2.
4. Engine prediction (µs, b1 / b16 / b48): s1 15.4 / 15.4 / 23.3; s2, s3, s4 10.3 / 10.3 / 18.4 (H-a).

## smem_ldsts — STS.128 at 4 warps, second experiment

1. Mechanism: inventory §3 "Plain shared loads/stores": STS register read 3.25 cycles per 128 B after the array.
2. Why now: open question 4 (1 of 2): STS.128 at 4 warps 7.02 measured vs 5.51 replayed (−22 %); cheap (the sweep gains 2, 3, 5, 6 warps for every kind).
3. Hypotheses, cycles per warp instruction per SM for STS.128 at 1, 2, 3, 4, 5, 6, 8 warps:
   - H-a current rule: monotone falling to the array's 4.0 (the engine's numbers).
   - H-b a per-sub-partition limit of ≈ 28 cycles per STS.128 (7.02 × 4): 23, 14, 9.3, 7.0, then 5 warps put two on one sub-partition: 56 / 5 = 11.2, 6 warps 9.3, 8 warps 7.0 → 4.0 only when the array binds; 5 warps slower than 4.
4. Engine prediction, STS.128 at 1, 2, 3, 4, 5, 6, 8 warps: 22.0, 11.0, 7.34, 5.51, 4.43, 4.00, 4.00 (H-a).

## Test set growth (28 §5 table; LOOP §8: queued measurements ride along)

Not probes; scored. TileLang 0.1.15, same pass configs as the HGEMMs (no TMA, no warp specialisation):

| Kernel | Shape | Cache | Front end |
| --- | --- | --- | --- |
| decode attention GQA split-KV, partial kernel | B 1, 32 query heads, 8 KV heads, KV length 4096, D 128, 8 splits, BN 64, 2 stages | warm, rotating | generic |
| decode attention, combine kernel | 32 heads × 8 splits → D 128 | warm | generic |
| w4a16 GEMV | N 4096, K 4096, int4 two per byte, fp16 scales per 128 | warm, rotating | generic |
| RMSNorm with weight | 512 × 4096 fp16 | warm, rotating | generic |
| row softmax | 512 × 4096 fp16 | warm | generic |

Engine predictions (µs): decode partial warm 23.7, rotating 81.0; combine 3.35; GEMV warm 51.8, rotating 431; RMSNorm warm 5.23, rotating 21.0; softmax 7.09. The generic front end is the one FA uses (busiest SM, block replacement, chip shares). Rotating rows: every load a DRAM miss at the DRAM rate shared by the blocks in flight (as `dram_rotate`). The split-KV sequence (partial + combine back to back) is a multi-kernel sequence the engine cannot replay: its two kernels are scored one by one, the sequence is `unsupported: multi-kernel sequence`.
