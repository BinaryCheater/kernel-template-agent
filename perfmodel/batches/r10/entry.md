# Batch r10: entry blocks

Written before the timed run (write-probe). The engine's replay of every case is in `predicted.json`. SASS checked after the compile: the hot loops hold the intended LDGSTS and LDG.

## Found before the batch, no GPU

HGEMM and FP8 now run on the generic front end. On the same measurements warm τ stays 0.60 → 0.64 at 2048³: warm ranking is not decided by the template path. The engine decides it.

## load_curve — L2 latency against offered load (contention group; questions 32, what-if magnitude)

1. Mechanism: inventory §5 L2 latency, `ldgsts_l2_latency_clk` fixed; L2 return by share.
2. Why now: what-if magnitude median 27 %, bottleneck 6 of 14; hypothesis of the supervisor, unverified: fixed latencies under load.
3. Sweep: G 1, 12, 24, 48, 96, 144 blocks, Z 0, 16, 64, 256, 1024 passes between the load warps' copies; warp 0 times a lone 512-byte copy.
   - H-fixed (replay): 392 cycles at every point, 394 at G 144.
   - H-queue: unloaded 392 × (1 + ρ / (2 (1 − ρ))) with ρ the chip's bytes per cycle over 817: rising from G 24, Z 0, 588 at ρ 0.5, 1570 at ρ 0.8.
   - H-share: the lone copy waits behind this SM's own return queue only: flat in G at each Z, rising as Z falls.

## copy_depth — copies in flight before issue waits (inventory §4 LSU queue)

1. Mechanism: address stage and queue; the engine has no depth limit.
2. Why now: same group; mio throttle stalls seen by ncu upstream.
3. Sweep: K 4–256 copies per warp, 1 and 4 warps, warm and after a flush.
   - H-unbounded (replay): t128 k16 381, k64 1965, k256 7725 cycles to issue; landed 1412, 4633, 17459.
   - H-depth D: issue follows the landings past D copies in flight on the SM: issue time → landed time for K × warps ≫ D, 32 < D < 512.

## dram_spread — DRAM latency and spread under load (inventory §5)

1. Mechanism: `ldg_dram_latency_clk` 1100 fixed, a rate queue.
2. Why now: same group. Not replayed: the chase's addresses are data.
3. Sweep: 47 blocks reading through 256 MiB with Z spacing, one lane chasing a random cycle; hop p10, p50, p90, p99.
   - H-fixed (engine rule): p50 1100 until the plateau of 236 GB/s, spread small.
   - H-rising (upstream dramlat: 421 ns idle, 554 at 121 GB/s, 1159 at 208): p50 1050, 1385, 2900; p90 well above p50.

## wq_* — question 36 and question 32, second experiment

`c128x256x32` t256 above the K 64 rows: K 32 and K 64 at s2 and s3, each whole, without compute, without stores. The part where K 32 stays above K 64 holds the mechanism: H-compute (the gap goes with `nocompute`), H-load (stays in `nocompute`), H-store (goes with `nostore`). Question 32: s2 vs s3 at 3072×1536×2048, same parts.

## Breadth, blind

`hand_*`: 10 hand-written CUDA kernels; `stress_*`: 16 generated mixes (seed 1010). Predictions in `predicted.json`, committed before the run. No parameter is taken from them.

## Counters

Passes `5u`, `5v`: issue stall reasons per warp, MIO queues, L1 to crossbar stalls, L2 port activity, for the per-unit comparison.
