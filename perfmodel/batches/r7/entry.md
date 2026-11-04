# Batch r7: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json`, engine rules as at the end of round 6 (one line changed: the HGEMM trip count takes stages 0 as 1). New harnesses were smoke-run once on spark before this block; nothing from that run is recorded.

## Selection (LOOP §3, supervisor guidance)

- 28 §1 acceptance items never tested: what-if direction and magnitude, Kendall τ, best variant. 
- Worst rows: RMSNorm / softmax graph −25 / −21 % (chip L2 share 16–32 %, stores 12–41 %), decode rotating and sequence +13 to +17 % (DRAM share 84 %), w4gemm −13 %.

## What-if set (`wi_*`, launch.json `whatif`: base unit and axis)

One change at a time from a base kernel, each compiled and replayed from its own SASS, the base in the same batch; predicted Δ = predicted(changed) / predicted(base) − 1, measured Δ likewise, from the `kernel` rows.

| Base | Changes |
| --- | --- |
| HGEMM 2048³ 128×128×32 s3 t128, warm and rotating | stages 1, 2, 4, 6; direct load (no cp.async); BK 16, 64; BM 64, BN 64; BM or BN 256 with t256; threads 64, 256; raster panels 4, 8; compute removed (mma gone, the C tile kept); loads removed (no global reads, mma on one shared tile); stores removed (C to shared memory) |
| HGEMM 128×256×64 t256 s3; 64×128×32 t128 s4, warm and rotating | stages 1, 2: the overlap question at other load/compute balances |
| FA S2048 H8 D128, warm (rotating for 4) | stages 1, 3, 4; direct; BN 32, 128; BM 128; t64; t256 at BM 128; stores removed; second mma and V removed; exp2 removed |
| GEMV w4a16 4096², warm and rotating | n_part 2, 8; reduce 16, 64; 8 or 32 bytes per lane; compute removed (dequantisation and multiply) |
| RMSNorm 512×4096, warm and rotating | threads 64, 256; fp32 (bytes per lane); compute removed; stores removed; rotation depth ×2, ×4 |
| softmax 512×4096, warm | threads 64, 256; stores removed |
| decode partial, warm and rotating | stages 1, 3; split 4, 16; BN 32; S 2048, 8192; rotation depth ×2, ×4 |
| w4gemm c128, warm | mma removed; dequantisation removed (question 21: which part carries the −13 %) |

Score: direction right when the signs agree, or when the measured change is within 2 % of zero and the predicted one within 2 % too. Magnitude error = |predicted Δ − measured Δ| / |measured Δ| on pairs with |measured Δ| ≥ 5 %. Bottleneck: the base's top attribution unit against which removal (compute, loads, stores) gains most, measured.

## Blind groups

`hgemm 3072x1536x2048` (14 variants, warm and rotating), `fa 1536x12x96` (10 variants): no engine has seen these shapes; Kendall τ, Spearman, inversions and best-variant at first score.

## l2mix — what the chip L2 read share depends on (question 15, new row: RMSNorm / softmax graph)

1. Mechanism: inventory §4 chip L2 share under stores.
2. Why now: `sim.l2_chip_read_share` 0.45 from one st256 point; kernels want near 1, strided stores more than 1.
3. Sweep: L loads and S stores per thread (32 B per load): L:S 4:1 … 1:4; interleaved or phased; stores 32 / 16 / 4 B contiguous, 16 and 32 B at 64 B apart; grids 96, 512. Share s = (both − stores alone) / loads alone.
   - H-const: s = 0.45 everywhere (`predicted.json`).
   - H-phase: interleaved 0.45, phased near 1 (a row kernel's loads and stores do not run together in a thread, the store's buffering does not overlap the reads).
   - H-ratio: s falls with the store share: the two streams overlap as far as the shorter one lasts.
   - H-sector: one L2 lookup path in sectors: both = read sectors / R + write sectors / W, s = 1 for every shape, stores cost by sectors touched.
4. Engine prediction: `predicted.json`.

## dram_burst — DRAM rate of a short read burst (questions 12, 24)

1. Mechanism: inventory §5 DRAM read rate.
2. Why now: decode reads 17 MB in at most 68 µs (250 GB/s), above the 237 GB/s plateau; the DRAM-bound short rows all replay slow.
3. Sweep: n × 786 KB per launch, n 5–85, t256 and t1024, a walk over 256 MB.
   - H-flat: bytes / 237 + latency + launch (`predicted.json`).
   - H-burst: short bursts run above the plateau.
4. Rotation depth rows: H-cold: rotation depth changes nothing (replay: Δ 0). H-partial: part of a rotating set survives in the L2 (the warm curve still loses only 0.85 at 32 MiB), time grows with the depth.
