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

## After the run (first score)

r7 valid: clock 2.492 → 2.475 GHz (drift −0.65 %), canaries −0.13 / +0.25 %. Units that did not run: shared memory above the limit (FA s3, s4, BN 128; decode s3; HGEMM 128×256×64 s3, whose two rows at s1 were scored against s2; two FA blind rows).

First score, engine as committed before the run:

| Group | n | median | worst | Kendall τ | Spearman | inversions | best variant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hgemm 3072×1536×2048 warm | 14 | 5.6 % | −60.4 % | 0.71 | 0.85 | 7 | yes |
| hgemm 3072×1536×2048 rotating | 14 | 7.6 % | −58.4 % | 0.78 | 0.93 | 5 | no (+6.0 %) |
| fa 1536×12×96 | 8 | 6.8 % | −11.3 % | 0.86 | 0.95 | 0 | yes |

What-if at first score: direction 79 of 98, median magnitude error 26 %, 29 of 59 within 25 %. By axis: stages 16/18, tile 10/12, threads 13/20, cp.async vs direct 3/3, compute removed 8/9, loads removed 2/2, stores removed 7/7, bytes per lane 4/6, raster 4/4, rotation depth 0/4, blocks 3/4, shape 4/4. Bottleneck: 5 of 10 bases agree; the 5 others have only one or two removals of the three, so the largest measured gain is not a full comparison.

Wrong directions by cause:

- **GEMV mk4, r16, np2** (+73, −84, −68 %): front end: the main loop's passes (D794; np2 warm stays, +73 %).
- **Rotation depth** (decode, RMSNorm, 0/4): the replay has no state between rotations; decode 68.9 → 78.2 → 77.2 µs at depth 67, 134, 268 MB, replay 78.8: rotating at 67 MB is not cold. Questions 12 and 24 answered: the +13 % is the row's cache state.
- **HGEMM stages 1, rotating** (+18 %, measured −1.8 %) and 64×128 s1 (+4, −3 %): the overlap. The replay spreads the DRAM rate evenly over the SMs (inventory: `A`, fair share); a block that loads while others compute takes more than its share. Warm it holds: s1 −2.8 / −6.6 %.
- **HGEMM t256** (−9, +1 %), **bk16** (−5, +2 to +7 %): the replay gains from more warps or shorter loads, the GPU does not.
- **decode split 16 warm** (−12, −0.6 %), **FA exp2 removed** (−4, +6 %: the kernel's values change), **RMSNorm t64 / t256**: launch of a 9 µs kernel (b2b −3 %), not the span.

Overlap, stages 1 against the base, measured / first: HGEMM s1 vs s3 warm −6.6 / −2.8 %, rotating −1.8 / +18.1 %; 128×256×64 s1 vs s2 +13.2 / +28.0 %, +20.3 / +43.6 %; 64×128 s1 vs s4 −2.4 / −2.4 %, −3.0 / +4.1 %; FA s1 vs s2 −15.5 / −15.6 %, −13.2 / −12.0 %; decode s1 vs s2 −12.4 / −15.9 %, +1.7 / −3.3 %. Sign right in 8 of 10; the replay overstates what overlap gains under DRAM load.

Spills: the largest blind errors are one mechanism: 256×256×32 s2 t256 870 µs (−60.4 %), HGEMM t64 +228 % (+82 %), FA without the second mma +186 % (+27 %, the fragment leaves the mma's layout and spills): each loop with 255 registers per thread (r7b `spill`).

**dram_burst**: H-flat survives: 3.9–67 MB per launch within −5.9 to +4.8 % of the replay, 187–237 GB/s with launch; no burst above the plateau.

**l2mix** (grid 512, share s = (both − stores) / loads): H-const rejected: s 0.25–0.71 at contiguous 32 B, above 1 strided; H-phase rejected: phased and interleaved alike (11871 / 11950, 6563 / 6560, 11211 / 11335); H-sector half right: stores cost by what they touch, not bytes (2 × 16 B 3765 = 1 × 32 B 3826; 2 × 4 B 414), reads and stores add less a constant: both ≈ loads + stores − 2200 (r7b, D791).

**RMSNorm**: the probe with the same bytes (L2 S2, phased, 11871 cycles) is not the gap: compute removed gains 10 % measured, 2 % in the replay; the block's arithmetic sits on the path beside the chip-bound streams.
