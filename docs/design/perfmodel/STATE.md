# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-04 at the end of round 1. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| GPU batches | r1 (probes + 10 HGEMMs; valid: clock drift −0.03 %, canaries +0.4 % / −0.4 %), r1fa (FA forward; valid). Clock 2.494 / 2.496 GHz, FFMA canary 490 µs, memory canary 2460 µs (the references for later batches) |
| Test set | 10 TileLang HGEMMs at 2048³ (one shape, variants of tile, stages, threads, raster) and 1 TileLang FA forward (S 2048, H 8, D 128), warm. Decode attention, w4a16 GEMV, RMSNorm/softmax, grouped GEMM: not yet built |
| Scorecard, test set | HGEMM 2048³: median 4.1 %, worst −12.4 %, 9/10 within 10 %, Spearman 0.65, 6 inversions, counts check 10/10, bottleneck 9/10. FA forward: −7.0 %, counts 1/1, bottleneck 1/1. The model is fast on 9 of 11 rows |
| Scorecard, development rows | gemm1024 full median 4.4 % (was 5.3), worst −16.2 %, Spearman 0.82, 23 inversions; nocompute 5.2 % (was 6.7); noload 2.4 % (was 3.4); tma full 8.9 % (was 11.2) |
| Probe set | 8 KTA probes, 72 points, all registered as table rows (tolerance 10 % per point; the open points below are the exceptions): smem_ldsts, shfl, ex2_ffma, namedbar, l1hit, store_tail (P-1), l2share_copy (P-2), dram_rotate |
| Engine | New `smem_op` class (LDS, STS, SHFL); BAR.ARV does not block; streaming LDG DRAM latency; 11 parameters from probes (gb10.json `sim.value_source`) |
| Not computed yet | the "mechanisms only" column (LOOP §4.4) |

## Error table: top valid rows

Valid = counts check passed and cache state declared: the 11 test-set rows.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| hgemm2048 c128x128x32_s3_t128_sw8 | −12.4 % | tensor queue (mma) 39 %, stall after mma 24 %, store queue 8 % |
| hgemm2048 c128x128x32_s3_t128 | −7.3 % | tensor queue (mma) 41 % |
| fa_fwd s2048 h8 d128 | −7.0 % | stall after mma 23 % |
| hgemm2048 c128x256x32_s3_t256_sw4 | −5.4 % | tensor queue (mma) 46 % |

The two raster variants (sw8, sw4) measure the same as their unpanelled twins (221.7 vs 221.6 µs, 234.8 vs 236.7 µs); the model gives them 6–8 % from fewer DRAM misses.

## Open questions (experiment counts per mechanism, LOOP §3)

1. **cp.async delivery under load (P-2, chip L2 share)** — 1 of 2. Per-SM delivery falls already at 16 blocks (13.6 µs vs 10.5), below the chip limit; none of flat chip rate / bytes-in-flight / lower chip rate survived. Next experiment: stages × blocks sweep (1, 2, 3 stages at 1, 16, 48 blocks) to test a load-dependent L2 latency with Little's law.
2. **HGEMM 2048³ model fast** — 0 of 2. Tensor queue tops the path on 8 of 10 rows. Candidate: tensor pipe throughput or clock under full-chip mma load (the clock kernel is FFMA-only).
3. **Raster panels give no gain on the GPU** — 0 of 2. L2 policy is `A` in the inventory (§5 row "L2 capacity and policy").
4. **STS.128 at 4 warps** — 1 of 2. 7.02 measured vs 5.51 replayed (−22 %); all other STS points within 6 %.
5. **Issue limit exceeded** — 1 of 2. `ex2_ffma E0_F32` (198 cycles for 280 instructions per sub-partition) and upstream F2I at 16 warps. Excluded (D758).
6. **DRAM latency: copy 1560 vs streaming LDG 1100** — 0 of 2. Only LDG uses 1100.
7. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 % (the knee). Below 3 % of any valid row's path: not refined.
8. Decode rows' cache state (tma m128, m32): needs a rotating-input re-measurement of those upstream kernels. Not one of our test kernels.
9. Silent default left: `BAR.RED` is treated as blocking. A TMA kernel without warp specialisation still gets the default latency.
10. Two clocks: settled for KTA batches (measured per batch, D757); upstream rows keep 2.52 / 2.54.

## Frozen parameters

None frozen at `unidentified` yet. Values marked `fitted` that the table depends on: `launch_stream_us`, `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Still `assumed`: `sim.default_var_latency_clk` 20 and `sim.rbar_release_clk` 8.

## Fit ledger

| Name | Component | Value | Data | Rows on whose path it sits (share) |
| --- | --- | --- | --- | --- |
| launch_stream_us | launch | 2.6 µs | upstream ablation harness (fitted) | gemm1024 nocompute rows up to 12 %; test-set rows about 1 % |

No fit was added in round 1.

## Round reports

```
Round 1: one batch (r1 + r1fa for FA) — probes for LDS/STS, SHFL, MUFU.EX2 beside FFMA, BAR.SYNC/BAR.ARV, L1 hits, DRAM streaming, the C store (P-1) and cp.async chip-L2 delivery (P-2); first test set (10 HGEMM 2048³ variants, 1 FA forward) with timings and ncu counters.
Question: the test kernels had no measurement; inventory mechanisms they use (shared loads/stores, shuffles, ex2 beside FFMA, named barriers) were absent or assumed; the worst upstream loads-only rows (−23.7 %, −18.4 %) put 46 % / 28 % of their path on chip-L2 share and stores.
Hypotheses: per probe in perfmodel/batches/r1/entry.md (current rule / array-rate / per-instruction-cost; non-blocking arrive; lower chip store limit / store tail; flat / bytes-in-flight / lower chip L2 rate).
Result: LDS, STS, SHFL take the shared-memory array for max(2 cycles, 128 B/clk), MIO slot 5 cycles, latency 28 (SHFL), 24 + bytes/32 (LDS); BAR.ARV does not wait (12 cycles flat); BAR.SYNC base 4, not 13; L1 hits 1 cycle per line, not 2; streaming DRAM latency 1100 cycles, plateau 236 GB/s as modelled; MUFU and FFMA on separate pipes as modelled. P-1: the store rule holds (no per-block tail); chip limit 122 B/clk, 16-byte stores 11 cycles per pair. P-2: no hypothesis survived.
Change: machine.yaml class smem_op (LDS, STS, SHFL); sm.py smem_op handler, BAR.ARV non-blocking, LDG DRAM latency; gb10.json: 7 new and 4 changed values, all `measured` with batch r1 as source; 8192 row deleted.
Scorecard: test set first scored: HGEMM median 4.1 %, worst −12.4 %, Spearman 0.65, 6 inversions, counts 10/10, bottleneck 9/10; FA −7.0 %. With the old engine on the same measurements: median 3.8 %, Spearman 0.62, 6 inversions. Probe rows: median 0.1–6 % (was 6–86 %). gemm1024 full median 5.3 → 4.4 %, Spearman 0.835 → 0.823, inversions 22 → 23. 66 rows better by more than 2 points; 3 worse (hgemm c128x128x32_s3 −4.6 → −7.3 %, gemm1024 c64x64x32_st3_th128 nocompute +7.5 → +9.9 %, c64x64x32_st3_th256 full +1.4 → +3.5 %). By LOOP §5 the round did not improve the model cleanly.
New mechanism: yes — inventory: LDS/STS X → M, SHFL X → M, ex2 beside FFMA X → M, BAR.ARV X → M, BAR.SYNC base A → M, clock A → M (KTA batches); store port and chip store limit re-measured; chip L2 share marked as a wrong rule.
Fits: none added; largest fit share on a path: launch 12 % (gemm1024 nocompute rows), about 1 % on test-set rows.
Next: worst valid row hgemm2048 c128x128x32_s3_t128_sw8 (−12.4 %): tensor queue 39 %, a measured mechanism, so the probe tests the rule: mma throughput and clock under full-chip tensor load; with P-2's second experiment (stages × blocks) in the same batch.
```
