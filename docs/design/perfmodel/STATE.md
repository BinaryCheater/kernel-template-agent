# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-04 at the end of round 2. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| GPU batches | r1, r1fa (round 1); r2 (probes, HGEMM and FA re-measured, softmax), r2b (decode, combine, GEMV), r2c (RMSNorm) in round 2, all valid: clock 2.485 / 2.485 / 2.484 GHz (drift −0.7 % to −0.8 %), canaries within +0.6 % |
| Test set | 10 HGEMM 2048³ variants and FA forward (warm); round 2 added decode GQA split-KV partial (warm, rotating) and combine (warm), w4a16 GEMV 4096² (warm, rotating), RMSNorm 512×4096 (warm, rotating), softmax 512×4096 (warm). The split-KV sequence as a whole: `unsupported: multi-kernel sequence`. MoE grouped GEMM: not built |
| Scorecard, test set (with fits / mechanisms only) | HGEMM 2048³ (r2 measurement): median 4.3 / 4.6 %, worst −12.1 / −12.3 %, Spearman 0.60 / 0.60, inversions 6 / 6, counts 10/10, bottleneck 9/10. FA −6.6 / −6.8 %. Decode partial warm −7.5 %, rotating +18.0 % (counts check fails both: not valid); combine +35 / +13 %; GEMV warm +215 %, rotating +1079 %; RMSNorm warm −44.5 %, rotating −48.1 %; softmax −21.7 % (counts check fails). Fits move only launch-dominated rows (combine, RMSNorm warm, softmax) |
| Re-measurement | HGEMM r1 → r2 (second session): every variant within 0.4 % except c128x128x64_s2 (−0.9 %); FA +0.0 % (254.1 µs) |
| Probe set | 10 KTA probes: round 1's 8 plus `mma_pair` (tensor pipe in the GEMM warp tile's loop, 1/48/96 blocks, start skew) and `l2share_stages` (P-2, stages × blocks); `smem_ldsts` gained 2, 3, 5, 6 warps |
| Engine | rules unchanged in round 2; batch front end extended (probe builders, rotating rows in the generic front end); parameter statuses `l2_aggregate_B_per_clk_cpasync`, `sts_read_clk_per_128B` → `unidentified` |

## Error table: top valid rows

Valid = counts check passed and cache state declared.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| gemv_w4a16 n4096 k4096 rotating | +1079 % | load latency incl. DRAM queue 99 %; defaults: front end 4224 (ldg_lines), issue-only 6864 |
| gemv_w4a16 n4096 k4096 warm | +215 % | chip L2 share queue 87 %; same defaults |
| rmsnorm m512 n4096 rotating / warm | −48.1 / −44.5 % | DRAM latency 86 % / launch 50 % |
| decode combine warm | +35.0 % | launch 78 % |
| hgemm2048 c128x128x32_s3_t128_sw8 | −12.1 % | tensor queue (mma) 39 %, stall after mma 24 % |

The GEMV and RMSNorm rows sit on the generic front end's load defaults (every LDG taken as 4 lines, 512 B, whatever its width: the `front end` default count) and, warm, on its all-hit L2 assumption (ncu without flush: L2 sector hit rate 6 % for GEMV, 2 % for decode, while GEMV warm runs at 506 GB/s).

## Open questions (experiment counts per mechanism, LOOP §3)

1. **Sharing between co-resident blocks** — 1 of 2. `mma_pair`: with a start skew the older block finishes almost as if alone (317 k cycles; alone 311 k; replay 356 k, +12 %), the pair's span is 694 k (replay 632 k, −9 %). The replay's sharing comes from the tensor pipe's queue order (swapping the issue pick to oldest-block-first changes no number). It is the replay's raster gap: the panelled twin's prologue desynchronises its pair. Next: find and change the queue-order rule, rescore the twins.
2. **HGEMM 2048³ model fast** — 1 of 2. Not the tensor rate and not the clock: the warp tile's loop is replayed within 1.8 % at 1 and 48 blocks and −1.0 % at two blocks per SM; the in-kernel clock under full-chip tensor load is 2.493 GHz (alone 2.500). The loss is outside the mma loop (loads, block turnover, stores).
3. **Raster panels give no gain on the GPU** — closed as an L2 question: the replay gives both twins the same DRAM traffic and ncu the same hit rate (93.2 / 92.8 % warm, r2; lookup misses 4.3 vs 8.5 MB). Moved to question 1.
4. **Generic front end for non-GEMM kernels** — new, a front-end defect (system bug, no probe): LDG line counts by width and warm L2 hits (GEMV, RMSNorm); decode and softmax fail the counts check (branch fallbacks).
5. **Issue limit exceeded** — 1 of 2 (unchanged): `ex2_ffma E0_F32`, upstream F2I at 16 warps. Excluded (D758).
6. **DRAM latency: copy 1560 vs streaming LDG 1100** — 0 of 2 (unchanged).
7. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 %: below 3 % of any valid row's path.
8. Decode rows' cache state (tma m128, m32): needs a rotating re-measurement of those upstream kernels.
9. Silent default: `BAR.RED` treated as blocking.
10. Two clocks: settled for KTA batches (D757).

## Frozen parameters

| Parameter | Value | Status | Why |
| --- | --- | --- | --- |
| `sim.l2_aggregate_B_per_clk_cpasync` (P-2) | 817.7 B/clk | `unidentified` | 2 experiments: 1 stage unchanged at 16 blocks (no latency rise), 2–4 stages −16 to −25 % at 16 blocks with the chip at 0.98 TB/s; flat share, Little's law and a lower chip rate all fail. Candidate on reopening: a share below the chip (TPC pair or GPC) |
| `sim.sts_read_clk_per_128B` | 3.25 | `unidentified` | 2 experiments: STS.128 at 1–6 warps 23.1, 12.0, 8.7, 7.0, 5.6, 4.35; the rule is 4–22 % fast at 2–5 warps; a per-sub-partition limit (5 warps slower than 4) fails |

Values marked `fitted` that the table depends on: `launch_stream_us`, `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Still `assumed`: `sim.default_var_latency_clk` 20 and `sim.rbar_release_clk` 8.

## Fit ledger

| Name | Component | Value | Data | Rows on whose path it sits (share) |
| --- | --- | --- | --- | --- |
| launch_stream_us | launch | 2.6 µs | upstream ablation harness (fitted) | gemm1024 nocompute rows up to 12 %; test-set rows about 1 % |

No fit was added in rounds 1 and 2. The mechanisms-only column (D760) shows the launch fit (2.6 vs 2.05 us) moving the launch-dominated new rows by up to 22 points (combine +35 to +13 %), the HGEMMs by 0.2-0.3.

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

```
Round 2: batches r2 + r2b + r2c — probes mma_pair (tensor pipe, block sharing, clock under load), l2share_stages (P-2 second experiment), STS.128 at 2–6 warps; HGEMM/FA re-measured; 7 new test rows (decode GQA, combine, w4a16 GEMV, RMSNorm, softmax; warm and rotating).
Question: worst valid row hgemm2048 sw8 −12.4 % (tensor queue 39 %); raster twins measure equal, the replay gives the panel 5.5 %; P-2 and STS.128 at 1 of 2.
Hypotheses: mma rule / clock drop under full-chip tensor load / chip tensor limit; block sharing fair / oldest-first; P-2 flat share / Little's law / lower chip rate; STS rule / per-sub-partition limit.
Result: mma rule holds (+1.8 % at 1 and 48 blocks, −1.0 % at 2/SM), clock 2.493 GHz under load (alone 2.500); the older block of a skewed pair runs almost as alone (317 k vs 311 k; replay 356 k); raster: same DRAM traffic in the replay, same L2 hit in ncu, so the gap is the replay's block sharing; P-2: none survived; STS: none survived.
Change: no engine rule; P-2 and STS parameters frozen `unidentified` (D764); mechanisms-only column (D760); ncu without its cache flush (D761); generic front end takes rotating rows (D763).
Scorecard: HGEMM (r2 data, engine same) median 4.1 → 4.3 %, worst −12.4 → −12.1 %, Spearman 0.65 → 0.60, inversions 6 → 6; mechanisms only 4.6 % / 0.60 / 6; new rows GEMV +215 / +1079 %, RMSNorm −45 / −48 %, combine +35 %, softmax −22 %, decode −7.5 / +18 % (counts check fails).
New mechanism: no; inventory warp scheduler M → P (between blocks), tensor pipe confirmed at chip scale, clock under tensor load measured, chip L2 share and STS `unidentified`.
Fits: none added; mechanisms-only gap largest on launch-dominated new rows (combine 22 points), HGEMM 0.2.
Next: worst valid row GEMV rotating +1079 %: the generic front end's LDG line count by width and its warm all-hit L2 (a front-end defect, fixed without a probe); then block sharing in the tensor queue (question 1) for the HGEMM ranking.
```
