# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-04 at the end of round 3. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| GPU batches | r1, r1fa (round 1); r2, r2b, r2c (round 2); r3 (round 3: probes l2reuse, launch; every test kernel re-measured; FP8 GEMM), all valid: r3 clock 2.485 GHz (drift −0.7 %), canaries within +0.6 % |
| Test set | 10 HGEMM 2048³ variants, FA forward, decode GQA split-KV partial (warm, rotating) and combine, w4a16 GEMV 4096² (warm, rotating), RMSNorm 512×4096 (warm, rotating), softmax 512×4096; round 3 added 2 FP8 GEMMs 2048³ (e4m3, QMMA.16832). The split-KV sequence: `unsupported: multi-kernel sequence`. MoE grouped GEMM: not built |
| Scorecard, test set (with fits / mechanisms only; r3 data) | HGEMM: median 4.4 / 4.7 %, worst −12.0 / −12.3 %, Spearman 0.62 / 0.62, inversions 8 / 8, counts 10/10, bottleneck 9/10. FP8 GEMM −4.3 / −8.9 %. FA −6.7 %. GEMV warm −9.9 %, rotating +21.5 %; RMSNorm warm −35.1 %, rotating −41.5 %; softmax −28.2 %; combine +23.3 %; decode partial warm −5.4 %, rotating +15.0 %. Counts check (R1) passes on every test row, now including global-load sectors per block for LDG kernels. No fitted parameter left on a KTA batch row |
| Re-measurement r2 → r3 | HGEMM within 0.5 % per variant, FA 0.0 %; small kernels move more between sessions (RMSNorm warm −8.7 %, softmax +6.3 %, decode warm −3.8 %) |
| Probe set | 12 KTA probes: round 2's 10 plus `l2reuse` (footprint 1.5–32 MiB × load kind cg / ca / cs × cold / warm / four passes in one kernel; profiler points) and `launch` (empty and tiny kernels over grids, back-to-back, known-work spans) |
| Engine | round 3: LDG lines, sectors and L1 hits from lane addresses (D765); FCHK fast path (D770); warm L2 misses by footprint (D766); `launch_event_us` for batch rows (D769); scheduler unchanged (D768) |

## Error table: top valid rows

Valid = counts check passed and cache state declared.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| rmsnorm m512 n4096 rotating | −41.5 % | load latency incl. DRAM queue 84 %, launch 12 % |
| rmsnorm m512 n4096 warm | −35.1 % | launch 50 %, chip L2 share 28 % |
| softmax m512 n4096 warm | −28.2 % | launch 41 %, ALU queue 10 % |
| decode combine warm | +23.3 % | launch 91 % |
| gemv_w4a16 rotating | +21.5 % | load latency incl. DRAM queue 89 % |
| hgemm2048 c128x128x32_s3_t128_sw8 | −12.0 % | tensor queue 39 %, stall after mma 24 % |

RMSNorm rotating runs at 207 GB/s on the GPU and 354 GB/s in the replay, above the DRAM plateau: the rotating front end counts only the reads as DRAM traffic at `dram_read_GBs`, while RMSNorm writes as much as it reads (Y, 4 MB).

## Open questions (experiment counts per mechanism, LOOP §3)

1. **Sharing between co-resident blocks** — closed at 2 of 2 (D768): fair sharing; the replay's rule holds. The mma_pair residuals (span −9 %, first +12 % at s16384) are the skew loop's speed (below).
2. **Raster panels give no gain on the GPU, the replay gives 5.5 %** — reopened from question 1, 0 of 2: not L2 traffic (round 2), not the sharing rule (round 3). Candidate: the panelled prologue (integer division, a fallback branch) in the replay.
3. **HGEMM 2048³ model fast** — 1 of 2 (unchanged): the loss is outside the mma loop (loads, block turnover, stores).
4. **Taken-branch cost** — new, 0 of 2: a 4-instruction FFMA loop takes 23.0 cycles per iteration on the GPU, 17.6 in the replay (the stall counts add to 18): about 5 cycles per taken BRA (probe launch work points; mma_pair skew). Below 3 % of every valid test row's path: not refined until a row points at it.
5. **Launch hidden behind short kernels** — new, 1 of 2: combine (2.50 µs measured in total) is below `launch_event_us` 2.80; the probe's event − span is bimodal (2.1 / 3.2 µs) and an empty kernel takes 2.43 µs. Combine +15.3 → +23.3 % by this change (compensating mechanism missing, LOOP §5).
6. **Softmax and FA after the FCHK fix** — compensating mechanism missing: softmax −21.6 → −28.2 %, FA −5.4 → −6.7 % once the slow-division path is no longer replayed.
7. **DRAM writes in rotating rows** — new, 0 of 2: RMSNorm rotating (reads 4 MB, writes 4 MB) faster than DRAM in the replay.
8. **Hits on pending misses** — new, 0 of 2: four passes in one kernel take about the cold time at small footprints (l2reuse in4 rows up to +188 %: the replay serialises passes that the GPU merges on outstanding lines). No test row attributes to it. Related: the l2reuse warm rows at 18–27 MiB are 35–41 % fast with the measured miss fraction (the replay spreads misses evenly; a cyclic scan beyond capacity misses in runs and exposes the DRAM latency); no test row's footprint is there yet.
9. **Issue limit exceeded** — 1 of 2 (unchanged): `ex2_ffma E0_F32`, upstream F2I at 16 warps. Excluded (D758).
10. **DRAM latency: copy 1560 vs streaming LDG 1100** — 0 of 2 (unchanged).
11. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 %; tiny-kernel store path at large grids (launch tiny rows): below 3 % of any valid row's path.
12. Decode rows' cache state (tma m128, m32): needs a rotating re-measurement of those upstream kernels.
13. Silent default: `BAR.RED` treated as blocking.

## Frozen parameters

| Parameter | Value | Status | Why |
| --- | --- | --- | --- |
| `sim.l2_aggregate_B_per_clk_cpasync` (P-2) | 817.7 B/clk | `unidentified` | 2 experiments: 1 stage unchanged at 16 blocks (no latency rise), 2–4 stages −16 to −25 % at 16 blocks with the chip at 0.98 TB/s; flat share, Little's law and a lower chip rate all fail. Candidate on reopening: a share below the chip (TPC pair or GPC) |
| `sim.sts_read_clk_per_128B` | 3.25 | `unidentified` | 2 experiments: STS.128 at 1–6 warps 23.1, 12.0, 8.7, 7.0, 5.6, 4.35; the rule is 4–22 % fast at 2–5 warps; a per-sub-partition limit (5 warps slower than 4) fails |

Values marked `fitted` that the table depends on: `launch_stream_us` (upstream ablation rows only since round 3), `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Still `assumed`: `sim.default_var_latency_clk` 20, `sim.rbar_release_clk` 8, `sim.fchk_fast_path`.

## Fit ledger

| Name | Component | Value | Data | Rows on whose path it sits (share) |
| --- | --- | --- | --- | --- |
| launch_stream_us | launch | 2.6 µs | upstream ablation harness (fitted) | gemm1024 nocompute rows up to 12 %; no KTA batch row since round 3 (replaced by the measured `launch_event_us`, D769) |

No fit was added in rounds 1–3.

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

```
Round 3: batch r3 — probes l2reuse (warm-L2 residency by footprint and load kind) and launch (launch term as the harness times it); LDG front end from lane addresses, FCHK fast path; every test kernel re-measured; 2 FP8 GEMMs added.
Question: GEMV +1079 / +215 %, RMSNorm −48 / −45 %, combine +35 % sat on the generic front end's 4-line LDG default, its all-hit warm L2 and the fitted launch (78 % of combine); decode partial and softmax failed R1; HGEMM Spearman 0.60, 6 inversions (block sharing 1 of 2).
Hypotheses: L2 full 24 MiB LRU / smaller effective capacity / no-allocate for some kinds; ncu sees the harness state / ncu refills; launch constant / grows with threads / additive vs hidden; sharing fair / oldest-first.
Result: warm re-reads miss 0.02 to 16.5 MiB, 0.13 at 20, 0.62 at 24, 0.85 at 32 MiB (smaller capacity), .cs lines not kept beyond 1.5 MiB; under ncu every launch refills the L2 (the 6 % hit rate is the profiler); launch 2.80 µs beyond the span (bimodal), empty kernel 2.43 µs to 131 k threads; sharing fair, oldest-first rejected (skew loop is 23 cycles/iteration, not 18).
Change: lanes.py + sm.py (LDG lines, sectors, L1 hits from lane addresses; FCHK); gb10.json l1_unified_bytes, l2_warm_miss_by_MiB, l2_warm_miss_evict_first_by_MiB, launch_event_us (measured, r3); generic front end warm misses by footprint; R1 gains LDG sectors; D765-D770.
Scorecard: GEMV +1079 → +21.5 %, +215 → −9.9 %; RMSNorm −48 / −45 → −41.5 / −35.1 %; combine +35 → +23 %; softmax −22 → −28 %; R1 decode 0/2 → 2/2, softmax 0/1 → 1/1; HGEMM 0.62 / 8 inversions on r3 data, unchanged by every change (r2 → r3 within 0.5 %); one ablation per change (r3/entry.md, after the run): lanes, FCHK, L2 curve, launch all kept (probe- or counter-backed); combine and softmax worsen: open questions 5, 6.
New mechanism: yes — inventory: L2 capacity A → M (curve, empirical), launch M re-measured as the harness times it, L1 hit path driver list → lane addresses, warp scheduler P → M (fair between blocks).
Fits: launch_stream_us off every KTA batch row (measured launch_event_us instead); none added; largest fit share on a path: launch 12 % (upstream gemm1024 nocompute).
Next: worst valid row RMSNorm rotating −41.5 % (DRAM load latency/queue 84 %): the rotating front end ignores DRAM write traffic and uses the datasheet read rate — a probe of mixed read/write streaming (question 7); then launch hidden behind short kernels (question 5).
```
