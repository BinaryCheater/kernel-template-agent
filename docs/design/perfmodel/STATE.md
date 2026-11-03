# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-04 at the end of round 4. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| GPU batches | r1, r1fa (round 1); r2, r2b, r2c (round 2); r3 (round 3); r4 (round 4: probes dram_rw, launch sequences; every test kernel re-measured and timed three ways; HGEMM rotating; MoE grouped GEMM), all valid: r4 clock 2.485 GHz (drift −0.7 %), canaries within +0.5 % |
| Test set | 10 HGEMM 2048³ variants warm and (round 4) rotating, FA forward, decode GQA split-KV partial (warm, rotating) and combine, w4a16 GEMV 4096² (warm, rotating), RMSNorm 512×4096 (warm, rotating), softmax 512×4096, 2 FP8 GEMMs 2048³ (e4m3), (round 4) MoE grouped GEMM 8 experts × 256 tokens, K 2048, N 1024, two tiles. Every test kernel has three rows (D773): `kernel` (one launch between events), `b2b` (50 back-to-back launches, per launch), `graph` (50 launches in one CUDA graph, per launch). The split-KV sequence: `unsupported: multi-kernel sequence`. NVFP4 GEMM: not built |
| Scorecard, test set (with fits / mechanisms only; r4 data, `kernel` rows) | HGEMM warm: median 3.9 / 4.1 %, worst −13.4 / −13.6 %, Spearman 0.77 / 0.77, inversions 4 / 4, counts 10/10, bottleneck 9/10; HGEMM rotating: 7.3 %, worst −12.5 %, Spearman 0.93, 2 inversions. FP8 GEMM −4.4 / −9.0 %. FA −6.5 %. RMSNorm rotating +16.0 %, warm −34.4 %; softmax −27.1 %; combine +23.3 %; GEMV rotating +24.5 %, warm −10.6 %; decode partial warm −8.4 %, rotating +15.2 %; MoE +231 % / −63 % (generic front end). `graph` rows (launch 0.6 µs): RMSNorm warm −50.5 %, softmax −35.0 %, combine +5.5 %; `b2b`: combine +57.8 %. R1 passes on every test row except MoE 1/2. No fitted parameter on a KTA batch row |
| Re-measurement r3 → r4 | HGEMM within 1.6 % per variant (s4 +1.5 %), FA −0.3 %, small kernels within 3.2 % (decode warm +3.2 %, GEMV rotating −2.4 %) |
| Probe set | 13 KTA probes: round 3's 12 plus `dram_rw` (read:write mixes over 30 MiB planes, rotating); `launch` gains sequence points (b2b and graph, FFMA spans 0.1–151 µs; graph empty grids) |
| Engine | round 4: DRAM writes paced in rotating rows (D771); warm HGEMM DRAM reads by the warm-miss curve on A + B (D772); launch sequences max(submit, span + gap) (D773) |

## Error table: top valid rows

Valid = counts check passed and cache state declared.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| moe c128x128x32_s3_t128 kernel (also b2b, graph) | +231 % | DRAM queue 89 % (generic front end: every load misses by the 44 MiB footprint curve, the GEMM's reuse ignored) |
| moe c64x128x32_s4_t128 kernel (R1 fails) | −63 % | DRAM queue 80 % |
| decode combine b2b | +57.8 % | launch gap (span 0.3 µs: below 0.5 µs the GPU runs back-to-back launches at the submission rate, 1.9 µs) |
| rmsnorm warm graph | −50.5 % | chip L2 share 56 % (span replayed 2.8 µs, measured ≈ 6.4) |
| softmax warm graph | −35.0 % | ALU queue 17 %, stall after FP32 14 % (span replayed 4.2 µs, measured ≈ 6.8) |
| rmsnorm warm kernel | −34.4 % | launch 50 %, chip L2 share 28 % |
| gemv_w4a16 rotating graph | +30.2 % | load latency incl. DRAM queue 95 % |

The `graph` rows carry almost no launch (0.6 µs): RMSNorm and softmax are slow in the kernel itself, not in the launch.

## Open questions (experiment counts per mechanism, LOOP §3)

1. **Sharing between co-resident blocks** — closed at 2 of 2 (D768): fair sharing.
2. **Raster panels** — 1 of 2: on the GPU the panel gains nothing warm (sw8 +0.1 %, sw4 0.0 %) and 3.5 % / 0.8 % rotating (r4); the replay gives sw8 5.1 % warm even with almost no warm DRAM traffic (D772), so the replay's gain is not DRAM. Candidate: the panelled prologue (integer division, a fallback branch) or the taken-branch cost (question 4).
3. **HGEMM 2048³ model fast** — 1 of 2: warm −1 to −13 %, rotating −4 to −12.5 % (r4); the loss is outside the DRAM path (loads from L2, block turnover, stores); the 128-thread 128×128×32 variants are the worst both ways.
4. **Taken-branch cost** — 0 of 2: the FFMA loop takes 23.0 cycles per iteration on the GPU, 17.6 in the replay; now visible on the launch probe's sequence rows (b2b / graph work points −16 to −21 %); still below 3 % of every test row.
5. **Launch hidden behind short kernels** — closed at 2 of 2 (D773): not hidden; back-to-back span + 2.74 µs, graph span + 0.60 µs at spans 0.7–151 µs. Below 0.5 µs span the back-to-back gap shrinks to the submission rate (empty chain 1.90 µs, combine 1.91 µs): new question 14.
6. **Softmax and FA after the FCHK fix** — compensating mechanism missing (unchanged); softmax graph −35 % shows the kernel span itself.
7. **DRAM writes in rotating rows** — closed at 1 of 2 (D771): reads / 237 + writes / 197 GB/s (dram_rw r4). RMSNorm rotating −41.5 → +16.0 % (now slow: the replay's DRAM queue for a 512-block, one-row-per-block stream; question 15).
8. **Hits on pending misses** — 0 of 2 (unchanged); no test row.
9. **Issue limit exceeded** — 1 of 2 (unchanged). Excluded (D758).
10. **DRAM latency: copy 1560 vs streaming LDG 1100** — 0 of 2 (unchanged).
11. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 %; tiny-kernel store path at large grids; graph of 4096 empty blocks 2.52 µs (replay 0.67): below 3 % of any valid row's path.
12. Decode rows' cache state (tma m128, m32): needs a rotating re-measurement of those upstream kernels.
13. Silent default: `BAR.RED` treated as blocking.
14. **Back-to-back launches of sub-0.5 µs kernels** — new, 0 of 2: combine b2b +57.8 %.
15. **Small-kernel spans: RMSNorm warm and softmax replayed 1.6–2.3× fast** — new, 0 of 2 (graph rows −50.5 / −35 %); RMSNorm rotating +16 %. 512 blocks of 128 threads, one row each: block turnover and the chip L2 share for warm LDG traffic are the candidates.
16. **MoE grouped GEMM on the generic front end** — new (D774): +231 % / −63 %, R1 1/2; a front-end defect (GEMM reuse, per-expert B), fixed by routing it through the HGEMM front end with E operand matrices, checked against R1, not a probe.
17. **Write-only streams** — new: no DRAM pacing without reads (dram_rw 0:2 −85 %); no test row.

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

```
Round 4: batch r4 — probe dram_rw (DRAM read/write mix), probe launch sequence points (b2b, graph); every test kernel timed three ways (kernel / b2b / graph); HGEMM rotating; MoE grouped GEMM added.
Question: worst row RMSNorm rotating −41.5 % (DRAM 84 %, writes not paced); worst-inverted HGEMM pair c128x128x64_s2_t128 vs c128x256x32_sw4 (measured −4.8 %, replay +6.2 %): the diff is 87 µs of DRAM latency+queue on s2 only, because every warm replay read A+B from DRAM; launch term 41–91 % of small rows.
Hypotheses: DRAM mix reads-only / sum of rates / one rate; warm HGEMM reads A+B from DRAM / hits the L2; launch in a sequence additive gap / hidden behind the kernel.
Result: dram_rw 267, 428, 289, 432, 320 µs = reads/237 + writes/197 (sum survives); HGEMM rotating costs +0 to +9.4 % over warm (warm mostly L2 hits; panels gain only cold); b2b span + 2.74 µs, graph span + 0.60 µs (additive), b2b at sub-0.5 µs spans = submission rate 1.9 µs.
Change: sim.dram_writes_paced (D771), sim.hgemm_warm_reads_by_curve (D772), launch_seq rule with launch_b2b_gap_us 2.74 / launch_graph_gap_us 0.60 measured (D773); gen.py write_bytes, b2b/graph rows, HGEMM rotating, MoE (D774).
Scorecard: RMSNorm rotating −41.5 → +16.0 %; HGEMM median 4.4 → 3.9 %, worst −12.0 → −13.4 %, Spearman 0.62 → 0.77, inversions 8 → 4 (D772 off: 0.59 / 8); HGEMM rotating 0.93 / 2; combine graph +5.5 %, b2b +57.8 %; graph rows show RMSNorm warm −50.5 %, softmax −35 % in the kernel itself; MoE +231 / −63 %; others within re-measurement.
New mechanism: yes — DRAM read/write mix X → M in KTA (dram_rw), launch sequences X → M (b2b, graph), L2 for warm HGEMM: launch-order sim → measured warm-miss curve.
Fits: none added; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute). FP8 rows: QMMA replayed at the HMMA hold (16 cycles); the tensor-pipe rate per data type is not exercised (no per-type parameter).
Next: MoE through the HGEMM front end with per-expert B (front end, R1); then small-kernel spans (question 15: RMSNorm warm graph −50 %) by a row-reduction probe; HGEMM pair sw8 vs s3_t128 (question 2, prologue / taken branch).
```
