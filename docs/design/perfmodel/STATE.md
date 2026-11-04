# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-04 at the end of round 7. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| GPU batches | r1 … r6b as before, all valid; round 7: r7 (160 kernels: one-change variants, larger blind groups; probes l2mix, dram_burst), r7b (probes rot_depth, spill, store_lines), valid: r7 clock 2.483 GHz, drift −0.65 %, canaries within 0.25 %; r7b drift +0.1 %, canaries within 0.5 % |
| Test set | As round 6 plus: blind groups `hgemm 3072x1536x2048` (14 variants, warm and rotating), `fa 1536x12x96` (8 ran), and the what-if set `wi_*` (98 changes from 8 kinds of base kernel, launch.json `whatif`) |
| Position against 28 §1 (user's targets; "accepted" is the user's word) | **Ranking** τ ≥ 0.8: not met. HGEMM 2048³ warm 0.64, rotating 0.78; 3072×1536×2048 first score 0.71 / 0.78; FA blind 0.86, gemm1024 0.75, tma 0.87; groups of 3–4 0.33–1.00. Best variant: 11 of 13 groups (missed: 4096×1024×2048 warm +2.7 %, 3072×1536×2048 rotating +6.0 %). **What-if** direction every time: not met, 79 of 98 first score, 77 now; magnitude: median 26 %, 28 of 59 within 25 %. Bottleneck: 5 of 10 bases agree, and only where all three removals ran is it a full test (HGEMM, both agree). **Time** median ≤ 10 %: met, 7.8 % on 91 `kernel` rows; none above 25 %: 2 rows, 256×256×32 t256 +31 / +38 %. **Probe ladder**: no per-level budget set; l2mix, store_lines and spill rows still have large single points (median 9.9, 43, 43 %) |
| Scorecard, `kernel` rows (latest batch per row) | HGEMM warm: median 5.7 %, worst −9.0 %, Spearman 0.78, τ 0.64, 4 inversions, R1 10/10, bottleneck 9/10; rotating 9.4 %, −11.5 %, 0.89, τ 0.78, 3. FP8 −14.7 / −15.5 %. FA −3.8 %. MoE +7.4 %. w4gemm 13.0 %, −15.4 %. RMSNorm −17 %, graph −22.6 %; softmax −14.2 %, graph −21.9 %; GEMV 10.7 %, −12.5 %; decode partial +6.1 %, sequence +9.7 %. gemm1024 full 5.8 % (was 3.4), tma full 9.7 % (was 5.0). R1 passes on every row with counters. No fitted parameter on a KTA batch row |
| Blind (r7, first score, engine of round 6) | hgemm 3072×1536×2048 warm 5.6 %, −60.4 %, τ 0.71, Spearman 0.85, 7 inversions, best yes; rotating 7.6 %, −58.4 %, τ 0.78, 0.93, 5, best no (+6.0 %); FA 6.8 %, −11.3 %, τ 0.86, 0.95, 0, best yes. r6 as before: median 9.4 %, worst −30.1 % |
| Probe set | 26 KTA probes: round 6's 21 plus `l2mix`, `dram_burst`, `rot_depth`, `spill`, `store_lines` |
| Engine | Round 6: branch redirect (D780), instruction buffer (D781), instructions beside the tensor pipe (D782), mma stream by operand sets (D783), LDS/STS by lane extent (D784), chip L2 read share under stores (D785), launch steps (D786), sequence program form (D779), lane address before the step (D788). Round 7: stores by sectors behind a buffer (D790), reads and stores with an overlap (D791), stack path (D792), rotating miss by window (D793), loop passes from the counter (D794), what-if scorecard (D789) |
| Rule depth (rows changed in rounds 5–7) | 16 marked in the inventory: 9 structural, 7 empirical (branch floor, beside mma ×2, operand curve, L2 curve, launch steps, DRAM rate by run) |

## Error table: top valid rows

Valid = counts check passed and cache state declared.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| hgemm 3072×1536×2048 256×256×32 s2 t256, rotating / warm | +38.0 / +31.4 % | L1 (stack path) 81 % |
| rmsnorm graph, softmax graph | −22.6 / −21.9 % | stores, chip L2; the arithmetic beside them is off the path (compute removed −10.3 % measured, +4.6 % replayed) |
| hgemm 3072×1536×2048 64×64×32 s4 t64 | +18.3 % | tensor 77 % |
| fp8gemm c128 | −15.5 % | tensor 64 %, store queue 11 %: compensating, question 25 |
| w4gemm c128 warm | −15.4 % | tensor 32 %, shared array 14 % |

Largest what-if misses (direction wrong now): GEMV n_part 2 warm (−11.8, replay +73 %), reduce 16 (−1.3, +67 %; rotating +3.4, −38 %), HGEMM s1 rotating (−1.8, +13.6 %), HGEMM t256 (+1, −9 %), decode split 16 warm (−0.6, −12 %), RMSNorm compute removed (−10.3, +4.6 %), BK 16 (+2 to +7, −5 %), rotation depth (+5 to +13, 0 %).

Worse than round 6: HGEMM warm 1.8 → 5.7 %, rotating 5.6 → 9.4 %, FP8 −7.3 → −15.5 %, w4gemm −12.7 → −15.4 %, gemm1024 full 3.4 → 5.8 %, tma full 5.0 → 9.7 %: all from D790 (ablation in `perfmodel/batches/r7b/entry.md`).

## Open questions (experiment counts per mechanism, LOOP §3)

1. **Sharing between co-resident blocks** — closed at 2 of 2 (D768): fair sharing.
2. **Raster panels** — 1 of 2: on the GPU the panel gains nothing warm (sw8 +0.1 %, sw4 0.0 %) and 3.5 % / 0.8 % rotating (r4); the replay gives sw8 5.1 % warm even with almost no warm DRAM traffic (D772), so the replay's gain is not DRAM. Round 5, no experiment: the replay's difference is `queue (stg)`, 10.4 vs 6.2 % of the path (s3_t128 vs sw8): the 4-byte store path under block sharing, or the panelled prologue. No longer an inversion above the 2 % threshold (the worst now: sw4 vs c64x128x32_s4, measured +2.2 %, replay −3.5 %).
3. **HGEMM 2048³ model fast** — 1 of 2: warm −1 to −13 %, rotating −4 to −12.5 % (r4); the loss is outside the DRAM path (loads from L2, block turnover, stores); the 128-thread 128×128×32 variants are the worst both ways.
4. **Taken-branch cost** — closed at 1 of 2 (D780): next instruction at least 11 cycles after a BRA in arithmetic loops, stall count included; none in loops of stores or barriers (why not: open, no row). The 14 cycles of mma_types are question 22, not the branch.
5. **Launch hidden behind short kernels** — closed at 2 of 2 (D773): not hidden; back-to-back span + 2.74 µs, graph span + 0.60 µs at spans 0.7–151 µs. Below 0.5 µs span the back-to-back gap shrinks to the submission rate (empty chain 1.90 µs, combine 1.91 µs): new question 14.
6. **Softmax and FA after the FCHK fix** — compensating mechanism missing (unchanged): FA −7.5 %; softmax now under question 15.
7. **DRAM writes in rotating rows** — closed at 1 of 2 (D771): reads / 237 + writes / 197 GB/s (dram_rw r4). RMSNorm rotating −41.5 → +16.0 % (now slow: the replay's DRAM queue for a 512-block, one-row-per-block stream; question 15).
8. **Hits on pending misses** — 0 of 2 (unchanged); no test row.
9. **Issue limit exceeded** — 1 of 2 (unchanged). Excluded (D758).
10. **DRAM latency: copy 1560 vs streaming LDG 1100** — 0 of 2 (unchanged).
11. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 %; tiny-kernel store path at large grids; graph of 4096 empty blocks 2.52 µs (replay 0.67): below 3 % of any valid row's path.
12. Decode rows' cache state — closed for the KTA rows (rot_depth, depth rows): rotating at 67 MB is not cold; at 134 MB and more the replay is within 1 %. Upstream tma m128, m32 not re-measured: no KTA harness for warp-specialised kernels.
13. Silent default: `BAR.RED` treated as blocking.
14. **Back-to-back launches of sub-0.5 µs kernels** — closed at 1 of 2 (D786): two steps, combine b2b +2.5 %.
15. **Chip L2 share under stores** — reopened by RMSNorm / softmax graph, closed (D790, D791): 0.45 was a missing mechanism: stores count sectors touched, behind a 470 KB buffer; reads add, less 2200 cycles (empirical). It was not RMSNorm's gap: that is the arithmetic (question 27).
16. **MoE grouped GEMM on the generic front end** — closed (D775, D776): +9.5 / +2.3 %, R1 2/2; no routing through the HGEMM front end needed.
17. **Write-only streams** — no DRAM pacing without reads (dram_rw 0:2 −80 %); no test row. store_lines adds the same path: above about 8 MiB written per launch each MiB costs the DRAM write rate; not in the engine.
18. **Store burst** — closed (D790): a 470 KB buffer, one of the store_lines line's two parameters.
19. **Strided partial-sector stores** — closed (D790): by sectors touched.
20. **GEMV rotating +22 %** — closed (D788, no experiment): an engine defect, a load overwriting its own address register; +10.0 %, DRAM queue still 94 % of the path.
21. **Fused dequantisation** — 2 of 2, closed: mma removed −37.5 % measured, −42 % replayed; dequantisation removed −24.8, −23.3 %: both parts right to within 5 points, no single cheap cause; −15.4 % sits with question 25.
22. **The mma stream's rate** — closed at 2 of 2 (D783): 16 at one operand set; + 0.05 per A set, `unidentified`; a host-compiled NOP-spaced stream on one warp 17.7, on no TileLang row (mma_issue, mma_types stay −5 %); `noload` ablation rows pay for the curve.
23. **Rotating HGEMM at 1024³** — new, 0 of 2: −8 to −11 %, warm within −3 %; 2048³ rotating −4 to −7.6 %, 4096×1024×2048 −2 to −4 %: the cold path is short at small shapes.
24. **Decode rotating and sequence** — closed (D793): +6.1 / +9.7 %, the cache state of the row, not the DRAM path; `dram_burst` rejects a burst above the plateau.

25. **Compensating mechanism missing (D790)** — new, 0 of 2: with stores priced by the probes, GEMM rows are fast by 4 more points (HGEMM warm −9.0, FP8 −15.5, w4gemm −15.4 %); the old 122 B/clk hid something else on their path. Candidates: the third store slope at 8 MiB (C is 8 MiB), store ablation −4.5 % replayed, −6.6 % measured.
26. **Retention under rotation** — new, 0 of 2: `LDG.E.EF` in rot_depth follows the plain curve, r3's `.cs` curve is cold from 12 MiB; decode keeps 18 % at 67 MB, RMSNorm without stores 55 % at 32 MiB, above the curve. Row depth rows 0/4 in direction.
27. **Arithmetic beside chip-bound streams** — new, 0 of 2: RMSNorm compute removed −10.3 % measured, +4.6 % replayed; softmax t256 −18 %, replay 0: first in the next selection.
28. **DRAM share by demand** — new, 0 of 2: the replay gives each SM a fixed share; stages 1 under DRAM load: measured −1.8 %, replay +13.6 %. The overlap question: first in the next selection.
29. **Threads on HGEMM and on GEMV** — new, 0 of 2: t256 replay −9 %, measured +1 %; GEMV n_part 2, reduce 16 warm +67 to +73 %, measured −12 to −1 %.

## Frozen parameters

| Parameter | Value | Status | Why |
| --- | --- | --- | --- |
| `sim.l2_aggregate_B_per_clk_cpasync` (P-2) | 817.7 B/clk | `unidentified` | 2 experiments: 1 stage unchanged at 16 blocks (no latency rise), 2–4 stages −16 to −25 % at 16 blocks with the chip at 0.98 TB/s; flat share, Little's law and a lower chip rate all fail. Candidate on reopening: a share below the chip (TPC pair or GPC) |
| `sim.sts_read_clk_per_128B` | 3.25 | `unidentified` | 2 experiments: STS.128 at 1–6 warps 23.1, 12.0, 8.7, 7.0, 5.6, 4.35; the rule is 4–22 % fast at 2–5 warps; a per-sub-partition limit (5 warps slower than 4) fails |

| `sim.mma_operand_clk_per_set` | 0.05 cycles per A set | `unidentified` | 2 experiments (r6, r6b); cause not shown, cross term rejected on mma_pair (D783) |
| `sim.stg_read_overlap_clk` | 2200 | measured on l2mix, empirical | replaces the share 0.45 (D791); within 10 % over L:S 4:1 to 1:4, grid 512 only |

Values marked `fitted` that the table depends on: `launch_stream_us` (upstream ablation rows only since round 3), `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Still `assumed`: `sim.default_var_latency_clk` 20, `sim.rbar_release_clk` 8, `sim.fchk_fast_path`.

## Fit ledger

| Name | Component | Value | Data | Rows on whose path it sits (share) |
| --- | --- | --- | --- | --- |
| launch_stream_us | launch | 2.6 µs | upstream ablation harness (fitted) | gemm1024 nocompute rows up to 12 %; no KTA batch row since round 3 (replaced by the measured `launch_event_us`, D769) |

No fit was added in rounds 1–6. Round 6 values come from probes; the operand curve and the read share were chosen against probes, with test rows in view: their ablations are in `perfmodel/batches/r6/entry.md`.

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

```
Round 5: batches r5 + r5b — one L2 model for every front end (engine, no probe); probes mma_types (tensor hold per type), row_reduce (a short block's steps), st256 (32-byte stores and loads); fused w4a16 GEMM added.
Question: worst rows MoE +231 / −63 % (DRAM queue 89 %, GEMM reuse ignored), RMSNorm warm graph −50.5 %, softmax graph −35 % (span fast in-kernel); FP8 rows at the HMMA hold with no per-type measurement.
Hypotheses: L2: reuse from addresses + curve / per-front-end rules; mma: one hold / hold ∝ elements / shorter with f16 accumulators; short blocks: engine / block turnover / chip store limit / chip L2 share.
Result: one hold (17.7 in the probe's loop, every type); no turnover (gap +14 cycles, blocks overlap); loads at the chip L2 share (833 B/clk); the store at 1024 B per instruction, chip 468.7 B/clk, which the engine counted as 128 B; L1 fills sectors, not lines. NVFP4 does not compile for sm_121a.
Change: l2.py with sim.l2_replay (D775), ko trip counts (D776), 256-bit store width with stg_chip_B_per_clk_32B 468.7 and stg_tpc_clk_per_instr_32B 8.4 measured, L1 sector hits (D777); mma_types: no parameter.
Scorecard: MoE +231 / −63 → +9.5 / +2.3 %, R1 2/2; RMSNorm warm −34.4 → −15.6 %, graph −50.5 → −27.4 %, rotating +16.0 → +7.3 %; softmax −27.1 → −18.1 %, graph −35.0 → −23.3 %; GEMV rotating +24.5 → +22.2 %; decode rotating +15.2 → +12.8 %. Worse: combine +23.3 → +25.0 %, graph +5.5 → +10.5 %, GEMV warm −10.6 → −12.6 %, FA −6.5 → −7.5 %. HGEMM unchanged: median 3.9 %, Spearman 0.77 → 0.77, inversions 4 → 4; rotating 0.93 / 2. FP8 unchanged: tensor rate was not the cause.
New mechanism: yes — L2 row: one engine rule; store port and chip limit at 32 B per lane X → M (st256); tensor pipe per type M (mma_types); L1 by sector.
Fits: none added; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute).
Next: worst valid rows combine b2b +59.9 % (question 14, submission rate), GEMV rotating +27.7 % (question 20, DRAM queue 95 %), RMSNorm graph −27.4 % (question 15, compute steps); pair sw4 vs c64x128x32_s4; test set can grow by rotating FP8.
```

```
Round 6: batches r6 + r6b — blind rows (other shapes and sizes, first scored before any change); probes branch, mma_issue, lds_narrow, launch_short, mma_pipe; decode split-KV as a sequence.
Question: blind 18 rows median 9.4 %, worst −30.1 %, nearly all fast; combine b2b +59.9 %, GEMV rotating +22 %; the tensor pipe (43 % of the worst HGEMM row) was 16 or 17 depending on the probe.
Hypotheses: branch add / floor / stops the sub-partition; fetch free / 9 per 8; mma per pass / hold / beside-free / beside-costly; LDS by width / by extent; launch max / step; stream rate by warp / by operands / by compiler.
Result: branch floor 11 in arithmetic loops; fetch 1 per line past 64 lines; FFMA 0.9, ALU 0.3, copy 2.3 cycles beside mma; LDS/STS by extent (no test row); launch two steps; the 17 belongs to the host compiler, TileLang streams 16 + 0.05 per A set.
Change: redirect, instruction buffer, beside, operand sets, smem extent, L2 read share 0.45, launch steps (D780–D786); sequence form (D779); GEMV address defect (D788); blind column (D787). A per-warp 17 and a full L2 share were tried and withdrawn.
Scorecard: HGEMM warm median 3.9 → 1.8 %, worst −13.4 → −5.3 %; rotating 7.3 → 5.6 %, −12.5 → −7.6 %; Spearman 0.77 → 0.77, 0.93 → 0.93; inversions 4 → 4, 2 → 2. FA −7.5 → −2.5 %, FP8 −9.0 → −7.3 %, combine +25 → +1.4 %, b2b +59.9 → +2.5 %, GEMV rotating +22.2 → +10.0 %, RMSNorm warm −15.6 → −13.6 %, softmax −18.1 → −16.6 %. Blind rows now 3.9 %, worst −21.5 % (no longer blind). Worse: noload ablation rows (2.4 → 4.6 %, 4.4 → 8.8 %), w4gemm c128 warm −8.2 → −12.7 %, st256 ld_st_g512 +13.4 %: by LOOP §5 not a clean round.
New mechanism: yes — branch, fetch, beside mma, operand sets, LDS extent, L2 read share, launch steps, sequences: inventory rows X/P → M, the operand curve `unidentified`.
Fits: none in the §4 sense; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute).
Next: RMSNorm and softmax graph −25 / −21 %: read share by store shape (a third value needs a new row, question 15 is at its limit); w4gemm (question 21); decode rotating DRAM share (questions 12, 24); rotating 1024³ (question 23).
```

```
Round 7: batches r7 + r7b — first real what-if set (98 one-change kernels) and larger blind groups, predictions committed first; probes l2mix, dram_burst, rot_depth, spill, store_lines.
Question: 28 §1 what-if and τ never tested; RMSNorm / softmax graph −25 / −21 % (share 0.45 suspect); decode rotating and sequence +13 to +17 %; then the largest blind error, −60 %, spills.
Hypotheses: share constant / by phase / by ratio / by sectors; DRAM burst / flat; rotation cold / curve of the window; stack 4 cycles per line / chip rate / capacity; stores by bytes / sectors / lines.
Result: what-if at first score 79/98 right, median magnitude error 26 %, 29/59 within 25 %. Blind 14 variants τ 0.71 / 0.78, FA 8 variants 0.86; best variant 11/13 groups. Stores count sectors with a 470 KB buffer; reads add, less 2200 cycles; phase no effect, no DRAM burst; rotating at 67 MB is not cold; stack at 830 B/clk, 270 above 192 KB per SM.
Change: D790–D794 engine and front ends; D789 scorecard (τ, best variant, what-if, bottleneck); gb10.json 7 new values, all from probes; share 0.45 gone (the constant was a missing mechanism). Depth: 16 rows marked, 9 structural, 7 empirical.
Scorecard: decode +13.3 → +6.1 %, sequence +16.9 → +9.7 %, RMSNorm graph −25.2 → −22.6 %, 256×256 t256 −60 → +31 %, st256 13.8 → 7.7 %. Worse: HGEMM warm 1.8 → 5.7 %, rotating 5.6 → 9.4 %, FP8 −7.3 → −15.5 %, w4gemm −15.4 %, gemm1024 3.4 → 5.8 %, tma 5.0 → 9.7 %; Spearman 0.77 → 0.78, 0.93 → 0.89; inversions 4 → 4, 2 → 3. Not a clean round: D790 stays by probe, question 25.
New mechanism: yes — store buffer and sector limit X → M, stack path at chip scale A → M, rotating retention X → P; dirty write-back above 8 MiB measured, not in the engine.
Fits: none in the §4 sense; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute).
Next: wrong directions first: arithmetic beside chip-bound streams (27), DRAM share by demand at stages 1 (28), thread rows (29); then question 25.
Stopped: no. Position against 28 §1 in the table at the top: time median met, ranking and what-if not.
```
