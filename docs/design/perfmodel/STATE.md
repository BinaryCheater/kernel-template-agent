# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-05 at the end of round 11 (table at commit 183ab16). "Where things stand" below still describes round 9 except the first row; the current position is in the round 11 report and at the top of the error table. Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| Position against 28 §1, round 11 (on the 24 seen design groups; nothing here is a blind score, and "accepted" is the user's word) | Pairs ≥ 3 % right 654/682 (95.9 %; target 95). True best in top 3 22/24 (missed: NVFP4 kernel and graph; group spread under 5 %). First pick's loss worst 8.6 %, 2 groups above 3 %: s2 vs s3 at 3072×1536 and 1536×3072 warm. What-if direction on changes ≥ 3 % 84/90 (93.3 %; target 95), magnitude on changes ≥ 10 % median 17 % over 51. Time: 266 `kernel` rows median 4.5 %, 8 above 20 %, 3 above 25 % (`rmsb_m48_nostore` −45.5, `wi_rms_fp32_warm` −33.5, `wi_hg_t64_warm` −27.0). Bottleneck metric 11 of 19, not repaired yet |
| GPU batches | r1 … r6b as before, all valid; round 7: r7 (160 kernels: one-change variants, larger blind groups; probes l2mix, dram_burst), r7b (probes rot_depth, spill, store_lines), valid: r7 clock 2.483 GHz, drift −0.65 %, canaries within 0.25 %; r7b drift +0.1 %, canaries within 0.5 %; round 8: r8 (66 kernels, 6 probes), valid: clock 2.483 GHz, drift −0.58 %, canaries within 0.65 %; round 9: r9 (test rows, repeats, blind group), r9b, r9c (probes first_reader, first_share; two harness faults, D813), all valid: drift −0.61, 0.0, −0.18 %, canaries within 0.8 % |
| Test set | As round 6 plus: blind groups `hgemm 3072x1536x2048` (14 variants, warm and rotating), `fa 1536x12x96` (8 ran), and the what-if set `wi_*` (98 changes from 8 kinds of base kernel, launch.json `whatif`); round 8: NVFP4 GEMM (`nvfp4gemm2048_*`, 6 rows, R1 0 of 6, so not valid yet; the block-scaled mma stays `unsupported: mma block_scale, ptxas sm_121a`), row parts `rmsb_*`, `smb_*` (10 changes). The 28 §5 table is now covered; round 9: blind group `hgemm 1536x3072x2048` (13 variants, warm and rotating); NVFP4 valid, R1 6 of 6 |
| Position against 28 §1 (user's targets; "accepted" is the user's word) | **Ranking** τ ≥ 0.8: not met on six groups. HGEMM 2048³ warm 0.60 (repeat ceiling 0.91), rotating 0.78 (0.91); 3072×1536×2048 0.71 / 0.85 (0.97, 0.96); 1536×3072×2048 0.67 / 0.85 (0.97, 1.00); FA 0.86, NVFP4 0.87, gemm1024 0.73, tma 0.87. The warm groups are not at the measurement's ceiling. Best variant: 12 of 16 groups (missed: 2048³ rotating +2.5 %, 3072×1536×2048 warm +8.0, rotating +2.2, 1536×3072×2048 warm +8.6). **What-if** direction every time: not met, 88 of 108 on `kernel` rows, 90 reading short kernels from graph rows (D804, user's call); stage rows 4 wrong → 0. Magnitude: median 27 %, 31 of 66 within 25 %. Bottleneck: 6 of 14 bases. **Time** median ≤ 10 %: met, 4.6 % on 123 valid `kernel` rows; none above 25 %: met (worst −21.9 %, 2 above 20 %, both 256×256×32 t256). NVFP4 now valid, −9.3 %, worst −11.4 %. **Probe ladder**: no per-level budget set; first_reader 3.1 %, worst +7.3 %; first_share 8.9 %, +26.2 %; copy_dram 4.2 %, +22.4 %: warm 48-block rows (question 32) |
| Scorecard, `kernel` rows (latest batch per row) | HGEMM warm: median 2.8 %, worst −6.3 %, Spearman 0.78, τ 0.60, 4 inversions, R1 10/10; rotating 6.1 %, −8.1 %, 0.93, τ 0.78, 4. FP8 −6.0 / −8.5 %. FA −3.8 %. MoE +7.7 %. w4gemm 11.5 %, −13.3 %. RMSNorm −12.7 %, graph −12.8 %; softmax −14.2 %, graph −21.9 %; GEMV 9.7 %, −11.9 %; decode −4.1 %. gemm1024 full 3.2 %, tma full 6.6 %. R1 passes on every row with counters. No fitted parameter on a KTA batch row |
| τ split (2048³, one replay per engine and measurement) | Warm: 0.60 on r1, r2, r3, r4, r9, 0.56 on r8, under the round 7, 8 and 9 engines alike: the drop was r8's measurement, engine 0. Rotating: round 7 engine 0.73 on r4, r8, r9; round 8 engine 0.73, 0.64, 0.64: an engine loss of 0.09 on two of three measurements; round 9 engine 0.78 on all three |
| Blind (r9, first score, predictions before the run) | hgemm 1536×3072×2048 warm 3.3 %, −11.1 %, τ 0.69, Spearman 0.82, 6 inversions, best no (+8.6 %); rotating 7.2 %, −11.8 %, τ 0.69, 0.87, 9, best yes. Now 3.0 %, τ 0.67 / 5.2 %, τ 0.85 |
| Blind (r7, first score, engine of round 6) | hgemm 3072×1536×2048 warm 5.6 %, −60.4 %, τ 0.71, Spearman 0.85, 7 inversions, best yes; rotating 7.6 %, −58.4 %, τ 0.78, 0.93, 5, best no (+6.0 %); FA 6.8 %, −11.3 %, τ 0.86, 0.95, 0, best yes. r6 as before: median 9.4 %, worst −30.1 % |
| Probe set | 34 KTA probes: round 8's 32 plus `first_reader`, `first_share` |
| Engine | Round 6: branch redirect (D780), instruction buffer (D781), instructions beside the tensor pipe (D782), mma stream by operand sets (D783), LDS/STS by lane extent (D784), chip L2 read share under stores (D785), launch steps (D786), sequence program form (D779), lane address before the step (D788). Round 7: stores by sectors behind a buffer (D790), reads and stores with an overlap (D791), stack path (D792), rotating miss by window (D793), loop passes from the counter (D794), what-if scorecard (D789). Round 8: warps across blocks (D797), DRAM by demand (D798), copy latency (D799), stack path (D800), partial sectors (D801), dirty limit (D802), L2 miss per buffer (D803). Round 9: repeat ceiling (D806), misses by slice and wave (D807), followers wait on the fill (D808), miss-status entries (D809), copy latency plus burst (D810), lanes split at branches (D811) |
| Rule depth (rows changed in rounds 5–9) | 26 marked in the inventory: 15 structural, 11 empirical. Round 8 added 3 empirical on design variables, each a candidate for a structural rule: stack rates (400, 133 B/clk), 1.6 cycles per contiguous store, 244 B/clk for partly filled sectors. Round 9 added 5, all structural, no empirical; one leans on a frozen parameter (D810). Earlier: branch floor, beside mma ×2, operand curve, L2 curve, launch steps, DRAM rate by run |

## Error table: top valid rows

Valid = counts check passed and cache state declared.

| Row | Error | Attribution (top) |
| --- | ---: | --- |
| hgemm 3072×1536×2048 256×256×32 s2 t256, warm / rotating | −21.9 / −20.1 % | stack path 70 % |
| softmax graph | −21.9 % | launch, drain (question 27) |
| softmax 512×4096 warm | −14.2 % | launch 34 %, drain |
| w4gemm c128 warm | −13.3 % | tensor, shared array |

Reversed pairs above the measured spread: warm 2048³ 4, 3072×1536×2048 7, 1536×3072×2048 7; rotating 0, 2, 3. They share two families at all three shapes: `c128x256x32_s3_t256` measures 3–6 % above the K 64 rows, replay level or below (the path differs by `queue (mma)` against stall counts, 40–55 µs each way); `c128x128x32_s2` against `s3` at 3 resident blocks (measured −8 %, replay 0 to +5 %, `queue (chip L2 share)` −16 µs: question 32). No round 8 change moved a warm pair: warm predictions keep τ 0.60 under all three engines.

Wrong what-ifs (20, `kernel` rows): rotation depth 4 (question 26, at its limit); short rows the event step hides, graph rows right, 3 (rmsb m192); threads t256 and RMSNorm t256 rotating 3 (question 29); bk16 2; panels rotating 2 (measured −2.5 %, replay +1 %); GEMV mk16, r16 rotating 2; dec bn32, dec split16, FA noexp, rmsb m48 nostore 1 each. What they share: nothing beyond the groups named; the stage rows, which shared the miss spreading, are all right now.

Worse than round 8, by ablation: `dram_burst_past_l2` 3072×1536×2048 warm τ 0.76 → 0.71, `miss_by_wave` 0.74 → 0.71; `fill_wait` 2048³ rotating best variant yes → +2.5 %, what-if 89 → 88; `l1_pending_fill` NVFP4 τ 1.00 → 0.87 (all within spread). All stay by probe (LOOP §5).

## Open questions (experiment counts per mechanism, LOOP §3)

1. **Sharing between co-resident blocks** — closed at 2 of 2 (D768): fair sharing.
2. **Raster panels** — 1 of 2: on the GPU the panel gains nothing warm (sw8 +0.1 %, sw4 0.0 %) and 3.5 % / 0.8 % rotating (r4); the replay gives sw8 5.1 % warm even with almost no warm DRAM traffic (D772), so the replay's gain is not DRAM. Round 5, no experiment: the replay's difference is `queue (stg)`, 10.4 vs 6.2 % of the path (s3_t128 vs sw8): the 4-byte store path under block sharing, or the panelled prologue. No longer an inversion above the 2 % threshold (the worst now: sw4 vs c64x128x32_s4, measured +2.2 %, replay −3.5 %).
3. **HGEMM 2048³ model fast** — 1 of 2: warm −1 to −13 %, rotating −4 to −12.5 % (r4); the loss is outside the DRAM path (loads from L2, block turnover, stores); the 128-thread 128×128×32 variants are the worst both ways.
4. **Taken-branch cost** — closed at 1 of 2 (D780): next instruction at least 11 cycles after a BRA in arithmetic loops, stall count included; none in loops of stores or barriers (why not: open, no row). The 14 cycles of mma_types are question 22, not the branch.
5. **Launch hidden behind short kernels** — closed at 2 of 2 (D773): not hidden; back-to-back span + 2.74 µs, graph span + 0.60 µs at spans 0.7–151 µs. Below 0.5 µs span the back-to-back gap shrinks to the submission rate (empty chain 1.90 µs, combine 1.91 µs): new question 14.
6. **Softmax and FA after the FCHK fix** — compensating mechanism missing (unchanged): FA −7.5 %; softmax now under question 15.
7. **DRAM writes in rotating rows** — closed at 1 of 2 (D771): reads / 237 + writes / 197 GB/s (dram_rw r4). RMSNorm rotating −41.5 → +16.0 % (now slow: the replay's DRAM queue for a 512-block, one-row-per-block stream; question 15).
8. **Hits on pending misses** — closed at 1 of 2 (D809): an L1 hit on a line on its way waits for the first reader's fill: first_reader, a hinge within 5 %.
9. **Issue limit exceeded** — 1 of 2 (unchanged). Excluded (D758).
10. **DRAM latency: copy 1560 vs streaming LDG 1100** — closed at 1 of 2 (D799): 1420 for streaming copies.
11. BAR.SYNC at 32 warps +19 %; L1 chase −4.3 %; dram_rotate at 128 threads −13 %; tiny-kernel store path at large grids; graph of 4096 empty blocks 2.52 µs (replay 0.67): below 3 % of any valid row's path.
12. Decode rows' cache state — closed for the KTA rows (rot_depth, depth rows): rotating at 67 MB is not cold; at 134 MB and more the replay is within 1 %. Upstream tma m128, m32 not re-measured: no KTA harness for warp-specialised kernels.
13. Silent default: `BAR.RED` treated as blocking.
14. **Back-to-back launches of sub-0.5 µs kernels** — closed at 1 of 2 (D786): two steps, combine b2b +2.5 %.
15. **Chip L2 share under stores** — reopened by RMSNorm / softmax graph, closed (D790, D791): 0.45 was a missing mechanism: stores count sectors touched, behind a 470 KB buffer; reads add, less 2200 cycles (empirical). It was not RMSNorm's gap: that is the arithmetic (question 27).
16. **MoE grouped GEMM on the generic front end** — closed (D775, D776): +9.5 / +2.3 %, R1 2/2; no routing through the HGEMM front end needed.
17. **Write-only streams** — closed (D832): stores past the limit take the DRAM with reads; dram_rw 0:2 unchanged, no test row.
18. **Store burst** — closed (D790): a 470 KB buffer, one of the store_lines line's two parameters.
19. **Strided partial-sector stores** — closed (D790): by sectors touched.
20. **GEMV rotating +22 %** — closed (D788, no experiment): an engine defect, a load overwriting its own address register; +10.0 %, DRAM queue still 94 % of the path.
21. **Fused dequantisation** — 2 of 2, closed: mma removed −37.5 % measured, −42 % replayed; dequantisation removed −24.8, −23.3 %: both parts right to within 5 points, no single cheap cause; −15.4 % sits with question 25.
22. **The mma stream's rate** — closed at 2 of 2 (D783): 16 at one operand set; + 0.05 per A set, `unidentified`; a host-compiled NOP-spaced stream on one warp 17.7, on no TileLang row (mma_issue, mma_types stay −5 %); `noload` ablation rows pay for the curve.
23. **Rotating HGEMM at 1024³** — new, 0 of 2: −8 to −11 %, warm within −3 %; 2048³ rotating −4 to −7.6 %, 4096×1024×2048 −2 to −4 %: the cold path is short at small shapes.
24. **Decode rotating and sequence** — closed (D793): +6.1 / +9.7 %, the cache state of the row, not the DRAM path; `dram_burst` rejects a burst above the plateau.

25. **Stores by lanes' spread** — closed (D801): by whether lanes share a sector they do not fill, at any distance; the 4 KiB threshold is gone. HGEMM rotating Spearman 0.85 → 0.81 with all round 8 changes: not restored.
26. **Retention under rotation** — 1 of 2 (r11 `wr_*_rot*`: plain loads alone and with stores within 10 % of the curve at the window: the simple case holds; decode and RMSNorm's extra stays unexplained): `LDG.E.EF` in rot_depth follows the plain curve, r3's `.cs` curve is cold from 12 MiB; decode keeps 18 % at 67 MB, RMSNorm without stores 55 % at 32 MiB, above the curve. Row depth rows 0/4 in direction.
27. **Arithmetic beside chip-bound streams** — closed for RMSNorm at 2 of 2 (D809): W's lines were instant L1 hits; graph −22.6 → −12.8 %. Softmax graph −21.9 % unchanged: its loads are not several readers of one line; new question 34.
28. **DRAM share by demand** — closed (D798): the rate limits what has been read so far. HGEMM s1 rotating still +12 %, measured +1 %: 1100 gives +0.3 %, the probe gives 1420: new question 30.
29. **Threads on HGEMM and on GEMV** — GEMV closed (D797). HGEMM t256, 1 of 2: mma_ldsm rejects the pipe (overlap real, replay within 5 %), −9 %, measured 0: not found. warp_map T 96 n 2 runs faster than four sub-partitions allow: no row.
30. **Misses inside a copy step** — closed (D807, D808): misses by slice and wave, followers wait; stage what-ifs 4 wrong → 0, s1 rotating right. A miss brings its 32-byte sector (`pieces`).
31. **Intra-warp divergence** — closed at 0 experiments (D811): both arms from lane values, R1 6/6, −33 → −9 %. Memory in an arm at 32 lanes: no row shows it.
32. **Chip L2 under arithmetic for copies** — 2 of 2, not identified; a row points at it again: the first pick's loss of 8 % (28 §1), which needs a new experiment on the compute side, not on the L2 rate (D835): r10 `wq_m3072`: s2 211.1, s3 228.5 µs; without compute 147.9, 149.0; without stores 192.7, 209.2: the gap goes with compute, as question 36. Earlier: lean one-stage warm copies at 48 blocks run about 1070 B/clk, replay 21.2 k, measured 17.5 k; sharing a tile does not help (D 1–48 flat). 1070 in the table: τ 0.60 → 0.38, rejected (D812); the tile rate and the lean loop's rate differ.
33. **Event timing resolution** — D804, user's call. Rotation depth rows (question 26) stay wrong: decode and RMSNorm reuse nothing inside a launch (checked: reads = footprint), so promotion in the L2 is not what keeps their 5–18 %.
34. **Softmax graph −22 %** — new, 0 of 2: launch and drain on the path; 1 of 2 on question 27's mechanism ruled it out.
35. **Rotating replays below warm** — new, 0 of 2: 128×128×32 s3 and 3072 s2: replay rotating 7 % below warm, measured above. The trace shows tensor busy 3.02 → 3.23 of 4 and the store queue 19 → 10 µs; probably DRAM delays skew the two blocks on an SM (not tested). No measurement shows that gain.
36. **`c128x256x32` t256 above the K 64 rows** — 1 of 2 (r10 `wq_*`): the gap goes with compute. Measured K 32 s3 238.3, K 64 s2 223.7 µs; without stores 213.8, 202.1; without compute 120.3, 128.7: reversed. The compute part is 118 against 95 µs, replay 112 against 100. Counters: K 32 has mio throttle ×2.1, short scoreboard ×2.4, FMA active ×3.1. Candidate, not scored: the IADD.64 ALU hold (D820).
37. **Chip L2 rate by pattern** — closed at 2 of 2 (D835): contiguous copies 1850–2100 B/clk however arranged, spread lanes 25 sectors/clk, LDG 14 lines/clk; TileLang's tiles keep 817, what separates them is not found. Before: load_sat (r10c) moves about 1900 B/clk of contiguous 512-byte copies at G 144 Z 0 by the launch's bytes over its span, against the frozen 817 for TileLang tiles; copy_depth (r10) lands 42 B/clk on one SM against the port's 30.7. By lines per clock: rejected on l2share_copy (D817). Open.
38. **Copy issue on one warp** — new, 0 of 2: copy_depth about 45 cycles per copy per warp at 1 and 4 warps, replay 30. No queue depth (D816). Candidate, not scored: IADD.64 in the address code.
39. **Global atomics on shared bins** — closed (D834): 1.44 per clock per 4 KiB region, 12 chip-wide, merge gain by bytes; `hand_hist` −9.6 %. `ATOMG` with return not measured.
40. **Store merge against store_spread** — new, 0 of 2: the merge (D833, probe-backed, stays) took `store_spread L2_R4096_S8` from −4 to −59 %; `rw_*` G 768 still +68 to +112 %, `store_lines S4_W4_D32` +546 %, `stress_09` +25 %. The two families ask opposite things of the chip queue: a compensating mechanism, not found.
41. **Rows worse since the round 9 table, not attributed** — new: `l2mix` median 9.9 → 14.8 %, `arith_beside` 3.4 → 9.0 %, `st256` 8.2 → 11.3 %, decode 2.8 → 6.3 %. The round 11 store and DRAM switches do not move the three rows ablated, so round 10's or `b0a72de`'s changes are the candidates; there was no table between. Not checked further.
42. **Write step beside re-read lines** — new, 0 of 2: 1.5 × `stg_dirty_bytes` is a bound between HGEMM's free 9.4 MB and the stencil's 16 MiB that pays; nothing measured between.

## Frozen parameters

| Parameter | Value | Status | Why |
| --- | --- | --- | --- |
| `sim.l2_aggregate_B_per_clk_cpasync` (P-2) | 817.7 B/clk | `unidentified` | 2 experiments: 1 stage unchanged at 16 blocks (no latency rise), 2–4 stages −16 to −25 % at 16 blocks with the chip at 0.98 TB/s; flat share, Little's law and a lower chip rate all fail. Candidate on reopening: a share below the chip (TPC pair or GPC) |
| `sim.sts_read_clk_per_128B` | 3.25 | `unidentified` | 2 experiments: STS.128 at 1–6 warps 23.1, 12.0, 8.7, 7.0, 5.6, 4.35; the rule is 4–22 % fast at 2–5 warps; a per-sub-partition limit (5 warps slower than 4) fails |

| `sim.mma_operand_clk_per_set` | 0.05 cycles per A set | `unidentified` | 2 experiments (r6, r6b); cause not shown, cross term rejected on mma_pair (D783) |
| `sim.stg_read_overlap_clk` | 2200 | measured on l2mix, empirical | replaces the share 0.45 (D791); within 10 % over L:S 4:1 to 1:4, grid 512 only |
| `sim.l2_contig_B_per_clk` | 1900 | measured, l2_cap, l2_local, empirical | D835; tile copies do not use it |
| `sim.stg_merge_run_clk` | 32 | chosen between wr_r1w4 and HGEMM | D833 |
| `sim.stg_chip_wide_B_per_clk` | 244 | measured, store_spread 237–286, empirical | D801 |
| `sim.l2_aggregate_B_per_clk_cpasync` in the burst term | 817.7 B/clk | frozen, read by D810 | slope fits 1–16 tiles |

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
Result: what-if at first score 79/98 right, median magnitude error 26 %, 29/59 within 25 %. Blind 14 variants τ 0.71 / 0.78, FA 8 variants 0.86; best variant 11/13 groups. Stores count sectors with a 470 KB buffer; reads add, less 2200 cycles; phase no effect, no DRAM burst; rotating at 67 MB is not cold; stack at 830 B/clk, 270 above 192 KB per SM. RMSNorm's gap is its arithmetic, not the share.
Change: D790–D794 engine and front ends; D789 scorecard (τ, best variant, what-if, bottleneck); gb10.json 9 new values, all from probes; share 0.45 gone. Depth: 16 rows marked, 9 structural, 7 empirical.
Scorecard: decode +13.3 → +6.1 %, sequence +16.9 → +9.7 %, RMSNorm graph −25.2 → −22.6 %, 256×256 t256 −60 → +32 %, st256 13.8 → 9.0 %, kernel rows median 5.8 %. Worse: HGEMM warm 1.8 → 3.1 %, rotating 5.6 → 6.6 %; Spearman 0.77 → 0.77, 0.93 → 0.85; inversions 4 → 4, 2 → 4; tma 5.0 → 7.5 %. Not a clean round: D790 stays by probe, question 25.
New mechanism: yes — store buffer and sector limit X → M, stack path at chip scale A → M, rotating retention X → P; dirty write-back above 8 MiB measured, not in the engine.
Fits: none in the §4 sense; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute).
Next: wrong directions first: arithmetic beside chip-bound streams (27), DRAM share by demand at stages 1 (28), thread rows (29); then question 25.
Position: time median met; none above 25 % fails on 2 rows; ranking and what-if not met (table at the top). Upstream tma m128 / m32 not re-measured.
```

```
Round 8: batch r8 — warp placement, cold copies under arithmetic, lanes' spread, arithmetic beside limits, two warps per sub-partition, stack loads vs stores; NVFP4 GEMM; re-measured HGEMM, RMSNorm, softmax, GEMV.
Question: 21 of 98 what-ifs wrong; 256×256×32 t256 +32 / +39 %; τ 0.60 / 0.73.
Hypotheses: warps local / across blocks; DRAM share fixed / by demand, 1560 / 1100; extent / partial sector / rows; arithmetic hides / adds; overlap / none; stack one rate / stores at chip rate / capacity.
Result: across blocks; by demand, latency 1420 (neither); partial sector; hides (the replay was right); overlap (no gain for t256); stores at chip rate with LRU capacity. Round 7's stack rates came from a probe with its loads folded. Short kernels' event times step by 0.96 µs: part of the wrong what-ifs was that.
Change: D797–D803 engine and front ends; 8 new or changed values, all from probes; `stg_wide_extent_B` gone; store-only kernels track lanes. A cumulative chip L2 rule was tried and withdrawn. D804 graph rows beside kernel rows.
Scorecard: t256 +32 / +39 → −20 / −16 %; rows above 25 %: 2 → 0; median 5.8 → 4.6 %; GEMV n_part 2 +73 → +1 %, spill median 43 → 7 %; direction 77 → 79 of the first 98, 85/108. Worse: HGEMM τ 0.60 → 0.56, 0.73 → 0.64 (re-measured rows), Spearman 0.77 → 0.73, 0.85 → 0.81, FP8 −8.5 → −10.3 %: not a clean round, all stay by probe.
New mechanism: yes — warp placement P → M, DRAM by demand A → M, stack path empirical, partial sectors empirical; dirty limit X → M; divergence stays X, now with a row.
Fits: none in the §4 sense; largest fit share on a path unchanged (launch 12 %, upstream gemm1024 nocompute).
Next: question 27 (loads with arithmetic), 30 (misses in a copy step), then t256; question 31 before NVFP4 counts.
Position: time met on valid rows, first time none above 25 %; ranking and what-if not met; τ ≥ 0.8 warm is at the measurement's ceiling; NVFP4 not valid (table at the top).
```

```
Round 9: batches r9 + r9b + r9c — HGEMM re-measured, blind group 1536×3072×2048, NVFP4; probes first_reader, first_share (a read of a line on its way; shared tiles at the chip's scale).
Question: HGEMM τ fell two rounds (0.60 → 0.56 warm, 0.73 → 0.64 rotating); round 8 called warm "at the ceiling". Split: warm is r8's measurement alone (engine 0, 0.60 on five of six measurements); rotating is the round 8 engine, −0.09 on two of three measurements. Ceiling 0.91 warm, 0.91 rotating: the claim does not hold. Rotating over warm follows the waves: +17 to +22 µs at 6, +4 to +6 at 3.
Hypotheses: H-pending (followers wait on the fill) / H-nowait or H-own (instant hit, own read) / H-first (only one reader pays) / H-even (spread over readers and waves) / H-line against H-sector.
Result: H-pending: cold 1086 … 82 by delay, warm 360 … 82, replay within 5 %; later waves 17.0 k, warm 17.1 k, even 33–36 k; 48 blocks on a tile +742 per step, one reader paying +0; a miss brings its sector; copy DRAM latency 1075 unloaded, 1420 was 16 tiles of burst. Valid data needed three runs (harness faults, D813).
Change: misses by slice and wave, fill waits, miss-status per line and sector, copy latency 1075 + burst / 817 B/clk, lanes split at branches (both arms, no schema change); table: repeat ceiling. D806–D813.
Scorecard: 2048³ rotating τ 0.64 → 0.78, warm 0.56 → 0.60 (measurement); 3072 0.75 / 0.78 → 0.71 / 0.85; RMSNorm graph −22.6 → −12.8 %; NVFP4 −36 → −11 %, R1 0 → 6 of 6; what-if 85 → 88 of 108; median 4.6 %, worst −21.9 %, none above 25 %. First score blind 1536×3072×2048: τ 0.69 / 0.69, 3.3 / 7.2 %, best no / yes. Not met: τ on six groups, what-if, bottleneck 6 of 14. By LOOP §5 not a clean round: four rows worse by ablation, all stay by probe.
New mechanism: yes — inventory: divergence X → P, miss-status M, cp.async spread X → P, DRAM queue by wave, DRAM latency unloaded + burst; 5 rules, all structural.
Fits: none added; largest share unchanged. Rule depth: round 8's 3 empirical rules on design variables stay candidates, none replaced.
Next: reversed pairs above the spread: `c128x256x32` t256 against the K 64 rows (question 36), with the s2 / s3 family (question 32's second experiment); then softmax graph (34).
Round 10: batches r10 (test rows, wq_*, hand_*, stress_*, load_curve, copy_depth), r10b, r10c (harness repairs: load_sat, dram_spread), r10w (14 library operators, read from WP28): front end, contention, breadth.
Question: warm HGEMM τ 0.60 under three engines; what decides it, and is contention why what-if and bottleneck do not move. No test parameter taken from any of it.
Hypotheses: A template path decides warm rank / B the engine does. Contention: H-fixed / H-queue (L2), H-unbounded / H-depth (copies), H-fixed / H-rising (DRAM). Q36: compute / load / store.
Result: B. On r9's measurements (tables base4 → gen6, engine of this round's front end) warm τ 0.60 → 0.64 at 2048³, 0.74 → 0.76, 0.67 → 0.69, rows within 0.3 % bar a few. Rotating median 6.1 → 3.0 %, 1024³ 11.2 → 3.4 %, 3072 rotating 4.3 → 3.3 %, 1536 rotating 5.2 → 8.7 %, worst +11.4; FP8 6.0 → 4.5 %; R1 10/10, 14/14, 13/13 warm on the first generic table (gen0). L2 latency flat: 380 at G 1, 396 G 48, 553 at G 144 Z 0 (r10, r10c), H-queue rejected, but load stayed under saturation. Copies: no depth, H-depth rejected; landed = issue + 325 at every K. DRAM p50 about 940 to the second-highest load, 1205 with p90 2411, p99 6391 at the highest (r10c): flat, then a tail; H-rising rejected below saturation; the rates printed there are not trustworthy (L2 hits). So the supervisor's contention hypothesis is not supported as a latency queue. Q36 gap goes with compute. Per-unit, table l0 (a rejected engine, busy ratios unaffected in kind): tensor 1.06, FMA 0.80, ALU 0.42, fill 1.80, LSU 1.02. Per-pipe counters then showed why ALU is low: IADD and IADD.64 had no pipe.
Change: one front end: `generic_kernel` for HGEMM, FP8, upstream GEMMs; `simulate_kernel`, the FP8 byte-equivalent fixture removed; ws.py keeps its own (TMA rows). Misses by wave from the L2 replay, sectors of 32 B, reader share of DRAM time. Rejected: L2 share by demand, chip rate by lines (D817). Not scored: opcode classes (D820).
Scorecard: table `off` (commit 254e3ca, `--sets batches --fast`, r10 measurements, no opcode change, no attribution): 2048³ warm 3.4 %, τ 0.73, best yes, 1 inversion; rotating 2.6 %, τ 0.73, best yes, 1; 3072 warm 3.8 %, −22.6, τ 0.76, 5; rotating 3.3 %, τ 0.87, 4; 1536 warm 3.0 %, τ 0.69, 5; rotating 8.7 %, +11.4, τ 0.90, 2; what-if 103/118, magnitude 28 %, 34/79. The rise of 2048³ warm to 0.73 is r10's measurement of those rows plus the engine, not separated. Bottleneck, the 123-row time and the full error table: not recomputed; `error-table.*` still round 9.
Breadth, blind (first score, predictions committed first; hand, stress r10; library r10w): hand 10 rows median 23.1 %, worst −96.8 (hist), mandel +58, triad −37, stencil −38, τ 0.69. Stress 16 rows 12.3 %, +443.5 (stress_09), stress_11 −87, τ 0.66. Library 14 of 24 replayed, 94.5 %, −99.6: run-time loop counts without arguments, R1 0 of 7; 9 `unsupported: control bits`, E05 `unsupported: polls a run-time value`. Library rows measure the front end's inputs, not timing.
New mechanism: no scored one. Opcode ranking (table l0, its default flag too broad): class `other`: IADD.64 4.21 %, IADD 3.76 %, MOV 1.52 %, UIADD3 1.34 % of a mean row. Convergence section written, never rendered; hand counts in `table.CLASSES` unchecked.
Fits: none added. Not done: step 4 stall families (no attribution in the fast tables, the code's share column is unverified), step 6 render, step 7 stress analysis and probes above 3 %, step 8 cubin patch, inventory rows, tests after the last commits, gate.
Next: one table on HEAD with attribution, ablate b0a72de by reverting it; if it stays, Q36, Q38 again; then writes of warm kernels above the L2 (triad reads / 237 + writes / 197 GB/s = 880 µs, measured 894, untested in the engine).
Stopped: step 5, the user stopped the round, no full table at the end.
```

```
Round 11: batches r11 (l2_cap, wr_*, atom_*, rw_*, 5 blind hand kernels), r11b (l2_local, wr_r1w1 E 5–12): throughput caps, write path past the L2, global atomics, store merge, unknown load addresses; testing tools first.
Question: on the worst hand and stress rows instruction counts equal the hardware: mechanisms were missing. hand_hist −96.8 %, stress_09 +443 %, hand_stencil −38, hand_triad −37, load_sat +765. `b0a72de` unscored; no table on HEAD.
Hypotheses: L2 H-port / 817 / bytes-1900 / lines / sectors; arrangement H-scatter / page / share / none. Writes H-now / add / max / reads-only. Atomics H-now / bin / line / slice. Stores H-now / distinct / lines. Unknown addresses: found before the batch, no GPU.
Result: contiguous copies 1850–2100 B/clk whatever the arrangement, spread lanes 25 sectors/clk, LDG 14 lines/clk: no single byte or line rate; H-none on the arrangement. Writes: H-add holds within 15 % once F + W is well past the L2: reads and write-back take turns on the DRAM, and a launch that writes past the limit pays from its first byte (W 6 MiB 12.9 µs, 7 MiB 38.6). Atomics: none of the three alone: rate by 4 KiB region up to 12 per clock chip-wide, plus merging on few words (one word 29 per clock). Stores: H-distinct in part: L2 sectors / L1 sectors 1.2 at G 48, 4 at G 768: they merge while queued; H-lines: a contiguous run costs 1.6 per instruction, misaligned too. Stencil's unknown addresses were per-lane predicates, not the cause of its error: 16 MiB written.
Change: lane predicates (D831); writes share the DRAM, dirty limit full from the start, written bytes in a single-use read's window (D832); store merge in the queue, pair by lines or sectors (D833); atomics by region (D834); chip L2 by pattern, question 37 closed, tiles keep 817 (D835); `b0a72de` stays (D830), ALU busy ratio 0.42 → 1.01. New parameters 13, all from r11, r11b, marked in gb10.json; 2 chosen, not measured: `stg_merge_run_clk` 32, the 1.1 single-use threshold, and the 1.5 bound. Testing tools D829; scorecard has 28 §1; LOOP §5, §8 updated.
Scorecard (commit 183ab16, against the round 9 table, r10 and r11 changes together): hand_hist −96.8 → −9.6 %, stress_09 +443 → +25.5, hand_triad −37 → +3.1, hand_stencil −38 → −15.6, load_sat +765 → +28.5 worst. 2048³ warm 2.8 → 3.2 %, τ 0.60 → 0.73, 1 inversion; rotating 6.1 → 2.1 %, τ 0.78 → 0.73; 3072 warm τ 0.71 → 0.80, 3 inversions; rotating 0.85 → 0.89; 1536 warm 0.67 → 0.72; rotating 0.85 → 0.87, median 5.2 → 8.4 %. Hand set median 23.1 → 9.6 % (15 rows), τ 0.90; stress 12.3 → 10.3 %. Blind, r11 (predictions committed, engine before these changes): hand_gups −71, hand_copy −35, hand_scale −39, hand_smear +167, hand_stencil512 −2 %. Now, no longer blind: −9, +21, +11, −1.5, −1.5 %. Those five were not used for parameters, but the rules were written with them in view: no blind score for the new rules exists. Against 28 §1: pairs ≥ 3 % 654/682 (95.9 %), best in top 3 22/24, first pick's loss worst 8.6 % (2 groups above 3 %), what-if direction ≥ 3 % 84/90 (93.3 %), magnitude ≥ 10 % median 17 % over 51; all on seen rows. Worse, by LOOP §5 a record: store_spread R4096 −4 → −59 % (merge, stays, question 40); l2mix, arith_beside, st256, decode not attributed (question 41).
New mechanism: yes — inventory: atomics by region X → M; store merge new; DRAM shared by writes P → M; chip L2 by pattern; 6 rows; 4 structural, 2 empirical (L2 pattern rates, merge gain).
Fits: none added under LOOP §4; largest fit share unchanged.
Testing cost: 16 local runs 11782 CPU s, 39 min wall at 6 processes, nice 19; one --full on spark 8461 CPU s, 11.6 min at 18. One local run cost 4685 CPU s by a tool fault (mechanisms-only pass on a selection, fixed) and a wide --focus; several later focus runs were wide too. The sentinels alone are 431 CPU s, not the 190 planned: rows with attribution cost twice.
Next: first pick's loss at 3072×1536 and 1536×3072 warm (question 32, on the compute side); what-if direction 6 wrong (gemv r16, np2, bk16, dec bn32, FA noexp); question 40; `rmsb_m48_nostore` −45.5 %.
```
