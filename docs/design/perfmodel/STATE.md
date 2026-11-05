# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-05. The current position is in the rows below, the last round report and at the top of the error table; older positions, closed questions and reports 1 to 15 are in [STATE-archive.md](STATE-archive.md). Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| **Blind position against 28 §1, round 17 (the number that counts: first scores of 95 new kernel rows, r17, predictions committed before the run, engine of round 16, the recipe of D874)** | Pairs ≥ 3 % right 157/199 (78.9 %; target 95): **not met**; HGEMM 64/73 (87.7 %), the others 93/126 (73.8 %); real pairs 151/184, off HGEMM 87/111 (78.4 %), without warm GEMV 67/91 (73.6 %). True best in top 3 9/11: **not met**. First pick's loss worst 33.0 %, 5 of 11 above 3 %: **not met**. What-if direction ≥ 3 % 37/47 (78.7 %): **not met**. Magnitude ≥ 10 % median 38 % over 31: **not met**. Time median 11.1 %, ≥ 30 µs 7.6 %: **not met** on all rows; 15 above 25 %, worst +114.5 %: **not met**. Of 54 rows off by more than 10 %: 0 defects, 34 known rules wrong on the row, 20 no rule covered |
| **Blind position against 28 §1, round 16 (the number that counts: first scores of 103 new kernel rows, r16, predictions committed before the run, engine of round 15, the recipe of D874)** | Pairs ≥ 3 % right 192/238 (80.7 %; target 95): **not met**; HGEMM 61/70 (87.1 %), the others 131/168 (78.0 %); pairs whose order holds in every pass 187/221 (84.6 %), off HGEMM 126/152 (82.9 %). True best in top 3 8/11: **not met**. First pick's loss worst 17.3 %, 6 of 11 above 3 %: **not met**. What-if direction ≥ 3 % 43/54 (79.6 %): **not met**. Magnitude ≥ 10 % median 27 % over 25: **not met**. Time median 11.2 %, ≥ 30 µs 6.2 %: **not met** on all rows; 13 above 25 %, worst +164 %: **not met**. Of 56 rows off by more than 10 %: 0 defects, 35 known rules wrong on the row, 21 no rule covered |

Real pairs off HGEMM, blind, rounds 13 to 17: 74, 71, 72, 83, 78 %; without warm GEMV 75, 73, 73, 79, 74 %.

## Rejected hypotheses, an index

One line each: what was tried and failed, so nobody tries it again without a new row. Details in `docs/decisions.md` and `STATE-archive.md`.

| Topic | Rejected | Where |
| --- | --- | --- |
| Sharing between blocks | Oldest-first as the sole rule; a per-warp stream rate of 17 | D768, round 6 |
| Chip L2 | Full read share under stores; cumulative chip rule; queue depth (H-queue, H-depth); rate rising below saturation | rounds 6, 7, 10, D812, D816, D817 |
| Two blocks on an SM | Fast mode hidden by averaging or the slowest SM (H-mean, H-max); lockstep (H-none) | D849 |
| Copies | Issue queue on an SM (fixed by rate 10 clk per copy); copies from cold slower | D869, r16 |
| Thread limit | 1024 threads per SM in place of 1536 | D863 |
| Records | H-now, H-next: stores to the same lines | D860 |
| Atomics | Repeats change the rate (H-repeat) | D850 |
| Turnover | Fixed cost per block; each wave a fixed cost (H-wave) | D872, D890 |
| Twin sectors | A time window (H-window); a queue that merges (H-queue) | D886 |
| GEMV's read | Pitch alignment (H-align), width (H-width) alone at rows of bytes / 237 GB/s | D886 |
| Stores apart | Partly filled sectors cost a read (H-fill) | D887 |
| Rows warm | The store path (H-store), the read path (H-read), launch floor alone | D890 |
| FA one stage | Second mma alone (H-mma), exp2 (H-exp), final stores (H-store) | D890 |
| Stage cost | Shared memory, registers, depth, code of the third stage (H-smem, H-regs, H-code, H-depth, H-own) | r14 |
| DRAM | TLB cost to 1 GiB; L1 random replacement; half the DRAM rate as a kind of read (replaced by bursts) | D877, D879, D880 |

## Open questions (experiment counts per mechanism, LOOP §3)

Round 17 changes: 51 given a rule by blocks with an unmeasured threshold (D889), 52 and 53 at 1 of 2, 48 stays closed (`fs_*`, D890), 55 and 56 new.

Round 15 changes: 47 closed by a probed rate in place of the fit (D869), 48 closed at 2 of 2 without a rule (D871), 50 closed at 2 of 2 without a rule (D870), compute alone at turnover closed at 2 of 2 without a rule (D872), 51 new. Round 14 changes: 44 closed at 2 of 2 for an index per lane, 47 closed at 2 of 2 by a fit (D864), 48 at 1 of 2, 49 closed at 1 of 2, 50 at 1 of 2. Round 13 changes: 43 closed at 2 of 2 for uniform addresses, 45 at 1 of 2, 46 closed, 47–50 new; the entries below carry them. Round 12 changes: 2 at 2 of 2, 39 closed for `ATOMG` except few words, 40 and 41 closed, 43–46 new; the entries below carry them.

3. **HGEMM 2048³ model fast** — 1 of 2: warm −1 to −13 %, rotating −4 to −12.5 % (r4); the loss is outside the DRAM path (loads from L2, block turnover, stores); the 128-thread 128×128×32 variants are the worst both ways.
6. **Softmax and FA after the FCHK fix** — compensating mechanism missing (unchanged): FA −7.5 %; softmax now under question 15.
9. **Issue limit exceeded** — 1 of 2 (unchanged). Excluded (D758).
23. **Rotating HGEMM at 1024³** — new, 0 of 2: −8 to −11 %, warm within −3 %; 2048³ rotating −4 to −7.6 %, 4096×1024×2048 −2 to −4 %: the cold path is short at small shapes.
26. **Retention under rotation** — 1 of 2 (r11 `wr_*_rot*`: plain loads alone and with stores within 10 % of the curve at the window: the simple case holds; decode and RMSNorm's extra stays unexplained): `LDG.E.EF` in rot_depth follows the plain curve, r3's `.cs` curve is cold from 12 MiB; decode keeps 18 % at 67 MB, RMSNorm without stores 55 % at 32 MiB, above the curve. Row depth rows 0/4 in direction.
32. **Chip L2 under arithmetic for copies** — 2 of 2, not identified; a row points at it again: the first pick's loss of 8 % (28 §1), which needs a new experiment on the compute side, not on the L2 rate (D835): r10 `wq_m3072`: s2 211.1, s3 228.5 µs; without compute 147.9, 149.0; without stores 192.7, 209.2: the gap goes with compute, as question 36. Earlier: lean one-stage warm copies at 48 blocks run about 1070 B/clk, replay 21.2 k, measured 17.5 k; sharing a tile does not help (D 1–48 flat). 1070 in the table: τ 0.60 → 0.38, rejected (D812); the tile rate and the lean loop's rate differ.
33. **Event timing resolution** — D804, user's call. Rotation depth rows (question 26) stay wrong: decode and RMSNorm reuse nothing inside a launch (checked: reads = footprint), so promotion in the L2 is not what keeps their 5–18 %.
34. **Softmax graph −22 %** — new, 0 of 2: launch and drain on the path; 1 of 2 on question 27's mechanism ruled it out.
35. **Rotating replays below warm** — new, 0 of 2: 128×128×32 s3 and 3072 s2: replay rotating 7 % below warm, measured above. The trace shows tensor busy 3.02 → 3.23 of 4 and the store queue 19 → 10 µs; probably DRAM delays skew the two blocks on an SM (not tested). No measurement shows that gain.
36. **`c128x256x32` t256 above the K 64 rows** — 1 of 2 (r10 `wq_*`): the gap goes with compute. Measured K 32 s3 238.3, K 64 s2 223.7 µs; without stores 213.8, 202.1; without compute 120.3, 128.7: reversed. The compute part is 118 against 95 µs, replay 112 against 100. Counters: K 32 has mio throttle ×2.1, short scoreboard ×2.4, FMA active ×3.1. Candidate, not scored: the IADD.64 ALU hold (D820).
38. **Copy issue on one warp** — new, 0 of 2: copy_depth about 45 cycles per copy per warp at 1 and 4 warps, replay 30. No queue depth (D816). Candidate, not scored: IADD.64 in the address code.
42. **Write step beside re-read lines** — new, 0 of 2: 1.5 × `stg_dirty_bytes` is a bound between HGEMM's free 9.4 MB and the stencil's 16 MiB that pays; nothing measured between. Rows now: `wi_rms_fp32_warm` −33.5 %, `wi12_rms_fp32_warm` −25.3 %: 8 MiB written, each element read twice, pays, HGEMM does not (`rmw_*` show a line read then written does not pay; what separates these is open).
45. **Hashed and random stores** — 1 of 2 (r13c `hs_*`, `hb_*`, D851): 6.5 sectors per clock to 4 MiB, 16 MiB free, past that 64 MiB hashed 208 µs, by blocks 516 µs: not separated. `hs_f16` 230 µs: hot words. Loaded addresses still at plain rates: `bl13_bucket` −78 %, `bl13_permute` −59 %, `bl_scatter` −39 %. Before: `bl_scatter` −39.9 %, `bl_fanout` −22.8 %, `bl_aos_f1` −47.6 %, `bl_tally2d` −52.5 %, `bl_mixrwa_noatom` −59 %: lanes apart at the chip; rates of plain stores in the replay. `bl_csr` +174.7 %: the warm curve at 24 MB against 5 % misses measured.
51. **Write limit beside single-use reads** — new, 1 of 2 (r15b `wl_*`, D873): softmax rows writing 6.56 to 7.65 MiB run 14 to 18 µs with no step; `wr_r1w1` 7 MiB 38.6 µs, 6 MiB 12.9. `sim.stg_dirty_once_bytes` stays; softmax 1920×1792 +130 to +210 %, `wl_*` +150 to +191 %. What separates the two is not found; a candidate, untested: 112 blocks against 1920.
52. **Rotating order under bursts** — 1 of 2 (r17 `tw_*`, `gr_*`, D886, D891): no window by time; twins merge between warps and blocks asked at once, not between two instructions of a warp, whatever the gap; T 32 unexplained; GEMV's read alone flat. GEMV rotating follows bytes per instruction, the replay follows compute. Before (D877): rows moved towards the GPU in level, the order worsened: decode 36×2304×144 rotating pairs 8/11 → 5/11, loss 0.1 → 18.9 %; `gemv14_np2_r64_rotating` +12.6 → +25.5, `gemv14_mk16_rotating` +5.6 → +16.3. Cell read × DRAM × row × c. Compensating mechanism missing; T 32's 1.4 may be the same one: twin sectors asked close in time.
53. **Stores in part of a sector** — 1 of 2 (r17 `pw_*`, `px_*`, D887, D888): a rate by sectors and lines; `pw_f12_*_one` still −33 to −70 %. Before: `mt_w_t128` +44, `mt_w_t32` −22, `mx_*` +73 to +104 % at first score, `bl16_recordthree` +68 %. Cells write × L2 × row, rmw × L2 × order.
54. **Past the rules of this round** — new, 0 of 2: `wr_r1w4_4m` +0.5 → +52, `hb_f64` −31 → +37: writes with reads beside them, or by blocks, at the burst rule.
55. **Decode rotating by tile** — new, 0 of 2: r17 `dec17_*` BN 64 68 to 74 µs, BN 32 56, the replay 54 to 56 flat; bytes alike, warm alike; first pick's loss 33 %. Cell read × DRAM × row × c. The burst count does not separate them (bursts/reads 1.03, 1.06).
56. **Rows past the register limit** — new, 0 of 2: `rms17_t16_warm` −47 %, `sm17_t16_warm` −41 %: 576 words a thread at N 4608.

## Frozen parameters

| Parameter | Value | Status | Why |
| --- | --- | --- | --- |
| `sim.l2_aggregate_B_per_clk_cpasync` (P-2) | 817.7 B/clk | `unidentified` | 2 experiments: 1 stage unchanged at 16 blocks (no latency rise), 2–4 stages −16 to −25 % at 16 blocks with the chip at 0.98 TB/s; flat share, Little's law and a lower chip rate all fail. Candidate on reopening: a share below the chip (TPC pair or GPC) |
| `sim.sts_read_clk_per_128B` | 3.25 | `unidentified` | 2 experiments: STS.128 at 1–6 warps 23.1, 12.0, 8.7, 7.0, 5.6, 4.35; the rule is 4–22 % fast at 2–5 warps; a per-sub-partition limit (5 warps slower than 4) fails |

| `sim.mma_operand_clk_per_set` | 0.05 cycles per A set | `unidentified` | 2 experiments (r6, r6b); cause not shown, cross term rejected on mma_pair (D783) |
| `sim.stg_read_overlap_clk` | 2200 | measured on l2mix, empirical | replaces the share 0.45 (D791); within 10 % over L:S 4:1 to 1:4, grid 512 only |
| `sim.l2_contig_B_per_clk` | 1900 | measured, l2_cap, l2_local, empirical | D835; tile copies do not use it |
| `sim.stg_merge_run_clk` | 32 | chosen between wr_r1w4 and HGEMM | D833 |
| `sim.stg_merge_lines` | 4 | measured, store_merge: 4 lines merge, 8 do not, nothing between | D838 |
| `sim.stg_chip_wide_B_per_clk` | 244 | measured, store_spread 237–286, empirical | D801 |
| `sim.l2_aggregate_B_per_clk_cpasync` in the burst term | 817.7 B/clk | frozen, read by D810 | slope fits 1–16 tiles |

Values marked `fitted` that the table depends on: `launch_stream_us` (upstream ablation rows only since round 3), `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Still `assumed`: `sim.default_var_latency_clk` 20, `sim.rbar_release_clk` 8, `sim.fchk_fast_path`.

## Fit ledger

| Name | Component | Value | Data | Rows on whose path it sits (share) |
| --- | --- | --- | --- | --- |
| launch_stream_us | launch | 2.6 µs | upstream ablation harness (fitted) | gemm1024 nocompute rows up to 12 %; no KTA batch row since round 3 (replaced by the measured `launch_event_us`, D769) |
| ~~copy_inflight_sm~~ | removed round 15 (D869), replaced by the measured `copy_issue_sm_clk` | | | |

No fit was added in rounds 1–6. Round 6 values come from probes; the operand curve and the read share were chosen against probes, with test rows in view: their ablations are in `perfmodel/batches/r6/entry.md`.

## Round reports

```
Round 16: batch r16 (blind set, memory-hierarchy sweep of the coverage matrix: mr_*, mk_*, mt_*, mw_*, mx_*, ms_*, ml_*, md_*, mh_*, mst_*, copy_cold): the first round to take mechanisms from the hardware's list.
Question: rounds 12-15 found no new module while real pairs off HGEMM stayed near 72 %. Which cells of the matrix are empty, what do they hold; round 15 left the memory hierarchy open.
Hypotheses: DRAM H-kind / H-shape / H-small / H-evict; threads H-load / H-store / H-sync / H-kernel; write limit H-bytes / H-blocks / H-reads; FA H-global / H-shared / H-none; L1 H-128 / H-192 / H-rand; far H-flat / H-bank / H-tlb / H-lines; copies H-same / H-free.
Result: one rule holds over the whole DRAM sweep: the DRAM moves 64-byte bursts, a sector asked without its twin pays the whole burst: "half the DRAM rate" was that. The L2 keeps 128-byte lines. No TLB cost to 1 GiB. L1 LRU 128 KiB. Past the L2 in-order stores meet no DRAM limit, hashed ones one burst per lane. Starting threads past 245 K costs time. Copies issue at the same rate cold. FA's mod 32 and the write limit not found.
Change: D877 bursts, D878 lines, D881 stores, D883 launch by threads; D879, D880, D882, D884 record results. D875, D876 core table and matrix.
Scorecard: core table of a841f67, against the same rows of the full table of 1d7d6b0: 431 kernel rows: median 6.9 → 7.2 %, above 25 % 54 → 51; 14 rows better by 3 points, 10 worse. `dl_*` −42 → +12; RMSNorm 1920×1792 first pick's loss 36.3 → 0, softmax 31.7 → 0; `hand_stencil` −16 → +8. Worse: decode 36×2304×144 rotating, loss 0.1 → 18.9 %, pairs 8/11 → 5/11; GEMV rotating rows to +25; `wr_r1w4_4m` +0.5 → +52; `hb_f64` −31 → +37. Sweep `mr_*`, first score median 43 %, now about 12 %. The core's scores do not compare with a full table's: 54 groups, pairs 537/660, top 3 44/54, loss worst 18.9 %, what-if 113/143, magnitude 39 % over 88.
New mechanism: yes — inventory: DRAM burst, L2 lines, L1 capacity, launch by threads, stores opening bursts: from probes, none from a failing row.
Fits: none added, none removed; `launch_thread_*` empirical, from probes.
Matrix: 406 cells, M 60 → 73, m 18 → 20, empty 328 → 313. Round 15's 51 misses: outside measured cells 9 → 2, in M cells 9 → 20. Round 16's 56 misses: M 21, m 34, empty 1.
Core back-test (D875): full tables of rounds 12-15: 51 of 60 families that moved show in the core (85 %), 19 of 31 that worsened (61 %), pair score's direction 3 of 3.
Testing cost: below the core 1892 CPU s of 3000: held. Predictions 787, selection 1105. Core 3438, 199 s wall, above its estimate (D885); no full. GPU: one batch, 40 min; preflight dropped 6 of 248 (shared memory). Time about 1 h 40.
Not solved: 28 §1 not met on any row. GEMV rotating 7/28 blind, real 4/17; rotating orders of decode, worse after the rule. RMSNorm warm blind 0/5, level −14 to −35 %. FA mod 32. Write limit `wl_*` +180 %. Stores in part of a sector. DRAM channels and banks not reached. `bl16_stackhist` −85 %, `bl16_stridewalk` +160 %.
Next: question 52 first: read × DRAM × row × c, twin sectors by time; then question 53; empty cells of atomics and TMA past the L2.
Blind set: the recipe of D874: HGEMM 3200×1280×1600, GEMV 7168×2688, FA 2240×7×112, decode 28×2240×112, rows 896×3584 (1344×2240 refused by TileLang at T 128 and above, before any result), own kernels `hand16/`. Rule: off HGEMM a factor 7 in every size, HGEMM a factor 25 in M. T 512 ran.
```

```
Blind block, round 16: r16, first scores, predictions at 6640750, engine of round 15: not mixed with anything below.
| 28 §1 | Target | HGEMM, one new shape | Off HGEMM, 9 groups | All 11 groups |
| Pairs ≥ 3 % right | ≥ 95 % | 61/70 (87.1 %) | 131/168 (78.0 %) | 192/238 (80.7 %), not met |
| Real pairs right | report | 61/69 (88.4 %) | 126/152 (82.9 %) | 187/221 (84.6 %) |
| Best in top 3 | every group | 2/2 | 6/9 | 8/11, not met |
| First pick's loss | ≤ 3 % | 0.0, 4.8 % | worst 17.3 %; 5 of 9 above | 6 of 11 above, not met |
| What-if direction ≥ 3 % | ≥ 95 % | | | 43/54 (79.6 %), not met |
| Magnitude ≥ 10 % | median ≤ 20-25 % | | | 27 % over 25, not met |
| Time | median ≤ 10 %, none > 25 % | 3.7, 3.2 % | decode 15.0, 24.1 %; FA 5.5, 8.3 %; GEMV 12.6, 10.1 %; RMSNorm 14.8, 3.1 %; softmax 19.3 %; hand 26.5 % | 11.2 %, ≥ 30 µs 6.2 %; 13 above 25 %, worst +164 %, not met |

Real pairs off HGEMM, rounds 13 to 16: 74 %, 71 %, 72 %, 83 %. It moved, before any engine change of this round: engines of 15 and new shapes, so it is not this round's rules: the shapes are others, GEMV warm alone gives 28/28 of those pairs. One round; not a trend.
Misses above 10 % on `kernel` rows, 56 of 103: defect 0; known rule wrong 35; no rule 21; cells in `29` §7a.
```

```
Round 17: batch r17 (blind set, tw_*, gr_*, pw_*, px_*, rr_*, tv_*, fs_*): rules for the cells that hold blind pairs and first-pick losses, no wider sweep.
Question: ranking of round 16's pairs and losses by cell: read × DRAM × order × c 21 pairs (GEMV rotating), read × L2 × row × c 6 pairs and 17.3 % (rows warm), read × DRAM × row × c 6 and 8.4 % (decode rotating), copy × DRAM × order × c 6 and 4.8 %, FA 3 and 9.5 %.
Hypotheses: twins H-instr / H-window / H-queue / H-local; GEMV read H-flat / H-align / H-width / H-table; stores H-sectors / H-lines / H-fill; rows H-store / H-read / H-launch; turnover H-wave / H-none; FA H-mma / H-exp / H-store.
Result: no time window for twin sectors: D877 stands; twins merge across warps and blocks asked at once. GEMV's read alone is flat at bytes / 237 GB/s; only runs crossing a burst pay a little. Stores apart cost by sectors and by lines. Turnover holds (H-none). Rows warm and FA's one stage: every hypothesis rejected or unsupported.
Change: D886 burst edges, unknown addresses bounded; D887 stores by sectors and lines; D888 dirty limit by lines; D889 once-limit by blocks (threshold unmeasured); D892 core. Probes replayed before and after each rule: `dl_*`, `mr_*`, `mk_*` unchanged; `mw_*` worse (D887).
Scorecard: core table of 7083a2e against the core of a841f67: 431 kernel rows median 7.2 → 7.4 %, above 25 % 51 → 47; 25 rows better by 3 points, 32 worse; pairs 883/1018 → 892/1019, top 3 63 → 63, loss worst 22.5 → 18.9 %, what-if 113/143 → 115/143, magnitude 39 → 33 %. Better: softmax 1920×1792 +144 to +211 → −18 to −26, `wl_*` +190 → −25, `hs_*`, `mx_*`, `mt_*`. Worse: `bl15_chunked_w6` −19 → −31, `rf_w*_m4` −5 → −15, `bl13_recordfill_w4` −30 → −38, `mw_o_f120` −3 → −14. The list grew by 14 rows, so the pair totals are not on the same rows.
New mechanism: no new module; `dram_burst_edge` and stores by lines refine known rows.
Fits: none added; `dram_burst_edge` empirical over two probe points; `stg_dirty_once_blocks` assumed.
Matrix: 406 cells, M 73 → 75, m 20 → 18, empty 313. Round 16's 56 misses: M 27, m 28, empty 1. Round 17's 54: M 25, m 28, empty 1.
Core back-test: worsened families 19/31 → 30/31 in sample (D892), moved 85 → 90 %.
Testing cost: below the core 2218 CPU s of 2500: held. Predictions 1234, selection 944, scratch 40 (estimate). Core 3499, 201 s wall. GPU: one batch, 30.4 min: profiler 23.0, timing 5.4, cool-downs 2.0; preflight dropped 14 of 199 (shared memory). Time about 1 h 30.
Not solved: 28 §1 not met on any row. GEMV rotating 18/35: the replay follows compute (D891). Decode rotating by tile, first pick's loss 33 % (55). Rows warm: the gap stays, no rule. T 32's twins. `pw_f12_*_one`. Real pairs off HGEMM fell back: round 16's 83 % was not a trend. D889's threshold.
Next: question 55 and GEMV's critical path against the DRAM's rate (D891), one experiment left on 52; `--full` is due in round 18.
Blind set: the recipe of D874: HGEMM 3456×1296×1728, GEMV 7200×2592 group 72, FA 2016×9×144, decode 27×2160×144 G 9, rows 720×4608, own kernels `hand17/`. Rule: a factor 9 off HGEMM, 27 on it; D 108 and 72 refused by TileLang, before any result. Preflight dropped most large-tile FA variants and two of decode: groups smaller than round 16's, FA's 14/14 rests on few pairs.
```

```
Blind block, round 17: r17, first scores, predictions at b27fd52, engine of round 16: not mixed with anything below.
| 28 §1 | Target | HGEMM, one new shape | Off HGEMM, 9 groups | All 11 groups |
| Pairs ≥ 3 % right | ≥ 95 % | 64/73 (87.7 %) | 93/126 (73.8 %) | 157/199 (78.9 %), not met |
| Real pairs right | report | 64/73 (87.7 %) | 87/111 (78.4 %) | 151/184 (82.1 %) |
| Best in top 3 | every group | 2/2 | 7/9 | 9/11, not met |
| First pick's loss | ≤ 3 % | 4.5, 1.4 % | worst 33.0 %; 4 of 9 above | 5 of 11 above, not met |
| What-if direction ≥ 3 % | ≥ 95 % | | | 37/47 (78.7 %), not met |
| Magnitude ≥ 10 % | median ≤ 20-25 % | | | 38 % over 31, not met |
| Time | median ≤ 10 %, none > 25 % | 3.1, 3.6 % | decode 11.1, 20.6 %; FA 17.4, 15.6 %; GEMV 10.5, 15.4 %; RMSNorm 27.2, 4.3 %; softmax 22.8 %; hand 13.1 % | 11.1 %, ≥ 30 µs 7.6 %; 15 above 25 %, worst +114.5 %, not met |

Real pairs off HGEMM, rounds 13 to 17: 74, 71, 72, 83, 78 %. Without warm GEMV: 75, 73, 73, 79, 74 %. Warm GEMV alone gave 28/28 and 20/20 in the last two rounds. D877-D883's first blind score: no step above rounds 13-15 once warm GEMV is out.
Misses above 10 % on `kernel` rows, 54 of 95: defect 0; known rule wrong 34; no rule 20; cells in `29` §7a.
```
