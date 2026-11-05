# Performance model: state

The single state file of the GB10 programme (28 §7, LOOP.md product 7). Updated 2026-11-05 (round 19). The current position is in the rows below, the last round report and at the top of the error table; older positions, closed questions and reports 1 to 15 are in [STATE-archive.md](STATE-archive.md). Table: [error-table.md](error-table.md). Inventory: [29](../29-mechanism-inventory.md). Batches: `perfmodel/batches/<batch>/` (entry blocks, SASS, predictions, batch record, measured values); raw outputs `data/perfmodel/batches/<batch>/` (not in git).

## Where things stand

| Item | State |
| --- | --- |
| **Blind position against 28 §1, round 19 (the number that counts: first scores of 103 new kernel rows, r19c, predictions committed before the run, engine of round 18 with D903: first blind score of D895 to D898, the recipe of D874)** | Pairs ≥ 3 % right 224/247 (90.7 %; target 95): **not met**; HGEMM 67/70 (95.7 %), the others 157/177 (88.7 %); real pairs 206/223, off HGEMM 139/153 (90.8 %), without warm GEMV 121/135 (89.6 %). True best in top 3 9/11: **not met**. First pick's loss worst 19.0 %, 5 of 11 above 3 %: **not met**. What-if direction ≥ 3 % 46/54 (85.2 %): **not met**. Magnitude ≥ 10 % median 56 % over 25: **not met**. Time median 9.8 %, ≥ 30 µs 9.4 %: median **met**, 13 above 25 %, worst −73.3 %: **not met**. Of 51 rows off by more than 10 %: 0 defects, 32 known rules wrong on the row, 19 no rule covered |
| **Blind position against 28 §1, round 18 (the number that counts: first scores of 96 new kernel rows, r18, predictions committed before the run, engine of round 17, the recipe of D874)** | Pairs ≥ 3 % right 213/255 (83.5 %; target 95): **not met**; HGEMM 71/76 (93.4 %), the others 142/179 (79.3 %); real pairs 206/242, off HGEMM 135/166 (81.3 %), without warm GEMV 110/138 (79.7 %). True best in top 3 6/11: **not met**. First pick's loss worst 25.3 %, 5 of 11 above 3 %: **not met**. What-if direction ≥ 3 % 41/52 (78.8 %): **not met**. Magnitude ≥ 10 % median 53 % over 32: **not met**. Time median 10.6 %, ≥ 30 µs 8.5 %: **not met** on all rows; 24 above 25 %, worst −81.5 %: **not met**. Of 49 rows off by more than 10 %: 9 defects, 24 known rules wrong on the row, 16 no rule covered |
| **Blind position against 28 §1, round 17 (the number that counts: first scores of 95 new kernel rows, r17, predictions committed before the run, engine of round 16, the recipe of D874)** | Pairs ≥ 3 % right 157/199 (78.9 %; target 95): **not met**; HGEMM 64/73 (87.7 %), the others 93/126 (73.8 %); real pairs 151/184, off HGEMM 87/111 (78.4 %), without warm GEMV 67/91 (73.6 %). True best in top 3 9/11: **not met**. First pick's loss worst 33.0 %, 5 of 11 above 3 %: **not met**. What-if direction ≥ 3 % 37/47 (78.7 %): **not met**. Magnitude ≥ 10 % median 38 % over 31: **not met**. Time median 11.1 %, ≥ 30 µs 7.6 %: **not met** on all rows; 15 above 25 %, worst +114.5 %: **not met**. Of 54 rows off by more than 10 %: 0 defects, 34 known rules wrong on the row, 20 no rule covered |

Real pairs off HGEMM, blind, rounds 13 to 19: 74, 71, 72, 83, 78, 81, 91 %; without warm GEMV 75, 73, 73, 79, 74, 80, 90 %. Pairs above the passes' spread, without warm GEMV: 84, 78, 77, 84, 76, 84, 94 % (D901). One round: not a trend yet.

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
| Common cause of flat pairs | One SM standing for all, by itself: once the DRAM charge is whole the replay's queue binds and its order follows bursts; D891's "replay follows compute" was the lost charge | D895 |
| Decode by tile | Rows an instruction spans (H-span); L2 read sectors (H-L2); burst count per instruction | D899 |
| Frames | A step at the SM's capacity (H-step); T 16's half-empty warps (H-lanes as a cause) | D897 |
| GEMV at a pitch | Edges alone, warps not mattering (H-edges) | D898 |
| GEMV's wrong pairs | The gap outside the scale table (H-a), the table's lone sectors real by variant (H-none) | D906 |
| Rows warm | Warm reads as the cause (H-load) | D907 |
| DRAM | TLB cost to 1 GiB; L1 random replacement; half the DRAM rate as a kind of read (replaced by bursts) | D877, D879, D880 |

## Open questions (experiment counts per mechanism, LOOP §3)

Round 19 changes: 58, 59 new, 51 reopened (D907); 55, 57 not run: the ladder did not select them; 52's `np8` remainder was D903. Round 18 changes: 52 closed at 2 of 2, 55 and 56 at 1 of 2, 57 new. Round 17 changes: 51 given a rule by blocks with an unmeasured threshold (D889), 52 and 53 at 1 of 2, 48 stays closed (`fs_*`, D890), 55 and 56 new.

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
51. **Write limit beside single-use reads** — reopened (D907): D889's limit by blocks does not hold at 832×6656; see 59. Before: 1 of 2 (r15b `wl_*`, D873): softmax rows writing 6.56 to 7.65 MiB run 14 to 18 µs with no step; `wr_r1w1` 7 MiB 38.6 µs, 6 MiB 12.9. `sim.stg_dirty_once_bytes` stays; softmax 1920×1792 +130 to +210 %, `wl_*` +150 to +191 %. What separates the two is not found; a candidate, untested: 112 blocks against 1920.
52. **Rotating order under bursts** — 2 of 2, closed (r18 `gq_*`, `tw_t*`, D895, D898): twins merge on one warp to a block; warps per row cost about 5 % a doubling, no rule, `np8` rows keep it. D891's reading replaced by D895. Before: 1 of 2 (r17 `tw_*`, `gr_*`, D886, D891): no window by time; twins merge between warps and blocks asked at once, not between two instructions of a warp, whatever the gap; T 32 unexplained; GEMV's read alone flat. GEMV rotating follows bytes per instruction, the replay follows compute. Before (D877): rows moved towards the GPU in level, the order worsened: decode 36×2304×144 rotating pairs 8/11 → 5/11, loss 0.1 → 18.9 %; `gemv14_np2_r64_rotating` +12.6 → +25.5, `gemv14_mk16_rotating` +5.6 → +16.3. Cell read × DRAM × row × c. Compensating mechanism missing; T 32's 1.4 may be the same one: twin sectors asked close in time.
53. **Stores in part of a sector** — 1 of 2 (r17 `pw_*`, `px_*`, D887, D888): a rate by sectors and lines; `pw_f12_*_one` still −33 to −70 %. Before: `mt_w_t128` +44, `mt_w_t32` −22, `mx_*` +73 to +104 % at first score, `bl16_recordthree` +68 %. Cells write × L2 × row, rmw × L2 × order.
54. **Past the rules of this round** — new, 0 of 2: `wr_r1w4_4m` +0.5 → +52, `hb_f64` −31 → +37: writes with reads beside them, or by blocks, at the burst rule.
55. **Decode rotating by tile** — 1 of 2 (r18 `dr_*`, D899): the gap needs rows off 64 bytes and 16-byte lanes; lone sectors per byte alike at BN 32, so why is open; plain-load half lost to a sizing error of `dp_*`. Before: r17 `dec17_*` BN 64 68 to 74 µs, BN 32 56, the replay 54 to 56 flat; bytes alike, warm alike; first pick's loss 33 %. Cell read × DRAM × row × c. The burst count does not separate them (bursts/reads 1.03, 1.06).
56. **Rows past the register limit** — 1 of 2 (r18 `sp_*`, D896, D897): lanes through spills; a frame slot's first write pays the DRAM's write rate once frames and stores pass the dirty limit. Threshold not located between 7 and 12 MB; `sp_n6144_t16` +28 %. Before: `rms17_t16_warm` −47 %, `sm17_t16_warm` −41 %: 576 words a thread at N 4608.
57. **Past the rules of this round** — new, 0 of 2: `dl_w4_t32_rotating` +11 → −42 %, `dl_w8_t32` likewise: 8 to 16 loads a burst on one warp run 59 to 67 µs where 2 run 44: D898 removed a double charge that hid a per-block cost. `stl_l_n64` +121 → +226, `bl17_stackrun` +115 → +225: stack kernels under D896, D897. `fa15_c64x64_s2_t64` −17 → +43 at 62361dc (before the first-write fix; not replayed since).
58. **GEMV's scale table** — new, 1 of 2 (r19c `gs_*`, D906): the replay's gap on `r64`, `mk16`, `np2_r64` is the table's lone sectors, counted per instruction: 8.20 against 7.38 MB by variant, measured flat. The table adds about 4 µs in the full kernel, about 5.8 alone. Keying twins by pass fails `tw_*`. Cell read × DRAM × row × c. Next: the same load beside a saturated stream and alone, by rate; 25 of 27 wrong pairs at the old shapes sit here.
59. **Rows past 4 MiB written** — new, 1 of 2 (r19c `rs_*`, D907): warm rows 21.7 µs at 7.9 MB written, 54.5 at 11.1, replay 12.8, 25.7; on the line to 4 MB; stores hold it, reads do not; the replay moves with T, measured flat. Cell write × L2 × cap × c.

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

```
Round 18: batch r18 (blind set, dr_*, dp_*, sp_*, gq_*, tw_t16_g0, tw_t64_g0): why real pairs stood still, by replay first; high effort trial (D894).
Question: real pairs off HGEMM without warm GEMV 75, 73, 73, 79, 74 % over five rounds while 15+ rules went in: is there one cause in the engine? Hypotheses by replay on seen rows, then questions 55, 56, 52.
Hypotheses: common cause H-charge (the DRAM charge is not the bytes the L2 replay counts: charge restored gives GEMV rotating ≥ 20/26) / H-oneSM (charge right, the one SM misjudges a chip-wide limit: queue and replay part ways); decode H-span / H-copy / H-align / H-L2; frames H-rate / H-step / H-lanes; pitch H-edges / H-warps / H-twins.
Result: a common cause found in one form, not for every family. H-charge holds on GEMV rotating and rows past the register limit: the address replay lost a load, and one lost execution discounted all of them, by variant (0.0, 0.37, 0.99 of the misses, same bytes); spills lost addresses (643 MB for 15.9). Once charged, queue and replay agree: H-oneSM has nothing left on these rows. No common cause for decode (BN 64 needs rows off 64 bytes and 16-byte lanes, why open), for rows warm, for FA. Part of the flat line is noise: hard pairs 84, 78, 77, 84, 76, 84 % (D901). Frames: H-rate. Twins: a step at one warp a block. H-warps about 5 % a doubling, no rule.
Change: D895 unknown executions alone, stride continued; D896 lanes through spills; D897 frames count at the dirty limit, first write of a slot pays; D898 twins on one warp; D899, D901 records; D900 cost. `review.py`, `STATE-archive.md`, `counter_sets`.
Scorecard: full table of 62361dc (before D897's first-write fix) on seen rows, first score → now, real pairs off HGEMM without warm GEMV: r13 75 → 78, r14 73 → 76, r15 73 → 76, r16 79 → 83, r17 74 → 85, r18 80 → 88 %: seen, not blind. `gemv 7200x2592 rotating` 13/26 → 24/26, loss 16.4 → 2.1 %. Against the core of 7083a2e, same 436 kernel rows: median 7.3 → 7.3 %, above 25 % 48 → 55, 1 row better by 3 points, 9 worse: blind rows are not in that list. Worse: `hgemm t64`, `c256 t256` −20 → +200 % in that table, by D897 as first written, then fixed: `hgemm_m1792n2560k2304_t64` −21 → −5 % by replay at 587202a, the others not replayed; `dl_w4_t32_rotating` +11 → −42; `stl_l_n64`, `bl17_stackrun` +225.
New mechanism: yes — frames at the dirty limit; no new module.
Fits: none added; `stg_dirty_bytes` reused, threshold not located.
Matrix: 406 cells, M 75, m 18, empty 313. Round 18's 49: M 23, m 25, empty 1.
Testing cost: below the table 2495 of 2500 CPU s: held, at the edge: predictions 1414, overrides 871 (723 a first fix dropped, D895), scratch 158, selection 52. No sentinels before `--full`: no room (D900). Full 19056 CPU s, 1131 s wall, one, of 62361dc. The final engine is 587202a: D897's first-write fix came after it. A second full was allowed once by the supervisor up to about 2500 CPU s; printed first: `sm.py` is in `engine_hash`, so 5601 rows, 3005 replays, about 19000 CPU s: not run, no exception used, `refuse_second` untouched. So `error-table.*` is of 62361dc, not of the final commit: 659 rows read `sim.stack_counts_dirty` and are stale, HGEMM T 64 and T 256 among them; round 19's first table replaces it. GPU: one batch, 10.1 min: timing 4.4, profiler 3.5; 12 preflight drops. Time about 1 h 30.
Not solved: 28 §1 not met on any row. `np8` −18 to −22 %. Decode BN 64. Rows warm level (`rms18` warm median 45 %). `bl18_histbytes` −81.5 %. `dp_*` pitch rows and `gq_w8_r32_np8` unread, 6 own kernels dropped and `bl18_gatherhash` failed: sizes of mine. The high effort trial cannot be judged here: this blind score is round 17's engine; round 19's blind is the first of D895-D898.
Next: round 19's blind scores D895-D898 at a new shape, its predictions on 587202a or later; a table of the final engine first if the round needs one; question 57; 55's second experiment with `dp_*` sized right; `full.py` could flag a load's share of unknown executions so this class shows before it costs five rounds.
Blind set: the recipe of D874: HGEMM 3712×1392×1856, GEMV 8800×3168 group 88, FA 2288×11×112, decode 33×2200×112 G 11, rows 704×5632, own kernels `hand18/`. Rule: a factor 11 off HGEMM, 29 on it; D 112 for shared memory: FA kept 8 of 10 variants, 24 pairs a group against 7.
```

```
Blind block, round 18: r18, first scores, predictions at e242da5, engine of round 17 (the flags of D895 off): not mixed with anything below.
| 28 §1 | Target | HGEMM, one new shape | Off HGEMM, 9 groups | All 11 groups |
| Pairs ≥ 3 % right | ≥ 95 % | 71/76 (93.4 %) | 142/179 (79.3 %) | 213/255 (83.5 %), not met |
| Real pairs right | report | 71/76 (93.4 %) | 135/166 (81.3 %) | 206/242 (85.1 %) |
| Best in top 3 | every group | 2/2 | 4/9 | 6/11, not met |
| First pick's loss | ≤ 3 % | 0.0, 2.9 % | worst 25.3 %; 5 of 9 above | 5 of 11 above, not met |
| What-if direction ≥ 3 % | ≥ 95 % | | | 41/52 (78.8 %), not met |
| Magnitude ≥ 10 % | median ≤ 20-25 % | | | 53 % over 32, not met |
| Time | median ≤ 10 %, none > 25 % | 2.9, 4.3 % | decode 21.5, 23.2 %; FA 8.4, 7.7 %; GEMV 10.8, 17.7 %; RMSNorm 45.2, 5.3 %; softmax 32.9 %; hand 15 rows' worst −81.5 % | 10.6 %, ≥ 30 µs 8.5 %; 24 above 25 %, worst −81.5 %, not met |

Real pairs off HGEMM, rounds 13 to 18: 74, 71, 72, 83, 78, 81 %. Without warm GEMV: 75, 73, 73, 79, 74, 80 %. All on engines before D895. Printout: `python -m kta_perfmodel.review 18`.
Misses above 10 % on `kernel` rows, 49 of 96: defect 9; known rule wrong 24; no rule 16; cells in `29` §7a.
```

```
Round 19: batch r19c (blind set, gs_*, rs_*; r19, r19b the same, failed the clock band): the first round under D902: locate, then rules.
Question: real pairs off HGEMM near 80 % with more than 15 rules added: which misses are defects of the model, where do GEMV rotating's wrong pairs and the rows' gap sit.
Hypotheses: gs_* H-sc / H-a / H-none; rs_* H-foot / H-store / H-load.
Result: the work check found a defect in its first run: lane 0 stood for the warp: every warp of a 2-D block lost passes (D903); sectors now equal the counter on GEMV. Scale table H-sc holds; rows H-foot holds, H-load rejected.
Change: D903 `tx0`, guards and exits per lane; D904 tools; D905 `settle`; D906, D907 records; D908 cost. No parameter moved, no rule added past the defect.
Scorecard: blind, first scores against round 18 (engine of round 17): pairs 83.5 → 90.7 %, real off HGEMM 81 → 91, without warm GEMV 80 → 90, hard 84 → 94; top 3 6/11 → 9/11; what-if 78.8 → 85.2 %; time median 10.6 → 9.8 %; `gemv rotating` 13/26 → 23/26, median 17.7 → 4.4 %. Worse: magnitude 53 → 56 %; rows warm within-group 11.8 → 37.8, softmax 10.1 → 60.7. Error within a group off HGEMM without warm GEMV 13.0 → 24.1 %; without rows 12.7 → 7.4 %. D903 cannot be told from D895-D898 here: no ablation (D908). Core table of 39f00b2 against the same 436 kernel rows of the full table of 62361dc: median 7.3 → 7.2 %, above 25 % 55 → 48, 7 rows better by 3 points, 1 worse, pairs 366 → 367 of 453; the `hgemm t64` and `c256 t256` rows of D897 return: +200 → −5 to −10. Worse: `stu_t64_n1024_g48` −42 → −68.
New mechanism: no new module.
Fits: none added, none removed.
Located: step 1 on all rows: GEMV `np8`, `r64`, `gr_w8_*`: defect, fixed. Copy paths: the model's column, not the counter's meaning. Cleared: L2 and sector traffic on rows warm, RMSNorm, softmax, decode, FA. Open: `lib_*` (library code, no replay), `hand_mandel`, `stl_l_n1024` (loops from data), FMA and ALU busy uncalibrated. Step 2, GEMV rotating, 5 pairs: no value flips: structure. Step 3: `reduce` and `mk` leave the direction. Step 4 `gs_*`: the scale table. Rows warm: step 1 clean, step 4: past 4 MiB written, stores. Steps 5, 6 not run. Next family after GEMV was the rows by wrong pairs and level; decode, FA not taken.
Retired: none; no full table.
Matrix: 406 cells, M 75, m 18, empty 313 (write × L2 × cap × c stays m). Round 19's 51: M 21, m 30, empty 0. Time, core table, 436 kernels: M 67, m 15, empty 3, beside the matrix 13 %; scorecard families carry 0 to 2 % in empty cells: no sweep needed; `compute × L2 × order × c` and `copy × DRAM × row × c` are the m cells that carry most. `read × L1 × order × c` is the empty cell with time: 12 kernels' worth, own kernels and sweeps. Hand placement of round 18's misses agrees with the top cell on 9 of 49: decode's time is its copies and compute, not reads: round 19 placed by it.
Testing cost: below the core 2996 of 2500 CPU s: over, by predictions 2158 (D908). Core 3682, 212 s wall. GPU: three batches, 33 min, one valid, 11.6 min: timing 5.0, profiler 4.0; 6 preflight drops, large tiles as round 18; own kernels 17 of 17 ran. Time about 2 h 30.
Not solved: 28 §1 not met on pairs, top 3, first pick, what-if, magnitude, worst row. Rows warm past 4 MiB, the model's worst group now. Decode rotating, loss 19 %. FA's loss 10.5, 8.8 %. GEMV warm level −9 to −16 %, order right. The scale table's charge. `bl19_atomtwo` −73 %. Short own kernels at the launch floor, 2.5 µs against 3.
Next: question 59 first: a rule from `rs_m*` with a sweep between 4 and 11 MB, D889's limit retired or replaced; question 58's second experiment; decode rotating's loss by the ladder; the blind recipe's size, as its replay cost now breaks the limit (the user's); `--full` is due in round 21.
Blind set: the recipe of D874: HGEMM 3968×1488×1984, GEMV 10400×3744 group 104, FA 2704×13×112, decode 39×2600×112 G 13, rows 832×6656, own kernels `hand19/`. Rule: a factor 13 off HGEMM, 31 on it; D stays 112 for shared memory.
```

```
Blind block, round 19: r19c, first scores, predictions at 866802a, engine of round 18 with D903: not mixed with anything below.
| 28 §1 | Target | HGEMM, one new shape | Off HGEMM, 9 groups | All 11 groups |
| Pairs ≥ 3 % right | ≥ 95 % | 67/70 (95.7 %) | 157/177 (88.7 %) | 224/247 (90.7 %), not met |
| Real pairs right | report | 67/70 (95.7 %) | 139/153 (90.8 %) | 206/223 (92.4 %) |
| Best in top 3 | every group | 2/2 | 7/9 | 9/11, not met |
| First pick's loss | ≤ 3 % | 0.0, 0.6 % | worst 19.0 %; 5 of 9 above | 5 of 11 above, not met |
| What-if direction ≥ 3 % | ≥ 95 % | | | 46/54 (85.2 %), not met |
| Magnitude ≥ 10 % | median ≤ 20-25 % | | | 56 % over 25, not met |
| Time | median ≤ 10 %, none > 25 % | 2.8, 3.6 % | decode 15.6, 18.0 %; FA 8.0, 8.2 %; GEMV 12.4, 4.4 %; RMSNorm 50.9, 9.4 %; softmax 62.3 %; hand 17 rows' worst −73.3 % | 9.8 %, ≥ 30 µs 9.4 %; 13 above 25 %, worst −73.3 %, median met, none above 25 not met |

Real pairs off HGEMM, rounds 13 to 19: 74, 71, 72, 83, 78, 81, 91 %. Without warm GEMV: 75, 73, 73, 79, 74, 80, 90 %. Hard pairs without warm GEMV 84, 78, 77, 84, 76, 84, 94 %. Error within a group, blind, %: `python -m kta_perfmodel.review within`. Printout: `python -m kta_perfmodel.review 19`.
Misses above 10 % on `kernel` rows, 51 of 103: defect 0; known rule wrong 32; no rule 19; cells in `29` §7a.
```
