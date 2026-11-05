# Batch r20: entry blocks

Written before the run. Engine: `3c5c828`, the same as the end of round 19 (no change to the simulator since), so this is the first blind score of nothing new: r20 scores the engine again at a new shape; a rule entered this round gets its first blind score in round 21. No parameter comes from a blind row of this batch. Three passes, cool-downs as D847; `counter_sets` main; sweeps without counters except `rw_*`, which read the L2's sector hit rate.

## Blind, the recipe (D874) with shapes capped

Families, variant kinds and what-if axes as round 19. New shapes by a rule chosen before any result: the largest whole factor below 11 not used yet, a factor 10 off HGEMM and 28 on it: HGEMM 3584×1344×1792, GEMV 8000×2880 group 80, FA 2080×10×112, decode 30×2000×112 G 10, rows 640×5120. D stays 112. The cap at about round 18's size is pending the user's confirmation (D908); predictions took 1248 CPU s against 2158 in round 19. 13 own kernels and 4 what-ifs in `hand20/`.

## rw_*, sw_*, wb_* — question 59

Rows warm on the line to 4 MB written, then 21.7 µs at 7.9 MB, 54.5 at 11.1, replay 12.8, 25.7; stores hold it; measured flat by T, the replay moves with T (D907). Sweep at N 5632: M 440 to 968 (5 to 10.9 MB written) at T 128; T 32, 512 at two sizes; without the load (`rw_nl_*`), without the store (`rw_ns_*`); three splits of about 9.9 MB by blocks (`rw_m1760n2816`, `rw_m604n8192`; 1408×3520 refused by TileLang); softmax at four sizes. `wb_*`: reads and stores apart: W 6, 9, 12 MiB beside R 0 to 16 MiB, one read and one store a pass, as many blocks as the larger; W 9 R 9 at 16 and 4 KiB a block.

- H-W: the limit is on bytes written: `wb_w9_*` pay at every R, `wb_w6_*` at none; `rw_nl_*` keep the gap.
- H-F: the limit is on the footprint against the L2 (about 14 to 16 MiB): `wb_w6_r12`, `r16` pay, `wb_w9_r0` does not; `rw_nl_*` lose it.
- H-T: past the limit the time is a chip-wide rate of the excess bytes: flat in T and in the split by blocks; the slope near 8 to 10 µs per MB (`rs_m*`, `rr_m1792`), about 100 GB/s. Otherwise a rate per block: the split moves it.

Replay (`predicted.json`): `rw_m*` T 128 9.7 to 25.1 µs; at M 880 T 32 15.0, T 128 14.9, T 512 37.7: it moves with T; `wb_w9_*` 16 to 79, `wb_w6_*` 9 to 31: it moves with R; split at 9.9 MB 14 to 19.

## cb_*, fas_*, dw_* — decode rotating, FA's loss: one suspect

§3b step 1 on rows that exist. Both families lose at D mod 32 = 16 and not at D 64, 96, 128: `fad_*` D 64 to 144 +7, −17, −1, −25, −5, −23 %; decode BN 64 D 96, 128 within 5 %, D 112, 144 −15 to −25; FA one stage D 96 0 %, D 80, 112 −21 to −30. BN 32 and `c64x32`, which copy 8 bytes a lane, stay within −2 to −9 % at the same D. D884 tried this with plain loads and copies into a conflict-free layout and found nothing flat by D. What D 80, 112, 144 share: TileLang's layout at an odd number of K tiles. By its `cp_async_gs<16>` address expressions, lanes of a warp that fall on one bank quarter of shared memory (32 banks of 4 B, as `bank_conflict_ways` for ldmatrix): 4.0, 5.0, 4.7, 7.0, 4.0, 8.9 at D 64 to 144. Computed from `kernel.cu` before this batch. Copies write through the array ldmatrix reads; the engine charges a copy's write by bytes, not banks.

Every `cb_*` row runs about 30 µs (2 µs timer steps, D901). `cb_q{4,8,16,32}`: copies into plain layouts of that many lanes on a quarter; `cb_tl{5,7,9}`: TileLang's layout; same contiguous source, alone and beside ldmatrix (`_ld`). `cb_rows_*`: D 112's source rows with both destinations. `cb_*_rotating`: DRAM behind it. `fas_d*`: the one knob in FA at one stage, `dw_d*_bn*`: in decode warm, as `dr_*` rotating.

- H-bank: time grows with lanes on a quarter: `cb_q32` > `cb_q8` > `cb_q4`, `cb_tl7`, `cb_tl9` between; the replay is flat at all of them.
- H-beside: alone flat, beside ldmatrix it grows: the array is fought for only when something reads it.
- H-source: destinations do not matter, `cb_rows_q4` and `cb_rows_tl7` alike and both above `cb_q4`: the rows' source addresses are the cost.
- H-none: all flat: bank quarters are a coincidence, and the D sawtooth is compute (the `% 7` address code, FCHK paths).

## gb_* — question 58, second experiment

The scale table's load adds about 4 µs in the full kernel, 5.8 alone (`gs_*`); the replay counts twins per instruction (D886), `tw_*` needs that. Alone rows have six times the rows, or they fall under the timer's steps. Rows: the table every pass or every 4th pass, one instruction or five with the same lanes (`i1`, `i5`), beside GEMV's stream and without it, rotating.

- H-free: beside a saturated stream the table costs its bytes whatever the instructions: `i1` and `i5` alike beside it, apart alone.
- H-always: twins per instruction hold beside the stream too: `i5` about 2× the table's cost in both.
- H-none: `i1` and `i5` alike in both: then `tw_*`'s rule does not reach loads of 2 bytes.
