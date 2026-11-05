# Batch r17: entry blocks

Written before the run. Engine: round 16's (no change to `src/` since 7aedb11c); no parameter comes from a blind row of this batch. Preflight as D854, three passes as D847. Sweeps come from the ranking of cells by blind pairs and first-pick losses of round 16, printed first: read × DRAM × order × c 21 wrong pairs (GEMV rotating), read × L2 × row × c 6 and loss 17.3 % with 19 rows above 10 %, read × DRAM × row × c 6 and 8.4 %, copy × DRAM × order × c 6 and 4.8 %, FA 3 and 9.5, 5.7 %. Each sweep gives its cell a rule; none widens the matrix.

## Blind, the fixed recipe (D874)

Families, variant kinds and what-if axes as round 16. New shapes by a rule chosen before any result: off HGEMM a factor 9 in every size, HGEMM a factor 27 in every size: HGEMM 3456×1296×1728, GEMV 7200×2592 group 72, FA 2016×9×108, decode 27×2160×108 G 9, rows 720×4608 (N must be a multiple of 512 at T 512, as round 16 found, so N is 9 × 512). 13 own kernels and 4 what-ifs in `perfmodel/testset/hand17/`, same classes as `hand16/`. T 512 runs. Predictions committed before the run.

## tw_* — read × DRAM: when and by whom the twin sector is asked (question 52, experiment 1 of 2)

1. Mechanism: D877's burst: a sector asked without its twin by the same instruction pays 64 bytes. `mr_o_*` show a twin 4 instructions later pays too; T 32 runs 1.4 times, not 2; GEMV's 128-byte instructions at a pitch of 1344 sit 10 % over the footprint at the DRAM rate or not, by shape.
2. Sweep: 8 MiB rotating, one thread one burst: first sector, a gap of 0 to 512 loop passes, second sector; the second after the first's data (`tw_dep`); two sectors of different bursts (`tw_far`) and of a line (`tw_l`), controls; T 32, 512; twins asked by two warps of a block (`tw_sw`), two blocks of an SM (`tw_ss`), blocks on two SMs (`tw_sc`), against first sectors alone at as many asks (`tw_s0`).
   - H-instr (D877 as it stands): any gap, even 0, pays twice: `tw_g*` all as `tw_far`, about 68 µs at 8 MiB.
   - H-window: the controller merges twins within a window: `tw_g*` rise from about 35 µs to 68 with a knee at the window; `tw_dep` at the top.
   - H-queue: twins merge while the first waits in a queue, so the knee moves with load: T 32 lower than T 512 at the same gap.
   - H-smem: sharing does not matter: `tw_sw`, `tw_ss`, `tw_sc` as `tw_g0`; H-local: merging holds inside an SM: `tw_sc` as `tw_s0`, `tw_ss` as `tw_g0`.

## gr_* — GEMV's read alone (read × DRAM × order, row × c)

1. Mechanism: GEMV rotating measures 42 to 50 µs, 10.3 MB at 237 GB/s gives 43.6. The replay gives 42.8 to 56.6 by variant. It charges the scale table's unknown-address loads by bursts, past the bytes they can read (seen in the replay: bursts/reads 1.07 to 1.30 where the addresses are unknown).
2. Sweep: 8 MiB rows, 2, 4, 8 bytes per lane × pitch 1280, 1344, 1360, warps as rows; thread shapes of the variants at 4 bytes; `gr_sc` with a scale table.
   - H-flat: all at bytes / 237 GB/s: within 5 %: the order in GEMV comes from elsewhere.
   - H-align: pitch 1344, 1360 slower than 1280 at 4 bytes (lines and bursts split).
   - H-width: 2 and 8 bytes faster than 4 as in `gemv16_mk*` at 1344 only.
   - H-table: `gr_sc` adds its own burst per lane, about 2 µs more than its bytes.

## pw_*, px_* — write and read-modify-write × L2: lanes against sectors and lines (question 53, experiment 1 of 2; cells write × L2 × row, rmw × L2 × order)

1. Mechanism: `mt_w_*` T 32, 128, 256, 512 run 6, 12, 27, 38 lanes per clock with 1, 4, 8, 16 lanes on a line, `mw_o_*` 7 with 2, `mx_*` 12 with 4 at 4-byte lanes. The replay gives one rate to T 32 and T 128: +47, −20 %.
2. Sweep: 4 and 12 MiB at T 128: 8-byte lanes 8 to 128 bytes apart, sectors filled, and one store only; 4-byte lanes one store; in place at 4 MiB.
   - H-sectors: rate by sectors per clock, about 12: `pw_*_s32` and `s16` the same sectors per clock; `s128` falls.
   - H-lines: a store costs by L2 lines it opens: time per instruction rises with lines, sectors do not matter: `_one` rows as filled rows at the same stride, per instruction.
   - H-fill: partly filled sectors cost a read of the line or a mask: `_one` slower per lane than the filled rows.

## rr_* — rows warm by parts (read × L2 × row × c; RMSNorm warm 0/5, −14 to −35 %)

1. Mechanism: measured flat at 12.7 µs from T 16 to 256 at 6.4 MB read and 6.4 MB written, the replay falls with threads. Decode and GEMV warm also sit 2 to 5 µs under.
2. Sweep: 896×3584 at T 32, 128, 512 without the store, the load, the compute; M 224, 448, 1792 at T 128.
   - H-store: flatness is the store path, about 244 B/clk: `rr_nostore_*` falls away, `rr_noload_*` stays flat at about the same.
   - H-read: the read path: the opposite.
   - H-launch: a floor by blocks: `rr_m*` a line with an intercept above the launch's.

## tv_* — blocks that follow blocks on an SM (compute at turnover; D872 closed 2 of 2 without a rule; reopened by the rows above: decode warm's gap 1.9, 3.5, 5.4 µs at 1, 2, 3 waves)

1. Sweep: 400 passes of a short loop, T 32 and 128, 48 to 384 blocks, one block on an SM (a shared array) or as many as fit.
   - H-wave: each wave of blocks adds a fixed cost above the replay's: `tv_one_*` a line in waves with slope above the block's own time.
   - H-none: the replay's slope holds: the rows' gap is not turnover.

## fs_* — one stage against two on FA (copy × L2 × row; FA's first pick's loss 9.5, 5.7 %; `c64x64_s1` −21, −24 %, others −5 to −12 %)

1. Sweep: stages 1 and 2 at 2240×7×112 without the second mma, without exp2, ending in shared memory.
   - H-mma: the stage's gap is with the second mma: gone in `fs_nov_s1`.
   - H-exp: with the softmax's scalar code: gone in `fs_noexp_s1`.
   - H-store: with the final stores: gone in `fs_shared_s1`.
