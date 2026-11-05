# Batch r13c: entry blocks

Sweeps the blind misses of r13b asked for, and the tile trace of question 46. Written after r13b's scores; no parameter comes from a blind row, only from these sweeps. Replay predictions in `predicted.json`.

## tr_* — progress of each block (question 46, first GPU work)

1. Mechanism: blocks resident together on an SM; the replay has one SM stand for all and two modes, one with a block ahead by about ten tiles for a whole wave (`sw8` at 3072×1536: tiles of 2300 clk against 2500; a block's tail then overlaps the other's main loop).
2. Why now: first pick's loss of 8 % on seen groups.
3. Discriminating: thread 0 of every block stamps clock, global timer and SM at each k-tile. s2, s3, sw8 at three shapes, warm; four launches each.
   - H-none: on the GPU blocks of an SM stay within a tile or two in every wave, s3 and sw8 alike: the fast mode does not exist and the replay must not make it.
   - H-max: some SMs run staggered, others not, the kernel ends with the slowest SM: the fast mode exists and does not show in the kernel's time.
   - H-mean: every SM drifts over launches; the time is an average of modes.
   The trace also gives the SMs' spread at the end, which tells whether one SM can stand for all.

## atb_*, atk_* — atomics on few words (question 43, second experiment)

1. Mechanism: D834, D840: 1.44 per clock per 4 KiB region, up to 12 chip-wide. r12b `atw_*`: lanes per word 1 to 8 slow it down at 64 bins, `bl_atomblock` runs 22 per clock.
2. Sweep: every lane of a block on one word, the same word each time, 1 to 1024 words; a new word after P atomics, P 1 to 8.
   - H-now (replay): by regions: `atb_w1` about 330 µs.
   - H-repeat: an atomic on the word the queue already holds merges: time falls as P rises and at few words.
   - H-instr: lanes on one word are one request: flat in P.

## hs_*, hb_*, hl_* — hashed stores and loads (question 45, first experiment)

1. Mechanism: stores with lanes apart take `stg_chip_wide`, past the limit the DRAM's write rate, by footprint; hashed loads take the L2 line rate.
2. Sweep: 2 M hashed stores over 16 KiB to 64 MiB of the whole grid, or 4 to 256 KiB per block; as many hashed loads.
   - H-now (replay): stores flat up to the dirty limit, the DRAM's rate above; loads 14 lines/clk.
   - H-sectors: stores pay by distinct sectors the L2 must take; a block's small region merges.
   - H-lines-load: hashed loads run below 14 lines/clk whatever the footprint.

## rf_* — records, each thread's own 64-byte line (`bl13_recordfill`, `bl13_tilefill` slow in the replay, by 2 to 3 times)

1. Mechanism: D838: more than 4 lines per instruction do not merge.
2. Sweep: W of 16 words per record, over 4 and 16 MiB.
   - H-now (replay): each store pays its 32 sectors.
   - H-next: a store merges with the sector its own warp just wrote, as at 4 lines or fewer: time goes by whole lines, nearly flat in W past 8.
