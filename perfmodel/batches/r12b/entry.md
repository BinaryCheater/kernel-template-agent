# Batch r12b: entry blocks

Written before the timed run, engine unchanged in behaviour since r12's predictions (the speed commit reproduces the round 11 table to the last digit on the sentinels). Same round as r12. No parameter comes from a `bl_*` row: these sweeps are the probes those rows' misses asked for.

## Harness repairs, blind rows

`bl_memset*`: the grid was twice what the buffer holds, faulted in r12. `bl_tally2d`: launch bounds below its block, nothing ran; r12's 0.96 µs is in `invalid_cases.json`. `wi12_dec_*`: 256-byte D overran shared memory; D 128 here, S 2048, H 16. `wi12_fa_s3` left out for the same reason.

## wf_m* — dirty limit by bytes or footprint (`bl_rewrite` +275 %)

1. Mechanism: `stg_dirty_bytes`; the replay counts bytes written so far (`SM.st_bytes`).
2. Why now: `bl_rewrite` writes 67 MB over 5 MB, L2 hit 99 %, replay at the DRAM write rate.
3. Sweep: 64 MiB of stores over W 1, 4, 8, 16 MiB.
   - H-bytes (replay): all W alike, DRAM write rate, about 790 µs.
   - H-foot: W 1, 4 at the store rate; W 16 past the limit.

## rmw_e*, wr_e* — a written line that was just read (`bl_axpy_m8` +198 %)

1. Mechanism: D832, `stg_dirty_once_bytes`, `warm_window_writes`.
2. Why now: y += a x at 8 MiB replays past the limit, measures in the L2.
3. Sweep: E 4–16 MiB; `rmw`: y read and written, `wr`: y only written.
   - H-now (replay): `rmw` and `wr` alike from 7 MiB.
   - H-rmw: `rmw` pays only when x + y pass the L2.

## atw_*, atr_* — atomics by lanes on a word, atomics with a return (`bl_atomblock` +769 %, `bl_atomret` −99 %)

1. Mechanism: D834, merge gain by the buffer's bytes; `ATOMG` not modelled (inventory, question 39).
2. Why now: all lanes on one word of 64: 22 per clock, the replay's gain by bytes gives 2.5.
3. Sweep: 1–32 lanes per word, 64 and 1 M bins; fetch-add on 1–64 K bins.
   - H-bytes (replay): gain goes with bins only.
   - H-lanes: requests are distinct words per instruction: rate × lanes per word at any bins, up to the chip's 29.
   - H-ret: `ATOMG` rate as `REDG`, plus the round trip.

## stu_* — stack, every lane alike (`wi_hg_t64_warm` −27 %, 256×256×32 −22.7 %)

r12's `stk_*` gave each lane its own index: 10 × above the replay at 7.5 clk per lane, a different case from the rows that point here. Here one index per pass, T 64, 256; N 256, 1024 words; G 48, 96, 144.
   - H-now (replay): LRU of 192 KB per SM, 400 / 133 B/clk.
   - H-warp: capacity per warp.
   - H-rate: the rates off by a flat factor.

## store_merge, profiler points

The kernel filter of r12's points matched nothing; the mangled name was shown as `k`. Timing stands; points again for L2 write sectors at G 512.
