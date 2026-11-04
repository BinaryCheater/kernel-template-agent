# Batch r11b: what r11 left open

Written before the run; replay in `predicted.json`. Second experiment on question 37, and the step in the write path.

## l2_local — why contiguous copies move 2100 B/clk here and 817 in TileLang's tiles (question 37, 2 of 2)

1. Mechanism: chip L2 rate, `l2_aggregate_B_per_clk_cpasync` 817.7 `unidentified`.
2. Why now: r11 `l2_cap` cp16 2144 B/clk, HGEMM tile copies stay 817 (2100 for them: l2share_copy −42 %, first pick's loss 8 % sits on `queue (chip L2 share)`). `atom_*` showed 4 KiB regions mapping to one target.
3. Sweep: the same 512-byte copies from 8 MiB, G 48, 144, 480: scattered, each block walking its own region, all blocks in lockstep on the same addresses, a warp held in one 4 KiB page for a pass.
   - H-scatter: only scattered addresses reach 2100: walk, lockstep or page near 817.
   - H-page: page falls to near 817 (targets by 4 KiB), walk and lockstep stay high.
   - H-share: lockstep alone falls.
   - H-none: all four near 2100: the arrangement is not it; the tile's other half (8-line copies) carries the rate.

## wr_r1w1 E 5–12 — the step in writes past the L2

W 4 MiB pays nothing, 8 pays 66 %, 16 all (r11); HGEMM's 9.4 MB of rewritten C pays nothing; the engine's step sits at 12.6 MB.
   - H-step: nothing up to 8, all from 12: 8 to 12 by the rule; reread by HGEMM's protected reads.
   - H-ramp: linear from 4 to 12 MiB.
