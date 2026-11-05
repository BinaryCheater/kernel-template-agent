# Batch r15b: entry blocks

Written before the run, after r15. Engine at this commit carries this round's rules so far: `copy_issue_sm_clk` in place of the fit, the lanes' two buffers and float steps. Only sweeps: no blind rows.

## wl_* — the write limit beside single-use reads (new row: softmax 1920×1792 +131 to +205 %)

1. Mechanism: `sim.stg_dirty_once_bytes` 6.5 MiB, the midpoint of r11b's 6 MiB (12.9 µs) and 7 MiB (38.6 µs). The blind softmax rows write 6.56 MiB and measure no penalty; the replay pays it. A parameter may not come from a blind row, so the threshold is read here.
2. Sweep: softmax rows, T 128, N 1792, M 1920 to 2240: 6.56, 6.70, 6.84, 6.97, 7.11, 7.65 MiB written.
   - H-limit: a step between two of these rows, about three times the time: the threshold lies there.
   - H-kernel: no step up to 7.65 MiB: the limit does not hold for this kernel, the rule's condition is wrong rather than its number.

## copy_two, 5 to 7 warps (question 47)

1. r15: 8 warps of two blocks 80 clk per copy, 4 warps 48. Per SM and per sub-partition give the same at 8.
2. Sweep: one block of 5, 6, 7 warps, K 32.
   - H-sm: 10 clk per copy on the SM: warp 0 at 50, 60, 70.
   - H-smsp: 40 clk per copy on a sub-partition: warp 0 at 43 or 80 by whether it shares its sub-partition.
