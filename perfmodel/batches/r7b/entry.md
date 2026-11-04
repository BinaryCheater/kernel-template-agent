# Batch r7b: entry blocks

Three questions r7's first score opened, probes only, engine as r7 except the GEMV trip count (defect: the source loop `ko` matched only in kernels with an mma, so a GEMV whose main loop was not unrolled ran one pass).

## rot_depth — what a rotating launch leaves in the L2 (questions 12, 24)

1. Mechanism: inventory §5 L2 capacity and policy.
2. Why now: rotation depth rows: decode 68.9 µs at 67 MB, 78.2 and 77.2 at 134 and 268 MB, replay (all cold) 78.8; RMSNorm 41.3, 43.4, 43.3; RMSNorm without stores 20.7 at 33 MB. The rule "rotating = every read misses" is right from about 134 MB.
3. Sweep: 7.9 and 16.5 MB per launch, window 1–384 MiB.
   - H-cold: every read misses above the L2 (replay).
   - H-curve: the miss fraction is a function of the window alone, the warm curve carried on.
   - H-chunk: it depends on the chunk too.

## spill — stack traffic (new row: HGEMM t64 +228 %, replay +82 %; 256×256×32 t256 870 µs, replay 344)

1. Mechanism: inventory §3 local memory, "4 cycles per line".
2. Why now: largest blind errors of r7; a tile too big for its threads is a design choice the search must rank.
3. Sweep: 16–1024 stack words, 64 and 256 threads, 1, 48, 192 blocks, 0 and 96 KB of shared memory.
   - H-line: 4 cycles per line (replay).
   - H-L1: the stack lives in the L1: beyond its share per SM the traffic goes to the L2.
   - H-rate: a higher constant per access at every size.

## store_lines — what a store costs at the chip (question 19, question 18, question 15)

1. Mechanism: inventory §6 chip store limit.
2. Why now: l2mix: stores cost by the extent touched: 2 × 16 B = 1 × 32 B contiguous (3765, 3826), 32 B and 16 B at 64 B apart alike (23015, 23057), 4 B contiguous nearly free (414).
3. Sweep: S stores of W bytes, lanes D apart, grids 128, 256, 512.
   - H-bytes: by width (replay).
   - H-lines: by 64-byte lines touched, a burst then a rate.
   - H-128: by 128-byte lines.
