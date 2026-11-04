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

## After the run (round 7, step 4–5)

r7b valid: clock 2.491 → 2.494 GHz (+0.1 %), canaries +0.07 / +0.5 %.

**rot_depth** (evict-first loads as `dram_rotate`): 7.5 MiB per launch 6.7 µs at a window of 16 MiB, 15.9 at 24, 31.0 at 32, 35.1 at 48, 36.2 from 96; 15.7 MiB: 11.8–13.5 up to 24, 65.1 at 32, 71.1 at 48, 72 from 64. H-curve survives, H-chunk rejected: a function of the window, cold from 48–64 MiB; the warm curve holds (0.85 at 32 MiB). The kernels keep more: decode 0.82 of cold at 67 MB, RMSNorm without stores 0.45 at 32 MiB, not explained. Disagreement: these `LDG.E.EF.128` loads follow the plain curve, while r3's `ld.global.cs` curve is cold from 12 MiB (question 26).

**store_lines** (grid 512): by sectors touched: 128 sectors per block 3884 (32 B), 3873 (16 B), 4054 (4 B), 4005 (32 B at 64 apart); 256: 8873, 8902, 8614 (4 B at 32 apart), 10644; 384: 13802. Slope 420 B/clk, zero at 470 KB. H-bytes rejected (4 B and 32 B alike per sector). H-lines and H-128 rejected: 128 sectors cost the same whether they fill 32 or 128 lines of 128 B. Third slope: 512 sectors 23204, 768 83406, 1024 148841: each MiB above about 8 costs 1 / 79 B/clk, the DRAM write rate: dirty lines leave the L2. r1's 122 B/clk was TileLang's strided lanes at 32 and 48 blocks.

**spill**: single warp pace within −24 to +9 % of the replay at one block; 48 blocks at 256 threads 7.4 cycles per stack instruction per SM (one block 1.8): H-rate rejected, a chip limit at the L2's rate, 830 B/clk. Beyond 192 KB per SM at 48 blocks 23–25 cycles: 270 B/clk; 96 KB of shared memory does not lower that capacity (two blocks then fit): H-L1 survives in part: the stack's own capacity, not the L1 left beside shared memory.

A regression the first rule broke: `store_tail tl_b48` −38 %: TileLang's tile store spreads each instruction over 8 rows and keeps 244 B/clk of sectors; lanes wider than 4 KiB take that rate, unknown lanes the width rates (D790, question 25).

**Ablation, one change off, `--sets batches`** (all rules on: HGEMM warm 3.1 % / −6.9 %, 0.77 / 4; rotating 6.6 % / −9.1 %, 0.85 / 4; directions 77 of 98):

| Off | Moves |
| --- | --- |
| stores by sectors and buffer (D790) | HGEMM warm 2.2 % / −5.3 %, 0.77, 4; rotating 5.6 % / −7.6 %, 0.93, 2; FP8 −7.3 %; RMSNorm −29.4 %, graph −36.3 %, softmax graph −27.6 %, st256 14.7 %, l2mix 34.4 %, store_lines 49.9 %. Stays: probe-backed, question 25 |
| read share and overlap (D791) | RMSNorm graph −23.7 %, directions 76; no other test row moves |
| stack path (D792) | blind HGEMM −59.6 / −57.5 %, inversions 7 / 6 (on: +32.3 / +38.9 %, 6 / 5) |
| rotating by window (D793) | decode +13.3 %, sequence +16.9 %; directions 79 |
