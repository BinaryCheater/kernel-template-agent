# Batch r15: entry blocks

Written before the run. Engine: round 14's rules with the lanes as a class of integer instructions (22af4e2 onward); no parameter comes from a blind row of this batch. Preflight as D854; three passes as D847.

## Blind score of rules D854-D864 and the lanes

1. Why now: the rules of rounds 13 and 14 have no blind score. The lanes were rewritten as a class; this is their first test too. The check of this round finds 99.8 % of memory operations resolved on the seen kernels; the blind rows are the test of that number at address forms not seen.
2. New rows, predictions committed before the run (`predicted.json`): HGEMM 2688×1920×1728 (8 variants, warm and rotating); GEMV 6144×3072 group 48 (7 variants), FA 2304×7×104 (7), decode 36×2304×104 with 12 groups (7), RMSNorm 1920×1792 (6 thread counts), each warm and rotating; softmax 1920×1792 (6) warm; 14 own kernels in `perfmodel/testset/hand15/` with 4 one-change variants: division by 7, by a run-time count, remainders, a packed triangle, a Morton curve, clamps, a select between buffers, bit counts in the index, a negative stride, 64-bit products, an index through a float, a periodic stencil, non-power chunks, a 28-byte record.
3. The HGEMM part is also the test of the fit `copy_inflight_sm` (D864): first pick's loss at a new shape, warm; before the fit it was 10.7 % blind at 2304×1792.

## copy_two — copies on their way, one and two blocks (question 47 by a new row: fit on a design variable)

1. Mechanism: `sim.copy_inflight_sm` 56, fitted on `qs*`: with two blocks on an SM, copies from the L2 not yet landed hold an SM at 56; the next copy's issue waits. D816 found no depth with one block. The capacity was never read.
2. Why now: a fit on a design variable; LOOP §4.6 says a mechanism must replace it.
3. Sweep: 48 B blocks of 128 threads (4 warps) with 48 KiB of shared memory, B 1 and 2 per SM; each warp issues K copies with no wait, K 2 to 64; warp 0's cycles to issue, median over blocks and rounds. 32 threads at K 4, 16, 64.
   - H-sm: a queue of 56 per SM whatever the blocks: issue per copy steps up at 8 K B near 56: K about 14 at B 1, K about 7 at B 2.
   - H-block: 28 per block: K about 7 at both B.
   - H-two: the fit as written: B 2 only, K about 7; B 1 no step up to K 64 (D816).
   - H-none: no step at any K: the third stage's cost is not a queue on issue.
4. No replay case: the row is read from warp 0's stamps; a builder follows if a rule comes of it.

## dl_*, gd_*, rd_* — what differs at the DRAM limit (question 50, second experiment)

1. Mechanism: reads at the DRAM rate `dram_r_GBs`; rotating GEMV 5120×2560 measured 31.1 to 35.2 µs, RMSNorm 58.8 to 62.9 µs, replay flat within 1 %, real pairs 5/18 and 2/5. All variants read the same sectors (counters 238 k, 213 k).
2. Sweep `dl_*`: 8 MiB read in order, rotating, bytes per lane 4, 8, 16, threads 32, 128, 512. `gd_*`, `rd_*`: the rows `gemv14_{b,mk16,np2_r64}`, `rms14_t{16,512}` with a rotation four times deeper.
   - H-retain: what the last launch leaves in the L2 differs by variant: at depth 4 the rows become flat, and `dl_*` are flat.
   - H-request: the rate depends on requests, not bytes: `dl_*` fall with bytes per lane at fixed T, by the gap of `gemv14_mk4` to `mk16` (6 %).
   - H-flight: the rate depends on streams in flight: `dl_*` move with T at fixed bytes per lane; `gd_*` keep the gap.

## fad_* — FA by D (question 48, second experiment)

1. Mechanism: t128 rows lose 11 to 30 % at D 80, 112, 2 to 5 % at D 96, 128, with R1 exact and barrier stalls 12 times t64's. At t128 a thread holds half a row, D bytes: 80 and 112 bytes per thread do not fill 32-byte sectors; 96 and 128 do; at t64, 160 and 224 bytes do.
2. Sweep: S 768, H 4, c64x64 s2 t128, D 72 to 128 by 8; D 80, 96 at t256.
   - H-sector: the loss goes with D mod 32: D 96, 128 small; 72, 80, 88, 104, 112, 120 large; t256 D 80 (40 bytes per thread) large, D 96 (48) large too.
   - H-odd: the loss goes with chunks of 16 bytes per thread odd: D 72, 88, 104, 120 and 80, 112 alike, as H-sector; t256 D 96 (3 chunks) odd, D 80 (2.5): tells them apart only there.
   - H-smem: the loss goes with the block's shared memory: rises smoothly with D.

## ct_* — compute alone at block turnover (question 47, the part the fit does not touch)

1. Mechanism: compute without copies −10 % at block turnover, −3 % without (r14).
2. Sweep: HGEMM 1536×3072 c128x128x32 s2 t128, copies out, K 480 to 3840: blocks fixed, tiles double each step.
   - H-start: a cost per block: the intercept of time over K is above the replay's by the missing time.
   - H-tile: a cost per tile: the slope is above the replay's.

## T 512 warm (question 50)

`dl_w8_t512_warm`, `dl_w8_t128_warm`: 8 MiB warm at T 512 and T 128: whether the −21 to −34 % of T 512 warm rows shows on plain loads.
