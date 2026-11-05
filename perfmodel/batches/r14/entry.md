# Batch r14: entry blocks

Written before the run. Engine: round 13's table commit 52e182f in behaviour (tools and generator only since). No parameter may come from a blind row of this batch. The run starts with the preflight (D854): every kernel launched once at its own size, with a marked guard after each buffer; a kernel that faults, leaves a guard or hangs is dropped and listed. A trial of the same preflight before the predictions dropped six of 150: `dec14_bn128_*`, `fa14_c128x128_s2_t256_*`, `fa14_c64x128_s2_t128_*`, invalid launch (shared memory); their directories are out of the prediction. The guard was checked on a copy of `rf_w4_m4` with a buffer 1 KiB short: "left buffer 0", exit 3.

## Blind score of rules D848-D853

1. Why now: the rules of round 13 have no blind score.
2. New rows, predictions committed before the run (`predicted.json`): GEMV 5120×2560 (9 variants), FA 2560×6×112 (6), decode 40×2560×112 (7), RMSNorm 1280×2560 (6 thread counts), each warm and rotating; softmax 1280×2560 (6) warm; HGEMM 2304×1792×1792 (10 variants, warm and rotating); 15 own kernels in `perfmodel/testset/hand14/`: atomics at uniform addresses (countblock, minblock) and varying ones (histlane, xorwide, exchring), stores wide and hashed (stridefill, brevfill, hashput, maskput), records (recordinit, recordpatch, recordwide), stack with loads (stackscan, stackfilter, stackmerge), with 10 one-change rows.
3. Three passes, D847, each after a cool-down with its own clock (D846). The first is the first score.

## rf_* — records, repaired (question 49, first GPU work)

1. Mechanism: D838: more than 4 lines per instruction do not merge. `bl13_recordfill` +232 %, `bl13_tilefill` +99 %.
2. The grid was four times the buffer in r13c; now 64 blocks per MiB.
3. Sweep: W of 16 words of a 64-byte record per thread, over 4 and 16 MiB.
   - H-now (replay): each store pays its sectors: time rises with W.
   - H-next: a store merges with the sector its warp just wrote: time goes by whole lines, nearly flat in W past 8.

## stl_* — stack reads, index per lane (question 44, second experiment)

1. Mechanism: stack path by sectors, D852 for stores; `bl13_window` −74 %.
2. Sweep: N 16 to 1024 words of frame, 4 N reads, `u` a uniform index, `l` one per lane.
   - H-now (replay): reads at the plain stack rate whatever the index.
   - H-lanes: reads with an index per lane cost per lane as stores do (5.5 sectors per clock chip-wide): `l` several times `u`.
   - H-frame: the cost goes with the frame past the SM's capacity, the index does not matter.

## qs* — what a third stage costs (question 47, first experiment; questions 32, 36)

1. Mechanism: measured s3 7-9 % above s2 warm on four shapes, 3072×1536: 213.5, 229.4 µs; replay 216.8, 201.6. Without compute the gap goes (r10 148, 149), without stores it stays (193, 209). The loop: barrier; copies of tile k + s − 1; wait for tile k; barrier; compute k. Replay predictions in `predicted.json`.
2. Why now: first pick's loss on HGEMM warm. The tile trace perturbed s2 by 6 %; none of these kernels stamps anything.
3. Rows at 3072×1536×2048, s2 and s3 unless named, one change each:
   - `qs_s2_smem3`: s2 code with s3's shared memory. H-smem: the size of the allocation costs: 229 µs. Others: 213.
   - `qs_s2_reg12`: s2 with twelve more live words, the registers of s3 (202 against 190; the trace added 5 and s2 rose 6 %). H-regs: 229. Others: 213.
   - `qs_s2_nocopy`, `qs_s3_nocopy`: the rotation and barriers without a copy. H-code: the gap stays, about 16 µs. H-load: the gap goes with the copies.
   - `qs_s2_b1`, `qs_s3_b1`: the barrier before the copies out. H-barrier: s3 falls to s2; the stall counter on barriers is 2.4 times s2's in r10.
   - `qs_s3_wait1`: compute waits for one tile more. H-depth: the third tile in flight is what costs: the row moves towards s2's behaviour with a tile less of lookahead. Others: unchanged or slower.
   - `qs1_s{1,2,3}`, `qs2_s{1,2,3}`: 48 and 96 blocks, one and two per SM throughout. H-share: the gap exists only with two blocks on an SM. H-own: it exists at one.
