# Batch r13b: entry block

The timed run of r13, a third time. r13 and r13a are invalid and kept: clock 2.494 to 2.454 GHz, then 2.474 to 2.437 GHz; the limit is 1 %. Neither was imported and no score was read from them. The clock follows temperature, about 0.002 GHz per degree, so three passes in a row leave the band. Here: the GPU below 45 C first; the batch's clock and canaries around the first pass alone, which gives the first scores; each repeat after a cool-down, with its own clock; the profiler last, on the new rows only. The same binaries, the same predictions. `dec13_bn128_*` and `fa13_c128x128_s2_t256_*` left out: they fail at launch; those groups keep 7 variants.

Written before the timed run. Engine: round 12's table commit bacbba0 in behaviour (tools only since). No parameter may come from a blind row of this batch.

## Blind score of rules D838-D843

1. Why now: round 12's rules have no blind score, and round 12's blind ranking rests on one HGEMM shape.
2. New rows, predictions committed before the run (`predicted.json`):
   - design groups off HGEMM: GEMV 3072×6144 (9 variants), FA 1792×10×80 (8), decode 24×3072×96 (8), RMSNorm 768×3072 (6 thread counts, powers of two: TileLang's reduce refuses others), each warm and rotating; softmax 768×3072 (6 thread counts) warm;
   - one more HGEMM shape, 1792×2560×2304, 16 variants warm and rotating, 8 one-change rows;
   - 21 own kernels in `perfmodel/testset/hand13/`: stores (transpose, interleave, tilefill, recordfill), rewrites (smooth, swaprows, blockcount, decay), atomics (wordcount, blockmin, bitset, ticket, casmax, floatadd), hashed addresses (probe, cuckoo, permute, bloomadd, bucket), stack with loads (insort, window), with 22 one-change rows;
   - one-change rows for compute, loads and stores removed on GEMV, FA, RMSNorm, softmax, HGEMM.
3. Every row of the batch is timed three times, each in its own process: in order, in reverse order, after the profiler passes (`repeats` in measured.json). The first pass is the first score. What-ifs are read pass by pass, base and change from the same pass.

## Repeats of round 12's rows

`bl_*`, `wi12_*`, `wi12b_*`, `wk_*`, `hgemm_m2560n2048k1536_*`: rebuilt from the same source, not predicted, their first scores stand. They give round 12's 75 what-ifs three more passes in one batch, to separate batch noise from model error (brief, step 3).
