# Batch r19: entry blocks

Written before the run. Engine: 587202a with D903 (found by the work check on rows of r13 to r18, before any blind row of this batch): `threadIdx.x` wraps at the block's first dimension, guards and exits per lane; D895 to D898's first blind score. No parameter comes from a blind row of this batch. Three passes, cool-downs as D847; `counter_sets` main; strip-downs with counters: they read the L2's sector hit rate.

## Blind, the fixed recipe (D874)

Families, variant kinds and what-if axes as round 18. New shapes by a rule chosen before any result: off HGEMM a factor 13 in every size, HGEMM a factor 31: HGEMM 3968×1488×1984, GEMV 10400×3744 group 104, FA 2704×13×112, decode 39×2600×112 G 13, rows 832×6656. D stays 112, as round 18, for FA's large tiles under the shared-memory limit (a rule of D would give 128). 13 own kernels and 4 what-ifs in `hand19/`, bounds checked by hand: round 18 lost 6 to sizes of mine.

## gs_* — LOOP §3b step 4 on GEMV rotating's wrong pairs

After D903: 72/99 real pairs on seen shapes; 27 wrong, 22 of them in 3072×6144, 5120×2560, 6144×3072, where `r64`, `mk16`, `np2_r64` sit +16 to +26 %, `b`, `r16`, `mk4` about 0. Step 2, no value flips a pair (5 pairs, 8 parameters, ±30 %, `dram_read_GBs` moves every row alike): structure. Step 3 is in the group: from `b` the knob `reduce` 64 and the knob `mk` 16 add 12 % in the model and take 5 % off measured. By replay (seen rows): the charge rises 7.38 → 8.20 MB in the scale table: `b`'s one U16 load pays 1.0 of its sectors, `r64` and `mk16`'s five loads 2.0: lone sectors, twins counted per instruction (D886).

Sweep at 5120×2560 rotating, the four rows of those pairs (`b`, `r64`, `mk16`, `np2_r64`) × without the scale table (`nosc`), without the matrix (`noa`), without the packed words (`nobq`), and the scale table alone (`onlysc`). 2 µs timer steps at these lengths (D901): gaps under 5 % are not read.

- H-sc: the model's gap between `b` and `r64`, `mk16`, `np2_r64` is the scale table's: `nosc` rows alike in the model and measured; `onlysc`: replay 2× apart by variant, measured alike, near bytes / 237 GB/s at both.
- H-a: the gap holds without the table: `nosc` still 12 % apart in the model, and measured falling as `r64`.
- H-none: `onlysc` measured 2× apart as the replay says: the table's lone sectors are real and `r64` gains elsewhere.

## rs_* — LOOP §3b step 4 on rows warm

`rms18` warm −38 to −53 % at T ≥ 32, `sm18` warm −27 to −42 %, `rms17` −17 to −31 %, `rms16` less: the gap grows with the footprint; work check clean: reads 375/376, writes 352/352 sectors per block, instructions 1.00, rotating rows within +13 %. So the model does the same work and the time is elsewhere. r17's `rr_*` (896×3584) found 1 to 2 µs in every part, at a smaller shape.

Sweep at 704×5632: T 128, 512 without stores, loads, compute (`rs_nostore`, `rs_noload`, `rs_nocompute`), and rows 88, 176, 352 at T 128 (`rs_m*`).

- H-foot: the gap needs X and Y together near the L2 (15.9 MB, `l2_effective_bytes` 16.8): `rs_m*` on the replay's line, `rs_nostore` near the replay's 8 µs, sector hit rate below 100 %.
- H-store: the writes alone: `rs_noload` keeps it, `rs_nostore` does not.
- H-load: the warm reads alone: `rs_nostore` keeps it, `rs_noload` does not.
