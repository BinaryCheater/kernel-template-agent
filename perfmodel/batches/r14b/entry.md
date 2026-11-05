# Batch r14b: entry blocks

Written before the run, after r14. Engine at this commit carries this round's first rules. Only sweeps: no blind rows. Preflight trial: 29 of 29 ok.

## occ_* — blocks per SM by threads (question 50)

1. Mechanism: `max_threads_per_sm` 1536, "spec: driver", never probed: at T 512 three blocks per SM, at T 256 six, at T 128 twelve. The occupancy counters give registers and shared memory only.
2. Why now: RMSNorm and softmax T 512 warm −15 to −34 %, first pick's loss 22 % in r13b; an override to 1024 on those rows takes them to −9 to +7 %. That is a hint from test rows, so it needs a probe and no value comes from them.
3. Sweep: one warp of each block chases 64 loads, the rest wait at a barrier, so a block's life does not depend on how many others share the SM; 48 k blocks.
   - H-1536: one wave up to k 3 at T 512, 6 at T 256, 12 at T 128, 24 at T 64; a second past it: k 4, 7, 13, 25 about twice k 1.
   - H-1024: one wave up to k 2, 4, 8, 16: a second wave already at k 3, 5, 9, 24.
   - H-other: another limit, read from the steps.

## qs*_nocopy, qs2_*_b1, qs_s*_late — the third stage (question 47, second experiment)

1. r14: the gap goes with no copies, goes with the first barrier out, needs two blocks on an SM; shared memory, registers, the wait's depth do not matter. Compute alone runs 198 µs, replay 180.
2. `qs1_s2_nocopy`, `qs2_s{2,3}_nocopy`: compute alone at one and two blocks per SM throughout. H-pair: the replay's −10 % is there only at two blocks: mma between blocks. H-tile: it is at one block too.
3. `qs2_s{2,3}_b1`: the first barrier out at two blocks per SM, no block turnover. H-turn: the barrier's cost in s3 needs a block's end and a new one (qs2 keeps s3 ≈ s2 with or without it would say no).
4. `qs_s{2,3}_late`: the next tile's copies after the compute, only the wait before it. H-issue: compute waits on the issue of its own next copies: s3 falls to s2 or below. H-stay: the gap stays.
