# Batch r9: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json`; each hypothesis's replay of the probes is in `hypotheses.json`. Every probe's SASS was checked after a compile: the hot loop holds the intended LDG, LDGSTS, clock reads and barrier. The engine carries this round's rules as `assumed` until the batch decides them.

## Selection (LOOP §3, supervisor focus)

Found before the batch, no GPU needed:

- **τ split.** Two engines on two measurements of the same rows, HGEMM 2048³: warm 0.60 (r4) → 0.56 (r8) under both the round 7 and the round 8 engine, so the whole warm drop is the re-measurement. Rotating: 0.73 under the round 7 engine on both; the round 8 engine 0.73 on r4, 0.64 on r8.
- **The ceiling.** τ between repeats of the same rows: warm `kernel` 0.87–0.96 (5 batches), `graph` 1.00; rotating `kernel` 0.82, `graph` 0.87. "At the measurement's ceiling" does not hold for warm.
- **Rotating over warm follows the waves of blocks.** Measured +17 to +22 µs on the rows with 6 waves (one block per SM at 256 tiles, two at 512), +4 to +6 at 3 waves, −0.5 on panels. The replay spreads B's misses evenly over the launch; they sit in the first wave, at a rate the chip's DRAM limits.
- **NVFP4:** the sign branches are per lane, on random data; a warp runs both arms. With both arms the replay's instructions equal the counter exactly (312596).

## first_reader — a second reader of a line on its way (questions 8, 27)

1. Mechanism: inventory §5 L1 and DRAM latency; a miss-status entry per line (absent).
2. Why now: RMSNorm and softmax graph −22 %, several warps read W's lines together; the replay gives a second reader an instant L1 hit.
3. Sweep: lead warp, follower Z passes later (about 24 cycles a pass), 4 lines, LDG and cp.async, warm and rotating, with the follower alone as control.
   - H-pending (replay): the follower waits for the fill, a hinge: cold LDG 1137 at Z 0, 744 at 16, 360 at 32, L1 hit 88 from 48; warm 379 at 0, 178 at 8, 88 from 16; cp.async cold 1434 at 0, 657 at 32, 366 from 48.
   - H-nowait: an instant hit from Z 0: 88 (LDG).
   - H-own: the follower sends its own read: cold 1437 cp.async flat until the lead's fill lands, then 366.

## first_share — misses of shared tiles at the chip's scale (questions 30, 32; rows HGEMM rotating, s1 rotating)

1. Mechanism: inventory §5 DRAM queue per SM: even spreading; cp.async path.
2. Why now: rotating HGEMM −4 to −9 %, s1 rotating replay +18 %, measured +1 %, τ 0.64.
3. Sweep: `share`: 48 blocks, D reading each tile (1–48), Z 0, 256. `seq`: 2 and 4 waves of 48 blocks at one block per SM, the same tiles. `pieces`: one block, stride S 16–128 along a line.
   - H-pending (replay): rot d48 z0 38.3 k, z256 137 k; seq first 134.6 k, last 21.4 k; pieces s16 6734, s128 7670.
   - H-even (misses spread over readers and waves): rot d48 z0 25.9 k, z256 124.6 k; seq w2 first 67.2 k, last 67.3 k.
   - H-first (one reader misses, the others hit at L2 latency): rot d48 z0 21.9 k, z256 120.6 k; seq as H-pending.
   - H-sector (a miss brings its 32-byte sector, not the line): pieces s16 half the steps miss, 7135; s32 and up 7670. H-line (the replay): s16 6734, s32 6868, s64 7135, s128 7670.

## New test rows

- **Blind group `hgemm 1536x3072x2048`** (13 variants, warm and rotating): a shape not on the scorecard, N twice M; predictions committed before the run.

## Re-measurements

`hgemm2048_*`, `hgemm_m3072*`, `hgemm_m1024*`, `hgemm_m4096*`, `fa_s1536*`, `nvfp4gemm*`, `wi_hg_*`: repeats for the ceiling (every group gets a second batch), and what-if bases measured in one batch with their changes.
