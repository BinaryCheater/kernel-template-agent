# Batch r18: entry blocks

Written before the run. Engine: round 17's; `sim.loose_part_only` and `sim.ldg_continues` are in the code, both off, and change no replay; no parameter comes from a blind row of this batch. Three passes, cool-downs as D847; blind rows with the main counter set only, sweeps `dp_*`, `gq_*`, `tw_*` timing only (D894). Replay-only finding of step 2 (seen rows, overrides, before any probe was designed): one unknown execution of a load discounted all its executions: D895.

## Blind, the fixed recipe (D874)

Families, variant kinds and what-if axes as round 17. New shapes by a rule chosen before any result: off HGEMM a factor 11 in every size, HGEMM a factor 29: HGEMM 3712×1392×1856, GEMV 8800×3168 group 88, FA 2288×11×112, decode 33×2200×112 G 11, rows 704×5632. D is 112, the multiple of 16 nearest 110, to keep FA's large tiles under the shared-memory limit that cut round 17's group. 13 own kernels and 4 what-ifs in `hand18/`.

## dr_*, dp_* — read and copy × DRAM × row: decode rotating by tile (question 55, experiment 1 of 2)

1. Mechanism: `dec14`, `dec15`, `dec16`, `dec17` rotating: BN 64 runs at about 160 GB/s of footprint, BN 32 at about 200; warm alike. Replay flat. Bursts per instruction alike (bf 1.106 both, counted per instruction, r17 SASS). BN 64 copies 16 bytes per lane, BN 32 8; rows are D × 2 bytes long, so a 512-byte copy spans 2 or 3 rows and a 256-byte copy 1 or 2.
2. Sweep: 27×2160 at D 96, 112, 128, BN 32, 64 and BN 16 at D 144; then the same runs as plain loads: 16- and 8-byte lanes, runs of 288, 256, 64 bytes, pitch 1296, and contiguous.
   - H-span: the rate falls with rows one instruction spans: a BN 64 gap at D 96, 112, 128, growing as D shrinks; `dp_w16_r288_p1296` slower than `dp_w8_r288_p1296` by about 20 %, as much with `r64`.
   - H-copy: the gap is the copy path, not the DRAM's: `dp_*` alike by lane width; gap only on `dr_*`.
   - H-align: the gap needs rows not at a multiple of 64 (D 112, 144): D 96, 128 show none.
   - H-L2: BN 64 reads more sectors than the footprint (counters: 13.3 against 12.2 MB): the time follows L2 read sectors: `dr_*` in line with counters, about 5.3 µs per MB.

## sp_* — rows past the register limit (question 56, experiment 1 of 2; cell compute/stack × c)

1. Mechanism: `rms17_t16_warm` 47.5 µs against T 32 15.8, replay 25.8; `sm17_t16_warm` 58.8, 19.9, replay 35.5. Counters: registers 255, stack operations 9 times T 32's, L2 writes 380 k sectors against 208 k, reads 264 k against 222 k.
2. Sweep: T 16 with 1024 to 6144 words a row, T 32 at 6144, 9216, softmax two sizes, about the same bytes.
   - H-rate: stack stores past the SM reach the L2 at the chip's stack rate: time rises with stack bytes from the first spill, linear in words past the register limit, slope about 5.5 MB / 30 µs.
   - H-step: a step at the point the stack passes the SM's local capacity: flat below, jump above.
   - H-lanes: T 16's half-empty warps: T 32 at twice the words is as slow.

## gq_*, tw_t16_g0, tw_t64_g0 — GEMV's load at a pitch 16 mod 64 (question 52, experiment 2 of 2; read × DRAM × order, row × c)

1. Mechanism: after D895 `gemv17_*` rotating rows follow bytes per instruction, 24/26 real pairs, but `np8` rows sit −18 %: 52.5, 65.7 against 43.3, 53.7. T 32's twins 1.4 not explained.
2. Sweep: 8 MiB at pitch 1296: 4 bytes × 32 lanes, 1, 2, 4, 8, 16 warps a block, one row each; 2, 8 bytes and 16, 64 lanes at 8 warps; twins at T 16, 64.
   - H-edges: time by bytes per instruction alone, warps do not matter: all `gq_w4_r32_np*` alike; replay after D895 right.
   - H-warps: more warps of a block on neighbouring rows cost more: time rises with np, about +20 % at np 8.
   - H-twins: T 32's 1.4 is a count of warps asking at once: T 16 and T 64 on a line through 1.0 and 2.0.
