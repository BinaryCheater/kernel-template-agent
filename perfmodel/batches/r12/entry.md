# Batch r12: entry blocks

Written before the timed run. Predictions in `predicted.json` come from the engine at the commit recorded there, with no engine change since round 11's table (`183ab16`, then D836 tools only). No parameter is taken from a `bl_*`, `hgemm_m2560*`, `wi12_*` or `stk_*` row: a miss enters the selection rule of the next change as an open question, the row's first score stays as it is.

## Blind set (28 §1, the number that counts)

- `bl_*`: 19 kernels of our own in CUDA, `perfmodel/testset/hand12/`, none seen by the round 11 rules: writes past the L2 (`memset`, `revcopy`, `scatter`, `aos`, `axpy`), atomics with new contention patterns (`atomblock`: all threads of a block on one word; `atomline`: each lane its own line, no hash; `atomret`: with a return value, which round 11 did not measure; `atommax`; `atom64`), store-heavy loops (`rewrite`, `fanout`, `maskstore`), mixed read-write-atomic (`mixrwa`, `rankpush`, `tally2d`), plus `gather`, `jacobi3d`, `csr`.
- `bl_*_<tag>`: 17 what-if pairs from those kernels.
- `hgemm_m2560n2048k1536_*`: a new shape, 16 variants, warm and rotating, each a what-if from `c128x128x32_s3_t128` as well; 3 ablations warm.
- `wi12_*`: GEMV, RMSNorm, softmax, decode and FA what-ifs at shapes no what-if used.

Known limit of the front end, left as it is for the blind score: `SM.atomic_bytes` is the launch's `write_bytes`, the first buffer (D834), so `mixrwa` and `tally2d` do not declare their plain stores' bytes.

## Repeats, not blind

`wi_hg_{b,bk16,t64,sw8}_warm`, `wi_gemv_{b,np2,r16}_rotating`, `wi_dec_{b,bn32}_rotating`, `wi_fa_{b,noexp}_warm`: the wrong what-ifs measured again beside their bases, r7's values being one batch. `wi_fa_noexp` has 26 × the L2 misses of its base on r7's counters, which the shape does not explain.

## wk_* — where the wrong what-ifs and t64 sit

1. Mechanism: inventory stack path (t64), mma stream (bk16), raster panels (question 2).
2. Why now: bk16 measured +2.3 / +5.3 %, replay −6.5 / −4.2 %, the replay's `queue (mma)` drops 21 µs; `wi_hg_t64_warm` −27 %, `l1` path 62 %; sw8 is the model's first pick at 3072×1536 and 1536×3072 warm, measured +0.5 %, replay −7.4 %, so the first pick's loss of 8 % is the panel's, not s2 against s3.
3. Sweep: bk8; bk16, t64, sw8 with compute and with stores removed.
   - H-compute: the replay's gain at bk16 is in compute: bk16_nocompute against wi_hg nocompute equal.
   - H-store: the replay's gain at sw8 is in stores (`queue (stg)` 25.5 → 16.1 µs at 3072): sw8_nostore equals nostore measured and replayed; replay's gap goes.

## store_merge — which stores merge in the queue (question 40; `store_spread L2_R4096_S8` −59 %)

1. Mechanism: store merge in the queue (D833), `stg_merge_run_clk` chosen.
2. Why now: row −58.8 %, 85 % `store_queue`; `rw_*` G 768 +68 to +112 % asks for more merging, this row for less.
3. Sweep: L lanes per row 1, 2, 4, 8, 16, 32; step S 4–32 B along the row, rows 512 or 4096 B apart; G 128, 512. L2 write sectors from ncu at G 512.
   - H-now (replay): a warp's next store on its own sector within 32 clk merges, whatever the rows: L2 S8 and L2 S4 as cheap as L2 S32-equivalents' single pass, L2 sectors = distinct.
   - H-row: merges only inside one line per instruction (L 32): L2 S8 pays 8 × 16 sectors, about 13 000 cycles at G 256; L2 sectors = L1 sectors below L 32.
   - H-entries: E open sectors per SM: merging holds while an instruction's sectors ≤ E, then stops: a step in L.

## stk_* — stack path by threads and resident blocks (D800; `wi_hg_t64_warm` −27 %, 256×256×32 t256 −22.6 %)

1. Mechanism: stack path, `local_capacity_bytes` 192 KB per SM, rates empirical on spill_rw at 256 threads.
2. Why now: two rows on the `l1` path, both model fast, at 64 and 256 threads.
3. Sweep: T 64, 128, 256; N 64, 256 words; G 48, 144; N 1024 at G 96, T 64, 128.
   - H-now (replay): LRU of 192 KB per SM.
   - H-warp: capacity per warp slot: T 64 at N 256 (32 KB per block, 2 warps) misses like T 256.
   - H-rate: rates alone are off: error flat across N.
