# Batch r16: entry blocks

Written before the run. Engine: the repaired engine of round 15 (no change to `src/` since 1b9cb9c); no parameter comes from a blind row of this batch. Preflight as D854; three passes as D847. This batch carries the matrix sweep of LOOP §3a (D874): the memory hierarchy first, each sweep a row or column of the matrix, not one question each.

## Blind, the fixed recipe (D874)

1. Round 14's families and variant kinds, including T 512 where the launch allows it. New shapes by a rule chosen before any result: off HGEMM a factor 7 in every size, HGEMM a factor 25 in M: HGEMM 3200×1280×1600 (10 variants, warm and rotating), GEMV 7168×2688 group 56 (9), FA 2240×7×112 (10 asked, what the launch allows), decode 28×2240×112 G 7 (8), RMSNorm 1344×2240 (6 thread counts), each warm and rotating; softmax 1344×2240 (6) warm; 17 own kernels in `perfmodel/testset/hand16/`. Predictions committed before the run.

## mr_*, mk_* — read × DRAM × every pattern; what stays in the L2 (matrix: read × DRAM, 16 cells; retention, question 26 at a new form)

1. Mechanism: `dl_*`: 8 MiB read in order runs 66 µs, half the DRAM rate, replay 38, flat in bytes per lane; `dram_burst` 3.9 to 67 MB at the plateau. The two differ in kernel (`__ldcs` against plain loads, 48 blocks in step against 1024 blocks). Rotating GEMV, RMSNorm, decode, FA all sit here.
2. Sweep: order by footprint 1 to 64 MiB; rotation deeper ×2, ×4, ×8; plain against evict-first at T 32, 128, 512 at one width; lanes 64 to 4096 B apart; rows of 512 to 4096 B at 8 KiB pitch; hashed 8 and 32 MiB at T 128, 512; dram_burst's kernel plain and evict-first; 1, 12, 48 blocks of 256 KiB; retention: 8 MiB in 2, 3, 4, 6 sets, 4 MiB in 4, 6, 8 sets.
   - H-kind: the half rate belongs to plain loads: `mr_cs_*` at the plateau, `mr_cg_*` at 66 µs; `mr_b48_cs` fast, `mr_b48_cg` slow.
   - H-shape: it belongs to many blocks walking apart: `mr_b48_*` both fast, `mr_cs_*`, `mr_cg_*` both slow.
   - H-small: a launch below some size pays a fixed start: time = a + bytes / plateau: `mr_o_f*` a straight line with an intercept near 30 µs, independent of pattern.
   - H-evict: replacing lines of the previous set costs: the excess follows bytes in the L2 at the launch, `mk_*` rise with sets past 16 MiB in total.

## mt_* — rows warm by threads (matrix: read, write, rmw × L2 × row × c; RMSNorm 1920×1792 T 256 +30 %, first pick's loss 36 %)

1. Sweep: 1920 rows of 4 KiB, T 32, 128, 256, 512: loads, stores, loads then stores, loads then a block barrier.
   - H-load: T 256 costs on loads: `mt_r_*` rise with T. `dl_*` warm says no.
   - H-store: on stores: `mt_w_*` rise.
   - H-sync: on the block's barrier: only `mt_rb_*` rise.
   - H-kernel: none rises: it lies in the reductions' code path, which the rows replay already.

## mw_*, mx_* — write × L2 × cap, write × DRAM, RMW (matrix: write × L2 × cap × c, question 51 second experiment; write × DRAM; rmw × L2, DRAM)

1. Sweep: 5 to 12 MiB written alone and with as much read once beside them, by T 32, 128, 512, 1024 at 7 MiB; 16, 32, 64 MiB in order; 16, 64 MiB hashed; lanes 4 KiB apart; in place 4 to 32 MiB.
   - H-bytes: a step at a fixed byte count: same step in every row.
   - H-blocks: `wr_r1w1` 112 blocks steps, softmax 1920 blocks does not: the step moves with T at 7 MiB.
   - H-reads: only with reads beside: `mw_o_*` no step, `mw_rw_*` a step.

## ms_* — bytes a thread holds by 32 (matrix: read × L2 × row, read and write × shared × row, copy × L2 × row; question 48's mechanism)

1. Sweep: D 48 to 128 by 16: 16-byte global loads, shared stores and loads, 16-byte copies, each D bytes apart per thread.
   - H-global: the loss sits in global accesses: `ms_g_*`, `ms_c_*` saw-tooth by D mod 32.
   - H-shared: in shared memory: `ms_s_*` saw-tooth.
   - H-none: all flat: it sits in the mma's operand path, tensor load × shared.

## ml_* — L1 capacity and replacement (matrix: read × L1 × cap, inventory absent row)

1. Sweep: 8 passes of P KiB, 16 to 384, a block per SM; 32 to 128, two blocks per SM; hashed at 128, 192.
   - H-128: LRU over 128 KiB per SM (the engine): at 160 a cyclic pass hits nothing.
   - H-192: the stack's 192 KB: the knee sits there.
   - H-rand: not LRU: past the knee hits fall slowly.

## md_*, mh_*, mst_* — DRAM channels and banks, TLB, loads apart (matrix: read × DRAM × stride, hashed, cap; read × L2 × stride, hashed; own kernels with loads apart)

1. Sweep: 32 K lanes 128 B to 32 KiB apart; 256 K lanes hashed through 1 MiB to 1 GiB, T 128, 512; warm 4 MiB lanes 64 to 4096 B apart.
   - H-flat: the rate is the plateau's at every stride past the L2.
   - H-bank: rate falls at the strides that keep one channel or bank.
   - H-tlb: hashed rate falls past a reach, a latency of about 400 ns per miss.
   - H-lines: warm hashed and strided loads cost per line at the chip's L2 rate (14 lines/clk, D835), and the replay holds.

## copy_cold — copy × DRAM × order (the copy rate on DRAM copies, D869; rotating rows +10 to +18 %)

1. Sweep: copy_two's kernel, source 16 MiB on per launch through 512 MiB, K 4 to 64, B 1 and 2, T 32, 128, 256.
   - H-same: the same 10 clk per copy on an SM, cold.
   - H-free: issue does not wait for DRAM copies at that rate: lower than 10 clk per copy.
