# Batch r11: entry blocks

Written before the timed run (write-probe). The engine's replay of every case is in `predicted.json`, on commit recorded there. SASS checked after the compile: `ld`, `cp` hot loops hold LDG and LDGSTS; `k_atom` holds REDG.E.ADD; `k_rw` 17 STG. All causes below are inferred, none probed before this batch.

## Found before the batch, no GPU

`hand_stencil`: 5472 'ldg address unknown' came from lanes the tracker could not split: `SEL` on a compare, `VIMNMX`, `@P2 IADD`. Compares of exact lanes are now kept per lane; 0 unknown, R1 instructions and sectors match. The row went −38 → −43 %: the address was not the cause. 32 MiB (16 read + 16 written) on a warm launch against a 16.8 MB L2: `warm_miss` looks at the read footprint only and writes are not DRAM traffic on warm rows. Same family as `hand_triad` below.

## l2_cap — the chip's L2 read rate by pattern (question 37; `load_sat` +765 %)

1. Mechanism: inventory §5 chip L2 share, `l2_aggregate_B_per_clk_cpasync` 817.7 frozen `unidentified`; port 16.7 clk per 512 B (30.7 B/clk).
2. Why now: `load_sat` row +765 %, both share rules rejected in r10 (D817), the probes never reached saturation on hits only (r10c rates partly L1).
3. Sweep: every warp of 12, 48, 144, 480 blocks reads a warm 8 MiB without a pause: LDG, 4-byte lanes 4, 32, 128 B apart (1, 8, 32 lines per warp instruction); 16-byte copies, lanes 16, 64, 128 B apart. Rate = bytes / (span − launch).
   - H-port: per SM cap only: g 48 and up 48 × 30.7 = 1470 B/clk, all patterns equal.
   - H-817 (replay): chip 817 B/clk all patterns, g 144 equals g 480.
   - H-bytes-1900: chip about 1900 B/clk at any pattern (load_sat's reading).
   - H-lines: the chip takes L lines per clock: rate falls with lines per instruction: ld128 at 4/128 of ld4's byte rate if L alone limits.
   - H-sectors: S sectors per clock: ld32 and ld128 rate equal, below ld4 by 4/32 × 128 / 8.

## wr_* — stores and reads past the L2 (`hand_triad` −37 %, `hand_stencil` −43 %, question 17)

1. Mechanism: inventory §5 DRAM rate, `dram_writes_paced`, `stg_dirty_bytes`; L2 warm-miss curve by read footprint only.
2. Why now: two hand rows; reads / 237 + writes / 197 GB/s for triad is 907 µs against 894 measured, model 570. Arithmetic only.
3. Sweep: r reads and w stores per thread pass, E MiB of elements: (r, w) = (1, 0), (1, 1), (2, 1), (1, 2), (1, 4), E 4–64.
   - H-now (replay): reads and stores do not share the DRAM on a warm row, the read footprint alone sets misses.
   - H-add: every byte past the L2 pays the DRAM: read footprint F, written W; miss = warm_miss(F + W), time = F·miss / 237 + W·miss / 197: 1/1 at 64 MiB 615 µs, 32 MiB 310 µs, 8 MiB (fits) store limit only.
   - H-max: reads and writes overlap, time = max of the two: 1/1 at 64 MiB 340 µs.
   - H-reads-only: stores never churn a warm read: 1/1 at 8 MiB and (1, 0) at 16 MiB alike.

Rotating `wr_*_rotN` (question 26, first experiment): plain loads alone and with stores, window rot × (F + W): H-curve: the warm curve at the window; H-kept: decode and RMSNorm's 5–18 % above the curve.

## atom_* — global atomics on shared bins (question 39; `hand_hist` −96.8 %)

1. Mechanism: inventory atomics: per SM, pair and chip byte rates (`ldg_atomic`, upstream), none per target.
2. Why now: `hand_hist`: 4.2 M atomics at about 1.3 per clock on 1024 bins; replay 42 µs, measured 1323 µs.
3. Sweep: 16 atomics per thread, 1024 blocks, bins 1 to 1 M, and 16, 256 bins 128 B or 4096 B apart.
   - H-now (replay): flat in bins.
   - H-bin: one atomic at a time per word or sector, c cycles: rate = min(chip, bins / c): rises with bins to the chip limit.
   - H-line: per line: 1024 bins = 32 lines; 256 bins at 128 B apart as fast as 8192 contiguous bins would be.
   - H-slice: per L2 slice: rate flat at any bins inside a few lines, spread by slice hash; 16 bins 4096 B apart much faster than 16 bins 128 B apart.

## rw_* — stores that rewrite sectors (`stress_09` +443 %)

1. Mechanism: inventory stores: chip by sectors touched (D790), pair by instruction; lane spread (D801).
2. Why now: `stress_09` counters: 417792 sectors at the L1, 107364 at the L2: 3.9 writes per L2 sector. Model 36 µs, measured 6.6. Misaligned contiguous stores take the 15-cycle pair rate (no `line` rule when 32 lanes span two lines).
3. Sweep: 16 stores per thread, S elements apart (32 lanes: S 32 no overlap, S 8 stress_09's), first lane on a line or 3 elements off, G 48, 768.
   - H-now (replay): chip by every store's sectors, pair 30 per misaligned instruction.
   - H-distinct: the chip's bytes count distinct sectors: S 8 at 19/68 of S 32's chip bytes.
   - H-lines: the pair's rate goes by lines touched, 2 × 1.6 per misaligned instruction, not 30.

## Blind

`hand_gups`, `hand_copy`, `hand_scale`, `hand_stencil512`, `hand_smear`: predictions committed before the run, no parameter taken from them.

## Scored before the batch: `b0a72de` (opcode classes)

Tables on the same rows, `b0a72de` on / off: 3072×1536×2048 rotating median 2.6 / 3.3 %, τ 0.96 / 0.87, inversions 0 / 4, best yes / no (+2.2 %); warm τ 0.74 / 0.76, inversions 6 / 5; 2048³ warm and rotating τ 0.73 both, median 3.0 / 3.4 and 2.3 / 2.6 %; stress median 9.3 / 11.0 %, hand unchanged. It stays: counter-backed (r10 per-pipe), no group lost above its spread.
