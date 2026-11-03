# Batch r5b: entry block

Same round as r5: r5's row_reduce strided its stores (16 B per lane, 64 B apart), not the row kernels' contiguous 32 B per lane, and sm.py takes `STG.E.ENL2.256` as 128 B per warp instruction. Prediction: `predicted.json`, engine at this commit.

## st256 — 32-byte-per-lane global accesses

1. Mechanism: inventory §6 store port and chip store limit (measured at 4 and 16 B per lane only), §5 chip L2 share on `LDG.256`.
2. Why now: RMSNorm warm graph −50.5 %, softmax graph −35 %: 4.2 MB read, 4.2 MB written per launch in ≈ 6.4 µs; both use only 256-bit global accesses.
3. Sweep: stores, loads, both, grids 1, 2, 16, 48, 96, 512 × 128 threads, 8 KiB per block as RMSNorm.
   - H-engine: 128 B per instruction at the 4-byte rates (`predicted.json`).
   - H-1024-16B: 1024 B per instruction at the 16-byte chip limit 570.9 B/clk: st_g512 ≈ 7.3 k cycles.
   - H-1024-4B: at the 4-byte limit 122 B/clk: ≈ 34 k cycles.
   - H-new: an own chip rate: st_g512 = 4.19 MB / rate; ld_st the sum of ld and st.
4. Smoke run (one run on spark, nothing recorded): st g1 205, g48 194, g96 463, g512 8908; ld g1 568, g48 652, g96 1202, g512 4975; ld_st g512 11093 cycles.
