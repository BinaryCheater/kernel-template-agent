# Batch r1: entry blocks

Written before the run (write-probe). The engine's replay of each probe's own SASS is in `predicted.json` (engine at commit `ee95dac` plus the namedbar trip-count fix; the description's kernel clock 2.54 GHz for microsecond rows). Sources are under `perfmodel/probes/<name>/probe.cu`; the SASS of every probe was checked: the hot loop holds the intended instructions and nothing is folded (LDS.32/LDS.64 had been removed by the compiler in a first build because their results were dead; fixed before this batch).

Cache state: `warm` for every case except `dram_rotate` (`rotating`: four 63.75 MB buffers in turn, 255 MB > 24 MB L2).

Test set (28 §5, first cut): ten TileLang HGEMM variants at 2048³ (tile, stages, threads, raster panel) from the `gemm_fp16_tilelang` family template, and one TileLang FA forward (B 1, S 2048, H 8, D 128, BM 64, BN 64, 2 stages, 128 threads). Their replay predictions are in `predicted.json`; they are scored, not probes.

## P-1 store_tail — C-store path at partial and full grids

1. Mechanism: inventory §6 rows 1–2 (store port shared by a TPC pair, 15 cycles per 4 B/lane STG per pair, chip 137.8 B/clk; "partner busy whenever the share is ≥ 2" assumed).
2. Why now: store queue 28 % of the critical path of the two worst loads-only rows (−23.7 %, −18.4 %); store width is a search-space variable; store sharing a hardware variable. Every HGEMM in the test set ends with this epilogue.
3. Hypotheses, per-block µs for blocks 1, 2, 8, 16, 32, 48 (tl pattern / v16 pattern):
   - H-a current rule (engine): tl 3.15, 6.17, 6.17, 6.17, 6.18, 9.12; v16 0.83, 1.53, 1.53, 1.53, 1.57, 2.30.
   - H-b chip limit ≈ 102 B/clk instead of 137.8: tl 3.15, 6.17, 6.17, 6.17, 8.1, 12.1.
   - H-c H-a plus a fixed per-block tail of 1.1–2 µs: tl 4.3–5.1, 7.3–8.2, …, 10.2–11.1.
4. Engine prediction: H-a numbers above (`predicted.json`, `store_tail *`).

## P-2 l2share_copy — cp.async chip-L2 delivery with one block per SM

1. Mechanism: inventory §5 row 6 (chip L2 rate for copy traffic 817.7 B/clk shared by blocks in flight) and §5 row 5 (per-SM return port 30.7 B/clk).
2. Why now: the chip-L2 share queue is 46 % of the same rows' critical path; bytes in flight per SM (stages) is a design variable; all HGEMM variants stream through it.
3. Hypotheses, per-block µs for blocks 1, 16, 32, 48 (upstream's `nocompute_nostore` kernel of c128x256x64_st2_th256, 16 k-tiles of 48 KB):
   - H-a flat chip rate shared by blocks in flight (engine): 10.33, 10.34, 12.37, 18.43.
   - H-b delivery bounded per SM by bytes in flight (Little's law on the 350-cycle L2 latency), not by the chip: flat ≈ 10.3 at every block count.
   - H-c chip rate lower for 512 B, 4-line copies (≈ 600 B/clk): 10.3, 10.3, 16.5, 24.7.
   Upstream's 1-block record (12.8 µs, `ablation.jsonl`) is already 24 % above H-a's 1-block value, so the per-SM part is in question too.
4. Engine prediction: H-a numbers.

## smem_ldsts — plain shared loads and stores (LDS, STS)

1. Mechanism: inventory §3 "Plain shared loads/stores" (absent: no array time, default 20-cycle latency).
2. Why now: FA forward (6 LDS in its loop, STS in its epilogue) and the HGEMM loop (3 LDS per pass) use them; status absent; the bank/array rate is a hardware variable.
3. Hypotheses, cycles per warp instruction per SM at 1 / 4 / 8 / 16 / 32 warps:
   - H-a engine (issue-limited, latency 20): LDS.32 33.6, 8.4, 4.2, 2.1, 2.1; LDS.128 35.9, 9.0, 4.7, 3.0, 2.7; STS.32 4.5, 1.1, 0.57, 0.48, 0.40; STS.128 13.0, 3.3, 1.6, 0.83, 0.81.
   - H-b LDS/STS take the shared-memory array at 128 B/clk like LDSM/STSM: at ≥ 16 warps LDS.32 / STS.32 1.0, LDS.64 2.0, LDS.128 / STS.128 4.0; latency ≈ 30 cycles plus the chain's ALU ops at 1 warp.
   - H-c every shared instruction costs a full 4-cycle array slot regardless of width: 4.0 for every kind at ≥ 16 warps.
4. Engine prediction: H-a numbers.

## shfl — warp shuffles

1. Mechanism: inventory §3 "Shuffles SHFL" (absent: default latency 20, issue slot only).
2. Why now: FA forward's row max and row sum (8 SHFL per loop pass); status absent.
3. Hypotheses: latency (1 warp, 1 chain) and cycles per warp instruction per SM at 1 / 4 / 8 / 16 / 32 warps with 8 chains:
   - H-a engine: latency 20.0; 4.4, 1.1, 0.55, 0.32, 0.30.
   - H-b one shuffle per cycle per SM (one wavefront each in the shared LSU path): latency ≈ 25–30; 4.4, 1.1, 1.0, 1.0, 1.0.
   - H-c one shuffle every 2 cycles per SM: 4.4, 2.0, 2.0, 2.0, 2.0.
4. Engine prediction: H-a numbers.

## ex2_ffma — MUFU.EX2 beside FFMA from the same warps

1. Mechanism: inventory §2 "ex2 with FFMAs between them from one warp" (absent; MUFU measured alone: 8 cycles per instruction per sub-partition).
2. Why now: FA forward's softmax (34 MUFU beside 170 FMUL and 70 FADD per pass); status absent.
3. Hypotheses, cycles per loop pass per sub-partition (8 warps) for (E, F) = (8,0), (8,8), (8,16), (8,32), (0,32), (4,32):
   - H-a engine (separate pipes, MUFU-bound): 512, 512, 512, 512, 293, 335.
   - H-b MUFU and FFMA serialise in dispatch: 512, 576, 640, 768, 256, 512.
   - H-c MUFU at 4 cycles per instruction when FFMAs fill the gaps: 512, 512, 512, 512, 256, 256.
4. Engine prediction: H-a numbers.

## namedbar — block barrier release and BAR.ARV

1. Mechanism: inventory §7 "BAR.SYNC release 13 + 2 per warp" (base assumed) and "BAR.ARV, BAR.RED" (absent: treated as blocking).
2. Why now: every test kernel synchronises with BAR.SYNC (HGEMM 2, FA 4 per pass); named-barrier producer/consumer is the warp-specialised FA pattern; one of the two silent defaults of 29 §8.
3. Hypotheses: sync cycles per barrier at 2/4/8/16/32 warps; arrive cycles at L = 0/64/256:
   - H-a engine (13 + 2 per warp; BAR.ARV blocks until the barrier completes): sync 35, 39, 48, 66, 116; arrive 35, 255, 1017.
   - H-b release independent of the warp count: sync ≈ constant (≈ 20–30) at every count.
   - H-c BAR.ARV does not block: arrive ≈ constant below 30 cycles at every L.
4. Engine prediction: H-a numbers.

## l1hit — L1 hit path of plain global loads

1. Mechanism: inventory §5 "L1 hit path: 2 cycles per line, 45-cycle latency" (measured upstream; hit or miss is supplied by the driver).
2. Why now: decode attention, GEMV and norm kernels of the test set read through L1; the HGEMM and FA rows do not, so this is the cheap re-measurement of a design variable.
3. Hypotheses: chase cycles per hop; cycles per LDG.128 per SM at 4/8/16/32 warps:
   - H-a engine: 53.0; 8.1, 8.0, 8.0, 8.0.
   - H-b L1 at 128 B/clk (as the shared-memory array): 4.0 at ≥ 8 warps; latency ≈ 35–40.
4. Engine prediction: H-a numbers.

## dram_rotate — DRAM reads under rotating inputs

1. Mechanism: inventory §5 "DRAM read rate by run length" (237 GB/s contiguous), "DRAM latency 1560 cycles", "DRAM queue per SM".
2. Why now: the declared `rotating` cache state of decode-style rows; the DRAM rate is a hardware variable and the two TMA decode rows (+36.7 %, +20.0 %) wait for a declared cache state.
3. Hypotheses, µs per launch at T = 32, 64, 128, 256, 512, 1024 threads per block (48 blocks, one LDG.128 in flight per warp):
   - H-a engine (latency-bound until 237 GB/s): 1721, 862, 433, 285, 285, 285.
   - H-b the rate under rotation saturates near 204 GB/s (run length, page changes): plateau 330 µs.
   - H-c the latency under load is higher than 1560 cycles: the latency-bound points (T ≤ 128) proportionally slower than H-a.
4. Engine prediction: H-a numbers.
