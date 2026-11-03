# Batch r3: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json` (the engine at the batch's build commit, unchanged since round 2 apart from the new case builders; microsecond rows at the description's kernel clock until the batch measures its own). Every new harness was smoke-run once on spark (rc 0, buffers as intended; nothing recorded): the smoke outputs were seen before this block was written, so the hypotheses below are not blind for l2reuse and launch; the predictions are the engine's and were not tuned.

Cache state: `warm` everywhere except the `*_rotating` test kernels and the l2reuse `cold` / `in4` modes (a 96 MiB read flush before the timed launch). All round-2 test kernels ride along (a third session), plus two FP8 GEMMs.

## Selection (LOOP §3, supervisor guidance for round 3)

Worst valid rows: GEMV rotating +1079 %, warm +215 %; RMSNorm −48 / −45 %; combine +35 %. The GEMV and RMSNorm rows sit on the generic front end's defaults (every LDG 4 lines / 512 B; warm loads all L2 hits). The LDG line count is a front-end defect, fixed without a probe and checked against ncu sector counts (R1), not against times. The warm-L2 rule is a mechanism: probe `l2reuse`. Launch-dominated rows (combine 78 %, RMSNorm warm 50 %) sit on the fitted `launch_stream_us`: probe `launch`. The HGEMM ranking (Spearman 0.60, 6 inversions) points at block sharing on the tensor pipe (open question 1, 1 of 2): its rule change is evaluated on round 2's `mma_pair` data by ablation, no new probe.

## l2reuse — what "warm" means

1. Mechanism: inventory §5 "L2 capacity and replacement" (the generic front end's warm rule: every re-read a hit, whatever the footprint).
2. Why now: GEMV warm (8.4 MB) runs at 506 GB/s, twice the DRAM plateau, while ncu reports a 6 % L2 sector hit rate; RMSNorm warm 9.5 µs against 41 µs rotating. Harness check: with `ROT 1` every launch passes the same buffers, and between two timed launches there are only the two event records and the host synchronize (gen.py `HARNESS`); nothing else runs.
3. Sweep: F = 0.75 n MiB, n ∈ {2, 8, 16, 20, 22, 24, 27, 32, 43} (1.5–32.25 MiB), 16-byte loads of kind cg / ca (plain LDG, as the GEMV) / cs (evict-first, as dram_rotate), modes cold / warm (back-to-back) / in4 (four passes in one kernel after a flush), time from %globaltimer in the kernel. Six profiler points (n 2, 16, 32, 43 warm; n 16 cold; n 16 cs warm) with L2 read, lookup hit/miss and sysmem-fill sectors.
   - H-full: 24 MiB usable with LRU-like replacement: warm at the L2 rate (≈ 2 TB/s) up to ≈ 22 MiB, at the DRAM rate (≈ 236 GB/s) from 24 MiB (a cyclic scan thrashes).
   - H-part: a smaller effective capacity (upstream `l2_carryover_b2b`, `l2_effective_bytes` 16.8 MB): all hits to 16 MiB, then a gradual loss (0.82 at 20, 0.6 at 24, 0.13 at 32 MiB).
   - H-stream: some access kinds do not allocate: warm = cold for that kind at every F (expected for cs), cg and ca as H-full or H-part.
   - Engine (generic front end): warm is all hits at every F (warm n43 16.6 µs against cold 143 µs).
   - Counters: H-ncu-ok, the profiled launch sees the harness's cache state (warm: sysmem fills ≪ read sectors); H-ncu-flush, the profiler's replay refills the L2 (fills ≈ read sectors warm and cold alike), so ncu hit rates say nothing about warm rows.
4. Engine prediction: `predicted.json` (`l2reuse *`), e.g. ca warm n16 6.35 µs, n32 12.4 µs, n43 16.6 µs; cold n43 143 µs.

## launch — the launch term of event-timed rows

1. Mechanism: inventory §1 "Launch" (`launch_stream_us` 2.6 µs fitted; mechanisms only 2.05).
2. Why now: launch is 78 % of combine's path and 50 % of RMSNorm warm's; the fit sits on every test row.
3. Sweep: one launch between two events and a synchronize (as the harness), empty and tiny (one STG per thread) kernels, grids 1–8192 × 128 threads and 48–512 × 1024; 100 back-to-back empty launches per event pair; a dependent FFMA chain of I = 0–16384 iterations on 48 × 128 threads with its in-kernel span.
   - H-const: the event time is a constant launch term plus the kernel, at every grid (engine: 2.60 + replay).
   - H-grid: a dispatch term grows with the threads launched beyond what the launch latency hides.
   - H-add vs H-hide (work points): event − span constant (additive) or shrinking with the span (part of the launch hidden behind the kernel).
4. Engine prediction: empty 2.60–2.74 µs over the grids; work_i points 2.71 → 118.9 µs (`predicted.json`).

## Block sharing on the tensor pipe (open question 1, no new probe)

Round 2's `mma_pair` (batch r2): with a start skew of 16384 FFMA iterations the older block finishes in 317 k cycles (alone 311 k), the engine says 356 k; s4096 first 469 k, span 563 k (engine 482 k, 551 k); s0 first 530 k (engine 524 k). The change to test: the tensor pipe's queue serves the oldest block first (H-oldest of round 2's entry). Accepted only if the `mma_pair` rows improve; its effect on the HGEMM ranking is recorded by ablation (LOOP §5).

## Test set growth (LOOP §8)

| Kernel | Shape | Cache | Front end |
| --- | --- | --- | --- |
| FP8 GEMM (TileLang examples/gemm_fp8 pattern), 128×128×64 s3 t128 and 128×256×64 s3 t256 | 2048³, A M×K and B N×K e4m3, fp32 accumulation, fp16 C | warm | HGEMM front end on the byte-equivalent fp16 tile (block_K / 2, K / 2; `kta:fixture` in probeset.py): same bytes, lines and mma count; SASS `QMMA.16832.F32.E4M3.E4M3` (16 cycles on the tensor pipe, `mma_types.cu`) |

Engine predictions (µs): 115.1 and 122.5. All round-2 kernels are re-measured in this batch with the same SASS.

## After the run (round 3, step 4–5)

Batch valid: clock 2.4938 → 2.4755 GHz (drift −0.7 %), canaries +0.6 % / +0.4 %.

**l2reuse.** ca warm, µs (GB/s): 1.5 MiB 1.02 (1536), 6 MiB 3.36 (1873), 12 MiB 6.27 (2006), 15 MiB 8.99, 16.5 MiB 9.95, 18 MiB 14.9, 20.25 MiB 21.2, 24 MiB 71.4 (352), 32.25 MiB 124 (273); cold 235 GB/s at every size. Miss fraction (span between the all-hit 0.52 µs/MiB and the cold 4.45 µs/MiB): 0.02, 0.02, 0.08, 0.13, 0.62, 0.85 from 15 MiB up. cg the same. cs warm = cold from 6 MiB (all hits at 1.5 MiB). H-full rejected (24 MiB still 38 % hits, 18–20 MiB already missing); H-part survives (close to upstream `l2_carryover_b2b`); H-stream holds for cs only. in4 is not a clean re-read meter: at small sizes the four passes take the cold time (later passes hit lines still pending), so it is not used for the curve. Counters: H-ncu-flush — `lts__d_sectors_fill_sysmem` equals the read sectors in every profiled point, warm (n2, n16, n32, n43) and cold alike, and the profiled time is 62 µs against 6.3 µs un-profiled for n16 warm; `lts__t_sectors_lookup_hit/miss` cover a small subset; `srcunit_tex` lookups report 100 % hits even cold. → D766, D767.

**launch.** Empty kernel 2.40–2.46 µs for 1–1024 × 128 and 48–128 × 1024 threads; 2.62–2.91 at 262 k threads, 4.64 at 524 k, 6.69 at 1 M (H-grid beyond 131 k threads). tiny = empty within 0.1 µs up to 131 k threads. Back-to-back 1.96 µs per launch (1 block), 1.69 (512), 4.10 (4096). Work points, event − span: 2.30, 2.08, 3.26, 2.08, 2.62, 3.23, 3.55, 2.98 µs for spans 0.10–151 µs: neither constant nor shrinking (H-hide rejected), median 2.80 → `launch_event_us` (D769). The FFMA loop runs 23.0 cycles per iteration (replay 17.6): open question 4.

**Block sharing.** Tried as a rule change on r2's mma_pair: oldest-block-first gives s0 first 319 k (measured 530 k), s4096 314 k (469 k), s16384 310 k (317 k); the current rule 524 k, 482 k, 356 k. With the skew loop at 23 cycles per iteration (block 2 starts at ≈ 94 k and ≈ 377 k cycles) fair sharing predicts 464 k and 311 k / 688 k span: fair survives, oldest-first rejected; no change (D768).

**FP8 GEMM.** 123.1 and 137.6 µs measured, 115.1 and 122.5 predicted before the run (−6.5 %, −11.0 %); after the round's changes −4.3 %, −8.9 %; R1 2/2.

**Ablations** (scorecard with one change off, r3 data; error with the change off → on):

| Change | Rows that move | R1 |
| --- | --- | --- |
| LDG from lane addresses (D765) | GEMV rotating +1052 → +21.5 %, warm +220 → −9.9 %; combine +42.8 → +23.3 %; RMSNorm −46.8 / −36.9 → −41.5 / −35.1 %; softmax −31.3 → −28.2 %; decode −3.2 / +16.2 → −5.4 / +15.0 % | adds the sector check: GEMV 1360 vs 1344 sectors per block, RMSNorm 512 = 512, softmax 256 = 256, combine 96 = 96 |
| FCHK fast path (D770) | softmax −21.6 → −28.2 %, FA −5.4 → −6.7 %, decode −3.4 / +15.7 → −5.4 / +15.0 % | decode 0/2 → 2/2 (22516 vs 22413 instructions per block), softmax 0/1 → 1/1 (2756 = 2756) |
| warm L2 by footprint (D766) | FA −7.7 → −6.7 %, GEMV warm −11.0 → −9.9 % | — |
| launch_event_us 2.80 for 2.6 (D769) | combine +15.3 → +23.3 %; RMSNorm warm −37.4 → −35.1 %; softmax −30.3 → −28.2 %; HGEMM median 4.5 → 4.4 % | — |

HGEMM Spearman 0.62 and 8 inversions with every change on or off; on r2's data the same engine gave 0.60 and 6: the difference is the re-measurement (every variant within 0.5 % of r2; pairs near the 2 % noise threshold flip).
