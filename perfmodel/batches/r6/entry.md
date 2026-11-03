# Batch r6: entry blocks

Written before the timed run (write-probe). The engine's replay of every case's own SASS is in `predicted.json`, engine rules as at the end of round 5 (only the sequence front end and the case lists were added, no rule changed). New harnesses were smoke-run once on spark before this block; nothing from that run is recorded.

## Selection (LOOP §3, supervisor guidance)

- Blind set: all engine changes were guided by the test rows. HGEMM 4096×1024×2048 and 1024³ (3 tile/stage variants each, warm and rotating), FA S1024 H16 D64, RMSNorm 2048×1024 (warm, rotating), softmax 1024×2048: the predictions in this batch are their blind score; no engine change before they are scored.
- Fast bias on HGEMM, FP8, FA, w4gemm, softmax, RMSNorm with every measured mechanism holding: absent mechanisms, share by count × cost (LOOP §3). Candidates: taken branch, instruction fetch on long code, mma's issue side, LDS/STS by lane layout.
- Worst rows: combine b2b +59.9 % (question 14), GEMV rotating +27.7 % (question 20: address of `LDG.U16`). The second needs no probe: the lanes step runs before the address is read, so a load whose destination is its own address register is unknown; to be fixed after the blind score.
- Decode split-KV sequence: `decode_seq_*`, generic program form (launch.json family `sequence`: kernels in order, buffers by name, L2 carried, gap by launch mode).

## branch — taken branch and instruction fetch

1. Mechanism: inventory §1 "Branch and loop resolution", "Instruction cache / fetch" (X).
2. Why now: question 4, 23.0 vs 17.6 cycles per iteration; a family of fast rows. Count × cost: FA 83 and softmax 51 branch sites, HGEMM one taken back edge per k-step.
3. Sweep: loop of N FFMA, N 4–1024, warps 1, 4, 32; straight lines of 64–4096 FFMA, warps 1–32; forward branch over 8 FFMA in a loop, taken and not.
   - H-none: replay (`predicted.json`).
   - H-add: a taken branch adds ≈ 5 cycles to the warp beyond its stall count: replay + 5 per pass at every N.
   - H-floor: the warp's next instruction comes at least B cycles after a taken branch, the stall count included: replay + max(0, B − stall) (BRA stall 5–6 here, 15 in the mma loop).
   - H-smsp: the redirect stops the whole sub-partition: w32 loses ≈ 8 × as much per pass as w1.
   - Fetch: H-free: 1 cycle per instruction on straight code; H-fetch: 9 cycles per 8 instructions past a buffer (smoke: 1.125 at 256+).
4. Engine prediction: `predicted.json` (H-none).

## mma_issue — what an mma stream pays beside the pipe

1. Mechanism: inventory §2 tensor pipe, "mma issue beside FFMA" (X), §1 "Issue interference" (X).
2. Why now: tensor queue 43 % + stall after mma 23 % of the worst HGEMM row; mma_types' 14 cycles extra per pass of 16.
3. Sweep: N mma per pass, N 2–64; 8 mma per pass with M other instructions after each (FFMA, LOP3, LDS, ldmatrix), M 1–14; warp 0 mma beside warp 4 streaming FFMA, LOP3, LDS on the same sub-partition.
   - H-pass: the extra is per pass: cycles/mma = 16 + c/N.
   - H-hold: the stream's rate is 17 per mma: 17 + c/N.
   - H-nothing-beside: other instructions fit in the hold, replay's values.
   - H-B34 (upstream: +43 % beside FFMA): FFMA beside costs mma cycles, ALU and LDS less.
4. Engine prediction: `predicted.json`.

## lds_narrow — LDS and STS by element and lane stride

1. Mechanism: inventory §3 plain shared loads/stores, "Bank conflicts" (P: none for LDS/STS).
2. Why now: question 21, LDS latency 13–17 % of w4gemm's path, tile 64×128 costs 1.5× in both; `LDS.U8`, `LDS.U16` never probed.
3. Sweep: U8, U16, 32, STS.U16, STS.32; lanes contiguous, 16 and 64 B apart; 1, 4, 32 warps.
   - H-width: as round 1, by opcode width: same at every stride.
   - H-span: the array is held by the lanes' extent, one 128-byte line per cycle: 2, 4, 16 cycles at contiguous, 16, 64; latency grows with it.
4. Engine prediction: `predicted.json` (H-width).

## launch_short — the launch term below 2.5 µs

1. Mechanism: inventory §1 launch; `launch_seq` rule.
2. Why now: question 14, combine b2b +59.9 %, kernel +25 %; b2b 1.91 µs at span 0.3, 3.69 at 0.74.
3. Sweep: chain length 0–256 (spans 0.1–2.5 µs), one launch, b2b, graph.
   - H-max: max(submit, span + gap), the current rule.
   - H-step: a launch that finds the previous kernel finished skips the wait: submit below a span T, span + gap above.
4. Engine prediction: `predicted.json` (H-max).

## Sequence

`decode_seq_s4096_{warm,rotating}`, kernel / b2b / graph. Prediction: spans of the two kernels from their own SASS, combine's loads of `glse` and `Opart` hit (the partial kernel just wrote them), gaps by mode.
