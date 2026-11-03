# Batch r6b: entry block

Written before the timed run; `predicted.json` is the engine at this commit. Two batches, not one: r6's own results raised the question (as r5 → r5b).

## mma_pipe — the tensor pipe's rate by stream (question 22)

1. Mechanism: inventory §2 tensor pipe.
2. Why now: r6 `mma_issue` runs 17 cycles per mma on one warp, r2 `mma_pair` 16, upstream `mma_rate` 16 with two warps per sub-partition; the HGEMM rows replay at −15 % (s3_t128) with 16 and −1.6 % with 17. The tensor queue is 43 % of that row. The test set may not choose the number (LOOP §4): a probe has to.
3. Sweep: 16 independent mma per pass, A operands cycling through 1–8 register sets, B through 1–8; one warp and 8 warps (2 per sub-partition, the pipe-limited stream); built by the host nvcc (13.0, spacing 16: HMMA stall 15 + NOP) and the image nvcc (13.2, TileLang's, spacing 12).
   - H-warp: one warp feeds 17, two feed the pipe's 16, whatever the operands.
   - H-operands: the pipe's rate grows with the operand sets the stream cycles through: w8 above 16.
   - H-compiler: the 17 belongs to the NOP spacing of the host compiler: image build at 16 + loop share.
4. Engine prediction: `predicted.json` (per-warp 17, pipe 16). Smoke run seen before this block (nothing recorded): w8 16.0 at one set, 16.86 at a8_b1, 18.85 at a8_b8; image build w1 16.84 at one set, host 17.72.

## After the run

r6b valid: clock drift −0.02 %, canaries within 0.1 %. H-compiler for the 17: image build one warp 16.84 at one set (spacing 12, the replay's 16.83), host 17.72; two warps per sub-partition 16.015 in both. H-warp rejected: the per-warp rule tried from r6 predicts the image build 0.86 too slow, and mma_pair b1 +6.7 %. H-operands survives with a replay of the extra moves included: w8 above replay 0.06 / 0.09–0.31 / 0.41 at 2 / 4 / 8 sets of A, 0.03 at 8 of B; 0.6 at 4×4, 2.4 at 8×8. D783. Two experiments on the question: frozen `unidentified`.
