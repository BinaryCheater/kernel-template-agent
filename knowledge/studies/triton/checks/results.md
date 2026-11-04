# Witness results

Command: `.venv/bin/python knowledge/studies/triton/checks/replay.py`.

Observed: `PASS: 1224 grouped-raster shapes, 11 worker counts each; grouped lookup and split-N capacity`.

Domain: M tile counts 1..17, N tile counts 1..9, G 1..8, worker strides 1..11; grouped count vector [6,1,4]; unique N coordinates 0..255. Covers bijection, exact-once static scheduling, monotone prefix lookup and split-output byte capacity. Does not cover native layout reinterpretation, asynchronous ring lifetime, actual memory access or floating-point arithmetic.

Continuation command: `.venv/bin/python knowledge/studies/triton/checks/design_witnesses.py`.
Observed: `PASS: blocked coordinate bijection; spare-slot lifetime sets; short snake strips; unique completion issuers; scale index discriminator`.
This resolves the consequential coordinate/lifetime-set ambiguities in the new rules. Native asynchronous completion, proxy fencing, numerical rounding and failed-CLC-accessor behavior remain the explicitly named realization obligations.

Production command: `.venv/bin/python knowledge/studies/triton/checks/production_witnesses.py`.

Observed:

```
PASS: full-sort top-k oracle; 32-bit FP4 permutation and 1024 byte cases; Hopper digit/pointer identity; packed column counts; local row buckets/bitmap order; cyclic split tails and empty owners
PASS: 65536 FP4 packing-axis byte pairs
PASS: 4096 signed-code magnitude maxima
```

Top-k capacity1/2/4/8/16 over1/2/3/9 blocks uses distinct identity keys with duplicate scores and compares against full sorting. Bit interleave checks every basis bit and each byte value in each source position. Hopper scale coordinate flattening independently matches the pointer formula for1/2/4/8 warps; both MMA digit profiles are bijections. Vertical column counts check zero/all-ones/random rows for powers of two1..512, including packed-field capacity boundaries. Row buckets test the actual per-CTA interleaved enumeration, not a global stable-sort claim. Cyclic split checks lengths0..72, tile widths1/3/8/16 and participant counts1/2/5/11. Quantizer native casts, exceptional/extreme-scale behavior, descriptor consumer layout and peer visibility were not executed by these finite checks.
