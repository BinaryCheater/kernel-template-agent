---
name: write-probe
description: Write a microbenchmark (probe) that isolates one hardware mechanism for the performance model, with the model's prediction written down before it is measured, and register it as a permanent regression. Use whenever a parameter is to be measured or a mechanism hypothesis tested; never fit a parameter from a real kernel instead.
---

# Writing a probe

A probe is a small kernel that makes one mechanism of the resource network observable (27 §5): *saturate resource R*, *dependency latency along path P*, *contention between R1 and R2*, *hand-off delay of sync object S*, *pipeline of depth d*. It is the only legitimate source of a model parameter (28 §2, R3).

This skill fixes the form of a probe. Which mechanism to probe comes from the error table's attribution and the mechanism inventory (28 §3); whether probing it is allowed comes from the error budget (R2).

## Entry check (write these four lines first)

1. **Mechanism**: the inventory row, in the resource-network vocabulary.
2. **Why now**: the error-table row and attribution share that exceeds the budget, or the design variable it sits on.
3. **Competing hypotheses**: at least two, each with the number it predicts for the sweep below. If every hypothesis predicts the same numbers, the probe cannot discriminate — change the sweep or do not run it.
4. **Model prediction**: run the current engine on the probe's own SASS (the engine replays probes exactly as it replays kernels) and write the predicted values down **before** measuring. This is the step most easily skipped and the one that makes the method work.

## Form

- One mechanism per probe. If a second one leaks in (an address calculation on the ALU inside an FMA-rate loop, a loop branch inside a latency chain), either subtract it with a control variant or restructure.
- **Sweep, not a point**: at least two values of the controlling quantity (warps, in-flight requests, stride, stage depth, operand width) chosen so the hypotheses diverge.
- **Cycles from `clock64()`** inside the kernel, per warp, after a warm-up iteration; report per-operation cycles with the loop's own overhead measured by an empty-body control.
- **Check the compiler**: dump the SASS (`cuobjdump -sass` / `nvdisasm -hex`) and confirm the hot loop contains the intended instructions and nothing folded, hoisted or vectorised away. Upstream lost a conversion probe to folding; the SASS is part of the probe's record.
- **Fixed layout** so the engine can replay it: one `.cu` with `main` taking the sweep values as arguments; prints one CSV line per point: `probe,point,cycles_median,cycles_iqr,ops,warps,…`.
- Reuse first: the ~40 upstream probes (research repo, read-only) and the B1–B40 suite cover levels 1–5 of the ladder; copy the pattern, cite the source.

## Run

Through the `gb10-measure` skill (it supplies clock, canaries, exclusivity and the record). The probe's prediction goes into that batch's `batch.json`.

## After

- Compare measured against each hypothesis's prediction. Write which one survived, or that none did. Two rounds without a discriminating result → fix the parameter at the best value, mark it `unidentified` in the parameter file, stop (R2).
- Update the parameter file or the engine; every changed value carries the probe name and batch directory as provenance.
- **Register the probe as a regression**: add it to the ladder list with its level, its measured values and the tolerance; the CPU replay (`scripts/…` from 28 §5 P1) must reproduce it within tolerance from now on.
- Re-run the whole ladder and the error table; a probe that improves its own row but worsens another is not done.
- One paragraph in the state file: mechanism, hypotheses, numbers, decision.

## Never

- fit a parameter from a real kernel's time;
- measure before the prediction is written;
- keep a probe whose SASS was not checked;
- tune a probe's sweep until the model "passes".
