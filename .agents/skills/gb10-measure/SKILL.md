---
name: gb10-measure
description: Run one measurement batch on the spark GB10 (timings, microbenchmarks, profiler counts) so that the numbers are comparable with every other batch and usable as references for the performance model. Use for every GPU measurement in the performance-model work; it adds the protocol of docs/design/28 §4 on top of remote-run.
---

# A measurement batch on the GB10

One batch = one exclusive run on spark that produces numbers the model will be compared against. The numbers are worthless unless every item below is recorded, so the batch is refused (not "run anyway") when one is missing.

This skill decides *how* to measure. What to measure, and why, comes from the error table and the mechanism inventory (28 §3, §7).

## Before the batch

1. **Question.** Write one sentence: which error-table row or inventory mechanism this batch serves, and what the model predicts (28 §3, R2: predict first). A batch without a written prediction is a fishing trip; do not start it.
2. **Code identity.** Everything that runs (probe sources, kernels, drivers) is committed; `scripts/remote-run sync spark` deploys that commit. Record the commit. A dirty tree is refused by `sync` for this reason.
3. **Exclusivity.** One measurement job on spark at a time. Check `scripts/remote-run status spark <name>` for anything running, and check `nvidia-smi` for other processes. Profiling goes through the WP27 profiler helper (`spec/modules/target-exec.md`); never run `ncu` with elevated privileges by hand.
4. **Cache state**, declared per case and written into the batch record — the model takes it as input:
   - `warm`: same inputs, back-to-back, after warm-up;
   - `cold`: an L2 flush (a write larger than the 24 MB L2) before every timed run;
   - `rotating`: *n* input copies used in turn, *n* × footprint > L2 (decode-style streaming).
   If a case does not say, it is `warm`.

## The batch itself

Run in this order, all inside the same exclusive job:

| Step | What | Why |
| --- | --- | --- |
| 1 | environment record: `nvidia-smi -q` (clocks, power mode, temperature), driver and CUDA versions, `uptime` load, idle check | GB10's LPDDR is shared with the CPU; load moves bandwidth |
| 2 | **clock calibration kernel**: a fixed dependent FFMA chain of known cycle count, timed by events → effective GHz | the model predicts cycles; conversion uses this clock, not the nominal 2.5 GHz |
| 3 | **compute canary** and **memory canary** (fixed kernels with known reference times) | detect a sick machine or a wrong clock before spending the batch |
| 4 | the probes and kernels of this batch: N warm-up runs, M timed runs each (defaults 10 / 30), event timing, **no profiler attached** | these are the timing references |
| 5 | the same kernels under the profiler, **counts only**, fixed metric list (below), replay mode and cache control declared | work counts for the front-end check (R1) and absolute pipe-active cycles |
| 6 | canaries and clock kernel again | a drift between step 3 and step 6 invalidates the batch |
| 7 | environment record again | |

Probes use `clock64()` inside the kernel for cycle counts; the event timing of step 4 is for whole kernels only.

**Profiler metric list** (counts, never durations): `smsp__inst_executed.sum`, per-pipe `smsp__inst_executed_pipe_*.sum`, `sm__pipe_{tensor,fma,alu,fp64,xu}_cycles_active.sum`, `sm__cycles_elapsed.max`, `l1tex__t_sectors_pipe_lsu_mem_global_op_{ld,st}.sum`, `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{ld,st}.sum`, `lts__t_sectors_op_{read,write}.sum`, launch and occupancy attributes. `gpu__time_duration` is recorded but **never used as a timing reference**; pipe percentages are converted to absolute active cycles and divided by the un-profiled cycle count (step 4 time × step 2 clock) before any comparison with the model.

## After the batch

- **Validity.** The batch is `valid` only if both canaries are inside their bands at steps 3 and 6 and the clock drift is below 1 %. Otherwise it is `invalid` and nothing from it enters the error table or a parameter — but it is kept.
- **Record.** One directory per batch under the data root: `batches/<date>-<name>/` with `batch.json` (question, prediction, commit, cache state per case, environment before/after, clock, canary values, validity), raw outputs per case, the profiler CSV, and the `clock64` dumps. Reference the directory from the state file (28 §7).
- **Report** medians and IQR, never means; keep every timed sample in the raw output.
- A faster number obtained by re-running is not a better number. Repeat only with a reason written in `batch.json`.

## Never

- time under the profiler, or use profiler durations as references;
- mix batches with different cache states or clocks into one comparison;
- change spark's clock, power mode, driver or packages — ask the user (28 §7);
- measure while another measurement or profiling job is on the machine.
