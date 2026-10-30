# kta.measure

Layer 5. Specification: [spec/modules/measure-check.md](../../../spec/modules/measure-check.md). Built by: WP3.

## Responsibility

Measurement harness and versioned measurement protocols; comparisons and profiles, run on GPU workers through the Executor.

## Public interface

- `measure`, `compare`, `profile` (spec signatures; they resolve store and executor from the Context), and `Measurer` with the same methods for callers that hold their own store and executor.
- `protocol.ensure_v1`, `protocol.ensure_protocol`, `protocol.load_params`; `noise.noise_probe`, `noise.revise_protocol`.
- Return values carry the record ID: `Measured(id, body, times_us)`, `Comparison` (with the Report ID) (D224).

## Files

| File | Role |
| --- | --- |
| `bench.py` | worker-side job `{python} -m kta.measure.bench spec.json`: run or time one or two modules |
| `jobs.py` | control-side bench job builder (launch from the Build's `build_key` blob, buffers from the Problem) |
| `protocol.py` | `measure-v<n>` Protocol records |
| `runner.py` | `measure`, `compare` and Measurement records |
| `stats.py` | median, bootstrap CIs, significance, noise summary |
| `noise.py` | noise probe and the `measure-v2` revision |
| `profile.py` | Nsight Compute profiles (NotSupported where counters are admin-only) |
| `nvml.py` | clock, temperature, power samples (worker side) |
| `inputs.py` | input generators shared with checks |
| `selftest.py` | simple real kernels used by WP3's runs (D94) |
| `peak_kernels.py`, `peaks.py` | per-unit peak microbenchmarks, measured HwFacts and the HwModel (D283) |
| `baseline.py`, `scripts/torch_bench.py` | library baselines (torch.matmul) under the same Protocol, in the worker image (D284) |

- `microbench.py` (WP11): L2 working-set sweep, DRAM latency, launch latency; the T5
  benchmark format (`Microbench`, `repeat`, `write_repeat_fact`).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
