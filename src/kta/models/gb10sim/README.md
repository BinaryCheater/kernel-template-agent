# kta.models.gb10sim

The GB10 simulator copied from AI-Infra-Co-Design (`agentic/gb10`, commit `0eef4979390559f7cca72eac2b06e9cfc249ea30`, local checkout `/Users/dingyi/Projects/ai-infra-codesign-ventus`) to become KTA's single hardware-model engine (WP26; D680, D681). Design: `docs/design/25-hardware-model-sources.md`.

## Provenance

| KTA file | Upstream file | Upstream sha256 |
| --- | --- | --- |
| `sass.py` | `agentic/gb10/sim/sass.py` | `45d00766f4d1a1e2…` |
| `sm.py` | `agentic/gb10/sim/sm.py` | `20cfa4a67fa06a29…` |
| `ws.py` | `agentic/gb10/sim/ws.py` | `53d0cb68bb815efb…` |
| `kernel.py` | `agentic/gb10/sim/kernel.py` | `655d7dc63716da9e…` |
| `model.py` | `agentic/gb10/model.py` | `aca7cbb792ad158c…` |
| `validate.py` | `agentic/gb10/sim/validate.py` | `557d2fe33dd439de…` |
| `ws_validate.py` | `agentic/gb10/sim/ws_validate.py` | `29440ecf511c280c…` |
| `gb10.json` | `calibration/spark-6951/gb10.json` | `8daea49d75734ac6…` (unchanged) |

Not copied: `gemv.py`, `epilogue.py`, `decode_sweep.py`, `layout.py`, `shapes.py`, `schedtest.py`, `audit_values.py`, `agentic/gb10/{evaluate,fidelity,l2policy_fit}.py`, and the microbenchmark sources (`agentic/tilelang/microbench/`).

## Changes against upstream (copy step)

Behaviour-preserving only; the replay on slai checks it (`scripts/gb10sim-replay`, delivery note).

- `ruff format`; over-long comments and docstrings re-wrapped; `l` renamed `line` in `sass.parse`; `zip(pts, pts[1:])` → `itertools.pairwise(pts)`; one unused local removed; `min/max(d, key=d.get)` → `key=lambda q: d[q]`; a no-op conditional (`a if c else a`) reduced to `a`; list concatenations written as unpacking.
- Type declarations: the attributes upstream attaches to `Warp` after construction are dataclass fields with the same initial values; `ready` and the unit-free lists are floats; `_val`, `_imm`, `_pred` return `Any`; `main_loop` returns `Optional`; the base `on_event` has a docstring body; `pick`/`last_yield` hooks via `getattr(..., None)`.
- Paths: parameters from the `gb10.json` next to the code; the calibration directory (SASS dumps, ablation records) is an argument of `validate` and `ws_validate` instead of a path under the upstream checkout; results are returned, not printed or written under `runs/`; `scipy.stats.spearmanr` → `kta.models.regression.spearman`.

## Parameter step (WP26, D682)

- Every entry of `gb10.json` has a `status`: `measured` (a probe isolating one unit), `fitted` (a real kernel with one mechanism removed, or a residual: a calibration parameter a later benchmark should turn into a mechanism), `structural` (chip design, driver, or a guard), `assumed`, `withdrawn`. `sim.value_status` does the same per simulator value. Counts at this step: 30 measured, 11 fitted, 13 structural, 1 withdrawn; sim 40 measured, 2 assumed.
- Hardware numbers that upstream wrote in code are entries now (source starts with "KTA (WP26)"): `clock_under_load_GHz` 2.54 (upstream uses 2.54 in kernel/ws code and `clock_GHz` 2.52 in `validate.py`: two clocks, to reconcile), `launch_stream_us` 2.6, `graph_launch_us`, `graph_launch_us_per_block`, `dram_datasheet_GBs`, `dram_rate_floor_GBs`; existing entries replace the TMA rates (81e9, 1860e9), DRAM read rate (237e9), occupancy limits, SM count (48) and block setup (0.1).
- Left in code as structural ISA facts: 32 lanes, 512 B per warp copy instruction, 16 B chunks, 128 B lines, six dependency counters, the mma tile 16×8×16, the SASS control-bit layout, simulation cycle limits, unit conversions.
- Instructions whose result latency falls back to the assumed `default_var_latency_clk` are counted per opcode and reported by `SM.run()` as `assumed_latency` (instead of an error, so the replay stays comparable; turning them into errors waits until each such opcode has a probe).

## Next steps (WP26)

L3 runs this engine on a compiled candidate's SASS; L2 and L1 read the same engine and parameters.

## Performance-model step (PERFMODEL; docs/design/27, 28, 29)

- `machine.yaml` is the machine description (27 §2.2): hierarchy, resources and their scope, the opcode → operation-class table; `machine.py` loads it with `gb10.json` into a `Machine`, which every entry point takes as an optional `machine` argument (D743). `Machine.with_overrides` makes what-if copies.
- `probes.py` replays upstream microbenchmarks from their own SASS (D745).
- The error table: `scripts/perfmodel-table` → `docs/design/perfmodel/error-table.{json,md}` (D744).
