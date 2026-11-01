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

## Next steps (WP26)

Constants written in code (clock 2.54e3 cycles/µs in `kernel.py`, `ws_validate.py`; TMA 81e9 and 1860e9 B/s in `ws.py`; occupancy limits in `kernel.py`, `validate.py`; launch 2.6 µs) move into the parameter file with a status (measured / fitted / structural) and their probe; an unknown opcode becomes an error instead of `default_var_latency_clk`; L3 runs this engine on compiled SASS; L2 and L1 read the same engine and parameters.
