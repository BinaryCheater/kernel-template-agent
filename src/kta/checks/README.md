# kta.checks

Layer 6. Specification: [spec/modules/measure-check.md](../../../spec/modules/measure-check.md). Built by: WP3.

## Responsibility

Independent checkers: numeric correctness, sanitizer, anti-cheat, SASS expectations, space verification, and (WP24) Template proofs: K1–K3, composition, per-candidate checks, proof records with reuse and invalidation, weakest preconditions. Must not import `knowledge` (a checker may not reuse the generator's logic).

## Public interface

- `check_numeric`, `check_sanitizer`, `check_anticheat`, `check_sass` (keyword `ctx`, D229), `full_check` (keywords `case_id`, `shape`, `build_for`, D227; `sanitizer_shape`, D285)
- `verify_space`, `SpaceItem`, `SpaceReport`
- `Checker` (explicit store, executor, clock, rng, private directory); every check returns `Checked(id, body)` (D224).

## Files

| File | Role |
| --- | --- |
| `numeric.py` | outputs vs GPU FP64 reference, per input distribution |
| `reference.py`, `scripts/ref_torch.py` | reference job in the worker image (PyTorch) |
| `sanitizer.py` | compute-sanitizer memcheck and racecheck |
| `anticheat.py`, `elf.py` | nsys kernel trace, NaN coverage, determinism, linked libraries |
| `sass.py` | SASS opcode statistics and `SassExpect` checks; Gap on violation |
| `hidden.py` | hidden shapes from a private seed |
| `full.py` | `full_check`, `verify_space` |
| `inputs.py` | adversarial input distributions (incl. `large_dynamic_range`, D289) |
| `selftest.py` | correct GEMM and three faulty variants (acceptance) |
| `proof/` | WP24 checker: `encode.py`/`space.py` condition builder (Z3), `interp.py` replay, `contract.py` K1/K3, `models/` K2 models, `template.py`, `prove.py`, `compose.py` (Theorem 1), `candidate.py`, `wp.py`, `records.py` (premises, reuse, invalidation), `smt.py` |

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `models`, `measure`, `formulation`.
- Forbidden: `knowledge`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
