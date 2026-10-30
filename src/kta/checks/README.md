# kta.checks

Layer 6. Specification: [spec/modules/measure-check.md](../../../spec/modules/measure-check.md). Built by: WP3.

## Responsibility

Independent checkers: numeric correctness, sanitizer, anti-cheat, SASS expectations, space verification. Must not import `knowledge` (a checker may not reuse the generator's logic).

## Public interface

- `check_numeric`, `check_sanitizer`, `check_anticheat`, `check_sass` (keyword `ctx`, D229), `full_check` (keywords `case_id`, `shape`, `build_for`, D227)
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
| `inputs.py` | adversarial input distributions |
| `selftest.py` | correct GEMM and three faulty variants (acceptance) |

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `models`, `measure`, `formulation`.
- Forbidden: `knowledge`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
