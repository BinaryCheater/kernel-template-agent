# kta.checks

Layer 6. Specification: [spec/modules/measure-check.md](../../../spec/modules/measure-check.md). Built by: WP3.

## Responsibility

Independent checkers: numeric correctness, sanitizer, anti-cheat, SASS expectations, space verification. Must not import `knowledge` (a checker may not reuse the generator's logic).

## Public interface

- `check_numeric`, `check_sanitizer`, `check_anticheat`, `check_sass`, `full_check`
- `verify_space`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `models`, `measure`, `formulation`.
- Forbidden: `knowledge`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
