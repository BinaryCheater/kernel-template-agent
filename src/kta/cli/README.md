# kta.cli

Layer 9. Specification: [spec/modules/orchestrator-cli.md](../../../spec/modules/orchestrator-cli.md). Built by: WP0 (`doctor`); WP1 (`store`); WP6.

## Responsibility

The `kta` command: a thin layer that parses arguments, calls lower layers and formats output. Every subcommand supports `--json`; exit codes follow conventions §12. `output.py` is the only place in `src/` that prints.

## Public interface

- `kta doctor [--json] [--role control|worker]` (WP0)
- `kta store bench [--records N] [--queries N] [--seed N] [--role] [--bundle PATH] [--json]`, `kta store export IDS --out PATH`, `kta store import BUNDLE` (WP1, D157)
- remaining subcommands (WP6)

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`, `agents`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
