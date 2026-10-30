# kta.cli

Layer 9. Specification: [spec/modules/orchestrator-cli.md](../../../spec/modules/orchestrator-cli.md). Built by: WP0 (`doctor`); WP1 (`store`); WP2 (`worker`, `target`, `backend`); WP6.

## Responsibility

The `kta` command: a thin layer that parses arguments, calls lower layers and formats output. Every subcommand supports `--json`; exit codes follow conventions §12. `output.py` is the only place in `src/` that prints.

## Public interface

- `kta doctor [--json] [--role control|worker]` (WP0)
- `kta store bench [--records N] [--queries N] [--seed N] [--role] [--bundle PATH] [--json]`, `kta store export IDS --out PATH`, `kta store import BUNDLE` (WP1, D157)
- `kta worker register NAME`, `kta worker list`, `kta worker serve` (pull daemon: NotSupported) (WP2, D197)
- `kta target probe` (WP2, D188, D199)
- `kta backend smoke [--backend cuda|tilelang|all] [--seed N]`, `kta backend selftest lease|timeout|cpu-compile` (WP2, D193)
- `kta measure ...`, `kta check selftest` (WP3, D234); `kta init` (WP4, D212)
- WP6 management (D274): `kta case add YAML`, `kta reference add YAML`, `kta solve CASE [--steps N]`,
  `kta expand REFERENCE [--steps N]`, `kta loop list|show|step`, `kta process start|stop NAME`
  (`solve`, `maintain`; D273), `kta task list|show|create|submit|check|recheck|cancel|retry` (D278) (human/CLI
  submission path, D269), `kta inbox`, `kta budget [SCOPE] [--set key=limit]`,
  `kta coverage [--snapshot]`, `kta export SNAPSHOT DIR`
- WP6 agent-facing (no task-token auth yet): `kta show`, `kta query pool|tmpl|chain`, `kta why`,
  `kta card`, `kta note`, `kta submit [--dir out]`, `kta validate-template`, `kta validate-family`;
  `instantiate`, `build`, `sass` on a Family record (D371); `run`, `src` raise NotSupported
- WP6 follow-up: `kta runner start|stop` (agent runner, D369; per slot, quota order, task batch and cancel
  watcher in round 3, D372, D374, D375), `kta profile approve`, `kta unit close`,
  `kta agents ...` (registered from `kta.agents.commands`, D319); `kta case add` reads the seed case
  format (D363); client mode when `KTA_TASK_TOKEN` is set (`client.py`, D370)

- WP11 (`process_cmds.py`): `kta learn add|backlog`, `kta improve run`, `kta experiment status`,
  `kta calibrate hw`, `kta quotas`, `kta profile write`.

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`, `agents`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
