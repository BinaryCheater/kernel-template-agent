# kta.api

Layer 9. Specification: [spec/modules/api-web.md](../../../spec/modules/api-web.md). Built by: WP9a, WP9b.

## Responsibility

HTTP API (read and operation endpoints) for the web frontend; holds the single store write connection.

## Public interface

- read endpoints and SSE (WP9a): `read.py`, `events.py`; views in `runs.py`, `trace.py`, `cases.py`, `pool.py`
- `app.py`: `create_app(api)`, `open_api(ctx)`; `python -m kta.api` serves it (`scripts/api`)
- record conventions the views read: spec/modules/api-web.md "Record conventions" (D169)
- operation endpoints (WP9b): `actions.py` (routes, `GET /actions/capabilities`, `/actions/log`),
  `operate.py` (what each operation writes), `wiring.py` (the `Orchestration` port: `Wired`
  calls WP6's task queue and inbox and WP8's profile pointer, batch verify and task tokens,
  D355; what WP6 does not offer yet raises NotSupported), `auth.py` (`POST /write/record`)
- WP9b read views: `views.py` routes to `taskview.py`, `agentsview.py`, `studies.py`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`, `agents`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
