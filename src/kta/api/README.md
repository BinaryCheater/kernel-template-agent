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
  delegates to WP6's `Operations` and inbox and calls `kta.agents` for the profile pointer,
  batch verify and task tokens, D355), `auth.py` (`POST /write/record`), `agentapi.py` (the
  token endpoints `/agent/*` for the agent commands in client mode, D352)
- WP9b read views: `views.py` routes to `taskview.py`, `agentsview.py`, `studies.py`

## Settings and release (WP21)

- `controls.py`: `GET/POST /controls`, `GET /controls/versions`, `GET/POST /release`; writes need the operator token (`X-KTA-Operator-Token`, `[api] operator_token_file`).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`, `agents`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).

## WP14 (D449, D452)

`GET /pool/diff?a=&b=` (`pool.diff` over `kta.knowledge.pooldiff`); `common.body` and `GET /records/{id}` serve upcast bodies (`stored_body`, `upcast_from` for old records).

## WP22 (D587–D591)

`system.py`: `GET/POST /system`, `POST /system/proxy/test`, `POST /system/workers/{name}/test` (Settings → System). `mechanisms.py`: `GET /mechanisms` (the Mechanisms page: learning stages, FormulationPlan diffs, selection arms, canaries, what-if).
