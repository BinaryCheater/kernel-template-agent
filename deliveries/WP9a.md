# WP9a delivery: read API and web frontend (code stage)

Branch `wp/WP9a`, with `integration` @ `65a0e2d` merged (WP0–WP5, WP10a, WP13 and the code-stage gate). Code commit of the evidence runs: `68300fcbaf910edd19e0c718e3b5cd1c406aa6c5` (clean). This is a **code-stage** delivery (D320): the user paused real runs, so everything below was exercised on `origin="fixture"` records only. The final read-only acceptance on WP7's real records is listed under "Pending real runs". Evidence: `deliveries/WP9a.evidence.json`, with outputs under `~/kta-data/dev/WP9a-fixture-68300fc/evidence/` on the control host.

## What was done

- **State machines as data** (`src/kta/orchestrator/machines.yaml`, D96/D163/D164): task types T1–T6 and the solve loop, expand loop and task lifecycle machines, with step/wait/terminal states, the rules on transitions, and the tasks each transition creates. It is now owned by WP6. The frontend draws the diagrams from `GET /machines`, so WP6's changes appear without frontend edits. Structural tests are in `tests/unit/test_orchestrator_machines.py` (evidence claim 0).
- **Read API** (`src/kta/api/`, D166/D170–D173): a FastAPI app over the real WP1 store and ops tables, listening on loopback only (`[api]` in `config/kta.toml`; `python -m kta.api` refuses other hosts). Endpoints:
  - `/meta` (env and record counts by origin);
  - `/machines`;
  - `/runs/active` (loops, ops task rows mapped onto the task lifecycle as in D172, agent slots, jobs from open spans, workers, GPU leases);
  - `/inbox`;
  - `/budget`;
  - `/trace`, `/trace/summary`, `/trace/critical-path/{trace}`;
  - `/records/{id}`, `/records`, `/search` and `/lineage/{id}` (the full evidence chain of a Solution);
  - `/model/residuals`;
  - `/cases` (per-row round progress, % of bound, active task, last activity) and `/cases/{id}/timeline`;
  - `/snapshots`, `/families?snapshot=` and `/families/{id}/chain`;
  - `/coverage?snapshot=`, which returns 501 when no coverage Report exists;
  - `/events`, an SSE stream of ops events and new records.

  Errors map to 404/501/403/400/500. Pages derive everything from records, ops rows and spans, following the conventions in `spec/modules/api-web.md` (D169). Evidence: claims 0 and 2.
- **Web frontend** (`web/`, D160–D162, D165, D167, D174; design in `web/DESIGN.md`):
  - Built with React 19, React Router, Vite and TypeScript, with exact versions installed by `npm ci`.
  - React Flow + elkjs draw the state machines and chain DAGs; ECharts draws the swimlanes and sparklines.
  - Pages:
    - Overview: a machine selector with live state counts per node, and a Now column (needs you, agent slots, GPU lease, workers and pools, budgets, just finished).
    - Running, Pipeline (swimlanes plus critical path).
    - Cases list and case page (round timeline, candidates, references).
    - Result: the evidence chain of a Solution.
    - Families list and family page (chain DAG, steps, glue constraints, versions).
    - Coverage.
    - A record drawer, opened with `?rec=` on any page.
  - The data layer reads the API only. Live updates come from SSE, with re-fetches throttled to 2 s; `?static=1` turns the stream off for headless screenshots.
  - A **FIXTURE DATA** banner shows whenever `/meta` reports non-real origins.
  - Evidence: claim 1 for the build; claim 2 for the endpoints it reads.
  - Screenshots at 1440×900 against the fixture store are in the coordinator's notes folder, under `WP9a-review-fixes/`.
- **Gate check** (`tools/kta_gate/specific.py`, `wp9a`): the named API tests pass, `npm run build` passes, and `web/src` contains no sample data or fixture markers.

## Fixtures and simplifications

- `tools/kta_devseed/` (`__init__.py`, `__main__.py`, `knowledge.py`, `world.py`, `evaluation.py`; all files marked `# kta:fixture`), run via `scripts/seed-dev` (D168).
  - Writes a synthetic 55-minute world: toy Templates, Families, Cases, loops, tasks, spans, jobs, evaluation records, PoolSnapshots and coverage Reports.
  - Every record has `origin="fixture"`. The seeder refuses non-empty stores and checks afterwards that it wrote nothing else.
  - It is used for frontend development and by `tests/unit/test_api_views.py`. None of it is evidence of system behaviour.
- The web frontend contains no fixture or sample data (enforced by the gate). It shows the FIXTURE DATA banner whenever the store holds non-real records.
- `src/kta/api/opsread.py` reads WP1's ops tables with SQL directly, because `Ops` has no listing methods yet (D171). This is not a fixture, but it is a stand-in until those methods exist (see "Needed changes").
- Inherited and unchanged: `src/kta/store/bench.py` (`_body`, D156).
- No skipped tests.

## Known issues

- **Only fixture data so far.** No page has rendered real loops, jobs or Solutions yet; layouts and derived numbers (round progress, % of bound, explained share, GPU idle) are checked only against the seed.
- **Task intermediate states are inferred** (D172).
  - The ops tasks table stores only queued, leased, done, failed and cancelled.
  - `running` means an open `agent_session` span. `rejected` vs `escalated` is decided by attempts against `max_retries`.
  - submitted, checking and expired cannot be shown until WP1 stores them.
- **Inbox** shows only the two triggers the API can derive. Trigger generation is WP6's.
- **`~/kta-data/dev/WP9a/` holds 3 PoolSnapshot records marked `origin="real"`.**
  - An early seed run wrote them through the store's `snapshot()`, which has no origin parameter.
  - Nothing reads that store. The seed now writes snapshots itself as fixtures, into `dev/WP9a-fixture*`.
  - It was left untouched because deleting data needs the user's permission. The user should delete it before WP9a's real-records acceptance, since the full gate reads `dev/WP9a/kta.db`.
- **No hot reload for the API.** `scripts/api` must be restarted after API code changes.
- The SSE stream polls the store once a second.
- Vite warns about a chunk over 500 kB (ECharts plus elkjs). It is harmless for a local tool.

## New decisions

D160, D161, D162, D163, D164 (needs user review), D165 (needs user review), D166, D167, D168, D169 (needs user review), D170, D171, D172, D173, D174.

## Needed changes

- **WP1 (store/ops):**
  - `Ops` listing methods for loops, tasks, workers, locks, budgets, spans overlapping a time window and open spans. `opsread.py` would then be removed (D171).
  - Storing intermediate task states (running, submitted, checking, expired).
  - An `origin` parameter on `snapshot()`, so non-real snapshots are not marked real.
- **WP6 (orchestrator):**
  - Write loops, steps, tasks, agent sessions and inbox triggers following the D169 conventions in `spec/modules/api-web.md` (loop `data` fields, `loop_step` spans per state, agent workers named `agent:<n>`, `agent_session` spans with `task_id`).
  - `machines.yaml` is now WP6's; the frontend reads it as is.
- **WP2, WP7, WP8, WP10b:** follow D169 for job spans and lanes, GPU lease locks, Solution refs, and the judgement and coverage Reports.
- **Gate (coordinator):** WP9a only reads records, so its full-stage evidence cannot point at records in `dev/WP9a/kta.db` written by WP9a's commit. The full stage needs a rule for read-only WPs, for example claims that cite WP7's store and records.
- **User review:** D164, D165, D169 (in `docs/review-queue.md`).

## Pending real runs

- **Read-only acceptance on WP7's real records** (after WP7's real run; D320). Point the API at the store WP7 writes and open every page:
  - Overview: machines and Now column against live loops, tasks, agent slots, GPU lease and workers.
  - Running.
  - Pipeline: real job and GPU lanes and the critical path.
  - A case timeline across real rounds.
  - The Result evidence chain of a real Solution: Measurement, Check, Bound and judgement Report.
  - Family chain.
  - Coverage, or a 501 if WP8 has written none.

  Confirm that the FIXTURE DATA banner disappears when only real records are present. Record the claims with real `record_ids` and rerun `scripts/gate WP9a` (full stage).
- Fix any D169 convention mismatches found in WP6's, WP7's and WP8's real records. These go either into the API or back to the producer WP.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate --stage code WP9a` on the control host, at the code commit plus these delivery files.

```text
gate WP9a --stage code: PASSED (23 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 7 locked files unchanged
  PASS pyright
  PASS unit tests — 316 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 214 decisions
  PASS skipped tests — none
  PASS evidence — 3 claim(s)
  PASS WP9a API tests — 9 tests passed
  PASS web build — tsc + vite build
  PASS no sample data in web/ — web/src reads the API only
```
