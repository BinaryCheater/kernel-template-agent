# WP9b delivery: operations and second-version frontend (code stage)

Branch `wp/WP9b`, with `integration` @ `0475dea` merged (WP0–WP8, WP9a, WP9b rounds 1–2, WP6's wiring, WP10a, WP10b, WP11, WP13; code-stage gates). Code commit of the evidence runs: `dbec6a3c187eec3b475b473350f00a9a5d5b1856` (clean). This is a **code-stage** delivery (D320): the user paused real runs, so everything was exercised on `origin="fixture"` records only. Evidence: `deliveries/WP9b.evidence.json`, outputs under `~/kta-data/dev/WP9b-fixture-dbec6a3/evidence/` on the control host.

**Round 3: every operation is wired.**
- WP6 side: cancel, retry, create-task, priority, pause/resume and re-verification go through WP6's `Operations`.
- WP8 side: the profile pointer, batch verify and task tokens call `kta.agents` directly.
- Agent commands: client mode has token endpoints for show, query, validate, build and submit.
- One partial: cancel does not stop an agent step already running.

## What was done

- **Operation endpoints** (`src/kta/api/actions.py`, `operate.py`):
  - `POST /actions/{pause,resume,cancel,retry,budget,priority,create-task,reject,flag,approve,confirm}` as in `spec/modules/api-web.md`, plus `POST /actions/deprecate` and `POST /actions/reject-improvement`.
  - Every operation writes one HumanAction naming the operator. A missing or blank required reason (reject, deprecate, budget, reject-improvement, and the retry note) is a 400.
  - `GET /actions/capabilities` (all 17 operations wired; `partial` on cancel) and `GET /actions/log`.
  - Done by the API itself, because the store alone defines their effect: reject (dependents become stale), flag (`suspect`), deprecate, confirm, approve a protocol, approve or reject an improvement, budget limits.
- **WP6 and WP8 wiring** (`src/kta/api/wiring.py` `Wired`, D355, D358):
  - Cancel, retry, create-task (a `human:<id>` work unit, priority 100), priority, pause/resume and re-verification call `kta.orchestrator.operations.Operations`.
  - WP6 writes each HumanAction as the operator. The API's own `create_task` record from round 2 is gone.
  - Flagging a Solution, or a Measurement, Check or Bound a Solution rests on, starts WP6's re-verification: a new solve loop per Case (WP6 D368). Other kinds are flagged only, and the response says why (D343).
  - `/inbox` comes from `kta.orchestrator.inbox.inbox`, including WP11's `improvement_pending` items, each with approve and reject.
  - `approve_profile`, `batches.verify` and `guard.verify_task_token` are called directly. The token key comes from `[agents] token_key_file`, and the API never creates it.
- **Agent token endpoints** (`src/kta/api/agentapi.py`, D352) for WP6's client mode (WP6 D370):
  - `GET /agent/records/{id}`, `/agent/cards/{id}`, `/agent/why/{id}`;
  - `POST /agent/query`, `/agent/validate-template`, `/agent/validate-family`, `/agent/instantiate`, `/agent/build`, `/agent/sass`, `/agent/submit`.
  - Each needs a valid task token for a live task and uses the same functions as the local command.
  - Each call is a `cli` span on `process:api` under the agent's `X-KTA-Trace-Parent`, so it appears on the Task page.
  - Files are sent as `{path: base64}`: relative paths only, 32 MiB in all.
  - I sent the endpoint shapes to the WP6 session. The CLI client (`src/kta/cli/client.py`) is WP6's to change.
- **Read views:**
  - Task detail, blobs, agents (skills catalogue, profile versions, pointer history, batch locks), pool, experiments, improvements.
  - New: `GET /quotas` (process shares configured and in force, WP11 D401).
  - Experiments carry `progress` from the experiment process's state (status, runs started and finished, result, last error; WP11 D403). Planned runs per arm = runs_per_arm × cases.
  - The coverage endpoint already returns WP10b's `pending` and `report` (D354).
- **Web frontend** (`web/`; `web/DESIGN.md` §6):
  - Pages: Inbox, Task, Operations, Pool, Model, Experiments, Improvement, Agents, as in rounds 1–2.
  - Round 3 additions:
    - Coverage shows the direct test (references, all steps matched, nothing unexplained) and "Not measured yet" from `pending`, plus open Gaps by `missing` and `attribution`.
    - Inbox and Improvement have Approve and Reject for the system items of an improvement report.
    - Experiments show the process state.
    - Operations has a process-quota panel; Pause and Resume now work.
    - Running and Task have a Priority control.
    - A flag's result names the re-verification loops it started.
  - Checked in the browser on a fixture store: Operations and Experiments render without console errors (round-2 checks of retry, start T1, cancel and budget still apply).
  - Screenshots of the round-1 pages (fixture data) are in the coordinator's notes folder under `WP9b-code-stage/`.
- **Gate check** (`tools/kta_gate/specific_wp9b.py`, D346): 29 named API tests (operations through WP6, agents functions, token endpoints, improvement decisions, experiment state and quotas); every spec operation exercised; clean-HEAD web build; no sample data in `web/src`; the real-run acceptance is a WARN in the code stage.

## Fixtures and simplifications

- `tools/kta_devseed/supervision.py` (marked `# kta:fixture`, D349) and WP9a's `tools/kta_devseed/world.py` (lifecycle task states and `done_checks` Reports, D357) write the fixture world the pages are developed against. All records are `origin="fixture"`.
- `Unwired` in `src/kta/api/wiring.py` is only the default of a bare `Api` in tests. It raises NotSupported and is never a fake success.
- `tests/apihelp.py` drives the ASGI app in-process instead of adding httpx (D353).
- Inherited and unchanged: `src/kta/store/bench.py` (`_body`, D156).
- No skipped tests.

## Known issues

- **Only fixture data so far.** HumanActions and the AgentProfile written by the evidence runs into fixture stores are origin `real`: real actions, on fixture subjects.
- **Cancel does not stop a running agent step.** The Executor has no job cancellation (needs WP2 `JobExecutor.cancel`, as WP6 noted).
- **WP6's `Operations.request_reverify` writes its HumanAction before it checks the kind.** The API therefore only requests re-verification for Solutions and what they rest on. I told the WP6 session.
- **No `/agent/*` call has been made by a real agent.** `instantiate`, `build` and `sass` need a worker and a Target, so their unit tests only check the token. The CLI side of client mode is WP6's and not changed yet.
- **The operator is always `user`** (`[api] operator` is not in `config/kta.toml`).
- **Leftover partial fixture stores,** not deleted (deleting data needs the user's permission). Nothing reads them:
  - `~/kta-data/dev/WP9b-fixture` and `WP9b-fixture-b`: from two seed runs that failed halfway;
  - `WP9b-fixture-6c8a990`: an evidence directory whose smoke run stopped early.
- **Local preview servers are left running for the user:** API on 127.0.0.1:7320 over `dev/WP9b-fixture-d`, Vite on 127.0.0.1:5183 (`.claude/launch.json`, git-ignored).
- **Decision IDs D340–D359 are used up.** Round 3 revised D343, D345, D352, D354, D355 and D358 in place, each marked "revised in WP9b round 3", instead of opening new IDs. Further decisions need a new range.
- The Pool page filters by snapshot membership. Vite warns about a chunk over 500 kB.

## New decisions

D340, D341, D342 (needs user review), D343 (needs user review), D344, D345 (needs user review), D346 (needs user review), D347, D348 (needs user review), D349, D350, D351, D352, D353, D354, D355, D356, D357, D358, D359. Revised in round 3: D343, D345, D352, D354, D355, D358.

## Needed changes

- **WP6:**
  - use the `/agent/*` endpoints in `kta.cli.client` (shapes sent to the WP6 session);
  - write the `reverify` HumanAction only after the kind is known to be re-verifiable.
- **WP2:** `JobExecutor.cancel(job_id)`, so cancel can stop a running agent step (WP6 then watches it in `kta runner`).
- **Coordinator:** a new decision-ID range for WP9b's next round.
- **Schema (user review, D342):** HumanAction has no parameters field.
- **WP12:** write experiment records following D345 and WP11 D403.
- **User review:** D342, D343, D345, D346, D348 (in `docs/review-queue.md`).

## Pending real runs

- **From the frontend against real records in `dev/WP9b`:**
  - start a real T1 from Operations, starting from WP7's GEMM family;
  - cancel a task;
  - reject a real result.

  Check the HumanAction, StatusChange (stale marks) and TaskCard records. List them in the evidence manifest, add screenshots of real records, and run `scripts/gate WP9b` (full stage).
- **A real agent through client mode:** `kta show`, `query`, `validate-*`, `build`, `sass` and `submit` via `/agent/*` with its task token. The spans should appear under its session on the Task page.
- **Flag a real Solution** and follow the re-verification loop.
- **Pause and resume a process** while it runs.
- **Change a priority and see the lease order follow it.**
- **Approve a real profile pointer change;** verify a real batch.
- **Approve and reject real improvement items;** follow a real experiment's progress.
- **See real coverage** with `pending` emptied once instruction coverage and gap explanation have data.
- **Check that the FIXTURE DATA banner disappears** on a store with only real records.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate --stage code WP9b` on the control host, at the code commit plus these delivery files.

```text
gate WP9b --stage code: PASSED (24 pass, 1 warn, 0 fail)
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
  PASS unit tests — 538 passed
  PASS contract tests — 5 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 319 decisions
  PASS skipped tests — none
  PASS evidence — 3 claim(s)
  PASS WP9b API tests — 29 tests passed
  PASS operations tested — every spec operation is exercised
  PASS web build — tsc + vite build of HEAD in a clean directory
  PASS no sample data in web/ — web/src reads the API only
  WARN WP9b real operations — pending real run (code stage): start a T1, cancel a task and reject a result from the UI against real records in dev/WP9b; screenshots in the delivery note
```
