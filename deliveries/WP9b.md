# WP9b delivery: operations and second-version frontend (code stage)

Branch `wp/WP9b`, cut from `integration` @ `868885b` (WP0–WP5, WP9a code stage, WP10a, WP13). Code commit of the evidence runs: `acba0bd62aa47f9d220f908d09ae09a1599a15eb` (clean). This is a **code-stage** delivery (D320): the user paused real runs, so everything was exercised on `origin="fixture"` records only. WP6 (orchestrator, task queue) and WP8 (agents, profiles, batch environments) were not merged into this branch. Every operation that needs them answers 501 NotSupported and writes nothing; none is simulated (D341). Evidence: `deliveries/WP9b.evidence.json`, outputs under `~/kta-data/dev/WP9b-fixture-acba0bd/evidence/` on the control host.

## What was done

- **Operation endpoints** (`src/kta/api/actions.py`, `operate.py`, D340–D344, D347):
  - Endpoints: `POST /actions/{pause,resume,cancel,retry,budget,priority,create-task,reject,flag,approve,confirm}` as in `spec/modules/api-web.md`, plus `POST /actions/deprecate`.
  - Every operation writes a HumanAction (actor `human`, named by `[api] operator`, default `user`). The parameters go in the action text (`<verb> <JSON>`, D342) and the subject goes in refs.
  - `reason` is required for reject, deprecate and budget; `note` is required for retry. A missing or blank value is a 400, and so is any malformed body (request validation errors are mapped to 400).
  - Carried out by the API itself, because the store alone defines their effect:
    - **reject**: a human StatusChange, and the dependents become stale through `kta.store.stale.reject`;
    - **flag**: a StatusChange to `suspect` (D343);
    - **deprecate**: a StatusChange, no propagation;
    - **confirm** a Pattern or Structure;
    - **approve** a Protocol or an improvement Report;
    - **budget**: the limits of an existing scope, merged, with the previous limits kept in the HumanAction.
  - A status change is pre-checked against the store's rules, so a refused change writes nothing. HumanAction and StatusChange records cannot be judged.
  - `GET /actions/capabilities` lists each operation: what it writes, whether it is wired, and what it needs. `GET /actions/log` lists HumanActions newest first, with the verb and parameters parsed.
  - Evidence: claims 0 and 3.
- **Delegated operations** (`src/kta/api/wiring.py`, D341, D351):
  - An `Orchestration` port names each call that belongs to WP6/WP8: cancel, retry, create-task, priority, pause/resume, re-verification of a flagged record, approve a profile pointer change (`/actions/approve {profile}`), batch verify, and task-token verification.
  - Its implementation `Unwired` raises NotSupported naming the missing function. The request is still validated first, so an unknown task is 404 and a bad body is 400.
  - Evidence: claims 0 and 3, with 501 on cancel and create-task.
- **Task-token write endpoint** (`src/kta/api/auth.py`, D352): `POST /write/record` is 403 without a bearer token. With one, it is 501 until WP8's guard is wired. It only accepts a record whose `actor.task` is the token's task.
- **New read views** (`views.py`, `taskview.py`, `agentsview.py`, `studies.py`, D345, D350, D354):
  - `GET /tasks/{id}`: card, loop, the task's spans and their descendants, kta calls and jobs, records written in the task (AgentRun, Submission, Check), rejections, time composition, and HumanActions on the card.
  - `GET /blobs/{sha}`: text blobs, e.g. an AgentRun's prompt (TASK.md).
  - `GET /agents`: runtime skill directories with front matter, plus the catalogue rows when `skills/README.md` exists. Also AgentProfile versions with the fields changed between versions, the `current_profile` history from the ops log and approvals, and the batch environments' `env.lock.json`. Verify is delegated.
  - `GET /pool?snapshot=`: Patterns, Templates, Structures, Gaps and References, with reconstruction coverage and confirmations.
  - `GET /experiments`: protocols, arms with run counts, results and approvals.
  - `GET /improvements`: Gap attribution counts overall and per day, and the improvement reports.
  - `/inbox` items now carry `task`, `detail` and `actions`.
  - Evidence: claims 0 and 3.
- **Web frontend, second version** (`web/`, D348; design in `web/DESIGN.md` §6):
  - One operation dialog for every control (`web/src/components/actions.tsx`). It shows the subject, what will happen, the records written, the reason (marked required where the API requires it), and the result: the HumanAction as a record link, the records marked stale, and follow-ups not created.
  - Unwired operations are drawn as dashed, disabled buttons tagged "not wired", with what they need as the tooltip.
  - Pages:
    - **Inbox**: items by urgency with suggestions and controls; the ten design/30 §2 triggers, with which ones are derived today.
    - **Task detail** `/tasks/:id`: the task's own swimlane, time stats, kta calls, task card, done checks with failures, AgentRuns with the prompt blob, records written, your actions.
    - **Operations**: start a task (T1–T6 type cards, goal, input records with family quick-picks, priority), pause/resume by scope, a budget editor per scope and resource, the action log and the capability table.
    - **Pool**: tabbed tables with per-row flag/reject/deprecate/confirm, and a snapshot filter.
    - **Model**: predicted vs measured scatter and the largest errors.
    - **Experiments**: protocol cards with arm progress, results, and approve-and-freeze.
    - **Improvement**: Gap attribution per day, attribution meanings, improvement reports with approval.
    - **Agents**: runtime skills × task types matrix and skill cards, profile versions with changed fields and "make current", pointer history, and batch environments with their lock and verify state.
  - Controls on existing pages:
    - record drawer: flag, reject, deprecate, confirm, approve;
    - Running: links to Task detail, plus cancel and retry;
    - Result: flag and reject;
    - Overview: "Needs you" links to the Inbox.
  - The sidebar is regrouped and no page is disabled any more.
  - A budget change made through the UI was seen end to end: dialog, then HumanAction, then the new limit, then the action log.
  - Screenshots at 1440 px against the fixture store, with the FIXTURE DATA banner visible, are in the coordinator's notes folder under `WP9b-code-stage/`:
    - `01-inbox.png`, `02-tasks.png`, `03-operations.png`, `04-pool.png`, `05-model.png`, `06-experiments.png`, `07-improvement.png`, `08-agents.png`, `09-running.png`.
  - Evidence: claim 1 for the build.
- **Gate check** (`tools/kta_gate/specific_wp9b.py`, registered in `specific.py`, D346):
  - The named API tests passed, and every spec operation path is exercised in `tests/unit/test_api_actions.py`.
  - WP9a's clean-HEAD web build, and no sample data in `web/src`.
  - In the code stage the real-run acceptance is a WARN.
  - The full stage requires listed, real, clean human HumanActions in `dev/WP9b` to start a task, cancel a task and reject a result, and a screenshot path in this note.
  - Tests are in `tests/unit/test_gate_wp9b.py`.

## Fixtures and simplifications

- `tools/kta_devseed/supervision.py` (new, marked `# kta:fixture`, D349) extends WP9a's `scripts/seed-dev`. Two lines were added to `tools/kta_devseed/__main__.py` to call it. It writes:
  - patterns (one in category `other`), a structure, and four more gaps;
  - two AgentProfile versions, with `current_profile` set to v1 by the seeder;
  - two AgentRuns with a prompt blob;
  - an experiment protocol with runs and a result Report, plus a draft protocol;
  - an improvement Report;
  - one batch lock with `"fixture": true` under the store's data directory.

  All records are `origin="fixture"`. They exist only so that the new pages have something to render, and none of them is evidence.
- **Not a fixture, but a stand-in:** `Unwired` in `src/kta/api/wiring.py` raises NotSupported for every operation that needs WP6/WP8. It never returns a result.
- `tests/apihelp.py` drives the ASGI app in-process instead of adding httpx (D353).
- Inherited and unchanged: `src/kta/store/bench.py` (`_body`, D156).
- No skipped tests.

## Known issues

- **Only fixture data so far.** No operation has acted on a real record. The HumanActions that UI checks and evidence runs wrote into fixture stores have origin `real`: they were real actions, on fixture subjects. Their stores are `dev/WP9b-fixture-a` and `dev/WP9b-fixture-acba0bd`, and nothing reads them.
- **9 of 16 operations are not wired** until WP6/WP8 are merged: cancel, retry, create-task, priority, pause, resume, approve-profile, env-verify and write-record. Flagging works, but its re-verification task is not created.
- **The inbox still derives only two of the ten triggers** (D170). The rest come with WP6's inbox generator.
- **The operator is always `user`.** `[api] operator` is not set in `config/kta.toml`, which this WP does not own.
- **WP6's TaskQueue writes its own HumanAction** (`task cancel <id>`, actor name `kta task cancel`). Once wired, a cancel from the UI would not name the operator unless WP6 accepts a human actor (see "Needed changes").
- **Leftover directory `~/kta-data/dev/WP9b-fixture`:** a seed run that failed halfway (the integration schema has no `skill_bundles` yet) left a partial fixture store there. It was not deleted, because deleting data needs the user's permission. Nothing reads it.
- **Local preview servers are left running for the user.** They are the API on 127.0.0.1:7320 over `dev/WP9b-fixture-a` and Vite on 127.0.0.1:5183, started from `.claude/launch.json` (git-ignored). They use non-default ports because WP9a's servers occupy 7310 and 5173.
- **The Pool page filters by snapshot membership.** The fixture patterns are not snapshot members, so they show only with "include records outside the snapshot".
- **Runtime skills named by a profile but absent from the source tree** show as "no source" in the Agents matrix. In integration only `decompose-kernel` exists, and the catalogue README (D321) is WP8's.
- Vite still warns about a chunk over 500 kB.

## New decisions

D340, D341, D342 (needs user review), D343 (needs user review), D344, D345 (needs user review), D346 (needs user review), D347, D348 (needs user review), D349, D350, D351, D352, D353, D354.

## Needed changes

- **WP6 (orchestrator):**
  - `TaskQueue.cancel/retry/recheck/create` should accept the human `Actor` (and the retry note for TASK.md), so the HumanAction names the operator. Otherwise the API would have to write a second one.
  - A human-created task needs a work unit and process. Today `create(req, loop_id, process)` needs a loop.
  - Priority updates on queued tasks (with WP1 `Ops`).
  - Pause/resume semantics for the `global`, `process:`, `loop:` and `task:` scopes, honoured by the processes.
  - A re-verification task for a record set to `suspect`.
  - The runner stops the agent process on cancel.
  - Expose `inbox(svc)` so `/inbox` can use it.
- **WP8 (agents):**
  - Merge `approve_profile`.
  - A `verify_batch(batch) -> dict` behind `kta agents env verify`.
  - Token key access for `verify_task_token` from the API process.
  - The runtime skill catalogue table in `src/kta/agents/skills/README.md`.
  - Confirm D350: batches live under `<env data dir>/batches/`.
- **WP9b follow-up after the merges:** replace `Unwired` with a wired `Orchestration` that calls those functions (`src/kta/api/wiring.py` only; endpoints and frontend unchanged).
- **WP1:** an `Ops` method to set a task's priority. The `Ops` listing methods from D171 are still open.
- **Schema (user review, D342):** HumanAction has no parameters field. Adding `params: dict` would replace the `<verb> <JSON>` encoding.
- **WP10b, WP11, WP12:** write experiment, improvement and confirmation records following D345.
- **User review:** D342, D343, D345, D346, D348 (in `docs/review-queue.md`).

## Pending real runs

- **From the frontend against real records in `dev/WP9b`** (after WP6 and WP8 are merged and `Unwired` is replaced):
  - start a real T1 from Operations, starting from WP7's GEMM family;
  - cancel a task from Running or Task detail;
  - reject a real result from the Result page.

  Check that the HumanAction, StatusChange (and stale marks) and TaskCard records have the right fields and actor. List them in the evidence manifest, add screenshots of real records, and run `scripts/gate WP9b` (full stage).
- **Approve a profile pointer change** and check the HumanAction and the `current_profile` move.
- **Verify a real batch environment** built by `kta agents env build`.
- **Flag a real result** and check that WP6 creates the re-verification task.
- **Retry with a note** and check that the note reaches the next TASK.md.
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
  PASS unit tests — 358 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 231 decisions
  PASS skipped tests — none
  PASS evidence — 3 claim(s)
  PASS WP9b API tests — 15 tests passed
  PASS operations tested — every spec operation is exercised
  PASS web build — tsc + vite build of HEAD in a clean directory
  PASS no sample data in web/ — web/src reads the API only
  WARN WP9b real operations — pending real run (code stage): start a T1, cancel a task and reject a result from the UI against real records in dev/WP9b; screenshots in the delivery note
```
