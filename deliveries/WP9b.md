# WP9b delivery: operations and second-version frontend (code stage)

Branch `wp/WP9b`, with `integration` @ `528caab` merged (WP0–WP8, WP9a, WP9b's first round, WP10a, WP13; code-stage gates). Code commit of the evidence runs: `6c8a99013a8a170ed133b703f09ecc6d50e339f9` (clean). This is a **code-stage** delivery (D320): the user paused real runs, so everything was exercised on `origin="fixture"` records only. Evidence: `deliveries/WP9b.evidence.json`, outputs under `~/kta-data/dev/WP9b-fixture-6c8a990/evidence/` on the control host.

**This round wired the API to the merged WP6 and WP8 code (D355–D359).** Of the 16 operations, 13 are wired; priority, pause and resume still answer 501 NotSupported and write nothing. Flagging works, but its re-verification task is not created yet. The WP6 session is adding these; when it is merged, only `src/kta/api/wiring.py` changes.

## What was done

- **Operation endpoints** (`src/kta/api/actions.py`, `operate.py`, D340–D344, D347):
  - Endpoints: `POST /actions/{pause,resume,cancel,retry,budget,priority,create-task,reject,flag,approve,confirm}` as in `spec/modules/api-web.md`, plus `POST /actions/deprecate`.
  - Every operation writes a HumanAction. The parameters go in the action text (`<verb> <JSON>`, D342) and the subject goes in refs.
  - `reason` is required for reject, deprecate and budget; `note` is required for retry. A missing or blank value is a 400, and so is any malformed body.
  - Carried out by the API itself, because the store alone defines their effect:
    - reject: a human StatusChange, and the dependents become stale;
    - flag: a StatusChange to `suspect` (D343);
    - deprecate;
    - confirm a Pattern or Structure;
    - approve a Protocol or an improvement Report;
    - budget limits.
  - `GET /actions/capabilities` lists each operation: what it writes, whether it is wired, what it needs, and `partial` (what a wired call does not do yet). `GET /actions/log` parses every writer's HumanActions, including WP6's `task <verb> <id>`.
- **Wired to WP6 and WP8** (`src/kta/api/wiring.py` `Wired`, built by `open_api`, D355, D356, D358):
  - **cancel and retry**: WP6 `TaskQueue.cancel` / `TaskQueue.retry`. The lifecycle is checked: cancelling a terminal task is a 400. WP6 writes the HumanAction (`task cancel <id>`).
  - **create-task**: WP6 `TaskQueue.create`, in a `human:<ulid>` work unit, process `human`, default priority 100 (as `kta task create`). The API adds a `create_task` HumanAction with `refs.card`.
  - **`/inbox`**: WP6's `kta.orchestrator.inbox.inbox`, with `task`, `at`, last rejection and the controls added. WP9a's two derived triggers are gone (D357).
  - **approve a profile** (`/actions/approve {profile}`): WP8 `approve_profile`, which writes the HumanAction and moves `current_profile`.
  - **batch verify** (`POST /agents/batches/{batch}/verify`): WP8 `kta.agents.batches.verify`, which regenerates the batch and compares it with its lock.
  - **`POST /write/record`**: WP8 `guard.verify_task_token`, with the key at `[agents] token_key_file`. The API never creates the key: without one every token is refused.
  - **runtime skill catalogue**: `src/kta/agents/skills/README.md`, with its planned skills.
  - **Still NotSupported**: priority, pause, resume, and re-verification after a flag.
  - Evidence: claims 0 and 2.
- **Task states and rejections from WP6's records** (D357): tasks are shown in the lifecycle state stored in ops `tasks.state` (D265). The session and attempt heuristics of D172 are removed. Rejections are the task's `done_checks` Reports (D276), on the Task page, the case timeline and the inbox.
- **Read views:**
  - `GET /tasks/{id}`: card, spans, kta calls, AgentRuns, done-check verdicts, your actions.
  - `GET /blobs/{sha}`.
  - `GET /agents`: skills catalogue, profile versions with changed fields, pointer history, batch locks.
  - `GET /pool`, `GET /experiments`, `GET /improvements` (record conventions D345, D354).
- **Web frontend, second version** (`web/`, D348, D359; `web/DESIGN.md` §6):
  - Pages: Inbox, Task detail, Operations (start a task, pause/resume, budgets, action log, capability table), Pool, Model, Experiments, Improvement, Agents.
  - Controls on the record drawer, Running, Task, Result, Experiments, Improvement and Agents.
  - One dialog for every operation, with the reason marked where it is required. The result opens in a top-level modal, so it stays visible when its subject leaves the page.
  - Unwired controls are dashed and say what they need. Wired controls with `partial` say what they don't do yet.
  - Checked in the browser on a fixture store: retry an escalated task from the Inbox; start a T1 from Operations; open the new task; cancel it from the Task page; change a budget. Each showed the HumanAction and the new state.
  - Screenshots of the first round's pages (fixture data, banner visible) are in the coordinator's notes folder under `WP9b-code-stage/`: `01-inbox.png` … `09-running.png`.
- **Gate check** (`tools/kta_gate/specific_wp9b.py`, D346):
  - The named API tests passed. These now include the wired task queue, profile, batch, token and inbox tests.
  - Every spec operation is exercised.
  - The web build of HEAD runs in a clean directory, and `web/src` holds no sample data.
  - In the code stage the real-run acceptance is a WARN.
  - Tests are in `tests/unit/test_gate_wp9b.py`.

## Fixtures and simplifications

- `tools/kta_devseed/supervision.py` (marked `# kta:fixture`, D349) extends WP9a's `scripts/seed-dev`. It writes:
  - patterns, a structure and gaps;
  - two AgentProfile versions, with `current_profile` set to v1;
  - two AgentRuns with a prompt blob;
  - experiment protocols with runs and a result Report;
  - an improvement Report;
  - one batch lock with `"fixture": true`.

  All records are `origin="fixture"`, used only so that the pages have something to render.
- `tools/kta_devseed/world.py` (WP9a's fixture world, marked `# kta:fixture` in its package) now writes lifecycle task states and `done_checks` Reports instead of WP1's old states and Check records (D357).
- `Unwired` in `src/kta/api/wiring.py` remains the default of a bare `Api` in tests: it raises NotSupported and is never a fake success. `Wired` inherits it for the operations WP6 does not offer yet.
- `tests/apihelp.py` drives the ASGI app in-process instead of adding httpx (D353).
- Inherited and unchanged: `src/kta/store/bench.py` (`_body`, D156).
- No skipped tests.

## Known issues

- **Only fixture data so far.** HumanActions written by UI checks and evidence runs into fixture stores have origin `real`, and so does the AgentProfile the smoke script wrote: these are real actions, on fixture subjects.
- **Cancel does not stop a running agent process,** and WP6 names the cancel and retry HumanAction actors `kta task cancel` / `kta task retry`, not the operator (WP6 follow-up in progress).
- **The retry note is not passed to TASK.md yet.** It is only stored in the HumanAction.
- **Priority, pause, resume and re-verification answer 501** until WP6 provides them.
- **The operator is always `user`** (`[api] operator` is not set in `config/kta.toml`).
- **Leftover partial fixture stores, not deleted** (deleting data needs the user's permission). Nothing reads them:
  - `~/kta-data/dev/WP9b-fixture` and `~/kta-data/dev/WP9b-fixture-b`: from two seed runs that failed halfway;
  - `~/kta-data/dev/WP9b-fixture-6c8a990`: holds the evidence outputs; its smoke run stopped before any operation.
- **Local preview servers are left running for the user:** API on 127.0.0.1:7320 over `dev/WP9b-fixture-c`, Vite on 127.0.0.1:5183 (`.claude/launch.json`, git-ignored). They use non-default ports because WP9a's servers hold 7310 and 5173.
- **The Pool page filters by snapshot membership.** The fixture patterns are not members.
- Vite still warns about a chunk over 500 kB.

## New decisions

D340, D341, D342 (needs user review), D343 (needs user review), D344, D345 (needs user review), D346 (needs user review), D347, D348 (needs user review), D349, D350, D351, D352, D353, D354, D355, D356, D357, D358, D359.

## Needed changes

- **WP6 (in progress on `wp/WP6`):**
  - accept the human `Actor` on TaskQueue operations (then `Wired` passes the operator);
  - a priority update;
  - pause/resume for the `global`, `process:`, `loop:` and `task:` scopes;
  - a re-verification task for a record set to `suspect`;
  - stopping the agent process on cancel;
  - the retry note in the next TASK.md;
  - `TaskQueue.create` could write the human's HumanAction itself, so the API's `create_task` record can go (D358).
- **WP9b after that merge:** fill the remaining methods of `Wired` and drop `PARTIAL` entries (`src/kta/api/wiring.py` only).
- **Schema (user review, D342):** HumanAction has no parameters field.
- **WP10b, WP11, WP12:** write experiment, improvement and confirmation records following D345.
- **User review:** D342, D343, D345, D346, D348 (in `docs/review-queue.md`).

## Pending real runs

- **From the frontend against real records in `dev/WP9b`:**
  - start a real T1 from Operations, starting from WP7's GEMM family;
  - cancel a task from Running or Task detail;
  - reject a real result from the Result page.

  Check the HumanAction, StatusChange (stale marks) and TaskCard records. List them in the evidence manifest, add screenshots of real records, and run `scripts/gate WP9b` (full stage).
- **Approve a profile pointer change** on real profiles.
- **Verify a real batch environment.**
- **Write a record through `/write/record`** from a real agent's `kta` with its task token.
- **After WP6's follow-up:** flag a result and see the re-verification task; retry with a note and see it in TASK.md; cancel a running agent and see the process stop; pause and resume.
- **Check that the FIXTURE DATA banner disappears** on a store with only real records.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate --stage code WP9b` on the control host, at the code commit plus these delivery files.

```text
(filled in after the run)
```
