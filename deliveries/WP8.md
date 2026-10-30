# WP8 delivery: agents and runtime skills (code stage)

Branch `wp/WP8`, cut from `integration` @ `90aa678`, fast-forwarded to `65a0e2d` (D320), then merged with `integration` @ `868885b` for the scope addition D321/D322 (runtime skill catalogue, skill backup, batch environments). This is a **code-stage** delivery (D320: the user paused real data and measurements on 2026-10-31). No real Codex session has run. The code paths below are real; tests drive them with a stand-in for the CLI (`tests/fakes/fake_codex.py`). Real-run acceptance (T1 with WP7's chain, resume, induced write, user-level marker) is listed under "Pending real runs".

## What was done

Three kinds of claims below, labelled as such: code that exists and is tested with the fake CLI ("code"), facts read from the real Codex CLIs without any model call ("checked"), and nothing claimed as a real run.

- **Agent step through the Executor (code, D300).** `run_step` / `run_task` (`src/kta/agents/runner.py`) does the following:
  - loads the TaskCard and its AgentProfile;
  - decides between a new session and resume (`sessions.py`, D306);
  - builds the step bundle (`workspace.py`, D314);
  - issues a task token (`guard.py`, D305) and a whitelisted environment;
  - runs the launcher as an Executor `shell` job in the `agent` pool of `[agents] worker`;
  - writes an AgentRun record (refs `env`, `worker`, `card`, `profile`, `previous`; blobs `prompt`, `stdout`, `workspace`; time split `wall_s`/`tool_s`/`queue_s`);
  - updates the session state;
  - renews the task lease while the job runs.
- **Launcher on the agent host (code).** `python -m kta.agents.launch <job dir>` (`launch.py`, `hostfs.py`) runs on the agent host. It:
  - refuses to start when an AGENTS.md lies above the workspace;
  - installs the bundle into `<data>/work/<wu>/` and creates `<data>/agent-home/<wu>/` from the profile's home files only on first use;
  - symlinks credentials into the CLI home;
  - writes the context manifest before the CLI starts;
  - runs the CLI with stdin = prompt, cwd = workspace and `CODEX_HOME` = agent-home, optionally as `kta-agent` via sudo with umask 007;
  - enforces the wall limit, the idle limit and the GPU guard (D311, D316);
  - commits the workspace with git before and after the step;
  - restores credentials a CLI refreshed by rename;
  - packs the CLI home without credentials or symlinks, and the workspace, deterministically.
- **Codex adapter (code + checked, D303).**
  - `adapters/codex.py` builds `codex exec [flags] -` and `codex exec resume [flags] <id> -` and refuses the non-persistent session option.
  - `adapters/codex_session.py` locates rollout files, takes the thread id from `--json` output and parses session files into AgentEvents (D304).
  - The other adapters (`claude_code`, `pi`, `deepseek_harness`) keep the interface and raise `NotSupported`.
  - `adapters/capabilities.py` holds the admission table. pi 1.3.0 is installed on the control host and was checked from `pi --help` only.
  - `tests/contract/test_codex_cli_contract.py` checks every option the adapter uses against the real `codex exec --help` and `codex exec resume --help`. It passed against Codex 0.163.0 on this Mac; the options were also read from 0.117.0 help on spark.
- **Codex sandbox on spark (checked, D317).** I ran `codex sandbox linux` on spark with codex-v1's `config.toml` as `CODEX_HOME`. This is the sandbox runner only, with no model call, done by hand with ssh and the probe directory removed afterwards.
  - The default bubblewrap sandbox fails there (`kernel.apparmor_restrict_unprivileged_userns = 1`).
  - With `features.use_legacy_landlock` (now in the profile), workspace writes succeed, writes outside are denied, and network is blocked by default.
  - Reads anywhere are allowed.
  - The config file was parsed without error.
- **Sessions (code).** Each work unit has one session, pinned to one worker. The session state lives in the ops pointer `agent-session/<wu>`. The new-session rules are tested one by one (`test_session_rules`).
- **Context manifest (code).** It records, with hashes:
  - every workspace file;
  - the CLI home files (credentials as "credential: provided");
  - the CLI command, sha256 and version;
  - the environment (the token as `<provided>`);
  - AGENTS.md files above the workspace;
  - the user-level `~/.codex` (present, not used);
  - `/etc/codex/*`.
  After the run, protected files (AGENTS.md, TASK.md, inputs/, skills) are compared with the manifest.
- **Compliance accounting (code, D311).** `AgentRun.violations` records the kinds `denied`, `kta_command_not_allowed`, `forbidden_tool`, `write_outside_workspace`, `gpu_outside_kta`, `notes_not_updated`, `workspace_layout`, `context_file_modified` and `secret_in_session`. The end-to-end test induces an out-of-workspace write, and both the redirect analysis and the watched-path snapshot record it.
- **Work-unit close (code, D313).** `units.close_work_unit` archives the workspace (with git history) and the CLI home in a Report with subject `work_unit_archive`. It deletes the directories only after the stored hashes are confirmed.
- **AgentProfile `codex-v1` (code, D308, D309).**
  - `profile.write_profile` is idempotent. Its blobs are the prompt templates in `src/kta/agents/prompts/`: fixed prefix, step headers T1–T6, messages (rejection, budget_warning, human_note, new_session), the NOTES seed and the isolated `config.toml`.
  - Skills are referenced as `<name>@<sha256>` of a stored deterministic archive.
  - `approve_profile` sets `current_profile` only with a HumanAction.
  - No profile record has been written to a dev store yet.
- **`kta-agent` setup script (code, D312).** `scripts/kta-agent-setup` does a dry run unless `--apply` is given. **Not run on any machine**: `--apply` creates a system user, a group, ACLs and a sudoers rule, which needs your approval.
- **Batch environments (code, D321, D322, D318).**
  - `kta.agents.batches.build` writes `$KTA_DATA/<env>/batches/<batch>/` from one AgentProfile record's blobs only: `env.lock.json`, `agent-home-template/` (exactly `home_files`), `skills/<name>@<hash>/` and `prompts/`. It refuses an existing name.
  - `verify` regenerates the template into a temporary directory and compares it with the lock and with the files on disk.
  - `run_step` refuses a batch that fails verification, requires the batch's profile to be the card's profile, keeps a work unit in one batch, and takes prefix, step and message templates and skills only from the batch root.
  - On the agent host, the launcher keeps a copy of the batch root, checked file by file against the same lock, with `work/<wu>` and `agent-home/<wu>` inside it.
  - Credentials are linked per work unit at start; a credential-named home file is refused from the template. Locks, templates, blobs and archives hold no credentials.
- **Skill backup (code, D322).** `AgentProfile.skill_bundles` (added to `kta.core.schemas.process`, additive) maps every `<name>@<sha256>` to its deterministic tar.gz blob, written when the profile is recorded. `batches.export_profile` writes the profile and every blob back out as files.
- **`kta agents` commands (code, D319).** `kta.agents.commands.register(sub, make_ctx, emit)` adds `export-profile`, `env build` and `env verify`. Wiring the hook into `kta`'s parser is WP6's (see "Needed changes").
- **Runtime skill catalogue (D321).** `src/kta/agents/skills/README.md` lists name, task types, author WP and status, including `diagnose-model` and `write-microbench` (WP11, planned). It is kept apart from the construction skills in `.agents/skills/`.
- **Runtime skills.**
  - New: `use-kta`, `find-knowledge`, `write-family`, `write-template`.
  - `decompose-kernel` finished and translated to English. Its output directories are now `templates/` / `patterns/` (WP10a's request). It adds the compiler-decided transaction class (D106), the counting rule (D102), configuration scoping and the HwFeature naming rule (D108).
  - All skills contain method only.
- **Gate (D315).** `tools/kta_gate/specific_wp8.py` checks:
  - the named WP8 tests passed;
  - the five skills exist with front matter;
  - no non-persistent session option appears in `src/`;
  - full stage only: real AgentRuns with a resume, an induced violation, credential-free archives with a session file, and a manifest blob.

  The checks are unit-tested in `tests/unit/test_gate_wp8.py`.
- **Tests.**
  - `tests/unit/test_agents_runner.py`: start → resume → induced write → close through the real local Executor, plus refusal on an upward AGENTS.md.
  - `tests/unit/test_agents_units.py`: 23 unit tests.
  - `tests/contract/test_codex_cli_contract.py`: 3 contract tests.
  - `tests/unit/test_gate_wp8.py`: 4 gate tests.
  - `tests/unit/test_agents_batches.py`: the same profile and commit give byte-identical batch roots on two stores; existing and malformed names are refused; tampered, added and missing files fail verification; credentials are refused from templates; export writes every blob.
  - `test_runs_only_from_a_verified_batch`: with the source prompts and skills directories made unreachable and a marked `~/.codex/AGENTS.md` in the worker's HOME, a step still runs and the marker appears in no archive, workspace or manifest; a tampered skill file in the batch refuses the next start.

### Round 2 (merged `integration` @ `0475dea`: WP6 wiring, WP9b, WP10b, WP11)

All of this is code tested with the stand-in CLI; there were no real runs (D320).

- **Experiment arms (D430).** `kta.agents.arms` reads the arm from the task bookkeeping (`info["arm"]` in the ops pointer `task:<id>`, which the WP6 session is adding; the shape WP8 reads was sent to that session).
  - `zero`: input records of pool kinds, `find-knowledge` and `kta query` are left out.
  - `text`: the same, and the pinned snapshot rendered by `knowledge.render_text_knowledge` goes under `inputs/knowledge/`.
  - `pool`: unchanged.
  - Task-subject records are inputs in every arm.
  - The condition is recorded in the context manifest (`arm`), the new `AgentRun.arm` field (additive) and `refs.snapshot`.
  - `test_each_arm_shapes_workspace_manifest_and_agent_run` runs one step per arm through the real launcher and checks how the manifests differ: Template input, the find-knowledge skill and the text book are present exactly where specified; the fixed prefix is equal for `zero` and `text` and differs for `pool`. `test_arm_of_task_info` covers the bookkeeping forms and refusals.
- **decompose-kernel (D431)** now states WP10b's D381: `patterns/` are written on acceptance; `templates/` drafts start T3 tasks and are never written directly.
- **codex-v2 (D432).** I checked WP11's `codex-v2` against D321/D322. Its skill references equal its `skill_bundles`; it differs from `codex-v1` only in name, version, T4/T5 skills and bundles; both build and verify a batch (`test_codex_v2_has_bundles_and_builds_a_verified_batch`). The WP8 gate's skill check now includes `diagnose-model` and `write-microbench`.

## Fixtures and simplifications

- `tests/fakes/fake_codex.py` replaces the Codex CLI (the LLM boundary) in tests. It is never used from `src/`. It has no classes, so the "fake coverage" gate does not see it; `tests/contract/test_codex_cli_contract.py` checks it against the real CLI's options.
- `# kta:fixture` markers in `src/`: only the one inherited from WP1, `src/kta/store/bench.py` (`_body`, synthetic store-benchmark bodies, D156). WP8 adds none.
- Allowed simplification used: only the Codex adapter is implemented; `claude_code`, `pi` and `deepseek_harness` raise `NotSupported`.
- Not implemented (`NotSupported`): isolation `container`; a provider other than codex in `run_step`.
- The GPU device permission of `kta-agent` is not restricted. If `/dev/nvidia*` are world-accessible on DGX OS (the setup script reports this), the GPU guard detects GPU processes under the agent and stops them (spec/modules/agents.md allows this fallback; D311, D312).

## Known issues

- **The agent-side `kta` has no channel yet (D301).** With the agent on spark and the control plane on the Mac, `kta` inside the agent must talk to the API with the task token. Neither the CLI client mode (WP6) nor the token check in the API (WP9) exists. Until both do, a real agent step needs `[agents] worker` to be a `local` worker on the control host, where `kta` opens the store directly.
- **Codex behaviour not yet seen in a KTA run:**
  - whether `web_search="disabled"` and the feature switches take effect;
  - whether workspace `.agents/skills` are discovered;
  - whether `network_access = true` lets `kta` through under Landlock;
  - whether `resume` appends to the same rollout file (the parser handles both);
  - how the `--json` output and the session files of 0.117 look for an actual turn.
- **Sandbox reads are open (D317).** The Landlock sandbox does not restrict reads; only the `kta-agent` user (not yet created) keeps the store and user files unreadable.
- **The GPU guard uses `nvidia-smi --query-compute-apps`.** On GB10 with unified memory, whether that lists compute processes is untested.
- **Out-of-workspace write detection is heuristic for shell commands.** It parses redirects and `tee`, plus a depth-1 snapshot of the watched paths; a write by a script to an unwatched path is only caught when the sandbox denies it.
- **The `config` isolation path is the tested one.** The `config+user` path (sudo, umask, ACLs) is written but has not run.
- **Recurring violations are not routed yet.** Moving repeated violations into the pending queue is the orchestrator's inbox (WP6); `compliance.summary` gives the per-run counts.

## New decisions

D300, D301, D302, D303, D304, D305, D306, D307, D308, D309, D310, D311, D312, D313, D314, D315, D316, D317, D318, D319 (D318 implements the coordinator's D322); round 2: D430, D431, D432. Needing your review: D300, D301, D305, D307, D308, D310, D312, D317, D430 (in `docs/review-queue.md`).

## Needed changes

- **WP6 (cli, orchestrator):**
  - In `kta.cli.main.build_parser`, call `kta.agents.commands.register(sub, context_from_environ, emit)`, where `emit(args, data, text)` prints JSON with `--json`, else the text (D319). Until then `kta agents ...` does not exist on the command line; the functions are tested directly.
  - When `KTA_TASK_TOKEN` is set, `kta` runs as an API client: it uploads the files its arguments name and sends the token with every call (D301).
  - The orchestrator calls `kta.agents.runner.run_step(ctx, task, retry=RetryNote(...), note=...)` from a process at layer 9, binding a tracer, and moves task states after it returns.
  - A command for `profile.approve_profile` (human) and `units.close_work_unit`.
- **WP9 (api):** verify agent requests with `kta.agents.guard.verify_task_token(key, token, now, ops)`, with the key from `[agents] token_key_file` (D305).
- **Lower modules:** `kta.core.schemas.process.AgentProfile` gained `skill_bundles: dict[str, str]` (default empty), as spec/schemas.md §4.9 now specifies, and `AgentRun` gained `arm: dict` (default empty, D430; spec/schemas.md §4.9 updated). Both are additive. Otherwise none.
- **WP6 (round 2):** write `info["arm"] = {"name", "knowledge": "zero"|"text"|"pool", "snapshot"}` into the task bookkeeping of experiment tasks (a plain name plus `info["arm_condition"]` also works). `kta query` and the solve loop's retrieval must honour the pinned snapshot; the agents module only shapes what it puts into the workspace. `config/kta.toml` gained an `[agents]` section, which now also holds WP9a's `slots = 2` (one table, the merge had declared it twice), and `agent = 2` in spark's pools (additive). The `Ops` Protocol has no `task()` (SqliteOps has it, D152); `guard` uses its own small Protocol for it instead of changing `kta.store.base`.
- **spec/modules/agents.md:** an English addendum records D300–D304 (where the CLI runs, `run_step`, the extra adapter methods, event kind `other`).

## Pending real runs

What the full gate (`KTA_DATA=~/kta-data scripts/gate WP8`) still requires. Steps marked "approval" are system or credential operations you have to approve or do.

1. **Approval, credentials:** place the Codex subscription credentials for KTA in `~/.config/kta/auth/codex/auth.json` (mode 600) on the agent host. On spark that means copying `~/.codex/auth.json` or running `codex login` with `CODEX_HOME` set to that directory.
2. **Approval, system change:** run `scripts/kta-agent-setup --data ~/kta-work/data/dev/WP8 --apply` on spark (system user, group, ACLs, sudoers). The alternative is to use a profile version with isolation `config` for the first runs.
3. After WP6 (task queue, `kta` commands, CLI client mode) and WP7 (GEMM chain) are merged into `integration`: deploy with `scripts/worker-setup spark`, write `codex-v1` into the WP8 dev store, run `kta agents env build wp8-first --profile <id>` and `kta agents env verify wp8-first`, then enqueue and run a T1 starting from WP7's GEMM family with one dtype or shape condition changed, from lease to accepted or correctly rejected.
4. Reject once and retry through `resume`, and compare the two wall times.
5. Induce one out-of-workspace write (a human note), and attempt a direct GPU call from the workspace; both must appear in `AgentRun.violations`.
6. **Approval, user config:** put a marked `AGENTS.md` in spark's user-level `~/.codex/` for the duration of one T1 run, and check that the marker is absent from the session events. `~/.codex` on spark is shared with other projects of the `memx` account.
7. Check that the context manifest equals the workspace contents, that the archive unpacks with no credential file, and that `parse_session` parses it. List the AgentRuns in `deliveries/WP8.evidence.json`.

## Gate output

`KTA_DATA=~/kta-data scripts/gate --stage code WP8` on `wp/WP8` @ a12732d (round 2, the commit before this note's gate paste):

```
gate WP8 --stage code: PASSED (22 pass, 2 warn, 0 fail)
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
  PASS unit tests — 530 passed
  PASS contract tests — 5 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 322 decisions
  PASS skipped tests — none
  WARN evidence — evidence list is empty
  PASS WP8 tests — 14 named tests passed
  PASS runtime skills — 7 skills
  PASS non-persistent session options — none in src/
  WARN WP8 agent runs — pending real run (code stage): real T1, resume, induced write
```

The two WARNs are expected in the code stage: the evidence list is empty (no real runs yet) and the WP8 agent-run check is pending (see "Pending real runs").
