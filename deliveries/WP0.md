# WP0 delivery: repository skeleton and toolchain

Branch `wp/WP0`, based on `integration` @ `a6a4283`. Evidence runs used commit `cff3726` (clean).

## What was done

- **Python toolchain locked.** `.python-version` (3.12.13), `pyproject.toml` (uv, `uv_build`; runtime deps pydantic, fastapi, uvicorn, highspy, pyyaml, numpy; dev deps ruff, `pyright[nodejs]`, import-linter, pytest, types-pyyaml), `uv.lock`. Installs only via `uv sync --frozen` into `.venv`. Real run: `scripts/bootstrap` locally and `uv sync --frozen` on spark (evidence claims 0 and 2).
- **Module skeleton.** `src/kta/<module>/__init__.py` and `README.md` for all 15 modules (responsibility, public interface, allowed dependencies, trimmed from spec/modules).
- **Import layers enforced.** `[tool.importlinter]` in `pyproject.toml`: the 10-layer contract with same-layer siblings independent (`a | b`), plus "checks must not import knowledge". Acceptance tests `test_gate_blocks_layer_violation` and `test_gate_blocks_checks_importing_knowledge` run the real config on a synthetic tree.
- **core: errors, registry, config, context, logging, secrets, doctor, EnvManifest.**
  - `core/errors.py`: the conventions §3.3 hierarchy, `ExecutionError(kind)`.
  - `core/registry.py`: `Registry` on `ctx.registry`; `build_registry()` collects `register_implementations` hooks from `src/kta` packages only; fixture flags must match where the factory lives; fixtures are refused in dev/main (D77).
  - `core/config.py`, `core/context.py`: TOML config with content hash; `Context` with env, config, data dir (`KTA_DATA`/`KTA_WP`, D88), registry, clock, per-name seeded rng.
  - `core/logging.py`: JSON lines at `<data>/logs/<process>/<date>.jsonl`, required fields, contextvars for loop/task/job/session, secret redaction, DEBUG only in dev.
  - `core/envmanifest.py`, `core/locks.py`, `core/envprobe.py`: every EnvManifest field from spec/schemas.md (`backends` stays empty until WP2); lock comparison against `.python-version`, `uv.lock` (installed versions) and `web/.nvmrc` once `web/` exists; git commit/dirty, or `.kta-commit` in deployed trees.
  - `core/doctor.py` + `kta doctor [--json] [--role]` (`cli/main.py`, `cli/output.py`): lists interfaces and selected implementations; fails on fixture selection in dev/main and on lock mismatch or dirty code in main; exit codes per conventions §12. Real runs: control host (claim 1) and spark worker (claim 2, CUDA 13.0.3, driver 580.95.05, nvcc/ptxas 13.0.88, no lock mismatches).
- **Gate.** `scripts/gate [--quick] <WP>` → `tools/kta_gate` (D79). All common checks of spec/gate.md: ruff format/check, pyright (strict for core/store/orchestrator), import layers, `pytest -m "not gpu"`, residue, fixture declarations, acceptance lock, delivery sections, decision log (incl. review-queue cross-check), skipped tests, test-double placement and `src`→`tests` imports, kta doctor in dev, contract tests + fake coverage, error handling AST, print AST, secrets regex, evidence format and store verification (D84). Extra check: subprocess use outside `kta.execution` (D78). WP-specific checks implemented for WP0, WP10a (D87) and WP13 (D86); other WPs fail until they add theirs (D85). Unit tests in `tests/unit/test_gate_delivery.py`.
- **WP0 acceptance: the gate blocks each listed violation** (`tests/acceptance/WP0/test_gate_blocks.py`): layer violation, `FakeX` class in src, bare except (and swallowed broad except), print in src, a fake API key (generated at runtime, never printed back), a fixture implementation selected in dev (resolve raises `PermissionDenied`, doctor fails). A control test shows the real repository passes the scans.
- **`scripts/remote-run`** (`tools/kta_remote`, hosts in `config/remote.toml`, D89): `exec`, `bg`, `status`, `logs`, `sync` (committed HEAD only, writes `.kta-commit`). Real runs: sync + exec on spark (claim 2), exec on slai (claim 3), bg/status/logs on spark (claim 4).
- **`scripts/bootstrap`**: checks uv 0.11.x, installs a missing pinned interpreter only into `.uv/python` (D90), `uv sync --frozen`, `npm ci` once `web/` exists, creates the data dirs, runs `kta doctor`. Re-runnable (claim 0).
- **Repository hygiene.** `.gitignore` (venvs, caches, data, databases, logs, workspaces, agent-home, artifacts, `.env`, `secrets.env`, keys, `.claude/` worktrees); `tests/fakes/` and `tests/contract/` with READMEs and one example (`ManualClock`, contract-tested against `SystemClock`); `deliveries/evidence.schema.json`; empty `tests/acceptance/LOCK`.
- **Construction skills.** `.agents/skills/run-gate`, `write-delivery`, `remote-run` (generic part; WP2 completes data dirs and worker registration).
- **Environment check** (before WP0): `deliveries/ENV-CHECK.md`.

Evidence outputs are kept outside git under `~/kta-data/dev/WP0/evidence/cff37267acacbfd3f1e3c845194fcab9820f4e38/` on the control host.

## Fixtures and simplifications

None. No `# kta:fixture` markers. `tests/fakes/clock.py` (`ManualClock`) is a test double for the clock boundary and is contract-tested.

## Known issues

- No store yet: EnvManifest is printed by doctor but not written as a record (WP1). WP0 evidence therefore has no `record_ids`.
- The store branch of the evidence check (record exists, origin real, `actor.commit`/`actor.dirty`, GPU worker label) is unit-tested only for the missing-store paths; WP1 must exercise it against a real database and name the Actor fields as in D84.
- `EnvManifest.backends` is empty until WP2 probes TileLang and other backends.
- spark has uv 0.11.2, the control host 0.11.28; both read `uv.lock` without mismatch, and the manifests record each version.
- slai has no uv; nothing was deployed there beyond the empty project directory `/storage/kta-work` (WP2).
- `core/trace.py` (spans) and `core/ids.py`, `core/expr.py`, `core/schemas/` are WP1.
- Directories created during WP0: `~/kta-work` (spark), `/storage/kta-work` (slai), `~/kta-data` (control host, D92). Nothing was installed outside project directories; spark's uv download cache is its default location.
- During construction, `ruff format .` briefly reformatted Python-like code blocks inside Markdown specs; this was reverted before any commit, and `*.md` is now excluded from ruff.

## New decisions

D77 (needs user review), D78, D79, D80, D81, D82, D83, D84 (needs user review), D85, D86, D87, D88, D89, D90, D91, D92.

## Needed changes

- spec/conventions.md §3.2 revised for the registry object (D77, done in this WP; pending user review).
- WP1: name the Actor fields `commit` and `dirty` (D84); provide `kta.core.schemas.validate_body(kind, body)` so the WP10a gate check validates YAML against the models instead of only parsing it.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate WP0` on the control host, after the evidence runs.

```text
gate WP0: PASSED (22 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 0 locked files unchanged
  PASS pyright
  PASS unit tests — 40 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — no # kta:fixture markers
  PASS decision log — 92 decisions
  PASS skipped tests — none
  PASS evidence — 5 claim(s)
  PASS module READMEs — 15 modules
  PASS gate violation samples — 6 samples blocked
```
