---
name: run-gate
description: Run scripts/gate for a work package, read its output, and fix failures without weakening checks. Use before every commit (--quick) and before delivering a WP (full).
---

# Run the gate

## Commands

- Every commit: `scripts/gate --quick <WP>` (ruff, import layers, src scans, secrets, acceptance lock).
- Before delivery: `KTA_DATA=<data root> scripts/gate <WP>`. The full run adds pyright, `pytest -m "not gpu"`, contract tests, `kta doctor` in dev, delivery-note, decision-log, evidence and WP-specific checks.
- `KTA_DATA` must point at the data root holding `dev/<WP>/kta.db` for work packages whose evidence needs records (WP1–3, 7, 8, 9a, 9b, 10b, 11, 12). Without it the gate uses an empty temporary directory and those claims fail.

## Reading the output

Each line is `PASS`, `WARN` or `FAIL` with a reason and up to 20 findings (`file:line: what`). The block after `---- summary ----` goes verbatim into the delivery note's "Gate output" section. Exit code 0 means no FAIL.

## Fixing failures

| Check | Fix |
| --- | --- |
| ruff format / ruff check | `uv run ruff format .`, `uv run ruff check --fix .`, then fix what remains by hand |
| pyright | fix the types; core, store and orchestrator are strict |
| import layers | move the code to the right layer or pass the dependency through `Context`; never add an `ignore_imports` entry to get past it |
| residue | finish the code or raise `NotSupported("<reason>")`; a line may keep a marker only with `# kta:allowed <reason>` |
| test doubles | move the class to `tests/fakes/` and add a contract test |
| error handling | re-raise or wrap in a `KtaError` subclass; do not return defaults |
| print | log via `logging`, or output through `kta.cli.output` |
| subprocess outside Executor | go through the Executor (`kta.execution`) |
| secrets | remove the value from the file and from git history before committing; never print it |
| acceptance lock | locked acceptance tests cannot change; ask for the change under "Needed changes" |
| fixture declarations | list every `# kta:fixture` file in "Fixtures and simplifications" |
| decision log | rows need 4 non-empty cells and unique IDs; every `needs user review` row needs a review-queue entry |
| evidence | rerun the claim from a clean commit in `KTA_ENV=dev` and record its IDs |
| `<WP> specific checks` not implemented | add the WP's checks from spec/gate.md to `tools/kta_gate/specific.py`, with tests |

Never relax an assertion, tolerance or gate rule to get a PASS. If a rule is wrong, write it under "Needed changes".
