# Run and inspect Template replay

Use the canonical [library](../../../knowledge/library/README.md), [frozen cases](replay-cases/) and [measured results](replay-results.md). The experiment uses native persistent Codex/DSH sessions and direct remote GPU commands; no frontend or running API service is needed.

The existing environment is `/Users/dingyi/kta-data/dev/template-replay/`, with the GB10 reachable through the configured `spark-hk` alias and the recorded pinned worker image. Dependencies stay in the project's locked environment. Register the Mac's agent pool for the current code when preparing another campaign:

```bash
KTA_DATA=/Users/dingyi/kta-data KTA_ENV=dev KTA_WP=template-replay KTA_MODE=dev-mac uv run --frozen kta worker register mac --json
```

To build another isolated package, use `scripts/template-replay prepare CASE RUN --environment ENV.json --seeds SEED1 SEED2`, execute the evaluator-owned `worker.py --preflight` remotely under the shared GPU lock, and seal its returned receipt with `scripts/template-replay seal-preflight RUN RECEIPT.json`. `package RUN WORKSPACE --library LIBRARY` supplies full catalog access; repeated `--template NAME@VERSION` supplies a prescribed subset. The public package excludes evaluator source, private inputs, source studies and prior solutions. Direct remote compilation/API inspection is available; target validation and timing use `harness/evaluate` so the declared budget and raw samples are retained.

New packages also contain `harness/language_smoke.py`, a standalone vector-add example compiled and checked on GB10. Its receipt is retained in `/Users/dingyi/kta-data/dev/template-replay/environment/public-vector-example/`. This addition prepares future sessions; historical experiment packages remain unchanged.

Immediately before a new implementation session, write `RUN/session-started.json` with the current `epoch_s`. Keep the initial prompt in `RUN/prompt.md` and launch:

```bash
KTA_DATA=/Users/dingyi/kta-data KTA_ENV=dev KTA_WP=template-replay KTA_MODE=dev-mac scripts/template-session start --workspace WORKSPACE --state RUN/session-state --prompt RUN/prompt.md
```

Use `--provider deepseek_harness` for the installed DSH. Use `resume` with the same state and workspace to continue its native session. Homes and transcripts remain persistent; credentials stay in the native provider store and are excluded from archives. Working/source read isolation is instruction-only and audited through native tool records. Model identities are fixed to Sol/high and Flash/high and resolved settings are retained.

After closure, `PYTHONPATH=tools uv run --frozen python -m kta_template_replay.report RUN REPORT.md` checks final source against measured snapshots and emits the report plus a retained summary. The [accounting method](results/accounting-method.md) explains native reconciliation and API-equivalent estimates. Original snapshots remain immutable; append a new analysis filename when correcting an earlier interpretation.

Every completed run has an immutable `selected/` directory with the exact evaluated candidate, design note, selection receipt and `RERUN.md`. Follow that file to replay in a new directory without an AI call or changes to original evidence. `attempts/` preserves every evaluator source/configuration, failure, numerical check, environment snapshot and raw timing sample. `session-state/` retains prompts, model/harness settings, native conversations, tools, timestamps and available usage.
