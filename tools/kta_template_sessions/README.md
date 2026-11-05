# Recorded Template-replay sessions

Use an external frozen run package with `TASK.md`. Keep state outside that package:

```sh
KTA_DATA=/Users/dingyi/kta-data KTA_ENV=dev KTA_WP=template-replay KTA_MODE=dev-mac \
  scripts/template-session start --workspace /absolute/run-package \
  --state /absolute/run-state --prompt /absolute/prompt.md
```

Replace `start` with `resume` and supply the next prompt to continue the same native
session. Default limits: 3600 seconds active wall time, 600 seconds idle time. Queue
wait is separate. The package must have no ancestor AGENTS.md. The Mac agent pool
must be registered for the current code. The fixed model is `gpt-6.1-sol`, effort
`high`; native turn contexts retain the actual resolved settings. Fast service mode
is not enabled.

The experiment-local CodexAdapter extension uses the existing argv/parser and
Executor; runtime workspace/spec contracts are unchanged. It keeps native provider
configuration and credentials local, copies no personal instructions/skills, and
archives only native session transcripts. It leaves the package's top-level files
in place. Full-access permits the experiment's authorized direct remote operations.
Workspace/source/network restrictions are instructions, not enforceable read
isolation. Supply those rules in the frozen package; do not expose evaluator source.

`state/session.json` is published only after native identity/transcript retention.
Each turn directory retains prompt/settings, manifest, native archive, stdout,
stderr, report and result. AgentRun and spans also go to the dev Store. Failed turns
remain; interrupted native transcripts remain in `state/home/sessions/` for resume.
Native tool calls/timestamps and cumulative token/cache totals stay in transcripts.
Results additionally report active wall time, queue wait and differences in totals
for each resumed turn. Costs remain null because Codex reports no price. Do not sum
cumulative session totals across resumed turns.

Verified persistence smoke: `session-smoke-state` under
`/Users/dingyi/kta-data/dev/template-replay/`; a fresh turn answered READY and a
separate resume recalled the phrase from its previous context. Both native turn
contexts resolve to the requested model and effort. An initial CLI flag error is
retained separately as a failed AgentRun.

DSH is selected with `--provider deepseek_harness` and the same package/state/prompt
interface. Its installed CLI is
`/Applications/DeepSeek Harness.app/Contents/Resources/runtime/cli/bin/dsh`, version
`0.2.1-rc.3`. Installed model metadata maps `deepseek-flash` to
`DeepSeek-V4.1-Flash`; native desktop settings use `deepseek-account` with high
effort. The experiment overlay enables that native account/credential service,
freezes the same model/effort, redirects persistence to the state directory, and
disables automatic title-model calls. It preserves native credentials in place.
The overlay is validated with `--dump-config`. Real start/resume smoke passed: native
request headers independently confirm deepseek-account/deepseek-flash/high, and the
resumed session recalled its earlier phrase. Verification and normalized token
counts are in `dsh-session-smoke-state/smoke-verification.json` under the same data
directory. A first launcher parsing failure is preserved in native session artifacts;
DSH emits diagnostics between JSON stdout lines, which the adapter now retains.

DSH headless emits per-step usage, including retries; results sum these for the
current invocation and never subtract an earlier turn. Native compressed session
artifacts retain the complete events. The stdout adapter does not reconstruct
native tool durations or timestamps; AgentRun's required tool_s field is zero and
its arm explicitly marks that field unavailable. Raw native artifacts retain the
data for later analysis. Missing token/cache usage and cost remain null.

DSH's native inputTokens counts uncached input, while Codex input includes cached
input. The launcher normalizes DSH total input to native totalTokens-outputTokens;
it retains native uncached/cache-write details and raw step usage separately. Smoke
results written before that discovery remain unchanged; smoke-verification.json
provides corrected comparable input totals. Native `.zstd` sessions use concatenated
frames; `export-dsh.cjs` expands all frames with their original event timestamps and
records native request settings. Exported transcripts are also retained automatically
for subsequent DSH runs.
