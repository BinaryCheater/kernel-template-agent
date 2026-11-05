# Time, tokens and API-equivalent cost

Retain native billing as unavailable: neither harness returned a dollar charge. API-equivalent estimates use the observed token counts with the published rate card frozen on 2026-11-05. This measures token economics separately from account billing and GPU cost.

GPT-6.1 Sol Standard rates per million tokens: $2 uncached input, $0.10 cached reads, $2.50 cache writes and $10 output. The estimator processes each native request, applies the documented long-context multipliers when applicable, and verifies that request sums equal the retained session totals. Completed runs so far have zero cache-write tokens and maximum request inputs below 272,000. Source: [OpenAI model pricing](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

DeepSeek Flash rates per million tokens are $0.15–0.30 uncached input, $0.003–0.006 cached input and $0.60–1.20 output. Report the off-peak/peak estimate range; account billing is separate. Source: [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing/).

Codex native input includes cached reads and its saved usage snapshots are cumulative. DSH native input excludes cache reads; normalize using totalTokens minus outputTokens. Final analysis uses the latest complete native assistant/message snapshot. Interrupted turns can lack a final stdout step-end event even when the native message has complete usage; reconciliation retains both sources. Preserve both native and normalized fields. Native reasoning tokens, cache writes and timestamps remain available where the harness supplies them.

Session wall time includes tools and GPU waiting. Report GPU queue separately; active time subtracts that recorded wait. Evaluation/transport durations overlap the session wall and must not be added to it. Compiler-only time is unavailable; first-call compilation/execution is measured separately from steady-state timing. Every accepted performance table comes from the evaluator's raw samples, with paired production/candidate measurements and median/IQR.

Reproduce an estimate with `PYTHONPATH=tools uv run --frozen python -m kta_template_replay.cost RUN`; it creates `RUN/api-equivalent-estimate.json`. The [pinned rates](../replay-cases/pricing-2026-11-05.json) are retained; complete native sessions and evaluator inputs remain under KTA_DATA.
