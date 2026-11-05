# Measured kernel exports

Standalone TileLang implementations with FP16 inputs/output and FP32 accumulation/statistics:

| Source | Entry point | Validated computation |
|---|---|---|
| [gemm_codex.py](gemm_codex.py) | `entry(a, b)` | [Row-major GEMM suite](../../docs/runbooks/learndemo-1/replay-cases/gemm.md) |
| [attention_codex.py](attention_codex.py) | `entry(q, k, v)` | [Forward causal attention suite](../../docs/runbooks/learndemo-1/replay-cases/attention.md) |
| [attention_dsh.py](attention_dsh.py) | `entry(q, k, v)` | The same forward causal attention suite |

These are reviewable exports of selected independent-session implementations. Import ordering, output argument naming and an equivalent unbounded compiled-kernel cache decorator were normalized for repository style. Each exported file was then compiled and run independently on GB10 through the pinned evaluator; all three shapes and every numerical pattern passed. [sources.json](sources.json) links the original immutable source and the new validation receipt. Compilation/toolchain dependencies come from the recorded worker image, not the Mac control environment.

Model-comparison tables refer to original source snapshots and their measured batches. Export validation retains its own source identities, inputs, raw samples and environment snapshots under `/Users/dingyi/kta-data/dev/template-replay/export-validation/`.

[vector_add_smoke.py](vector_add_smoke.py) is the minimal language example supplied in future isolated packages. It was independently compiled and checked on GB10; it contains no target-kernel implementation.

For a replay without a model call, create a fresh run directory, copy the corresponding original `manifest.json` and `preflight.json`, then run from the repository:

```bash
scripts/template-replay remote-evaluate NEW_RUN examples/template_replay/gemm_codex.py --agent-active-s 0 --host spark-hk
```

Use the attention case's receipts for either attention export. A fresh destination protects original evidence. The command preserves the source, validation errors, paired timings, raw samples and GPU queue/environment records. [Immutable originals and exact rerun commands](../../docs/runbooks/learndemo-1/results/implementations.md), [results](../../docs/runbooks/learndemo-1/results/summary.md).
