# kta.execution

Layer 3. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

The only path to compute: jobs submitted to local or ssh workers (pull: interface only), capability-label matching, artifact transfer by hash, exclusive GPU leases. The only module allowed to start subprocesses (besides the read-only env probe in core).

## Public interface

- `Job`, `JobResult`
- `Executor`: submit, wait, run, quiet_lease
- `register_worker(ctx, name, host, transport)`

## Dependencies

- May import: `core`, `store`, `targets`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
