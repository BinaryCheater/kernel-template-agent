# kta.execution

Layer 3. Specification: [spec/modules/target-exec.md](../../../spec/modules/target-exec.md). Built by: WP2.

## Responsibility

The only path to compute: jobs submitted to local or ssh workers (pull: protocol in `transport/pull.md`, raises NotSupported), capability-label matching, execution pools, blob transfer by sha256, GPU-exclusive leases, timeouts and GPU health. The only module allowed to start subprocesses (besides the read-only env probe in core).

## Public interface

- `Job`, `JobResult` (from `kta.core.jobs`), `Executor` protocol, `JobExecutor` (`resolve("executor", ctx)`): submit, wait, run, cancel, quiet_lease, read_blob, put_blob, pending, lease_held
- `register_worker(ctx, name, host, transport)`; `registry.register(ctx, store, ops, config, commit)`, `label_matches`, `worker_configs`
- `WorkerConfig` (`[workers.<name>]` in config), transports `LocalTransport`, `SshTransport`, `PullTransport`
- Worker side: `python -m kta.execution.worker info|has|put|get|run|cancel|serve`; `sandbox.docker_argv` for container jobs
- `scripts/worker-setup <worker>` deploys a commit and registers the worker

## Dependencies

- May import: `core`, `store`, `targets`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).

## WP22 (D589)

`Job.requires["worker"]` restricts placement to the named worker; only the settings page's worker Test uses it.

## WP23 (D603, D607)

`hostpolicy.py`: host-lock mode per job and the unified-memory cap. Workers with `host_lock` run cpu/gpu-shared jobs under the lock shared and self-leased exclusive jobs exclusive; `quiet_lease` holds it exclusive through the worker command `hold`; waits are `lock_wait` spans on `host:<worker>`. `Transport.popen` for long-lived worker commands.
