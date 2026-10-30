# Pull transport protocol

Status: specified, not implemented in phase 1. `PullTransport.call` and
`python -m kta.execution.worker serve` (the `kta worker` daemon) raise `NotSupported`.

Use it when the control plane cannot open connections to the worker (another cluster,
NAT, a batch scheduler) but the worker can reach the control-plane API over HTTPS.

## Roles

- **Control plane**: keeps the job queue (the same in-process scheduler as the other
  transports: label matching, pools, GPU-exclusive leases) and exposes the endpoints below
  on the API service (WP9a).
- **Worker daemon** (`kta worker serve`): runs on the worker under its deployment of the
  control plane's commit, polls for jobs, runs them with the same code as
  `kta.execution.worker.run_job`, and uploads results.

## Authentication

Each worker gets a worker token issued by the control plane (`kta worker token <name>`,
stored on the worker in `~/.config/kta/worker-token`, mode 600). Every request carries
`Authorization: Bearer <token>`. Tokens are bound to one worker name and revocable; they
never appear in records, logs or git (conventions §8).

## Endpoints

All bodies are JSON unless noted. `<w>` is the worker name.

| Method and path | Body | Reply |
| --- | --- | --- |
| `POST /api/workers/<w>/register` | output of `worker info` (manifest, labels, devices, commit) | `{"worker": <Worker id>, "env_manifest": <id>}`; 409 if the commit differs from the control plane's |
| `POST /api/workers/<w>/heartbeat` | `{"healthy": bool, "running": [job ids]}` | `{"cancel": [job ids]}` |
| `POST /api/workers/<w>/lease` | `{"pools": {"cpu": free, "gpu-shared": free, ...}, "wait_s": 30}` | 204 (long poll timed out) or `{"job_id", "job", "device", "lease_s"}` |
| `GET /api/blobs/<sha256>` | — | raw bytes (`application/octet-stream`); the worker verifies the hash |
| `HEAD /api/blobs/<sha256>` | — | 200 or 404 (used before uploads) |
| `PUT /api/blobs/<sha256>` | raw bytes | 201; the server verifies the hash and refuses a mismatch (422) |
| `POST /api/jobs/<job_id>/result` | the dict `run_job` returns | 200; the control plane then writes job records as for ssh |

## Job lifecycle

1. The daemon leases a job; the control plane has already applied label matching, pool
   limits and the GPU-exclusive lease for the worker's device (the lease is held by the
   control plane on behalf of the job, so deferral of gpu-shared jobs works as for ssh).
2. The daemon downloads missing input blobs into its cache (`GET /api/blobs`), verifies
   them, runs the job (timeouts, process groups, GPU health check exactly as `run_job`),
   uploads stdout, stderr and outputs (`HEAD`, then `PUT` for missing ones).
3. It posts the result. A job whose lease (`lease_s`, renewed by heartbeats) expires
   without a result returns to the queue; a late result for a re-queued job is refused.
4. A heartbeat reply can cancel jobs; the daemon kills their process groups.

## Failure semantics

- Network errors while leasing or uploading are retried with backoff; the job result is
  idempotent on `job_id`.
- A worker whose heartbeat stops for 3 intervals is marked unhealthy (ops `workers`
  table) and its leased jobs are re-queued as `infra` failures.
- `gpu_hang` results mark the worker unhealthy, as for the other transports.
