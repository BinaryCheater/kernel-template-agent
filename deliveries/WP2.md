# WP2 delivery: targets, execution, backends

Branch `wp/WP2`, cut from `integration` @ `6762cc0` (WP0, WP1, WP13 merged). Code commit of all evidence runs: `6f5796469d10e9e7d82091cdbe37d043513d721c` (clean). The exception is the cached Builds in claim 6, which were compiled at the earlier clean commit `302bc15306aa` of this branch. Evidence outputs are kept outside git under `~/kta-data/dev/WP2/evidence/6f5796469d10e9e7d82091cdbe37d043513d721c/` on the control host. Records are in `~/kta-data/dev/WP2/kta.db`. Claim numbers below are 0-based indexes into `deliveries/WP2.evidence.json`.

## What was done

Real runs: the control plane was this Mac (`KTA_ENV=dev`), and the jobs ran on the real workers through the Executor's ssh transport.

- **Worker registration with probed labels (claim 0, spark; claim 5, slai).**
  - `scripts/worker-setup <w>` does the following, and is re-runnable:
    - deploys the committed HEAD to `<root>/src/<commit>`;
    - runs `uv sync --frozen`;
    - installs a project-local uv when missing (slai, D195);
    - builds the worker image when the worker has one and Docker is present;
    - runs `kta worker register`.
  - Registration probes the worker through its transport (`python -m kta.execution.worker info`) and writes an EnvManifest (role `worker`, `backends`, `image_digest`) and a Worker record.
  - spark: Worker `8426a3b7a9e3…`, EnvManifest `27e6395fae37…`. Its probed labels:
    - `gpu=sm_121`, `gpu_name=NVIDIA GB10`, `cuda=13.0`;
    - `backend.cuda=13.0.88`, `backend.tilelang=0.1.15`, `docker=29.1.3`;
    - `pool.cpu=18`, `pool.gpu-shared=2`, `pool.gpu-exclusive=1`;
    - `arch=aarch64`, `cpus=20`, `commit=6f57964…`.
  - slai: Worker `4bb85de8f7c8…`, EnvManifest `6751bdc76be9…`. Its labels are `arch=x86_64`, `cpus=32` and `pool.cpu=16`, with no GPU labels (D182).
- **Executor with `local` and `ssh` transports** (`kta.execution`): every job of the claims below went through it.
  - It matches workers by label (D182) and runs per-worker execution pools (D183).
  - Inputs go to the worker by sha256. The worker checks each blob, caches it, and returns stdout, stderr and the files under `out/` by hash. The control plane stores them after checking the hash again. This replaces WP1's manual bundle copy (D153).
  - Each job runs in its own process group with a timeout and is classified as `none`, `timeout`, `oom`, `gpu_hang`, `infra` or `nonzero`.
  - Each job gets `job_queue` and `job_run` spans on the lane `worker:<name>:<pool>`.
  - Job records carry `refs.env` and `refs.worker`.
  - `local` is exercised by unit tests (`tests/unit/test_execution_local.py`, real subprocess workers). `pull` has a protocol document (`src/kta/execution/transport/pull.md`), and both `PullTransport` and `kta worker serve` raise `NotSupported`.
- **Target probe (claim 1).** `kta target probe` ran a device-attribute job and 52 feature probes on spark: 52 compile jobs and 50 run jobs. It wrote:
  - Target `9a47aca83844…`;
  - 21 HwFacts (`method=documented`);
  - 52 HwFeatures. Each holds the probe source, the compiler stderr and the run output as blobs.
  - Results table below.
- **Both backends compile and run vector add correctly (claims 2 and 6).**
  - CUDA C++: `nvcc -cubin -arch=sm_121a` on spark.
  - TileLang 0.1.15: `tilelang.lower` runs inside the worker image (no network, no GPU), then the host nvcc 13.0 is mounted read-only and compiles with TileLang's headers (D187).
  - The harness loads both cubins the same way, through the CUDA driver API (D186).
  - Output equals A + B exactly (N = 2^20 fp32, 0 mismatches): Check `c0e8801fc397…` (CUDA) and `e7b6da247633…` (TileLang).
  - The Builds `e7a0416d06af…` (CUDA) and `102d7d1a434e…` (TileLang) came from the input-hash build cache (D189). They were compiled at `302bc15` and are listed in claim 6.
- **GPU-exclusive lease defers shared and cpu jobs (claim 3).** Report `d7a60b2a5a3e…` holds the span timestamps. Log excerpt from `~/kta-data/dev/WP2/logs/backend-selftest/2026-10-30.jsonl`:
  ```text
  05:56:08.988 GPU lease spark:0 taken by 01M65P2G4V93Q6P0QM2E9TB4PA (max 600s)
  05:56:08.988 job 01M65P2G4W2YC1CMXJFTPEYNMZ (run, gpu-shared) deferred: GPU lease on spark held by 01M65P2G4V93Q6P0QM2E9TB4PA
  05:56:08.988 job 01M65P2G4W2VGVYXHQTQ2EPAP2 (shell, cpu) deferred: GPU lease on spark held by 01M65P2G4V93Q6P0QM2E9TB4PA
  05:56:10.658 GPU lease spark:0 released by 01M65P2G4V93Q6P0QM2E9TB4PA after 1.670s
  05:56:10.659 job 01M65P2G4W2YC1CMXJFTPEYNMZ start on spark pool gpu-shared device 0
  05:56:10.659 job 01M65P2G4W2VGVYXHQTQ2EPAP2 start on spark pool cpu device None
  ```
  The lease is also held in the ops `locks` table. A lease held longer than `max_s` is force-released and written to `ops_events`; this is unit-tested with `max_s=1` (D192).
- **Timeout of a kernel that never ends, then recovery (claim 4).** Report `ad1bfb82058a…`. The spin kernel ended as `timeout` after 10.1 s, with exit code -9 (process-group SIGKILL). The post-timeout `nvidia-smi` health check passed. A vector add right after it passed: Check `0acabcdc9f15…`.
- **Second machine (claim 5).**
  - slai as a CPU worker: a C compile job selected by `requires arch=x86_64` produced Build `fb011367d187…` on slai.
  - The EnvManifests of spark and slai have the same `uv.lock` sha256 (`417823627ae5…`).
  - `scripts/bootstrap` in a git checkout on slai set up a control plane (`kta doctor` OK, uv 0.11.2, Python 3.12.13). In that checkout, `pytest -m "not gpu"` gave 190 passed (`slai-bootstrap-unit-tests.txt`).
- **Worker Docker image.**
  - Definition: `docker/worker/Dockerfile`. The base is NGC PyTorch 26.03 pinned by digest, plus `requirements.lock` installed with `--no-deps`.
  - Scripts: `docker/build` (prints the image ID), `docker/lock`.
  - Built on spark as `sha256:897c8784d961…` (Target.image).
- **Other deliverables.**
  - `targets/gb10/features.yaml` (52 PTX instruction families) and `scripts/worker-setup`.
  - The `remote-run` skill now covers worker data directories, registration and labels.
  - CLI: `kta worker register|list|serve`, `kta target probe`, `kta backend smoke|selftest`.
- **WP2 gate checks** (`tools/kta_gate/specific.py` `wp2`, D198; unit-tested in `tests/unit/test_gate_wp2.py`):
  - run `pytest -m gpu tests/acceptance/WP2`;
  - require the six named tests to pass;
  - check the listed real records in the dev store.
- **Acceptance tests** (`tests/acceptance/WP2/`, markers `gpu` and `acceptance`; 6 passed against spark and slai in about 80 s), one per acceptance item:
  - ssh registration with GPU labels;
  - probe records;
  - vector add on both backends, plus a build-cache hit;
  - lease deferral;
  - timeout recovery;
  - slai compile with the same lock hash.

### Probe results (spark, Target `9a47aca83844…`)

| Item | Value |
| --- | --- |
| `KTA_DATA` | control host `~/kta-data` (dev data `~/kta-data/dev/WP2/`); spark `~/kta-work/data`; slai `/storage/kta-work/data` |
| GPU | NVIDIA GB10, sm_121 (probe flags `-arch=sm_121a`) |
| Driver / CUDA / nvcc / ptxas | 580.95.05 / 13.0.3 / 13.0.88 / 13.0.88 |
| TileLang | 0.1.15 (worker image `sha256:897c8784d961…`; codegen in the container, compiled by the host nvcc 13.0) |
| slai cross-compile for sm_121 | **No**: slai has no CUDA toolkit, and installing one is outside WP2's allowed installs. slai has no `backend.cuda` label, so CUDA compile jobs never go there. It does run CPU compile jobs (claim 5) |
| sm_count, warp_size | 48, 32 |
| max threads per CTA / per SM, max CTAs per SM | 1024 / 1536 / 24 |
| smem per CTA (opt-in / default / reserved), per SM | 101376 / 49152 / 1024, 102400 bytes |
| registers per CTA / per SM / per thread | 65536 / 65536 / 255 (per thread: documented constant) |
| L2, persisting L2 | 25165824, 18874368 bytes |
| SM clock, memory clock, bus width | 2418000 kHz, 8533000 kHz, 256 bit |
| global memory | 128526274560 bytes (unified with the CPU) |
| cluster launch, portable cluster size | 1, 8 (portable size: documented constant) |

HwFeature support on gb10 (D188):

| Support | Count | Features |
| --- | --- | --- |
| yes | 48 | atom_cas_b64, atom_global_add, atom_global_add_f32, atom_shared_add, barrier_cluster, cp_async, cp_async_bulk, cp_async_bulk_prefetch, cp_reduce_async_bulk, cvt_e2m1x2, cvt_e4m3x2, distributed_shared_memory, dp4a, elect_sync, ex2_f16x2, fence_proxy_async, fma_bf16x2, fma_f16x2, globaltimer, griddepcontrol, ld_acquire, ld_global_nc, ld_global_v4, ld_global_v8, ld_st_global, ld_st_shared, ldmatrix, match_sync, mbarrier, mma_bf16, mma_e4m3, mma_f16_f16acc, mma_f16_f32acc, mma_f8f6f4, mma_mxf8f6f4_block_scale, mma_s8, mma_tf32, movmatrix, named_barrier, nanosleep, prefetch_l2, red_global_add_f16x2, redux_sync, setmaxnreg, shfl_sync, stmatrix, tanh_approx, vote_sync |
| no | 2 | tcgen05, wgmma (ptxas: "not supported on .target 'sm_121a'") |
| unknown | 2 | cp_async_bulk_tensor, multimem (compile-only entries: running them needs a host-encoded CUtensorMap or a multicast object) |

`yes` means that the instruction compiled for sm_121a and the probe kernel ran without error. Speed and numerical results were not checked.

## Fixtures and simplifications

- No `# kta:fixture` markers in `src/`.
- Allowed simplification used: container isolation is used only for external code (TileLang codegen). KTA's own jobs (harness runs, probes, nvcc compiles) run in the worker's host environment (D187).
- Deliberately simple first kernels (D94): vector add (N = 2^20 fp32) per backend, a spin kernel for the timeout test, and a one-function C file for the CPU compile job.
- `tests/unit/test_execution_local.py` registers the local worker a second time with hand-written GPU labels and device `0`, to test the lease scheduling rules on a machine without a GPU. Its jobs are shell commands and never touch a GPU. A health-check command that cannot answer (`nvidia_smi = "/nonexistent/nvidia-smi"`) stands in for a hung GPU. These are unit tests in env `test`; every claimed capability ran for real on spark and slai.

## Known issues

- **Scheduling scope (D184).** Queue, pools and gpu-shared deferral live in the one `JobExecutor` of a process. The GPU lease is mirrored in the ops `locks` table, but a second control-plane process does not see the first one's pool counts. WP6 should run one executor for the control plane.
- **Worker cache and deployments are never pruned.** `~/kta-work/src/<commit>` (one venv each) and `data/<env>/worker/cache` grow with every commit and job.
- **Timing path is a first version.** The harness times each launch with a pair of CUDA events, including launch overhead. WP3 owns the measurement protocol.
- **Features that cannot be exercised stay `unknown`:** TMA tensor copies need a host-encoded `CUtensorMap` and are not yet exercised. `setmaxnreg` and 256-bit `ld.global.v8` report `yes` on sm_121a; that is what the toolchain and hardware accepted, not a documented capability claim.
- **Unit tests in a worker deployment.** Worker deployments are `git archive` trees without `.git`. There, 3 gate-tool tests (`test_gate_delivery.py::test_evidence_format`, `test_gate_wp1.py::test_wp1_gate_accepts_a_valid_benchmark_record`, `test_gate_wp1.py::test_wp1_gate_rejects_dirty_control_or_unlisted`) fail because they need git history. In a git checkout (claim 5) all 190 pass.
- **Gate needs deployed workers.** The WP2 gate runs the GPU acceptance tests, which reach spark and slai over ssh. The commit under test must be deployed on both (`scripts/worker-setup spark`, `scripts/worker-setup slai`, or `scripts/remote-run sync` plus `uv sync --frozen`). The final delivery commit was deployed this way.
- **Data written during WP2** (all inside project directories):
  - spark: `~/kta-work/src/<commit>` for the WP2 commits, `~/kta-work/data/{dev/WP2,test-worker}`, and the Docker image `kta-worker:tilelang-0.1.15`;
  - slai: `/storage/kta-work/.uv` (uv 0.11.2, Python 3.12.13, uv cache), `/storage/kta-work/{src,data,tmp,bundles,checkout}`;
  - control host: `~/kta-data/dev/WP2`.
- The dev store also holds records from earlier construction runs (registrations and probes at `302bc15`, `c1d426d`, `766d60b`, `4fdc3e4`). Only the records listed in the evidence manifest are evidence.

## New decisions

D180, D181 (needs user review), D182, D183, D184 (needs user review), D185, D186 (needs user review), D187 (needs user review), D188 (needs user review), D189, D190, D191, D192, D193, D194, D195, D196, D197, D198 (needs user review), D199.

## Needed changes

- **Spec additions made in this WP** (spec/modules/target-exec.md "WP2 implementation notes", spec/modules/core-store.md):
  - `Job`/`JobResult` moved to `kta.core.jobs`, plus `JobRunner` (D180);
  - `Executor.read_blob`/`put_blob`;
  - `BuiltModule.entry` and `build_id` (D189);
  - `Ops.worker_health`/`record_event` and `trace.current_tracer` (D192).
- **WP3:** the harness (`kta.backends.harness.run_job`/`collect`) is the launch path for measurement. `gpu-exclusive` jobs (kind `measure`/`profile`/`microbench`) take the GPU lease automatically, and `Executor.quiet_lease(worker, device, holder)` covers a batch.
- **WP6:** hold a single `JobExecutor` per control plane (D184) and surface `JobExecutor.pending()` in the pending view.
- **Merging:** after merging WP2, every commit that should run remote jobs must be deployed with `scripts/worker-setup` (D181).

## Gate output

GATE_OUTPUT_PLACEHOLDER
