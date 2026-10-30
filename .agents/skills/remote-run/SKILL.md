---
name: remote-run
description: Run commands on the construction hosts (spark GPU worker, slai CPU worker) by hand during construction via scripts/remote-run. Business code must use the Executor instead.
---

# Remote runs during construction

Hosts, their project roots and PATH additions are in `config/remote.toml`:

| host | role | project root | notes |
| --- | --- | --- | --- |
| spark | GPU worker (GB10, sm_121, CUDA 13.0) | `~/kta-work` | uv in `~/.local/bin`, CUDA in `/usr/local/cuda/bin`; Docker with nvidia runtime |
| slai | CPU worker, backup | `/storage/kta-work` | root disk is full: write only under `/storage`; project-local uv 0.11.2 in `/storage/kta-work/.uv/bin` (D195); no CUDA toolkit |

Remote shells are non-interactive: rely on the configured PATH, not on login profiles.

## Commands

```bash
scripts/remote-run exec spark -- nvidia-smi                 # run in the project root, keep exit code
scripts/remote-run sync spark                               # deploy committed HEAD to src/<commit>, link `current`
scripts/remote-run exec spark -- sh -c 'cd current && uv sync --frozen'
scripts/remote-run bg spark noise-1 -- sh -c 'cd current && uv run --frozen pytest -m gpu'
scripts/remote-run status spark noise-1                     # running | exited <code>
scripts/remote-run logs spark noise-1 -n 100
```

- `sync` refuses a dirty tree. Deployed trees carry `.kta-commit`, which the EnvManifest reads as the commit.
- Background runs write `runs/<name>/{out.log,pid,exit}` under the project root.
- Install only inside the project root (uv venvs, containers). No system packages, no user-level installs, no changes to system configuration: stop and ask the user instead.
- Do not run GPU benchmarks while another job holds the GPU; GPU leases come with the Executor (WP2).

## Workers, data directories and registration (WP2)

Workers used by the Executor are `[workers.<name>]` in `config/kta.toml` (transport, ssh alias, root, PATH, env, image, pools). `config/remote.toml` holds the same hosts for manual runs.

| | spark | slai |
| --- | --- | --- |
| `KTA_DATA` on the worker | `~/kta-work/data` | `/storage/kta-work/data` |
| dev data of a WP | `~/kta-work/data/dev/<WP>/` | `/storage/kta-work/data/dev/<WP>/` |
| worker blob cache / job scratch | `data/<env>/worker/cache`, `worker/jobs/<job_id>` (D194) | same |
| deployments | `~/kta-work/src/<commit>` with its own `.venv`; `current` links the last sync | `/storage/kta-work/src/<commit>` |
| uv env | default | `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `TMPDIR` under `/storage/kta-work` (pass them to `uv` by hand: `env UV_CACHE_DIR=/storage/kta-work/.uv/cache UV_PYTHON_INSTALL_DIR=/storage/kta-work/.uv/python TMPDIR=/storage/kta-work/tmp uv ...`) |
| probed labels | `gpu=sm_121`, `cuda=13.0`, `backend.cuda`, `backend.tilelang`, `docker`, pools cpu/gpu-shared/gpu-exclusive | `arch=x86_64`, pool cpu (no GPU labels) |

```bash
export KTA_DATA=~/kta-data KTA_WP=WP2 KTA_ENV=dev   # control plane (this machine)
scripts/worker-setup spark          # deploy HEAD, uv sync, build the worker image, register (Worker + EnvManifest records)
scripts/worker-setup slai           # same; installs project-local uv on first run
uv run kta worker list              # registrations usable by the current commit
uv run kta target probe             # Target, HwFact, HwFeature records from spark
```

- The Executor only dispatches to a worker deployed and registered at the control plane's commit (D181): after every commit that should run remotely, run `scripts/worker-setup <worker>` again (or `scripts/remote-run sync` plus `uv sync --frozen` in `src/<commit>` when no new registration is wanted).
- Records produced by jobs are written by the control plane into `$KTA_DATA/dev/<WP>/kta.db`; outputs come back by sha256 (no bundle copying; D153 bundles remain for moving stores).
- Look at a worker's dev store only for debugging worker-side state: `scripts/remote-run exec spark -- sh -c 'ls data/dev/WP2/worker/cache | head'`. Evidence lives in the control-plane store.
- Worker images: `docker/build` on the worker (called by worker-setup); `docker/lock` regenerates `docker/worker/requirements.lock`.
