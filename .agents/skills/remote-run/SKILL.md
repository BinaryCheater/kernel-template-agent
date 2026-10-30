---
name: remote-run
description: Run commands on the construction hosts (spark GPU worker, slai CPU worker) by hand during construction via scripts/remote-run. Business code must use the Executor instead.
---

# Remote runs during construction

Hosts, their project roots and PATH additions are in `config/remote.toml`:

| host | role | project root | notes |
| --- | --- | --- | --- |
| spark | GPU worker (GB10, sm_121, CUDA 13.0) | `~/kta-work` | uv in `~/.local/bin`, CUDA in `/usr/local/cuda/bin`; Docker with nvidia runtime |
| slai | CPU worker, backup | `/storage/kta-work` | root disk is full: write only under `/storage`; no uv, no CUDA toolkit |

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

## To be completed by WP2

Data directories on each worker (`KTA_DATA`), worker registration, capability labels, and how to query a worker's dev store.
