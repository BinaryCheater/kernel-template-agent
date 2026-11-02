# LEARNDEMO-1: the learn loop in a clean, separate instance

Purpose: show the user, through the frontend, how the knowledge-accumulation (expand) loop
learns from a newly introduced repository. Branch `learndemo/1` (from `sysdebug/2`), worktree
`.claude/worktrees/learndemo-1`. It runs beside the SYSDEBUG instances and shares nothing with
them except the compute host.

## Instance

| | |
| --- | --- |
| mode | dev-mac (control plane and agent CLIs on the Mac, jobs on spark-hk) |
| data | `KTA_DATA=/Users/dingyi/kta-data-learndemo`, `KTA_ENV=dev`, `KTA_WP=LEARNDEMO1` |
| config | `/Users/dingyi/kta-data-learndemo/kta.toml` (dev-mac with API port 7320, agent pool 4) |
| API / web | `127.0.0.1:7320` / `127.0.0.1:5183` (`.claude/launch.json` → `learndemo-web`) |
| wrapper | `/Users/dingyi/kta-data-learndemo/kd <cmd>` sets the variables and runs `uv run <cmd>` in the worktree |

## Steps taken

1. `kd kta init` (seed only: 8 Templates, no reference decompositions).
2. `kd kta reference add docs/runbooks/learndemo-1/vllm_rms_norm.yaml` (Source, Problem,
   Reference; the WP10a decomposition is not imported).
3. `kd kta profile write codex-v5`, `kd kta profile approve <id>`, `kd kta agents env build <batch>`.
4. Controls: `release.stage` S3, `budget.agent_instances` start 3 (POST /controls).
5. `kd kta learn add <ref>`, `kd kta expand <ref> --analysis-only "<reason>"` (L0: no build,
   measurement, reproduction or ablation).
6. `kd scripts/worker-setup spark-hk` after every commit (jobs run the control plane's commit).
7. `kd kta process start learn`, `kd kta process start maintain`,
   `kd kta runner start --slot N --batch <batch>`.

Watch: `http://127.0.0.1:5183/references/<reference id>`.

## Lessons from the run

- Stop processes with `kta process stop` / `kta runner stop` and wait for them to exit. Killing a
  runner leaves its task `running` until the task lease (`budget.lease_s`, 1800 s) ends; killing
  a process left its loop lock for an hour before D723.
- The first real fetch failed on a pathspec without matches (fixed in intake, a2a1865).
