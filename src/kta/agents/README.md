# kta.agents

Layer 8. Specification: [spec/modules/agents.md](../../../spec/modules/agents.md). Built by: WP8.

## Responsibility

Task cards → prompts, step bundles, persistent sessions, provider adapters (Codex only),
isolation and task tokens, AgentProfiles, context manifests, compliance accounting.
Runtime worker skills live in `skills/`, prompt templates in `prompts/`.

The agent CLI runs as an Executor job on the agent host (`[agents] worker`, D300); the
job program is `python -m kta.agents.launch`, the only file here that starts processes.

## Public interface

- `run_task(ctx, task) -> AgentRun`; `run_step(ctx, task, *, retry, note, deps) -> StepResult` (D302)
- `AgentAdapter` (`start`, `resume`, `parse_session`, `last_launch`), `AgentEvent`, `Limits`, `Launch` (`base.py`)
- `parse_session(path)`: a Codex session file, CLI home or archive → events (`runner.py`)
- `profile.write_profile`, `profile.approve_profile` (AgentProfile `codex-v1`, pointer `current_profile`)
- `guard.issue_token`, `guard.verify_token`, `guard.verify_task_token` (for the API and agent-side `kta`)
- `units.close_work_unit`: archive and remove a work unit's workspace and CLI home
- `batches.build`, `batches.verify`, `batches.export_profile`, `batches.BatchRoot` (D322): batch roots `<data>/batches/<batch>/`
- `commands.register(sub, make_ctx, emit)`: `kta agents export-profile|env build|env verify` (wired by the cli)
- `adapters.capabilities`: the admission table per CLI

| File | Role |
| --- | --- |
| `runner.py` | card + profile → session decision → bundle → launcher job → AgentRun |
| `launch.py`, `hostfs.py`, `launchspec.py` | job program on the agent host: install, manifest, limits, GPU guard, git, pack |
| `adapters/` | `codex.py` + `codex_session.py` (parser); others raise `NotSupported` |
| `cards.py`, `workspace.py`, `prompts/` | AGENTS.md, TASK.md, messages, step bundle |
| `batches.py`, `commands.py` | batch roots (template, lock, verify), profile export, `kta agents` hook |
| `sessions.py`, `compliance.py`, `guard.py`, `profile.py`, `units.py`, `config.py` | as named |
| `skills/README.md` | runtime skill catalogue (D321); construction skills are in the repo's `.agents/skills/` |

- WP11: `profile.CODEX_V2` (adds `diagnose-model`, `write-microbench`); `batch_port.BatchEnvs`,
  registered as interface `batch_envs` for the experiment process.

## Controls (WP21)

- `policy.py`: context and loop policy from the controls snapshot; the runner records the version in `AgentRun.refs.controls`. AgentProfile policies of codex-v1/v2 are kept as recorded and no longer drive a run.

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
