# kta.agents

Layer 8. Specification: [spec/modules/agents.md](../../../spec/modules/agents.md). Built by: WP8.

## Responsibility

Task cards, prompts, worker workspaces, persistent sessions, provider adapters (Codex first), isolation, AgentProfiles, context manifests, compliance. Runtime worker skills live in `skills/`.

## Public interface

- `AgentAdapter`: start, resume
- `AgentEvent`, `Limits`
- `run_task(ctx, task) -> AgentRun`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`, `checks`, `solve`, `orchestrator`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
