

## D594 — Profile edits from the workflow page are draft versions (WP22)

- **Question:** how to make every part of the agent configuration editable from the frontend while runs stay reproducible.
- **Choice:** an edit writes a new AgentProfile version (refs.base, operator as author); approval moves `current_profile`. Model and effort stay profile-wide; cli_flags, isolation, auth, allowed commands and the skill list per type are not editable here.
- **Rationale:** profile records are immutable and referenced by runs; a per-task-type model needs an AgentProfile schema change (a field `model_by_type`), which is better decided by you.
- **Cost of changing later:** low for more editable fields (one entry in `EDITABLE_FIELDS`); a schema bump with upcast for per-type models.

## D595 — Batch of an approved draft (WP22)

- **Question:** batches are built per profile, and the runner refused tasks whose card named another profile than the configured batch, so an approved draft could never run.
- **Choice:** approval builds `<name>-v<version>`; the runner uses it for new work units of tasks without their own batch when the configured batch is from another profile.
- **Rationale:** new tasks pick up the approved profile without editing config/kta.toml; running work units keep their batch.
- **Cost of changing later:** low (one branch in `open_batch`).
