# kta.knowledge

Layer 4. Specification: [spec/modules/knowledge.md](../../../spec/modules/knowledge.md). Built by: WP4.

## Responsibility

Patterns, refined Templates, Chains and Families: loading, validation, Γ evaluation, retrieval, incremental reuse, text rendering, coverage.

## Public interface

- Patterns: `load_patterns`, `match_pattern`
- Templates: `validate_template`, `propose_template`
- Chains/Families: `validate_chain`, `load_family`, `validate_family`
- Scope and reuse: `scope_holds`, `reuse_plan` (optional `assignment` evaluates `p.*` expression bindings, D286)
- Retrieval: `retrieve_families`, `retrieve_templates`, `nearest_chains`, `diff_templates`, `card`
- Rendering/coverage: `family_overview`, `render_text_knowledge`, `coverage_report`
- Seed: `import_seed`, `import_references`, `seed_dir` (used by `kta init`)

Functions that read the pool take the `Store` as a parameter (D200). Validators return `Issue`s with severity `error` or `warning` (D201). The seed lives in the repo-root `knowledge/seed/` (D204).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
