# kta.knowledge

Layer 4. Specification: [spec/modules/knowledge.md](../../../spec/modules/knowledge.md). Built by: WP4.

## Responsibility

Patterns, refined Templates, Chains and Families: loading, validation, Γ evaluation, retrieval, incremental reuse, text rendering, coverage.

## Public interface

- Patterns: `load_patterns`, `match_pattern`
- Templates: `validate_template`, `propose_template`
- Chains/Families: `validate_chain`, `load_family`, `validate_family`
- Scope and reuse: `scope_holds`, `reuse_plan`
- Retrieval: `retrieve_families`, `retrieve_templates`, `nearest_chains`, `diff_templates`, `card`
- Rendering/coverage: `family_overview`, `render_text_knowledge`, `coverage_report`

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
