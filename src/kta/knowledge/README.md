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

## Knowledge model v2 (WP14, D440–D454)

- `capabilities.py`: `target_capabilities`, `template_availability` — Capability support per Target (declaration, else inferred from ISA probes, else unknown).
- `realisations.py`: `realisations_of`, `expected_sass`, `probed_on`, `validate_realisation` — code and SASS per backend × ISA range; schema-1 Template bodies read as a legacy realisation.
- `findings.py`: `findings` (Regime and Negative read as Findings; `prior` out of scope), `couplings`.
- `ladder.py`, `promotion.py`: one status ladder with evidence rules; `promote`, `demote` (stale propagation), `review`.
- `pooldiff.py`: `diff(store, a, b)` between two PoolSnapshots.
- `learning.yaml`, `learning.py` (WP20): learning-loop tables (admission classes, system features, tiers L0–L4, source classes, priority, performance core, ablation, SOTA levels, backends, O12) and their readers.
- `admission.py`: the admit step (`admit`, `admit_references`, `warnings.yaml`, `admission` Reports); `sota.py`: four-level SOTA coverage, system features, language gaps, per-backend expressibility; `ablation.py`: suspected pairs, designs, coupling estimates, proposed Couplings.
- Retrieval with a target hides Templates whose Capabilities are not `yes`; `retrieve.unavailable_templates` and `CoverageReport.availability` count them.
