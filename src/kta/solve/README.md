# kta.solve

Layer 6. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Enumeration and MIP solving of formulations; re-ranking with build information.

## Public interface

- `solve(f, case, shape_bucket, model, hw, shortlist=64, time_limit_s=60, rng=None, max_nodes=2_000_000) -> SolveResult`: < 10⁶ points → exhaustive enumeration; else linear → HiGHS MIP; else pruned enumeration with a node budget (D248, D249).
- `SolveResult`: candidates (best first, each a `Candidate` with `assignment`, `bucket`, `prediction`, `key`), method, space size, feasible, pruned and `pruned_by` constraint, exhaustive, `model_optimal` with its `scope`, MIP status and gap, notes.
- `rerank(candidates, builds, model, hw, k=8)`: builds keyed by `Candidate.key`; compile feedback through the model (D250).
- `shape_buckets(case, family)` (D254); `enumerate_space`, `solve_mip`, `space_size`.
- `record_solve(ctx, store, result, actor, ...)`: Candidate, Prediction and a `solve` Report (D251).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`, `backends`, `knowledge`, `models`, `measure`, `formulation`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
