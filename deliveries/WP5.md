# WP5 delivery: models, formulation, solve

Branch `wp/WP5`, cut from `integration` @ `c4081bb` (WP0, WP1, WP2, WP4, WP10a, WP13 merged). Code commit of the evidence runs: `4ad1b8347d2a7b05333d7e30deb294d12da1a9f6` (clean). WP5 is not record-backed (D84): evidence is unit and acceptance runs on the control host (`deliveries/WP5.evidence.json`, outputs under `~/kta-data/dev/WP5/evidence/`). The real run (GEMM family on three shapes, top 8 measured, rank correlation) belongs to WP7.

## What was done

All of the following are real implementations in `src/`, exercised by unit and acceptance tests (evidence claim 0). None has run against real hardware data yet; that needs WP3's HwModel and WP7's family.

- **models** (`src/kta/models/`):
  - `bound(case, shape, hw, work)`: per-unit time of the minimum work at `HwModel.peaks`; the bound is the largest, with its bottleneck unit. `min_work(problem, shape)` evaluates `Problem.params["min_work"]` (D243). A unit without a peak raises `NotSupported`.
  - `UnitModel` (frozen): `time_u = work_u / (peak_u · eff_u · occ)`, `total = max_u time_u + serial_coef · serial + serial_const_us + Σ γ_uv · min(t_u, t_v)`. `Prediction.per_unit`/`serial` are peak times in µs, `total` the model's prediction (D241). `version` is a hash of the parameters.
  - Compile feedback (`predict(..., feedback)`, `repredict`): occupancy from regs, smem and threads per CTA, factor `min(1, occ / 0.25)` (D242). `NoResidentCta` when nothing fits.
  - `fit(data)` (`fit.py`, D244): coordinate search on eff_u, non-negative least squares for the serial terms, and an interaction pair added only when an F test passes. Every fifth point is held out; the hold-out error and ordering accuracy are reported. The old version is unchanged.
  - `save_model` / `load_model` / `calibrate`: each version is a `Report` (subject `unit_model`) with parameters, parent, fit info and the (Prediction, Measurement) IDs it was fitted on (D245).
- **formulation** (`src/kta/formulation/`):
  - `assemble(family, chain, templates, case, hw, previous=None)`: one block per chain item (`blocks.build_block`) plus the glue constraints. Each constraint and cost term keeps its source (item key or `glue`, and the Template's constraint ID or cost index).
  - Binding semantics settle WP4's D205 (D246): family variable, constant, list-domain value, or an expression over family variables, `case.*` and `hw.*`. `p.*` bindings work the same way; an unbound `p.*` that is used fails assembly with its name. `p.instances` scales an item's cost terms (D247).
  - A Template variable bound to a constant leaves the search and is recorded in `Block.fixed`. Preconditions without `same_as` are constraints too (D253).
  - Incremental reuse: a block is reused from `previous` when its fingerprint (item, Template, referenced family variables) is unchanged. When one step changes, only its block is rebuilt.
  - `linear` / `cost_linear`, `hw_refs`, `case_refs`, `missing_hw`, `fingerprint`.
  - `per_unit_work` returns overlappable and serial work per unit (D240). Diagnostics: `violated` (which step's constraint excludes a point) and `cost_breakdown` (each step's cost term).
- **solve** (`src/kta/solve/`):
  - `solve(f, case, shape_bucket, model, hw, shortlist, time_limit_s, rng, max_nodes)`:
    - fewer than 10⁶ points: exhaustive enumeration;
    - otherwise, linear and exportable: MIP with HiGHS;
    - otherwise: pruned enumeration with a node budget, and a shuffled value order when an `rng` is given.
  - Enumeration (D248) checks each constraint as soon as its variables are set and narrows range variables by affine comparisons. Every excluded point is attributed to a constraint (`pruned_by`), so feasible + pruned = space when the search is exhaustive.
  - MIP (D249) uses one-hot menus, linear rows and objective, and no-good cuts for the shortlist. `model_optimal` is claimed only for an exhaustive enumeration, or a gap-0 MIP under a model without interaction terms. Its `scope` names the family, formulation fingerprint, model version, HwModel and bucket.
  - `rerank(candidates, builds, model, hw, k)`: builds are keyed by `Candidate.key`. Feedback comes from the Build plus the `threads` variable (D250). Unbuilt, failed and no-CTA candidates are dropped.
  - `record_solve`: writes Candidate, Prediction and a `solve` Report (D251). `shape_buckets(case, family)` gives one bucket per shape (D254).
- **Test family**: `tests/acceptance/WP5/gemm_family/` is a GEMM family whose chain uses all five seed Templates with `p.*` bindings. It is test data; its `gen.py` emits markers only. With a test HwModel it assembles into 10 variables, 12 constraints and 5 cost terms. It solves exhaustively on 256³, 1024³ and 4096³: the space has 1.53·10⁹ points, 3804 are feasible, and each solve takes about 0.2 s (evidence claim 1, informational).
- **Gate**: `wp5` in `tools/kta_gate/specific.py` (D252) checks:
  - the three named acceptance tests;
  - WP4's seed check;
  - every seed family plus the WP5 test family validates and assembles, and together they use all five seed Templates.

  It is tested in `tests/unit/test_gate_wp5.py`.
- **Tests**:
  - `tests/acceptance/WP5/test_formulation_solve_acceptance.py` (3 tests, the spec's three unit-test items);
  - `tests/unit/test_models_unit.py` (8), `tests/unit/test_formulation_solve_unit.py` (11) and `tests/unit/test_gate_wp5.py` (4);
  - helper `tests/solvehelp.py`.

## Fixtures and simplifications

- No `# kta:fixture` added to `src/`. Inherited, unchanged: `src/kta/store/bench.py` (`_body`, D156).
- Test data, not kernels or facts: the GEMM test family (`tests/acceptance/WP5/gemm_family/`), the linear test Template and family, and the `TEST_HW` HwModel (`tests/solvehelp.py`, peaks and limits are illustrative values, not GB10 measurements). The fit tests use synthetic measurements generated from a known model.
- Allowed simplification used: MIP export covers only linear comparisons (joined by `and`) and finite one-hot menus (range domains up to 4096 values). Everything else goes to enumeration (D249).
- No skipped tests.

## Known issues

- **The seed chain has no cost for the contraction itself, nor for operand DRAM traffic.** None of the five seed Templates carries tensor/FMA work of the GEMM or the A/B global reads; only copy issue, barriers, P = 0 DRAM, smem wavefronts and combine FMAs are costed. On the test family the top 8 candidates of every shape tie. WP7 needs one of two fixes: a Template that carries the contraction and operand traffic (for example WP10a's `output_tiling` / `compute_instruction_selection`, once importable), or family-level cost terms, which would be a schema change.
- **Variables that affect no cost term tie.** `mech` is an example. Ties are broken by the assignment's JSON, so they are deterministic but arbitrary.
- **Missing hw.* names.** `shared_resource_binding@1` reads `hw.smem_queue_depth`, `hw.smem_banks` and `hw.smem_bank_bytes`, which are not among WP2's `hw.*` limits (D199). Solving with a real HwModel raises `NotSupported` until these facts exist. No HwModel record with measured `peaks` exists yet; WP3 provides it.
- **Calibration limits.** Fitting uses pre-compile peak times only, because feedback is not stored in a Prediction. The occupancy knee (0.25) is not fitted, and spills are not used.
- **Interaction significance.** It uses its own F test (D244), not measure's rule: models is layer 4 and cannot import measure.
- **Solve scope.** One shape per solve: a bucket that spans several shapes, and the `max_normalized_latency` objective, are not handled. Variable `when` conditions are not used. Narrowing applies only to single comparisons that are affine in a range variable.
- **MIP counts.** For a MIP solve, `feasible` counts the solutions returned, not the feasible set.
- **WP13 finding 5.** It is supported by a shortlist plus `rerank(k)`, so a top-k set can be measured. The L2-capacity regime is not modeled separately.
- **Validator mismatch.** WP4's `validate_family` reports an expression bound to a Template variable (not a `p.*`) as `unbound_name`, although assembly accepts it (D246). The test family only uses expressions in `p.*` bindings.

## New decisions

D240, D241 (needs user review), D242, D243 (needs user review), D244 (needs user review), D245 (needs user review), D246 (needs user review), D247 (needs user review), D248, D249, D250 (needs user review), D251, D252 (needs user review), D253, D254.

## Needed changes

- **knowledge (WP4 owner):** `validate_family` should accept expression bindings for Template variables, consistent with D246. Today it reports them as `unbound_name`.
- **WP2/WP3:** HwFacts or limits for `smem_queue_depth`, `smem_banks` and `smem_bank_bytes`, which the seed's `shared_resource_binding@1` reads. Also an `HwModel` record with measured `peaks` per unit (dram, l2, smem, tensor, fma, issue, launch).
- **Seed modeling (WP7 / method discussion):** a step or family-level mechanism that costs the contraction and operand traffic (see Known issues).
- **Spec edits made in this WP:** `spec/modules/formulation-solve-models.md` signatures: `serial` per unit (D240), `assemble(..., previous)`, `solve(..., rng, max_nodes)`, `record_solve`, and `fit.py`.

## Gate output

Run: `KTA_DATA=~/kta-data scripts/gate WP5` on the control host, at the code commit plus these delivery files.

```text
gate WP5: PASSED (23 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 4 locked files unchanged
  PASS pyright
  PASS unit tests — 258 passed
  PASS contract tests — 2 passed
  PASS fake coverage — 1 fake(s) covered
  PASS registry (kta doctor, dev)
  PASS delivery note
  PASS fixture declarations — 1 declared
  PASS decision log — 178 decisions
  PASS skipped tests — none
  PASS evidence — 2 claim(s)
  PASS WP5 acceptance tests — 3 tests passed
  PASS seed validates — 8 Pattern, 1 Source, 9 Structure, 5 Template
  PASS seed formulations assemble — gemm_seed5_test: 12 constraints, 5 costs
```
