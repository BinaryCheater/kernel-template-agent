# kta.models

Layer 4. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5; L1 and the declarative model WP15; L2, what-if, L3 adapter and regression WP16.

## Responsibility

The declarative hardware model resolved to numbers (`MachineModel`: HwModel instance on its HwArch class), L1 steady-state pricing of effects, per-resource SOL and lower bounds, the per-unit calibration layer; calibration against measurements.

## Public interface

- `MachineModel.build(hw, arch=None)`; `load_arch`, `load_machine(store, hw_id)`, `latest_machine(store, target)` (D463).
- L1 (`l1.py`, D464/D465): `EvalTx`, `EvalStorage`, `EvalOverlap`, `EvalEffects` → `price(effects, mm, features=None) -> L1Result` (busy, bounds, walls, `unpriced`, `unmodelled`, `prediction()`); `price_parts`, `compose`, `check_walls`, `sol_us`. Unmodelled resources and unpriced classes cost no time and are reported. This is the interface WP16 (L2) and WP17 (MIP) consume (spec/modules/formulation-solve-models.md "Hardware model and L1").
- L2 (`l2.py`, `l2graph.py`, WP16, D500–D505): `simulate(effects, mm, features=None) -> L2Result` (total, per-resource busy and idle by category, ncu-named metrics and stalls, waves and busy intervals, notes); `moved(mm, peaks, walls)`. What-if (`whatif.py`, D504): `what_if(...) -> WhatIf` with the shadow price; `record_l2`, `record_what_if` write Reports. L3 (`base.py`, `l3.py`, D506): `L3Simulator`, `ExternalSassSimulator` (an Executor job; `simulate` raises NotSupported until run for real). Regression (`regression.py`, D507–D510): `spearman`, `results`, `summarise`, `admit`, `observe`, `record_version`, `scoreboard`.
- `bound(..., arch=None)` and `unit_time_us(..., arch=None)` accept an HwModel or a MachineModel; `UnitModel.predict_l1(result, mm, feedback=None)` (D466).

- `bound(case, shape, hw, work) -> Bound`; `min_work(problem, shape)` reads `Problem.params["min_work"]` (D243).
- `UnitModel` (frozen): `predict(per_unit_work, serial, hw, feedback=None)`, `repredict(prediction, hw, feedback)`, `fit(data)`, `version`, `to_dict`/`from_dict`. `Prediction.per_unit`/`serial` are peak times in µs; `total` is the model's prediction (D241). Compile feedback → occupancy factor (D242); `NoResidentCta` when nothing fits.
- Fitting (`fit.py`, D244): coordinate search on eff_u, non-negative least squares for the serial terms, F-test for pair interactions, every fifth point held out.
- `save_model`, `load_model`, `calibrate(ctx, store, model_id, pairs, actor)`: model versions as `Report` records with subject `unit_model` (D245).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
