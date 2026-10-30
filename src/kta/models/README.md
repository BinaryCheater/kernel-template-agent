# kta.models

Layer 4. Specification: [spec/modules/formulation-solve-models.md](../../../spec/modules/formulation-solve-models.md). Built by: WP5.

## Responsibility

Per-unit cost models and lower bounds; calibration against measurements.

## Public interface

- `bound(case, shape, hw, work) -> Bound`; `min_work(problem, shape)` reads `Problem.params["min_work"]` (D243).
- `UnitModel` (frozen): `predict(per_unit_work, serial, hw, feedback=None)`, `repredict(prediction, hw, feedback)`, `fit(data)`, `version`, `to_dict`/`from_dict`. `Prediction.per_unit`/`serial` are peak times in µs; `total` is the model's prediction (D241). Compile feedback → occupancy factor (D242); `NoResidentCta` when nothing fits.
- Fitting (`fit.py`, D244): coordinate search on eff_u, non-negative least squares for the serial terms, F-test for pair interactions, every fifth point held out.
- `save_model`, `load_model`, `calibrate(ctx, store, model_id, pairs, actor)`: model versions as `Report` records with subject `unit_model` (D245).

## Dependencies

- May import: `core`, `store`, `targets`, `execution`.
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
