# kta.core

Layer 0. Specification: [spec/modules/core-store.md](../../../spec/modules/core-store.md). Built by: WP0 (registry, errors, logging, doctor, config, context, env manifest); WP1 (the rest).

## Responsibility

Schemas (pydantic models per record kind), content-hash IDs, the expression language, configuration, the Context object, the implementation registry, errors, JSON-lines logging, spans, and `kta doctor`.

## Public interface

- `Context` (env, config, data_dir, registry, clock, rng; store/executor/target added by WP1/WP2)
- `registry.Registry.register/select/resolve`, `registry.resolve(interface, ctx)`, `registry.build_registry()`
- `errors`: `KtaError` and subclasses (conventions §3.3)
- `logging.setup(ctx)`, `logging.bind(**ids)`
- `doctor.run_doctor(ctx)`, `envmanifest.collect_manifest(role, root, cuda_home)`
- WP1: `canonical_json`, `record_id`, `Expr`, `load_config`, `schemas/`, `trace.span`

## Dependencies

- May import: none (bottom layer).
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).
