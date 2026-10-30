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
- `ids`: `canonical_json`, `record_id`, `UlidFactory` (WP1)
- `schemas`: one pydantic model per kind (`KINDS`), `Record`, `Actor` (with `commit`/`dirty`), `validate_body(kind, body)` raising `InvalidBody` (a `ValueError`) (WP1)
- `expr.Expr.parse(src, declared)` -> `free_vars`, `attr_refs`, `is_linear`, `eval(env, hw, p, case)`; `ExprError` (WP1)
- `trace`: `span(kind, name, lane, **attrs)`, `tracing(tracer)`, `new_tracer(sink, clock, ids, environ)`, `current_span_id()`, `child_environ()` (`KTA_TRACE_PARENT`) (WP1)
- `load_config(path)` (WP0)
- `jobs`: `Job`, `JobResult`, `JobRunner` (structural submit/wait/run/read_blob/put_blob), `DEFAULT_POOL` (WP2, D180, D183); `trace.current_tracer()` (WP2, D192)

## Dependencies

- May import: none (bottom layer).
- Enforced by import-linter (`pyproject.toml`, `[tool.importlinter]`).

## Schema versions (WP14, D440, D441)

- `schemas/upcast.py` and `KindSpec.upcasters`: old stored bodies are upcast on read; `Record.stored_body` / `to_json_dict()` keep the stored form (the ID), `Record.body` is the current shape.
- `schemas/levels.py`: levels 1–7 and the versioned mapping `paper-0to5@1` from the paper's 0–5.
- New kinds: Capability, CapabilityDecl (hardware), Realisation, Finding, Coupling (knowledge); Regime and Negative are legacy (`legacy_of = "Finding"`).
