# load_compute_store_role_pipeline@1: load compute store role pipeline

## When to consider

When compute should overlap both input movement and output stores, use separate load, compute and store roles with independently sized rings. A two-slot input ring and three-slot output ring need different modulo/phase counters. A ready barrier does not replace the async proxy fence, and output empty may be published only after TMA finishes reading that slot.

## Computation before and after

Three warp roles communicate through independent input and output generation rings.

Allocate input ring with full/empty barriers per load_slots and output ring with separate full/empty barriers per store_slots. Load role waits empty at slot i%load_slots and phase opposite floor(i/load_slots)%2, registers expected input bytes and issues TMA loads. Compute waits input full at that phase, captures input in registers, fences shared/async proxy access and signals input empty. Compute forms output, waits output empty at its independently derived phase, stores/fences it and signals output full. Store role waits output full, issues TMA stores, permits store_slots-1 groups in flight and signals the oldest output slot empty only after its store source reads complete. Drain final stores. Source uses one load warp and one store warp with 24-register worker budgets, plus default compute warps; actual allocation rounds the worker roles into warpgroups.

Recover the original computation: Serialize each generation load, computation and output store while erasing the two physical rings and three actor roles.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| load_slots | compile / software | {"lo": 1, "hi": 65536} | Input ring slots shared by load and compute roles. |
| store_slots | compile / software | {"lo": 1, "hi": 65536} | Output ring slots shared by compute and store roles. |
| compute_warps | compile / software | {"lo": 1, "hi": 65536} | Default compute-role warp count; source binds four or eight. |

## Coupled constraints and scope

Finite equal-trip three-role CTA pipelines with independent input/output ring lengths and native TMA/mbarrier support. This exposes the separate store-role ring/resources absent from warp_specialization@1 fields.

Scope predicate: `(p.role_warps_and_register_budgets_fit and p.all_roles_same_trip_count) and (p.barriers_initialized_and_input_output_phase_maps_match and p.proxy_fences_between_shared_and_async_accesses) and (p.store_source_reads_retire_before_empty_publication and p.final_stores_drained)`.

- Precondition `pre0` (semantic): `p.role_warps_and_register_budgets_fit and p.all_roles_same_trip_count` — Role resources fit rounded NVIDIA warpgroup allocation and every role processes the same generations.
- Precondition `pre1` (semantic): `p.barriers_initialized_and_input_output_phase_maps_match and p.proxy_fences_between_shared_and_async_accesses` — Ready/empty barriers and asynchronous proxy visibility match both rings.
- Precondition `pre2` (semantic): `p.store_source_reads_retire_before_empty_publication and p.final_stores_drained` — Output storage is not released before pending store reads finish.

## Local example and misuse

For a local instance of `work_assign`, identify the object `load_compute_store_role_pipeline` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/load_compute_store_role_pipeline@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/load_compute_store_role_pipeline@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `generation_identity` (proof): Every role reads the same generation and each ring is released only after its last reader.
- `proxy_and_store_completion` (test): Native ready/empty publication includes proxy fences and completed store-source reads.
- `role_progress` (test): Role allocation and waits have no circular missing-generation dependency.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
