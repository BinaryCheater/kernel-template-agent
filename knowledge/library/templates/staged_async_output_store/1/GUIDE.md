# staged_async_output_store@1: output staging ring

## When to consider

When completed output values are in registers but the selected asynchronous output engine reads shared memory, consider a shared staging ring. Preserve finalized bytes through source-read completion and retain a separate destination-write completion before observation; tune stages only within storage and backend guards. The pinned TMA epilogues derive stages from their layout/budget and use one committed group per subtile; direct stores, new clipping paths or a different completion protocol need separate bindings or variants.

## Computation before and after

Stage output subtiles through a shared ring and recycle slots after source-read completion.

For output subtile u, use shared slot u mod stages. Store its final register values into the slot, publish them with the required proxy fence and writer synchronization, issue its shared-to-global asynchronous copy, and commit the copy group. Wait until at most stages-1 committed groups remain source-unread before the next subtile can overwrite a slot. After the final subtile, drain all source reads before reusing/freeing shared storage. Retain a separate destination-write completion dependency before any observer reads the output.

Recover the original computation: Replace each staged copy by the corresponding logical output store at its actual destination completion; erase shared slots while retaining pending captured-value writes and output observation order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| stages | compile / software | {"lo": 1, "hi": 32} | Number of shared output slots; at most stages-1 committed copy groups may remain source-unread before the next iteration. |

## Coupled constraints and scope

A finite stream of already-computed output subtiles with legal shared-to-global asynchronous transfers. One copy group captures one subtile's shared bytes; source-read waits retire an ordered committed prefix. Stage bytes include padding/alignment. Output arithmetic, TMEM-to-register lifetime, vector-tail predicates and descriptor clipping are separate refinements. Shared storage is modeled here; transfer costs and latency have not been priced or measured.

Scope predicate: `p.output_tiles_disjoint and p.source_visibility_fenced and p.async_store_legal and p.source_read_wait_available and p.output_observed_after_write_completion`.

- Precondition `pre_layout` (semantic): `p.output_tiles_disjoint and p.source_visibility_fenced` — Each output subtile has a distinct destination and all source writers publish bytes before async use.
- Precondition `pre_backend` (implementation): `p.async_store_legal and p.source_read_wait_available` — Descriptor/layout and source-read completion semantics are established for the selected backend.
- Precondition `pre_observation` (semantic): `p.output_observed_after_write_completion` — Output readers wait for actual destination-write completion; a source-read wait alone grants no output-read permission.
- Constraint `c_capacity` (resource): `stages * p.stage_bytes <= p.capacity_bytes` — Padded shared slots fit the assigned output-staging budget.
- Constraint `c_backend` (implementation): `p.async_store_legal and p.source_read_wait_available` — Backend supports the chosen shared-to-global copy and a query/wait for completed source reads.

## Local example and misuse

For a local instance of `store`, identify the object `output_staging_ring` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/staged_async_output_store@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/staged_async_output_store@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: stages shared-memory output slots.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: output_slots
  level: smem
  scope: cta
  bytes: stages * p.stage_bytes
overlap: []
```

## Obligations and numerical checks

- `source_visibility` (test): Every async copy reads the finalized register-produced bytes after all shared writers have published them.
- `source_lifetime` (exhaustive): Each shared slot remains unchanged until every copy using its generation has finished reading it.
- `output_identity` (exhaustive): Every live output destination receives its intended subtile value exactly once with the original output bounds.
- `destination_visibility` (unspecified): Output readers observe all destination writes only after a backend completion event stronger than source-read completion where required.

No new arithmetic; the already-converted output values and bounds are preserved.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [physical_storage@1](../../physical_storage/1/GUIDE.md); Shared output generations survive through async source reads.

Structured contract: [template.yaml](template.yaml).
