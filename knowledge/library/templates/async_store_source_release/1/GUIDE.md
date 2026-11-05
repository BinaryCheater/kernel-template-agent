# async_store_source_release@1: async store source release

## When to consider

When shared-memory metadata is drained by an asynchronous global store, release its source buffer only after the store engine has finished reading it, and keep destination publication as a separate completion event.

## Computation before and after

When shared-memory metadata is drained by an asynchronous global store, release its source buffer only after the store engine has finished reading it, and keep destination publication as a separate completion event.

Write a complete logical metadata tile into an aligned shared slot. Join all source writers, execute the shared→async proxy fence and have one elected issuer issue and commit the bulk shared→global store. The store engine may continue independently. Before any later generation destructively rewrites that slot, the issuing thread waits for the previous committed group to finish reading its source; a CTA barrier then releases all writers. A source-read-release milestone permits refill but does not by itself permit a destination reader. Destination publication and final outstanding-store completion follow their own supported event, not merely the source-release wait.

Recover the original computation: Associate every store with the source generation it fully read before reuse and with its later complete destination generation. Collapsing the transfer at actual destination completion recovers one correct metadata store; source release only shortens source occupancy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| source_slots | compile / software | {"lo": 1, "hi": 8} | Shared source generations retained in rotation; source instance uses one512-byte slot. |
| store_bytes | compile / boundary | [512] | Physical metadata payload bytes per admitted bulk store. |

## Coupled constraints and scope

Source expert_specialization/es_sm100_mxfp8_blockscaled_group_quant.cuh:230-270 uses one aligned512-byte shared scale tile, thread0 bulk store/commit and next-tile cp_async_bulk_wait_group_read<0> plus CTA barrier before overwrite. Full128-row tiles supply the direct fully initialized live-byte instance; partial-tile padding and final kernel completion retain explicit native obligations. This outgoing transfer is outside async_transfer@1's into-on-chip scope.

Scope predicate: `(p.shared_source_aligned_and_allocated and p.live_source_bytes_initialized and p.padding_bytes_unobserved) and (p.source_writers_complete_and_async_proxy_fence_before_issue and p.one_issuer_owns_store_group) and (p.source_read_release_wait_before_every_destructive_refill and p.waiting_thread_releases_all_cta_writers) and (p.destination_generation_not_read_until_actual_store_completion and p.final_pending_stores_have_supported_completion) and (p.source_slot_and_destination_generation_maps_exact and p.pending_store_groups_fit_hardware)`.

- Precondition `pre_0` (implementation): `p.shared_source_aligned_and_allocated and p.live_source_bytes_initialized and p.padding_bytes_unobserved` — All live payload bytes are defined and storage is valid; any physical padding has no logical consumer.
- Precondition `pre_1` (implementation): `p.source_writers_complete_and_async_proxy_fence_before_issue and p.one_issuer_owns_store_group` — CTA writes are visible to the async engine before the issuing thread commits its store group.
- Precondition `pre_2` (implementation): `p.source_read_release_wait_before_every_destructive_refill and p.waiting_thread_releases_all_cta_writers` — The same issuer waits for all previous reads of a reused slot, then a CTA barrier lets other writers proceed.
- Precondition `pre_3` (implementation): `p.destination_generation_not_read_until_actual_store_completion and p.final_pending_stores_have_supported_completion` — Source release alone does not certify global payload readiness; external readers and kernel exit retain a supported completion contract.
- Precondition `pre_4` (implementation): `p.source_slot_and_destination_generation_maps_exact and p.pending_store_groups_fit_hardware` — Every issued source snapshot writes the intended destination generation within hardware bounds.

## Local example and misuse

For a local instance of `store`, identify the object `async_store_source_release` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/async_store_source_release@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/async_store_source_release@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Reusable bounded payload slots with explicit generation identity.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `logical_identity` (proof): Associate every store with the source generation it fully read before reuse and with its later complete destination generation. Collapsing the transfer at actual destination completion recovers one correct metadata store; source release only shortens source occupancy.
- `publication_and_lifetime` (test): Native readiness, index maps and memory lifetimes implement the declared complete generation.

The store engine reads a fixed source generation until its read-release event. A refill ordered after that event cannot change transferred bytes; a CTA barrier makes the issuer's wait govern every writer. Proxy visibility precedes issue. Independently waiting for actual destination completion prevents exposing a partially stored generation. No abstract source-release proof discharges the source kernel's final pending-store or partial-padding native contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
