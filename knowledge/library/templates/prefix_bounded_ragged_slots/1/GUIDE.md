# prefix_bounded_ragged_slots@1: conservative segment chunk storage from existing token prefix

## When to consider

When segment token prefixes already exist but rounded chunk-count prefixes would require another scan or host readback, derive safe noncompact chunk slots from segment index and token prefix. Allocate the matching count bound and keep gaps invisible to all consumers.

## Computation before and after

Bound-derived ragged chunk slots from original token prefixes, without a second chunk-count scan.

Define B(n,t,C)=m+floor((t-m)/C), m=min(n,t). Given original token prefix S_i, segment i starts its chunk records at b_i=B(i,S_i,C). Its live count is q_i=ceil(L_i/C), and record j occupies b_i+j for0<=j<q_i. Reserve capacity B(N,T,C); leave [b_i+q_i,b_(i+1)) unused. A valid previous-chunk state is b_i+j-1 only when local j>0; the first chunk uses its own sequence seed even if b_i>0. Keep dense logical task IDs and actual chunk counts separate from these storage slots. Do not initialize or consume gaps as identity records unless a different algorithm explicitly requires that. Changing C changes starts/counts and requires a coherent allocation/producer/consumer generation.

Recover the original computation: Remove unused capacity slots and concatenate the live local records of each segment. Each valid segment/chunk retains its original identity. The next bound prefix is at least the current start plus ceil(L_i/C), so record intervals cannot overlap and all fit final capacity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_elements | runtime / software | {"lo": 64, "hi": 2147483584, "step": 64} | Positive64-aligned CP piece length in the concrete source; choice trades chunk count, per-piece serial work and summary workspace. The bound itself is valid for any positive integer C. |

## Coupled constraints and scope

FlashInfer CP GDN varlen workspace uses this bound for local-transfer, local-state and fixed-state records, with D128 and64-aligned CP piece lengths. It also allocates64-token T tiles under the same mapping. Empty sequences own no chunk records; their final-state copy behavior is separate. Logical checks use smaller positive C to isolate combinatorial boundaries, not native menu claims. This is a conservative slot map, not aligned per-segment payload rounding or a dense chunk-prefix scan.

Scope predicate: `p.segments >= 1 and p.total_elements >= 0 and p.bytes_per_slot >= 1 and chunk_elements % 64 == 0 and p.slot_capacity == min(p.segments, p.total_elements) + (p.total_elements - min(p.segments, p.total_elements)) // chunk_elements and (p.nonnegative_segment_lengths_and_complete_token_prefixes) and (p.all_producers_and_consumers_use_the_bound_prefix_map) and (p.readers_visit_only_valid_local_chunks_and_never_fold_capacity_gaps) and (p.total_and_prefix_bounds_and_slot_byte_products_fit_index_types) and (p.token_prefix_metadata_and_payloads_ready_before_consumers) and (p.consumer_supports_noncompact_slot_map_and_selected_chunk_extent)`.

- Precondition `pre_segments` (implementation): `p.nonnegative_segment_lengths_and_complete_token_prefixes` — N ordered segment lengths L_i>=0 have exact token prefixes S_i and total S_N=T. Segment count and total are host-known; device metadata and host totals refer to the same immutable invocation.
- Precondition `pre_index` (implementation): `p.all_producers_and_consumers_use_the_bound_prefix_map` — For chunk size C, slot start for segment i is B(i,S_i,C), not a dense prefix of ceil(L_j/C). Local chunk j maps to B+j. Every descriptor/index reader uses this same conservative storage map.
- Precondition `pre_holes` (implementation): `p.readers_visit_only_valid_local_chunks_and_never_fold_capacity_gaps` — Segment i has ceil(L_i/C) live chunks; zero-length segments have none. Allocation gaps may remain stale because they are never read as states, summaries or outputs. A previous-chunk read occurs only for local j>0, not merely global slot>0.
- Precondition `pre_capacity` (implementation): `p.total_and_prefix_bounds_and_slot_byte_products_fit_index_types` — Reserve B(N,T,C) slots of the declared payload size. All index/stride products fit actual host/device integer types. Chunk extent and slot capacity remain coherent through replay.
- Precondition `pre_publish` (implementation): `p.token_prefix_metadata_and_payloads_ready_before_consumers` — Token prefixes publish before slot calculations; payload producers finish before valid consumers read them. This map avoids scanning derived chunk counts but does not remove the original token-prefix or payload publication dependencies.
- Precondition `pre_backend` (implementation): `p.consumer_supports_noncompact_slot_map_and_selected_chunk_extent` — Backend accepts this noncompact storage map and supported C. Dense logical task enumeration remains separate from storage addresses; kernels cannot infer live count from capacity or assume gaps are neutral records.
- Constraint `c_counts` (implementation): `p.segments >= 1 and p.total_elements >= 0 and p.bytes_per_slot >= 1 and chunk_elements % 64 == 0` — Finite nonnegative ragged domain and native64-token-aligned piece extent.
- Constraint `c_capacity` (implementation): `p.slot_capacity == min(p.segments, p.total_elements) + (p.total_elements - min(p.segments, p.total_elements)) // chunk_elements` — Capacity derives from segment count,total and C; it is not an independent live-count or dense-prefix parameter.

## Local example and misuse

For a local instance of `store`, identify the object `conservative_segment_chunk_storage_from_existing_token_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/prefix_bounded_ragged_slots@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/prefix_bounded_ragged_slots@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Potential holes between segment-local chunk intervals inside one host-known bounded allocation.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: chunk_record_capacity
  level: dram
  scope: kernel
  bytes: p.slot_capacity * p.bytes_per_slot
  note: Conservative record storage; useful live records may be fewer. Original prefix
    metadata and per-record payload choice are separate.
overlap: []
```

## Obligations and numerical checks

- `bound_and_disjointness` (proof): For every nonnegative prefix partition, B(i,S_i,C)+ceil(L_i/C)<=B(i+1,S_(i+1),C), and final B bounds all records.
- `live_coordinate_identity` (test): All source producers/consumers agree on segment/local chunk addresses and never treat gaps or prior-sequence slots as live predecessors.
- `metadata_generation` (test): Host count/total, device token prefixes, C and capacity refer to one coherent generation before consumers run.

No arithmetic reassociation; stale gaps, dense-prefix substitution or a changed C with old workspace offsets can silently feed another state generation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
