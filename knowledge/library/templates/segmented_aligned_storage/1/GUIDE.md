# segmented_aligned_storage@1: segment local padded regions

## When to consider

When a consumer requires aligned metadata groups but data remains concatenated without that padding, consider separate aligned regions for the metadata. Round each payload segment before taking its storage prefix, preserve the logical prefix for unpadded data and match the producer's segment-local layout. The MoE scale instance uses128 rows or4 scale columns, with requested zero fill and distinct token/scale origins. Rounding the fused total or substituting token offsets for padded scale offsets is invalid. Alignment is fixed by the consuming layout, not an independent tuning knob in this source binding.

## Computation before and after

Separate-round each payload segment and preserve logical/storage origins.

For payload segment length Lg, allocate Pg=ceil(Lg/alignment)*alignment elements on its axis. Build storage starts as prefix sums of Pg, retaining logical starts from the original lengths separately. Place each valid local coordinate x at storage_start[g]+x, then apply the selected segment-local physical layout. Initialize padding under the bound policy; require it to be unobserved or neutral. Consumers select the matching segment and use its storage origin and local coordinate; unpadded operands continue to use their logical origins. Empty segments occupy zero storage. Publish all storage origins before use. Reusing the logical prefix as storage prefix is legal only when every segment already meets alignment.

Recover the original computation: Discard each region's padding and invert its local physical layout; concatenate only valid payload coordinates in original segment order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| alignment | compile / boundary | {"lo": 1, "hi": 65536} | Alignment in payload-axis elements required by the selected consumer layout; the pinned scale layouts require128 rows or4 scale columns. |

## Coupled constraints and scope

Independent padding of finite payload segments with an explicit storage-origin map and supported segment-local layout. MoE scale rows use alignment128; reduction scale columns use alignment4 after ceil(tokens/sf_vec_size), while descriptor offsets retain token-equivalent units sf_vec_size times the padded columns. Data tensors remain unpadded. Prefix metadata storage/scan cost is separate and unpriced here. Native32_4_4 layout, descriptor proxy visibility and reduction-tail neutrality require their own implementation binding. Selecting the rule does not license changing the consumer's required alignment or reusing padded origins for unpadded operands.

Scope predicate: `p.segment_lengths_nonnegative and p.segment_table_complete and p.segment_metadata_immutable and p.producer_consumer_storage_map_agree and p.logical_and_storage_offsets_distinguished and p.padding_initialized and p.padding_unobserved_or_neutral and p.storage_layout_matches_padding and p.offsets_ready_before_use`.

- Precondition `pre_segments` (semantic): `p.segment_lengths_nonnegative and p.segment_table_complete and p.segment_metadata_immutable` — Every payload segment is described consistently from preparation through use.
- Precondition `pre_map` (semantic): `p.producer_consumer_storage_map_agree and p.logical_and_storage_offsets_distinguished` — Producer and consumer share segment-local storage origins; offsets for other unpadded tensors stay logical.
- Precondition `pre_padding` (semantic): `p.padding_initialized and p.padding_unobserved_or_neutral` — Missing payload coordinates are initialized under the bound policy and either unobserved or neutral under the original operation.
- Precondition `pre_layout` (implementation): `p.storage_layout_matches_padding and p.offsets_ready_before_use` — Selected per-segment physical layout and published storage-prefix metadata are supported.
- Constraint `c_layout` (implementation): `alignment == p.required_alignment and p.padded_extent_fits_allocation and p.prefix_fits_int32` — Padding is dictated by the bound consumer layout and its allocation/integer limits.

## Local example and misuse

For a local instance of `store`, identify the object `segment_local_padded_regions` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/segmented_aligned_storage@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/segmented_aligned_storage@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Independently padded payload regions and storage-prefix metadata.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: padded_payload
  level: gmem
  scope: device
  bytes: p.padded_payload_bytes
  note: Total layout-specific allocation supplied by the bound family; includes valid
    payload and padding, not an additional duplicate allocation.
overlap: []
```

## Obligations and numerical checks

- `payload_identity` (exhaustive): Every valid segment/local coordinate selects its original payload without collision or crossing segment padding.
- `prefix_identity` (test): Storage starts equal sums of separately padded segment extents and are distinguished from logical starts.
- `padding_policy` (test): Extra coordinates cannot change visible results and never expose uninitialized payload.

Wrong segment bases select another group's scales; non-neutral or live padding can change reductions.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
