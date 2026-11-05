# strided_plan_byte_chunk_transfer@1: strided plan byte chunk transfer

## When to consider

When a network operation moves a strided tensor region, flatten only its logical contiguous-inner byte stream and split each byte chunk back into source/destination-strided pieces. contrib/nccl_m2n/src/reshard_kernels.cuh:25-88 and reshard_user_window.cu:92-260,316-440.

## Computation before and after

When a network operation moves a strided tensor region, flatten only its logical contiguous-inner byte stream and split each byte chunk back into source/destination-strided pieces.

Decode outer iteration i from last outer dimension to first using idx=i%count[d],i//=count[d]; src/dst offsets are respective base plus sum idx*stride. For logical byte chunk [b,b+n), firstIter=b//inner, firstOffset=b%inner. Repeatedly copy min(inner-offset,remaining) bytes at mapped src/dst offsets, advance iteration and reset offset0. Signal once on final piece after all earlier pieces in the same ordered provider path. Ring forwarding reads the completed destination layout rather than original source layout. Direct per-inner-transfer signals are a different explicitly counted realization.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_bytes | compile / software | {"lo": 1, "hi": 1073741824} | Logical byte chunk size used for publication or forwarding. |

## Coupled constraints and scope

Contiguous-inner strided plans over registered windows; no transpose or packing equality is inferred outside supplied paired strides. Resource effects: Avoids flattening entire tensor; plan/chunk sizes determine number of puts and signals.

Scope predicate: `p.outer_counts_positive and p.inner_bytes_positive and p.transfer_plan_in_bounds and p.destination_regions_disjoint_or_ordered and p.plan_covers_requested_region_once and p.same_queue_piece_order_and_final_signal_covers_prior_puts and p.window_lifetime_valid`.

- Precondition `pre_0` (semantic): `p.outer_counts_positive and p.inner_bytes_positive and p.transfer_plan_in_bounds` — Mixed-radix outer coordinates and contiguous inner byte ranges fit registered source/destination windows.
- Precondition `pre_1` (semantic): `p.destination_regions_disjoint_or_ordered and p.plan_covers_requested_region_once` — Plan specifies a one-to-one logical transfer, with explicit ordering for any overlapping destinations.
- Precondition `pre_2` (semantic): `p.same_queue_piece_order_and_final_signal_covers_prior_puts and p.window_lifetime_valid` — Last-piece signal implies preceding pieces completed under provider contract, not arbitrary cross-queue order.

## Local example and misuse

For a local instance of `map`, identify the object `strided_plan_byte_chunk_transfer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/strided_plan_byte_chunk_transfer@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/strided_plan_byte_chunk_transfer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Avoids flattening entire tensor; plan/chunk sizes determine number of puts and signals.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coordinates` (proof): Mixed-radix iteration and inner offset reconstruct exactly the original tensor region coordinates.
- `chunk_cover` (proof): Piece intervals partition requested byte chunk without gaps/overlap, even when endpoints cut inner transfers.
- `publication` (proof): Observed final signal covers every piece and source is retained until provider no longer reads it.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
