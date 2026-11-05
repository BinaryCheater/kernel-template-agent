# address_sorted_indirect_gather@1: sorted request inverse map

## When to consider

When an indirect table gather touches random rows beyond translation/page reach, choose ascending source-row request order and retain original destination indices. Duplicate row IDs remain separate requests; row decode and output order stay unchanged. Sorting trades preparation and index storage for possible locality, requiring measurement on the intended memory mapping.

## Computation before and after

Sort independent gather requests by source address while scattering to original destinations.

For request list ids[i], choose original order or obtain sorted_ids[j]=ids[perm[j]] with a full permutation perm. Launch gather for sorted_ids[j] and write decoded row into output[perm[j]], applying the same row-local decode as original order. Do not deduplicate equal source IDs: each original output slot remains represented. Sorting affects access sequence only, not feature layout or numerical decode.

Recover the original computation: For each original output i, choose unique j with perm[j]=i; its gathered source is ids[i], so erasing the request permutation yields the original gather.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| request_order | runtime / software | ["original", "ascending_row_id"] | Whether to keep caller order or sort requests by row ID before launching gathers. |

## Coupled constraints and scope

Independent row gathers with row-monotone address mapping; pinned Engram implementation optionally torch.sorts row IDs, gathers FP8/E8M0 data from device or accessible host pointers and writes BF16 outputs in original order. Benefit is a locality hypothesis for translation/page reach and must be measured; sorting overhead may dominate.

Scope predicate: `p.request_reads_immutable and p.distinct_output_slots and p.request_ids_in_bounds`.

- Precondition `pre_independent` (semantic): `p.request_reads_immutable and p.distinct_output_slots and p.request_ids_in_bounds` — Each original request owns one distinct destination, though source row IDs may repeat; source and output do not destructively alias.
- Precondition `pre_order` (implementation): `p.row_id_order_matches_address_order and p.sort_returns_full_permutation` — Sorted-row realization assumes row-monotone source addresses and retains one original slot for every request.

## Local example and misuse

For a local instance of `transfer`, identify the object `sorted_request_inverse_map` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/address_sorted_indirect_gather@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/address_sorted_indirect_gather@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Nearby requests may share source memory translations/pages; no fewer logical rows are requested.. Storage: Sorted IDs and original-position permutation.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `permutation` (exhaustive): All original request slots are represented once, including repeated source IDs and equal-key ties.
- `decode_identity` (test): Reordered reads use the identical source generation, row decode, feature coordinates and destination slots.
- `visibility` (test): Sorted IDs and permutation are ready before gather and remain live through all destination writes.

No arithmetic reassociation in per-row decode. Other fused reductions and overlapping destinations are outside scope.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: indexed_gather (outside menu); Every permuted request retains the original indirect row semantics.

Structured contract: [template.yaml](template.yaml).
