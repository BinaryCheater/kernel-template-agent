# topk_boundary_tie_repair@1: canonical topk identity repair

## When to consider

When an available fast top-k has arbitrary boundary ties but a consumer needs deterministic token identity, retain strict winners and refill only the kth-value tie quota by an ascending-ID full-row scan.

## Computation before and after

When an available fast top-k has arbitrary boundary ties but a consumer needs deterministic token identity, retain strict winners and refill only the kth-value tie quota by an ascending-ID full-row scan.

Let b be the minimum returned top-k value. Compact all returned records with value>b. Set quota=k-count_above. Scan the full original row in ascending token-ID order, append the first quota records with value==b, then sort the k records by descending value and ascending original ID. Input selector tie identities no longer affect the result. Source packs positive FP32 probability bits in the high32 bits and the complemented unsigned token ID in low32 bits for a descending sort.

Recover the original computation: The above-boundary set is unchanged and complete. Replacing arbitrary kth-boundary IDs by the smallest equal-value IDs leaves the top-k value multiset fixed and establishes the requested total order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

TextSeal ops/sampling/textseal_selector.py:44-137, finite nonnegative FP32 probabilities and token IDs in the source signed-safe range. The scan covers the full vocabulary even when upstream top-k retained only k records. This does not repair an incorrect top-k value threshold or missing strictly larger values.

Scope predicate: `(p.finite_nonnegative_values and p.k > 0 and p.k <= p.length and p.original_ids_unique_and_ascending_in_scan) and (p.input_topk_value_multiset_exact and p.strictly_above_boundary_records_complete) and (p.full_row_boundary_scan_complete and p.boundary_quota_exact and p.paired_records_preserved) and (p.final_order_value_descending_id_ascending and p.packed_sort_key_preserves_order)`.

- Precondition `pre_input` (semantic): `p.finite_nonnegative_values and p.k > 0 and p.k <= p.length and p.original_ids_unique_and_ascending_in_scan` — Ordered finite values, unique original IDs and a complete ascending-ID traversal.
- Precondition `pre_topk` (semantic): `p.input_topk_value_multiset_exact and p.strictly_above_boundary_records_complete` — Fast selector returns the exact top-k values with arbitrary boundary-tie IDs; every record strictly above kth value is present.
- Precondition `pre_repair` (semantic): `p.full_row_boundary_scan_complete and p.boundary_quota_exact and p.paired_records_preserved` — All equal-boundary IDs can be visited and the remaining quota is filled by the smallest IDs.
- Precondition `pre_sort` (semantic): `p.final_order_value_descending_id_ascending and p.packed_sort_key_preserves_order` — Final comparator orders values descending then original IDs ascending; source positive FP32 bits and unsigned-ID complement packing must preserve that comparator.

## Local example and misuse

For a local instance of `reduce`, identify the object `canonical_topk_identity_repair` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/topk_boundary_tie_repair@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/topk_boundary_tie_repair@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-shard summaries and original row inputs.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `boundary_quota` (proof): Exactly k-count_above smallest boundary-tie IDs fill the remaining slots.
- `canonical_identity` (proof): Every arbitrary-tie exact top-k input is repaired to the same value/ID sequence.

There are fewer than k strict winners and enough equal-boundary inputs to fill the quota. Their first IDs are precisely the required canonical boundary set. Sorting the fixed set establishes the same total order for every upstream tie choice. Host exhaustive arbitrary-boundary-subset checks are checks/tie_repair.json; GPU compaction, sort packing and FP32 signed-zero behavior require native admission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
