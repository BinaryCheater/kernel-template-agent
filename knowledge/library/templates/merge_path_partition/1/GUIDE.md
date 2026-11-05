# merge_path_partition@1: merge path partition

## When to consider

When merging long sorted runs, partition by output diagonals so each executor receives a bounded amount of merged output even if one run dominates. Choose tile_items; search both input coordinates at each diagonal and preserve the same tie priority during local merge. Reject a partition computed for a different fallback tile size.

## Computation before and after

Output-diagonal partitions of a stable merge of two sorted sequences.

For output diagonal d search i in [max(0,d-B_length),min(d,A_length)] with j=d-i. Binary search advances i when B[j-1] is not strictly less than A[i], giving first-input priority for equivalent keys. Neighboring diagonals d,d+tile_items determine disjoint A/B slices. Each tile merges only those slices into its matching output interval; optionally partition each tile again by per-thread diagonals and serial merge short ranges. Clamp last diagonals to total length and move associated values with their key source indices.

Recover the original computation: Concatenate output intervals in diagonal order to obtain the stable global merge of the original two sequences.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| tile_items | compile / software | {"lo": 1, "hi": 65536} | Merged outputs per parallel tile. |

## Coupled constraints and scope

Two sorted random-access sequences, strict weak comparator, first-input-before-second equivalent ties, exact output-length arithmetic. Partition tile size must match any selected fallback consumer agent.

Scope predicate: `(p.inputs_sorted_same_strict_weak_order and p.comparator_consistent) and (p.diagonals_cover_output and p.partition_and_consumer_tile_sizes_equal) and (p.first_input_wins_equivalent_ties and p.output_ranges_disjoint)`.

- Precondition `pre0` (semantic): `p.inputs_sorted_same_strict_weak_order and p.comparator_consistent` — Inputs are sorted using the same comparator.
- Precondition `pre1` (semantic): `p.diagonals_cover_output and p.partition_and_consumer_tile_sizes_equal` — Partition launch and consumer agree on diagonal spacing including final tail.
- Precondition `pre2` (semantic): `p.first_input_wins_equivalent_ties and p.output_ranges_disjoint` — Tie rule is consistent across binary partition and serial merge.

## Local example and misuse

For a local instance of `map`, identify the object `merge_path_partition` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/merge_path_partition@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/merge_path_partition@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `stable_partition` (proof): Each diagonal identifies the unique stable-merge prefix and adjacent intervals contain every source once.
- `tie_and_payload` (test): Equivalent-key priority and associated value identity agree between search and local merge.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
