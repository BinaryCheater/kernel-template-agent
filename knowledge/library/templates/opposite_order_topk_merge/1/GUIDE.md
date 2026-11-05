# opposite_order_topk_merge@1: opposite order topk merge

## When to consider

When streaming exact top-k should keep only k-sized state, merge opposite-sorted retained and block top lists by pairwise max, then bitonic-sort the result. This needs neither a multi-run cursor nor an append/threshold buffer. Keep score/index key order identical at every block, and exclude padded zeros from valid keys.

## Computation before and after

Two bounded top-k lists in opposite order merge by pairwise maximum into a bitonic candidate list.

Scan disjoint input blocks. Retain each block top capacity keys under one total order. Sort the running retained list ascending, take the next block top list descending, and replace running state elementwise by max(left[i],right[i]). This bitonic sequence contains the largest capacity keys of their union; sort/bitonic-merge it before the next iteration. Finally sort retained keys descending and keep requested k. Source total key packs monotone floating value in high bits and reverse original index in low16 bits, so smaller expert index wins equal supplied floating-bit values.

Recover the original computation: Expand the stream into the union of all original candidates; retained state is its top-capacity multiset and final prefix is the requested exact top-k.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| capacity | compile / software | {"lo": 1, "hi": 65536} | Power-of-two retained candidate count, at least the requested k. |

## Coupled constraints and scope

Power-of-two bounded exact top-k, disjoint input blocks and common total keys. Floating signed zero and NaN must follow explicit key semantics; mathematical equality of differently encoded values is not inferred. Source expert count<32768 bounds low16 index encoding.

Scope predicate: `(p.capacity_power_of_two and p.common_total_key_order and p.input_blocks_partition_candidates) and (p.left_list_ascending_and_right_list_descending and p.padding_below_all_valid_keys) and (p.requested_k <= capacity and p.enough_valid_candidates)`.

- Precondition `pre0` (semantic): `p.capacity_power_of_two and p.common_total_key_order and p.input_blocks_partition_candidates` — Local top-k and retained state use the same total order over distinct original occurrences.
- Precondition `pre1` (semantic): `p.left_list_ascending_and_right_list_descending and p.padding_below_all_valid_keys` — Merge inputs are oppositely ordered and padding cannot win.
- Precondition `pre2` (semantic): `p.requested_k <= capacity and p.enough_valid_candidates` — Retained count covers requested outputs.

## Local example and misuse

For a local instance of `reduce`, identify the object `opposite_order_topk_merge` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/opposite_order_topk_merge@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/opposite_order_topk_merge@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `pairwise_union` (proof): Opposite sorted lists pairwise maxima retain the largest capacity elements of their union.
- `key_identity` (test): Packed keys preserve value order, original identity and admitted tie/exception semantics.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
