# tree_navigation_materialization@1: tree navigation materialization

## When to consider

When the same rooted candidate tree drives ancestor visibility and child traversal, materialize exact ancestry rows or first-child/sibling links while preserving node identity and sibling order.

## Computation before and after

When the same rooted candidate tree drives ancestor visibility and child traversal, materialize exact ancestry rows or first-child/sibling links while preserving node identity and sibling order.

Resolve each local node parent. For ancestor_rows, walk each node to the root and mark exactly self plus ancestors in its own output row; optional immutable verified-prefix columns are separate caller-owned visibility. For child_sibling, set first_child[parent] to its smallest local child and next_sibling[node] to the next larger local child of the same parent, else absent. Reverse-node insertion prepending to each parent gives this order. If input is ancestor_rows in topological order, the greatest earlier marked index is the immediate parent; the first later row containing a node is its first child, and later rows whose immediate parent matches are its siblings. Position is prefix_length+ancestor_count_excluding_self; retrieval identity remains the specified local-to-flat map. Physical boolean packing is a separate refinement.

Recover the original computation: Reconstruct parent from child/sibling membership or the greatest earlier ancestor. Exact tree edges and ancestry are unchanged, so both navigation and visibility observe the same rooted histories and original identities.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| input_view | compile / boundary | ["parent_indices", "ancestor_rows"] | Input topology representation supplied by the workload. |
| output_view | compile / software | ["ancestor_rows", "child_sibling", "both"] | Which exact topology views downstream consumers require materialized. |

## Coupled constraints and scope

Source EAGLE eagle.cuh:36-239 derives masks and ordered child/sibling links from selected-parent IDs; ngram.cuh:26-105 reconstructs links/depth from a dense ancestor mask. N<=1024, topological ancestor-closed selections, exclusive request/node writes. Non-topological IDs, arbitrary graph reachability, missing parents and concurrent topology updates are outside this version.

Scope predicate: `(p.finite_rooted_tree and p.root_index == 0 and p.every_parent_precedes_child and p.node_count <= 1024) and (p.parent_lookup_or_ancestor_input_exact and p.selected_nodes_ancestor_closed) and (p.sibling_order_ascending_local_index and p.local_and_flattened_ids_distinctly_mapped) and (p.output_rows_and_links_have_exclusive_writers and p.initial_zero_and_absent_sentinels_correct)`.

- Precondition `pre_0` (semantic): `p.finite_rooted_tree and p.root_index == 0 and p.every_parent_precedes_child and p.node_count <= 1024` — One acyclic rooted tree with fixed topological node indices; every nonroot has one valid earlier parent.
- Precondition `pre_1` (semantic): `p.parent_lookup_or_ancestor_input_exact and p.selected_nodes_ancestor_closed` — External candidate IDs resolve to the same local parent map, or input ancestor rows encode precisely that tree; ancestors are never omitted.
- Precondition `pre_2` (semantic): `p.sibling_order_ascending_local_index and p.local_and_flattened_ids_distinctly_mapped` — Children/siblings use increasing local slots; retrieval IDs separately map to flattened original identities.
- Precondition `pre_3` (semantic): `p.output_rows_and_links_have_exclusive_writers and p.initial_zero_and_absent_sentinels_correct` — Mask storage starts clear where OR is used; links use an unambiguous absent sentinel, with safe row ownership and capacity.

## Local example and misuse

For a local instance of `map`, identify the object `tree_navigation_materialization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/tree_navigation_materialization@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/tree_navigation_materialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-instance records and optional materialized view.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `representation_identity` (proof): Reconstruct parent from child/sibling membership or the greatest earlier ancestor. Exact tree edges and ancestry are unchanged, so both navigation and visibility observe the same rooted histories and original identities.
- `native_mapping` (test): Indices, padding, sentinels and output ownership implement the logical mapping.

No arithmetic approximation; original predicate or topology and identity must be preserved.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
