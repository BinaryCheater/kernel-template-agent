# subtree_reduce_root_broadcast@1: subtree reduce root broadcast

## When to consider

When every rank needs a reduction and the chosen communication topology is a rooted tree, reduce disjoint child subtrees upward and broadcast the single finalized value downward. all_reduce.h:86-225. SIMPLE half(+64 for large group) versus LL70/30 are heuristic bindings, not universal optimum.

## Computation before and after

When every rank needs a reduction and the chosen communication topology is a rooted tree, reduce disjoint child subtrees upward and broadcast the single finalized value downward.

Partition output into matching channel/chunk ranges. Each leaf sends local preprocessed input upward. Each internal node reduces its local contribution with one completed value per child subtree and sends one partial upward. Root combines all and applies final post operation exactly once. Broadcast finalized chunks down tree; internal nodes copy/forward and leaves store. Separate worker groups can overlap reduction of later chunks with broadcast of earlier chunks; same buffer lifetime and credits remain required. Root may fuse reduce-copy-send.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| reduce_warps | compile / software | {"lo": 1, "hi": 32} | Reduction worker budget when up/down stages use disjoint groups; root may fuse stages. |

## Coupled constraints and scope

Associative/grouping-admitted reductions on a valid rooted topology, source fan-in up to3 and root fan-in up to2. Resource effects: O(tree height) stages with per-edge streamed chunks; optional split worker budget.

Scope predicate: `p.rooted_tree_covers_each_rank_once and p.child_subtrees_disjoint and p.reduction_grouping_admitted and p.peer_transport_visible and p.stage_groups_and_barriers_disjoint and p.all_peers_progress`.

- Precondition `pre_0` (semantic): `p.rooted_tree_covers_each_rank_once and p.child_subtrees_disjoint and p.reduction_grouping_admitted` — No duplicate or omitted contributions; tree order satisfies arithmetic policy.
- Precondition `pre_1` (semantic): `p.peer_transport_visible and p.stage_groups_and_barriers_disjoint and p.all_peers_progress` — Chunk publication and independent up/down roles cannot deadlock through mismatched protocol participation.

## Local example and misuse

For a local instance of `map`, identify the object `subtree_reduce_root_broadcast` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/subtree_reduce_root_broadcast@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/subtree_reduce_root_broadcast@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: O(tree height) stages with per-edge streamed chunks; optional split worker budget.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `subtree` (proof): A partial sent by node v contains exactly one contribution from every node in its subtree.
- `finalization` (proof): Root post operation is applied after all contributions once; forwarded values never reapply it.
- `progress` (proof): All peer constructors, transfer steps and barriers have matching participation, including direct pointer exchange even if payload operation is buffered.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
