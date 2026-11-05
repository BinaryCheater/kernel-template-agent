# tagged_peer_reduction_rounds@1: tagged peer reduction rounds

## When to consider

When a small statistic must be replicated across a power-of-two peer group, use disjoint peer exchange rounds with epoch-tagged partial states and combine the received state once per round.

## Computation before and after

When a small statistic must be replicated across a power-of-two peer group, use disjoint peer exchange rounds with epoch-tagged partial states and combine the received state once per round.

For P=2^r peers, at round k each peer publishes its current statistic to partner rank xor2^k in storage dedicated to that round/channel. Poll all required payload tags for the current epoch, then combine local and received state. The represented peer subset doubles each round, provided partner selection joins disjoint subgroups. CTA0 may publish once while all local CTAs consume the result. Epoch reuse is legal only after previous readers and writers retire; source uses8-bit magic values and per-round payload slots. This is separate from a reusable inter-CTA up/down counter barrier.

Recover the original computation: Small replicated statistics across source power-of-two peer groups. Raw sums/counts may combine additively; centered moments require their own correct merge. GPU transport/tag atomicity is a backend obligation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| rounds | compile / software | {"lo": 1, "hi": 3} | r: log2(peer count); source peer groups2/4/8. |

## Coupled constraints and scope

Small replicated statistics across source power-of-two peer groups. Raw sums/counts may combine additively; centered moments require their own correct merge. GPU transport/tag atomicity is a backend obligation.

Scope predicate: `(p.partners_join_disjoint_peer_subsets and p.round_storage_disjoint and p.payload_tags_publish_complete and p.epoch_no_stale_alias and p.peer_progress and p.combine_associative_commutative and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.partners_join_disjoint_peer_subsets and p.round_storage_disjoint and p.payload_tags_publish_complete and p.epoch_no_stale_alias and p.peer_progress and p.combine_associative_commutative and p.rounding_admitted` — Partner topology, per-round storage, native tag visibility, epoch ownership, progress and statistic combine.

## Local example and misuse

For a local instance of `reduce`, identify the object `tagged_peer_reduction_rounds` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/apex/delta/tagged_peer_reduction_rounds@1/template.yaml`. Original outline: `knowledge/studies/apex/delta/tagged_peer_reduction_rounds@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `peer_cover` (proof): Induction on rounds gives exactly2^k distinct represented peers; final state includes every peer once.
- `tag_epoch` (proof): Every accepted payload belongs to the current round/channel/launch epoch.
- `publication` (proof): All data fragments are complete before their tags can be accepted by consumers.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
