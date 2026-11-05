# head_reset_scan_state@1: head reset scan state

## When to consider

When a scan must reset accumulated values at segment starts, lift the value to head-count plus suffix aggregate. The right state replaces the left suffix exactly when it contains a new head; otherwise extend it. This permits parallel ordered scans and cross-tile carry without treating resets as commutative reduction. A run crossing a tile boundary must remain one run, and the final suffix needs an explicit flush.

## Computation before and after

An ordered scan state carries head count and aggregate of the suffix after the most recent head.

Lift value operator op to pair (h,v), where h counts segment starts and v is the suffix aggregate since the last start. Combine A then B as h=A.h+B.h and v=B.v if B.h>0, otherwise op(A.v,B.v). Scan in original item order, carrying the same state across tile boundaries. A nontrivial-run encoder derives head/tail flags from neighboring values, feeds h=head&&!tail and v=!head||!tail, then uses head-prefix counts to compact offsets and the previous suffix aggregate as completed length. Explicitly flush the final suffix and run count at end of input.

Recover the original computation: Project each prefix state to its number of segment starts and the value aggregate of its final open segment.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Ordered segmented scans with an associative value operator and exact head markers. Nontrivial RLE binding ignores singleton runs via paired head/tail predicates; ordinary scan-by-key uses all segment heads instead.

Scope predicate: `(p.associative_value_operator and p.head_flags_match_segments) and (p.scan_order_preserved and p.cross_tile_neighbors_match and p.counts_fit)`.

- Precondition `pre0` (semantic): `p.associative_value_operator and p.head_flags_match_segments` — Value combine admits reassociation and head flags identify exact logical segment starts.
- Precondition `pre1` (semantic): `p.scan_order_preserved and p.cross_tile_neighbors_match and p.counts_fit` — Ordered scan and boundary context do not invent/split segments.

## Local example and misuse

For a local instance of `scan`, identify the object `head_reset_scan_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/head_reset_scan_state@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/head_reset_scan_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `lift_associativity` (proof): The lifted operator is associative but generally noncommutative.
- `boundaries_and_flush` (test): Tile neighbors and final suffix handling preserve each original segment exactly once.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
