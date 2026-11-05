# ordered_prefix_lookback@1: published tile prefix

## When to consider

When a scan spans CTAs, publish local aggregates so each tile can recover its exclusive prefix without a global barrier. Choose all-prefix lookback or fixed 32-tile anchors matching the source stable mode. Preserve older-before-newer operand order: concatenating A,B,C must yield ABC, not CBA. Reject stale status/payload pairs and launches that cannot schedule awaited predecessors.

## Computation before and after

Decoupled ordered predecessor lookback with optional fixed anchor publication.

Each tile first computes its ordered aggregate and local prefixes. Tile zero establishes the initial prefix. Other tiles publish PARTIAL aggregate; inspect predecessor tiles in descending windows, but combine values in increasing logical order. Stop at the nearest permitted INCLUSIVE predecessor and prepend its inclusive value to all later partial aggregates. Prepend older window aggregates to newer ones. This gives the exclusive tile prefix; combine it on the left of each valid local prefix. Publish INCLUSIVE for all tiles in all mode, or only tile indices divisible by 32 in anchors mode. Anchors mode waits for the designated predecessor anchor before reducing its fixed window. Guard partial-tile output stores.

Recover the original computation: Concatenate valid tile prefixes; ordered associative composition equals a scan of the original input.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| publication | compile / software | ["all", "anchors"] | Publish all completed inclusive prefixes or only fixed 32-tile anchors for stable source grouping. |

## Coupled constraints and scope

One-dimensional ordered scan with disjoint nonempty tiles, initialized state and valid-tail stores. Supports associative noncommutative operators. Floating point requires explicit reassociation policy; anchors alone is not cross-device bitwise assurance. Segmented resets are outside this version.

Scope predicate: `p.tiles_cover_in_order and p.local_scan_preserves_order and p.associative_contract and p.status_payload_consistent and p.predecessors_can_progress and p.fresh_tile_state`.

- Precondition `pre_op` (semantic): `p.associative_contract and p.status_payload_consistent and p.predecessors_can_progress and p.fresh_tile_state` — Associativity under the numerical policy, safe publication and predecessor progress are required.
- Constraint `c_valid` (semantic): `p.tiles_cover_in_order and p.local_scan_preserves_order and p.valid_store_guarded` — Tiles cover input once in order and inactive lanes never alter a visible valid prefix.

## Local example and misuse

For a local instance of `scan`, identify the object `published_tile_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/ordered_prefix_lookback@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/ordered_prefix_lookback@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Aggregate/inclusive payload and state per tile plus CTA scan scratch.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ordered_cover` (proof): Lookback covers each predecessor exactly once in increasing operand order.
- `publication` (test): A valid status identifies the matching completed aggregate or inclusive payload.
- `progress` (test): Awaited predecessors can execute; stale state cannot satisfy a new invocation.
- `determinism` (test): Anchor mode has a fixed reduction tree and native operator behavior under the admitted determinism policy.

Default grouping may depend on publication timing; fixed anchors constrain grouping but rounding and target policy still apply.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
