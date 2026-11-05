# stable_radix_digit_scatter@1: stable radix digit scatter

## When to consider

When integer or transformed scalar keys can be sorted by digits, scatter through bin base plus earlier-tile count plus local rank. Choose separate count/scatter launches or online onesweep publication. For digits [1,0,1] in tile0 and [0,1] in tile1, digit0 owns output positions0,1 and digit1 owns2,3,4 in original within-bin order. A homogeneous full tile can copy directly only after preserving its count-prefix effects.

## Computation before and after

One stable digit partition using global bin bases, per-tile count prefixes and local stable ranks.

For each digit bin b compute global base(b)=sum counts of smaller bins. Each tile computes its stable rank within bin and bin count. Destination of item x in tile t is base(b)+sum_{u<t} count(u,b)+rank_within_bin(x). Source may sort into shared memory by full local rank then subtract the tile-local digit base from the global offset. Separate upsweep materializes count prefixes; onesweep publishes PARTIAL digit counts, looks back to a COMPLETE prefix and then scatters within the same pass. The onesweep full-tile all-one-bin shortcut skips local ranking/scatter rearrangement but still completes count-prefix publication. Repeating stable passes from least to most significant digit sorts the complete transformed key.

Recover the original computation: Erase count tables and scatter order: outputs are a stable partition by this digit, and repeated LSD passes give full transformed-key order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| digit_bits | compile / software | {"lo": 1, "hi": 65536} | Bits in one digit pass; legal source menu is backend-specific. |
| count_schedule | compile / software | ["separate_upsweep", "onesweep_lookback"] | Precompute per-tile digit prefixes or publish them with online lookback. |

## Coupled constraints and scope

Fixed-width ordered keys, stable digit ranks, bounded counters and initialized output. Onesweep lookback progress/publication is a separate backend obligation; availability is not a speed claim.

Scope predicate: `(p.key_twiddle_order_preserving and p.local_digit_ranks_stable and p.payload_follows_key) and (p.bin_totals_exact and p.tile_digit_prefix_exact and p.counts_fit) and (p.output_capacity_sufficient and p.tail_items_excluded and p.count_publication_visible)`.

- Precondition `pre0` (semantic): `p.key_twiddle_order_preserving and p.local_digit_ranks_stable and p.payload_follows_key` — Ordered key transform and stable local ranks preserve key/payload identity.
- Precondition `pre1` (semantic): `p.bin_totals_exact and p.tile_digit_prefix_exact and p.counts_fit` — Global bin bases and earlier-tile bin counts are exact.
- Precondition `pre2` (semantic): `p.output_capacity_sufficient and p.tail_items_excluded and p.count_publication_visible` — All destinations exist and invalid tile items cannot appear as real keys.

## Local example and misuse

For a local instance of `map`, identify the object `stable_radix_digit_scatter` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/stable_radix_digit_scatter@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/stable_radix_digit_scatter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `position_bijection` (proof): Per-bin output intervals and stable within-bin ranks give unique complete destinations.
- `prefix_visibility` (test): Online prefixes read completed matching counts and keep tile order.
- `payload_stability` (test): Equal digits preserve original logical order and values follow the same key permutation.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
