# cluster_partial_destination_table@1: cluster partial destination table

## When to consider

When a cluster reduction needs its result at one CTA or all CTAs, choose destination tables before selecting the final writer. Four warps in each of two FP32 CTAs deliver eight scalars: 32 expected bytes per destination. Root mode sends eight stores total; all mode sends sixteen. Reject mismatched byte counts, reused phase-zero barriers and peers leaving before remote completion.

## Computation before and after

Sender-indexed scalar partial tables placed at either one cluster root or every cluster CTA.

Each warp reduces its local values to a lane-uniform scalar. Sender rank r,warp w owns slot r*warps+w. In all_ctas mode lane d sends that scalar to destination d for d<peers; in last_cta mode one elected lane sends only to rank peers-1. Each consuming CTA registers peers*warps*p.scalar_bytes expected bytes against its fresh phase-zero barrier, waits for transaction completion, then folds all sender slots with lane-strided reads and a warp reduction. Only designated destinations expose final results.

Recover the original computation: Project each destination table fold to the complete reduction of the disjoint sender warp partials; redundant copies expose the same admitted result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| destinations | compile / software | ["last_cta", "all_ctas"] | Deliver partials only to the last cluster rank or replicate to every rank. |
| peers | compile / software | {"lo": 1, "hi": 65536} | Resident cluster CTA count, at most 32 in the lane-selected realization. |
| warps | compile / software | {"lo": 1, "hi": 65536} | Contributing full warps per CTA. |

## Coupled constraints and scope

One-shot SM90+ resident cluster scalar float/double reductions. Tables have peers*warps slots per destination. Repeated phase-zero use requires another generation protocol. Source helper permits lane-strided folds of tables larger than one warp.

Scope predicate: `peers <= 32 and p.scalar_bytes >= 4 and p.scalar_bytes <= 8 and p.sender_slot_count == peers * warps and p.partial_partition_exact and p.fresh_phase_zero_barrier and p.cluster_initialization_published and p.remote_store_supported and p.transaction_completion_orders_reads and p.allocations_live_until_remote_access_retired and p.combine_order_admitted and p.participants_match_launch`.

- Precondition `pre0` (semantic): `p.fresh_phase_zero_barrier and p.cluster_initialization_published and p.remote_store_supported` — Destination barriers initialized once and published before remote stores.
- Precondition `pre1` (semantic): `p.transaction_completion_orders_reads and p.allocations_live_until_remote_access_retired` — Waits publish payload and allocations survive every sender/reader.
- Precondition `pre2` (semantic): `p.combine_order_admitted and p.participants_match_launch` — Numerical order and cluster participants match the requested operation.
- Constraint `c0` (semantic): `peers <= 32 and p.scalar_bytes >= 4 and p.scalar_bytes <= 8` — The lane-selected sender and float/double remote stores support the bound type.
- Constraint `c1` (semantic): `p.sender_slot_count == peers * warps and p.partial_partition_exact` — Sender table has one injective slot per warp partial.

## Local example and misuse

For a local instance of `reduce`, identify the object `cluster_partial_destination_table` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/cluster_partial_destination_table@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/cluster_partial_destination_table@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Source-bound physical buffer; exact byte count and capacity supplied by binding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `delivery` (proof): Each consuming destination receives every sender partial once in its injective slot.
- `byte_completion` (test): Each destination expects exactly the bytes actually sent before reading.
- `retirement` (test): No peer exits or reuses storage while remote accesses or consumer reads remain.

Address/storage transformation preserves values; arithmetic changes require separate obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
