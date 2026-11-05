# privatized_bin_accumulation@1: privatized bin accumulation

## When to consider

When many samples contend on the same output bins, accumulate per-worker tables and flush counts. Choose shared/global-private storage and optionally coalesce adjacent equal bins inside each thread. Bins [2,2,5,5,5] become updates (2,+2),(5,+3); invalid bins contribute nothing. Extra private capacity and initialization trade against fewer global atomics.

## Computation before and after

A per-worker bin table absorbs sample updates before combining nonzero bin totals globally.

Partition samples among workers. Initialize one private counter table per worker/channel. Decode each valid sample to a bin. Individual mode atomically adds one to its private bin. local_equal_runs mode forms maximal equal-bin runs in each thread item sequence and atomically adds each run length once, ignoring invalid bins. After all worker updates retire, flush nonzero private counters through the output-bin decoder with atomic additions to the initialized global histogram. Shared/global-private tables realize the same count decomposition; source blended mode assigns location by CTA parity.

Recover the original computation: Sum each private count by logical output bin to recover exactly the original sample histogram.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| table_location | compile / software | ["shared", "global_private"] | Where each CTA-private table lives. |
| update_grouping | compile / software | ["individual", "local_equal_runs"] | Count each decoded bin or coalesce adjacent equal decoded bins within one thread. |

## Coupled constraints and scope

Integer sample histograms with consistent channel/bin mapping and bounded counters. RLE grouping is thread-local decoded-bin equality, not sorting and not global run encoding.

Scope predicate: `(p.sample_partition_exact and p.bin_decoder_consistent and p.invalid_bins_ignored) and (p.private_tables_zeroed and p.final_table_zeroed and p.counts_fit and p.private_storage_fits) and (p.private_updates_atomic and p.private_updates_complete_before_flush and p.final_updates_atomic)`.

- Precondition `pre0` (semantic): `p.sample_partition_exact and p.bin_decoder_consistent and p.invalid_bins_ignored` — Each valid sample has one decoded bin and every private/final map agrees.
- Precondition `pre1` (semantic): `p.private_tables_zeroed and p.final_table_zeroed and p.counts_fit and p.private_storage_fits` — Initial counter state and resource capacity are explicit.
- Precondition `pre2` (semantic): `p.private_updates_atomic and p.private_updates_complete_before_flush and p.final_updates_atomic` — Concurrent updates and table-flush ordering are safe.

## Local example and misuse

For a local instance of `reduce`, identify the object `privatized_bin_accumulation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/privatized_bin_accumulation@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/privatized_bin_accumulation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `count_conservation` (proof): Every valid sample contributes exactly one, even when equal-bin runs are compressed.
- `flush_visibility` (test): Private table publication precedes flush and final atomic sums retain every worker contribution.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
