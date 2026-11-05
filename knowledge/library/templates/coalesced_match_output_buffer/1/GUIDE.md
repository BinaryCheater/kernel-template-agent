# coalesced_match_output_buffer@1: coalesced match output buffer

## When to consider

When hash probes can yield many matches per key, accumulate paired probe/match records in bounded shared buffers and reserve global output ranges per flush. For bucket_size2, max next step64 and buffer96, pending40 must flush before another all-match step. Reserving probe array at base100 but match array at base104 destroys pair correspondence even if each array stays in bounds.

## Computation before and after

When hash probes can yield many matches per key, accumulate paired probe/match records in bounded shared buffers and reserve global output ranges per flush.

Each probing subgroup ballots matching slots, counts matches per bucket, reserves a shared-buffer interval and writes (probe_key,matched_slot) pairs at ballot-prefix offsets. All probing subgroups in a flushing warp share a counter/buffer. Before another probe step can exceed capacity, synchronize, atomically reserve count global records once, broadcast base and cooperatively copy the buffered pairs. Reset shared count only after flush reads finish; flush the final remainder. Output order is unspecified unless a later ordering stage establishes it.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| buffer_multiplier | compile / software | [1] | Source shared buffer reserve in maximum-match steps. |
| flushing_lanes | compile / software | [32] | One warp shares a flush buffer; probing group size must divide it. |

## Coupled constraints and scope

NVIDIA open-addressing multimap/multiset retrieve_if/retrieve paths. Exact membership is separate from output reservation; insufficient caller output capacity has undefined source behavior and is excluded. Resource effects: Shared capacity=(buffer_multiplier*32*bucket_size)+32 records per warp; one global counter reservation per flush replaces one reservation per match, at cost of buffering and synchronization.

Scope predicate: `p.matches_complete_and_paired and p.output_capacity_sufficient and p.max_matches_per_step_bounded and p.buffer_capacity_valid and p.buffer_writers_synchronized and p.global_reservations_atomic`.

- Precondition `pre_0` (semantic): `p.matches_complete_and_paired and p.output_capacity_sufficient` — Probe traversal finds all admitted matches, preserves duplicate multiplicity and output allocation covers every emitted pair.
- Precondition `pre_1` (semantic): `p.max_matches_per_step_bounded and p.buffer_capacity_valid` — A probe step produces at most flushing_lanes*bucket_size records; buffer holds pending count plus the next step.
- Precondition `pre_2` (semantic): `p.buffer_writers_synchronized and p.global_reservations_atomic` — All shared writes complete before flush; atomic output reservation allocates disjoint global ranges.

## Local example and misuse

For a local instance of `map`, identify the object `coalesced_match_output_buffer` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/coalesced_match_output_buffer@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/coalesced_match_output_buffer@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared capacity=(buffer_multiplier*32*bucket_size)+32 records per warp; one global counter reservation per flush replaces one reservation per match, at cost of buffering and synchronization.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `multiplicity` (proof): Every matching slot yields one paired record per logical probe, including duplicate input keys.
- `local_capacity` (proof): Preflush threshold leaves room for an all-match next probe step.
- `output_ownership` (proof): Each flush owns a disjoint global interval, and probe/match arrays use exactly the same offset.
- `lifetime` (proof): Flush sees completed shared records and reset follows all reads.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
