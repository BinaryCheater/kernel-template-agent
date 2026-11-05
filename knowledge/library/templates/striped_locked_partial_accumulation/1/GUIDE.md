# striped_locked_partial_accumulation@1: striped locked partial accumulation

## When to consider

When many rows contribute the same parameter-gradient vector, choose a bounded number of striped partial slots instead of one partial per row. Hash row modulo groups, initialize on first lock acquisition and update the entire vector while locked. Reject a final reduction that reads never-initialized slots or overlaps active writers.

## Computation before and after

Independent row contributors are hashed to locked vector partial accumulators, then combined in a separate pass.

Contributor row r uses slot r%groups. Compute its partial vector. Acquire that slot mutex with compare-and-swap. If initialized flag is zero, set it and store the partial without reading old scratch; otherwise load the complete prior vector and add the partial under the lock. Publish all vector stores before releasing the mutex. After the contributor kernel completes, reduce only slots with actual contributors into final output. groups controls lock contention versus scratch and final-combine traffic; it does not split a row numerical statistic.

Recover the original computation: Expand each locked slot into the original row partial vectors in actual acquisition order, then combine all slots once.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| groups | compile / software | {"lo": 1, "hi": 65536} | Number of lock-protected vector partial slots. |

## Coupled constraints and scope

Vector partial accumulation from independent rows with whole-vector mutexes and fresh flags per invocation. Floating-point order depends on lock acquisition. Unused slots must be excluded or initialized by a separate policy.

Scope predicate: `(p.contributors_partition_complete and p.row_to_slot_is_mod_groups) and (p.locks_and_initialized_flags_zeroed and p.lock_release_publishes_whole_vector) and (p.final_pass_waits_all_contributors and p.read_only_actual_nonempty_slots and p.rounding_order_admitted)`.

- Precondition `pre0` (semantic): `p.contributors_partition_complete and p.row_to_slot_is_mod_groups` — Every row contribution enters one designated partial slot.
- Precondition `pre1` (semantic): `p.locks_and_initialized_flags_zeroed and p.lock_release_publishes_whole_vector` — A zero lock protects the entire vector and its first-writer flag.
- Precondition `pre2` (semantic): `p.final_pass_waits_all_contributors and p.read_only_actual_nonempty_slots and p.rounding_order_admitted` — Combiner sees complete nonempty slots and admits lock-arrival accumulation order.

## Local example and misuse

For a local instance of `reduce`, identify the object `striped_locked_partial_accumulation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/striped_locked_partial_accumulation@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/striped_locked_partial_accumulation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `first_writer` (proof): No slot is read before a completed initializing contribution, and updates cannot lose concurrent vector elements.
- `slot_combine` (test): Each original contribution enters one slot and every nonempty complete slot enters the final combine once.
- `progress` (test): Lock holders and waiting contributors can make progress under the launch/runtime contract.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
