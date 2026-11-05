# tombstone_preserving_probe_chain@1: tombstone preserving probe chain

## When to consider

When deleting from an open-addressed table, mark removed slots as tombstones so a search continues past deletions instead of mistaking them for the original end of a probe chain. KeysA andB collide and occupy consecutive slots. DeleteA: searchB must pass ERASED at A slot and findB. Writing EMPTY would terminate search early and falsely reportB absent.

## Computation before and after

When deleting from an open-addressed table, mark removed slots as tombstones so a search continues past deletions instead of mistaking them for the original end of a probe chain.

Probe for the key, stopping on never-occupied EMPTY but continuing past ERASED. If an equal live slot is found, atomically transition the observed slot to the erased sentinel state; success means this deletion removed one live entry. Later insertion may claim a tombstone using the appropriate expected-state compare-exchange. Do not replace the removed slot by EMPTY unless a full rebuild repairs all dependent probe chains. Reclaiming payload or reusing a key needs its own reader lifetime protocol.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA static open-addressing erase paths using stable slot CAS and explicit erased marker. Concurrent arbitrary erase/reinsert is not proved by insertion-only slot publication rules; supported operator combinations must establish their own lifetime contract. Resource effects: Retains probe-chain reachability without moving following entries; accumulated tombstones lengthen probing and may motivate a rebuild. No load-factor/performance threshold inferred.

Scope predicate: `p.empty_and_erased_sentinels_distinct and p.keys_exclude_sentinels and p.probe_sequence_consistent and p.atomic_slot_transition_supported and p.operation_concurrency_policy_defined`.

- Precondition `pre_0` (semantic): `p.empty_and_erased_sentinels_distinct and p.keys_exclude_sentinels` — Empty and erased keys are distinct states, neither a valid user key.
- Precondition `pre_1` (semantic): `p.probe_sequence_consistent and p.atomic_slot_transition_supported` — Find/insert/erase share probe sequence and slot removal uses the declared atomic state transition.
- Precondition `pre_2` (semantic): `p.operation_concurrency_policy_defined` — Phase ordering or a separately verified mixed-operation protocol defines erase/reinsert/find lifetime and ABA behavior.

## Local example and misuse

For a local instance of `map`, identify the object `tombstone_preserving_probe_chain` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/tombstone_preserving_probe_chain@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/tombstone_preserving_probe_chain@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Retains probe-chain reachability without moving following entries; accumulated tombstones lengthen probing and may motivate a rebuild. No load-factor/performance threshold inferred.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `probe_preservation` (proof): Removing one slot cannot make later colliding keys unreachable.
- `linearized_remove` (proof): Successful erase corresponds to the observed live key and failed interference is retried or reported consistently.
- `sentinel_semantics` (proof): Every operation distinguishes empty, erased and live states as required.
- `reader_lifetime` (proof): No consumer uses removed/replaced payload under a stale key observation without an admitted concurrency protocol.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
