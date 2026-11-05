# first_failure_prefix@1: first failure prefix

## When to consider

When the accepted length is the leading run of independently available true predicates, reduce the first false index instead of carrying a serial live-prefix flag.

## Computation before and after

When the accepted length is the leading run of independently available true predicates, reduce the first false index instead of carrying a serial live-prefix flag.

For predicates b_i,i=0..length-1, the serial form increments a count only while every preceding predicate remains true. The parallel form maps a true valid position to the endpoint length and a false position to its original index, then takes minimum together with length. Empty/all-true rows return length. Consumers may use that index to load one boundary/bonus record if the declared input has it; no extra matching positions beyond the first failure are accepted.

Recover the original computation: The minimum false index j has all b_i true for i<j and b_j false if j<length, exactly matching the serial leading-prefix count. Padding has no smaller value and cannot change j.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| strategy | compile / software | ["serial_live_prefix", "parallel_first_false"] | Serial leading-prefix recurrence or minimum failure-position reduction. |

## Coupled constraints and scope

Source DFlash dflash.py:8-63 versus326-347: length=block_size-1, b_i=(candidate[i+1]==target[i]), bonus=target[result]. Original full output buffer beyond commit_len is not required invalid. Nonempty vector reductions may implement empty logical length with padded neutral lanes.

Scope predicate: `(p.length >= 0 and p.predicates_available_without_prefix_side_effects) and (p.padding_uses_length_sentinel and p.accepted_length_not_total_match_count) and (p.same_predicate_semantics_and_original_positions)`.

- Precondition `pre_0` (semantic): `p.length >= 0 and p.predicates_available_without_prefix_side_effects` — Every valid predicate can be evaluated independently from already available inputs; skipping later predicate evaluation has no required side effect.
- Precondition `pre_1` (semantic): `p.padding_uses_length_sentinel and p.accepted_length_not_total_match_count` — Padding contributes only the neutral endpoint length; result means a consecutive leading prefix.
- Precondition `pre_2` (semantic): `p.same_predicate_semantics_and_original_positions` — No changed comparison/tie/NaN policy or position permutation is introduced.

## Local example and misuse

For a local instance of `reduce`, identify the object `first_failure_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/first_failure_prefix@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/first_failure_prefix@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-instance records and optional materialized view.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `representation_identity` (proof): The minimum false index j has all b_i true for i<j and b_j false if j<length, exactly matching the serial leading-prefix count. Padding has no smaller value and cannot change j.
- `native_mapping` (test): Indices, padding, sentinels and output ownership implement the logical mapping.

No arithmetic approximation; original predicate or topology and identity must be preserved.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
