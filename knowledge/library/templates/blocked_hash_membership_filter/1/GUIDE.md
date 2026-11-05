# blocked_hash_membership_filter@1: blocked hash membership filter

## When to consider

When approximate set membership is acceptable, map each key to one small bit block and test or atomically OR its deterministic word patterns under a shared policy. Two keys setting bits0 and3 in the same word must leave0b1001; non-atomic load/OR/store can leave only one bit. An absent key requiring those same bits is a permitted false positive; a completed inserted key returning absent is not.

## Computation before and after

When approximate set membership is acceptable, map each key to one small bit block and test or atomically OR its deterministic word patterns under a shared policy.

Split a key hash into block-selection and lower pattern bits. Map the upper part to a filter block; derive required word patterns from the lower part. Insertion atomically ORs those patterns into their designated words. Query checks (stored_word & required_pattern)==required_pattern for every required word and ANDs the results across its cooperative layout. Merge equal-policy filters by wordwise OR. Intersection/clear are separate operations with separate concurrency/semantic contracts. Conditional no-op OR may skip already set bits only under a valid memory-access model.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| words_per_block | compile / software | [1, 2, 4, 8, 16, 32] | Power-of-two block words, bounded by source layout. |
| word_bits | compile / software | [32, 64] | Native atomic-OR word format; policy and target must agree. |

## Coupled constraints and scope

NVIDIA blocked Bloom filter exact state representation with approximate membership. Native block/device/system atomic scope must encompass all writers. Source conditional non-atomic read optimization is not itself a portable race-free proof. Resource effects: One hashed block bounds scattered-word footprint; block width and pattern density trade memory/atomic traffic against false-positive rate. Statistical error needs separate analysis, no speed or FPR measured here.

Scope predicate: `p.hash_pattern_policy_identical and p.filter_initialized_zero and p.insert_or_atomic_and_visible and p.concurrent_clear_excluded and p.false_positive_membership_contract`.

- Precondition `pre_0` (semantic): `p.hash_pattern_policy_identical and p.filter_initialized_zero` — Insert and query use the same split hash, block mapping and bit-pattern policy from zero state.
- Precondition `pre_1` (semantic): `p.insert_or_atomic_and_visible and p.concurrent_clear_excluded` — All competing insertions preserve every bit through correct-scope OR and complete before absence is interpreted.
- Precondition `pre_2` (semantic): `p.false_positive_membership_contract` — Caller accepts collisions producing present for absent keys; no false-negative contract applies only after admitted insertions finish.

## Local example and misuse

For a local instance of `map`, identify the object `blocked_hash_membership_filter` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/blocked_hash_membership_filter@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/blocked_hash_membership_filter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One hashed block bounds scattered-word footprint; block width and pattern density trade memory/atomic traffic against false-positive rate. Statistical error needs separate analysis, no speed or FPR measured here.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `same_pattern` (proof): A key maps to identical block and required patterns in every insertion/query layout.
- `monotone_bits` (proof): Concurrent insertion cannot lose previously set required bits.
- `query_complete` (proof): All required words/lanes participate and are visible before a definitive absent result.
- `policy_merge` (proof): Merged filters have matching block extents and pattern/hash policy.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
