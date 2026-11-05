# value_lowbit_index_tag@1: value lowbit index tag

## When to consider

When ranking values widened from a narrow floating format leaves unused low mantissa bits, embed a stable source index in those bits so the comparison network transports value and index together.

## Computation before and after

When ranking values widened from a narrow floating format leaves unused low mantissa bits, embed a stable source index in those bits so the comparison network transports value and index together.

Let b=ceil(log2(N)), mask=2^b-1. Require each widened finite FP32 bit pattern to have zero low b bits. For positive x tag=(~index)&mask; for negative x tag=index&mask. Replace low bits by tag, rank encoded floats descending, then clear the tag bits and recover index with the inverse sign-dependent transform. Equal positive/negative nonzero values favor smaller indices. Restrict this version to finite nonzero values; signed zeros, infinities and NaNs need an explicit total-order extension. For source BF16 and N<=4096 b<=12 fits the16 zero low bits; FP16 leaves13. General FP32 inputs can lose rank-relevant bits and are excluded.

Recover the original computation: Finite nonzero narrow-float values widened exactly to FP32 with enough vacant mantissa bits. Ranking uses fewer register planes; this rule does not authorize truncating meaningful FP32 mantissa bits.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| index_bits | compile / software | {"lo": 1, "hi": 16} | b derived from row extent; source N power-of-two<=4096. |

## Coupled constraints and scope

Finite nonzero narrow-float values widened exactly to FP32 with enough vacant mantissa bits. Ranking uses fewer register planes; this rule does not authorize truncating meaningful FP32 mantissa bits.

Scope predicate: `(index_bits <= p.widened_zero_low_bits) and (p.finite_nonzero_values and p.indices_fit and p.descending_float_comparator)`.

- Precondition `pre0` (semantic): `p.finite_nonzero_values and p.indices_fit and p.descending_float_comparator` — Defined finite nonzero source domain and descending numerical compare.
- Constraint `c0` (implementation): `index_bits <= p.widened_zero_low_bits` — Tag replaces only provably unused bits.

## Local example and misuse

For a local instance of `select`, identify the object `value_lowbit_index_tag` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/value_lowbit_index_tag@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/value_lowbit_index_tag@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `value_restore` (proof): Clearing tag bits exactly restores every input value.
- `order_and_ties` (proof): Encoded comparisons preserve unequal-value rank and select smaller index for ties.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
