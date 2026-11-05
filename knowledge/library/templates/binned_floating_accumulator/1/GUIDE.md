# binned_floating_accumulator@1: binned floating accumulator

## When to consider

When reproducibility must survive changes in reduction partition rather than merely fixing a tree, select an exponent-binned partial-state representation. The source device path uses three primary and three carry collectors, paying more state and arithmetic than a scalar sum. Preserve bin alignment and renormalization endurance; replacing its low-bit deposit by ordinary addition changes the algorithm.

## Computation before and after

A floating sum uses exponent-indexed primary/carry collectors instead of one floating scalar.

Represent a partial sum using fold primary and fold carry floating values with an exponent-indexed bin anchor. Before deposit rebin to accommodate the maximum magnitude: shift retained collectors toward lower-priority bins and initialize newly exposed bins. Deposit an input by forced-low-bit floating additions into each primary collector, propagating the residual to the next. The highest-range bin uses the source compression/expansion path. Renormalize primary carry bits into the carry vector before the endurance bound is exceeded. Merge two partial accumulators by aligning bin indices, combining corresponding primary/carry components and renormalizing. Convert to output scalar only after the complete partial merge. Source device dispatch binds fold=3; manually batched unsafe deposits additionally require a declared magnitude bound and bounded count before renorm.

Recover the original computation: Interpret each binned state through its primary/carry decoding recipe; completed state represents the admitted reproducible sum of its deposited inputs.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| fold | compile / software | {"lo": 1, "hi": 65536} | Number of primary collectors, paired with the same number of carry collectors; device dispatch currently binds three. |

## Coupled constraints and scope

Finite float/double sum and the exact CCCL binned recipe, 13/40-bit bins, default fold3 in device dispatch. This is not an arbitrary exact superaccumulator, not scalar Kahan summation, and not a claim of correctly rounded real summation. Full numerical proof/native checks remain open obligations.

Scope predicate: `(p.input_is_finite_float_or_double and p.requested_sum_contract_admits_binned_rounding) and (p.bin_width_matches_type and p.native_rounding_and_bit_operations_match_recipe) and (p.collector_range_and_renormalization_bound_valid and p.merge_aligns_bin_indices)`.

- Precondition `pre0` (semantic): `p.input_is_finite_float_or_double and p.requested_sum_contract_admits_binned_rounding` — Input and result use the specified finite summation recipe.
- Precondition `pre1` (semantic): `p.bin_width_matches_type and p.native_rounding_and_bit_operations_match_recipe` — Float uses 13-bit bins, double 40-bit bins with the source bit-deposition recipe.
- Precondition `pre2` (semantic): `p.collector_range_and_renormalization_bound_valid and p.merge_aligns_bin_indices` — Collector shifts, carries and merged states respect bin/endurance invariants.

## Local example and misuse

For a local instance of `reduce`, identify the object `binned_floating_accumulator` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cccl/delta/binned_floating_accumulator@1/template.yaml`. Original outline: `knowledge/studies/cccl/delta/binned_floating_accumulator@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `encoded_sum` (proof): Rebin, deposit, merge and renorm preserve the specified encoded-sum interpretation within its numerical error contract.
- `order_independence` (test): For the admitted range/endurance, changing partition or merge order yields the declared reproducibility guarantee.
- `conversion` (test): Final scalar conversion follows the specified bin/carry rounding sequence.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
