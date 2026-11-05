# upper_bound_guarded_refinement@1: expensive bounded scalar predicate

## When to consider

When a scalar decision needs an expensive reduction but a cheap upper bound already rejects many tests, guard refinement with that bound. Preserve test identity and compute the exact threshold only for unresolved cases.

## Computation before and after

Cheap conservative upper bound guards an expensive pure threshold refinement.

The requested result is test_value < exact_threshold. Eager computes the exact threshold and compares. Guarded first tests test_value < cheap_upper_bound; if false, return false without the expensive reduction; otherwise compute the same exact threshold and compare. All data and random draws retain their original identity. If the exact value is also needed as later output, preserve its evaluation on that path instead of skipping it.

Recover the original computation: Reinsert the skipped pure threshold evaluation. Since exact_threshold<=cheap_upper_bound<=test_value on every skipped path, the original strict comparison was false.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| evaluation | compile / software | ["eager", "guarded"] | Always compute the exact threshold or skip it when a cheap upper bound already proves the strict comparison false. |

## Coupled constraints and scope

SGLang block verifier uses prefix scalar c in[0,1], residual M=sum(max(c*P-Q,0)) and interior acceptance h=M/(M+1-c). For normalized finite P,Q, M<=c implies h<=c. Reject coin>=c before scanning vocabulary. Boundary c=0/1 and final-prefix acceptance have separately defined direct thresholds; they are not evaluated by the interior formula. No probability-law claim follows merely from this work-skipping transformation.

Scope predicate: `(p.exact_threshold <= p.cheap_upper_bound and p.ordered_non_nan_comparison) and (p.expensive_threshold_evaluation_pure and p.skipped_evaluation_has_no_required_outputs) and (p.same_test_value_and_inputs_in_eager_and_lazy_paths and p.native_bound_rounding_conservative)`.

- Precondition `pre_bound` (semantic): `p.exact_threshold <= p.cheap_upper_bound and p.ordered_non_nan_comparison` — The cheap bound cannot be smaller than the exact threshold; both use the declared strict comparison and exclude NaNs.
- Precondition `pre_effects` (semantic): `p.expensive_threshold_evaluation_pure and p.skipped_evaluation_has_no_required_outputs` — The expensive threshold computation has no externally required effects; only its comparison result matters on skipped paths.
- Precondition `pre_identity` (implementation): `p.same_test_value_and_inputs_in_eager_and_lazy_paths and p.native_bound_rounding_conservative` — Moving the guard cannot change the test value, random identity, threshold inputs or conservative native upper bound.

## Local example and misuse

For a local instance of `contract`, identify the object `expensive_bounded_scalar_predicate` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/upper_bound_guarded_refinement@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/upper_bound_guarded_refinement@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Bound/test scalars; exact reduction scratch only on surviving paths.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bound` (proof): The actual native upper bound is conservative for all admitted inputs and rounding.
- `skip_equivalence` (proof): Each skipped path returns the same predicate and preserves required state and randomness.

An underestimated rounded bound can skip a true result. Threshold evaluation side effects or differently consumed random draws break equivalence.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
