# nilpotent_inverse_doubling@1: doubling unit lower inverse iterate

## When to consider

When a small unit-lower block is to be inverted on matrix hardware, initializeI-A and double the inverse polynomial with2X-X(I+A)X. For16 rows three updates suffice algebraically becauseA is nilpotent. This replaces serial row dependency with matrix products, while requiring explicit fragment orientation and a finite-precision inverse residual contract.

## Computation before and after

Three Newton-Schulz polynomial doublings invert a16-order unit-lower block in exact arithmetic.

SetX0=I-A. Repeat exactlythree timesXnext=2X-X*(I+A)*X, retaining the source product order and storage casts. In exact arithmetic this doubles the included Neumann-polynomial terms from2 to4,8,16; becauseA^16=0, the finalX invertsI+A. Publish the completed16x16 inverse before dependent lower-block assembly. Source stores an initial negative systemM=-(I+A), computesX*M, rounds, then accumulates(X*M)*X+2X under the stated numerical policy.

Recover the original computation: Erase intermediate iterates; after three exact updates(I+A)X=I=X(I+A) by nilpotence. Native implementation refines only an admitted approximate inverse contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Pinned16x16 unit-lower inverse building block, used inside four-block64-token feedback inverses. No norm-contraction assumption needed for exact nilpotent algebra; finite-precision error still requires admission. Not a general dense Newton-Schulz convergence rule.

Scope predicate: `p.order == 16 and p.strict_lower_A and p.inverse_cast_mma_residual_policy_admitted`.

- Precondition `pre_matrix` (semantic): `p.strict_lower_A and p.unit_diagonal_system and p.operand_maps_preserve_matrix_orientation` — A^16=0, systemI+A and MMA transpose/fragment maps retain multiplication order.
- Precondition `pre_numeric` (implementation): `p.inverse_cast_mma_residual_policy_admitted and p.finite_products` — Source BF16 operand casts and FP32 products are an admitted approximate realization; exact finite polynomial does not certify rounded inverse accuracy.
- Constraint `c_shape` (implementation): `p.order == 16 and p.doubling_steps == 3` — Pinned warp-local16x16 block, initialI-A then three doubling updates.

## Local example and misuse

For a local instance of `contract`, identify the object `doubling_unit_lower_inverse_iterate` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/nilpotent_inverse_doubling@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/nilpotent_inverse_doubling@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Warp fragments plus shared transpose image; intermediate rounding is explicit.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `polynomial` (proof): Each exact update squares the inverse error polynomial, giving degree16 after three steps fromI-A.
- `orientation` (test): MMA operand/transpose maps implement2X-X(I+A)X and not an incorrectly transposed/reordered product.
- `numeric_residual` (test): Rounded BF16/FP32 source sequence meets the declared inverse residual/error policy.

BF16 rounding after intermediate matrix products invalidates exact polynomial cancellation; small norm is not inferred and inverse residual must be checked for the source regime.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
