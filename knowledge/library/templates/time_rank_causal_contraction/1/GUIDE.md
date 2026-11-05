# time_rank_causal_contraction@1: time rank causal contraction

## When to consider

When each time step has multiple input/output rank channels, flatten time×rank for GEMM but apply causality by time quotient and compute the same-time rank block under its own coefficient.

## Computation before and after

When each time step has multiple input/output rank channels, flatten time×rank for GEMM but apply causality by time quotient and compute the same-time rank block under its own coefficient.

Flatten row i=t*R+r. Form QK^T on flattened rows but keep strictly past block iff floor(i/R)>floor(j/R). Same-time rows form an R×R block, all of whose rank pairs participate under the current-step coefficient, separately from shifted past-history scaling. Contract the past matrix against rank-expanded values, add the same-time block contraction, then apply requested rank-dependent output projection/gating and sum output rank. Source R1 reduces to strict triangular matrix plus scalar diagonal; R>1 uses full same-time block, not just elementwise diagonal.

Recover the original computation: MIMO causal contractions with grouped time coordinates; feature/head grouping and rank projection are workload inputs, not automatically interchangeable dimensions.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| rank_channels | compile / software | {"lo": 1, "hi": 64} | R: workload MIMO rank. |

## Coupled constraints and scope

MIMO causal contractions with grouped time coordinates; feature/head grouping and rank projection are workload inputs, not automatically interchangeable dimensions.

Scope predicate: `(p.flatten_map_bijective and p.time_coefficients_shared_within_step and p.same_time_block_complete and p.rank_projection_contract_fixed and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.flatten_map_bijective and p.time_coefficients_shared_within_step and p.same_time_block_complete and p.rank_projection_contract_fixed and p.rounding_admitted` — Known rank layout, correct time coefficients, complete diagonal blocks and declared rank projection.

## Local example and misuse

For a local instance of `contract`, identify the object `time_rank_causal_contraction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/mamba/delta/time_rank_causal_contraction@1/template.yaml`. Original outline: `knowledge/studies/mamba/delta/time_rank_causal_contraction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `temporal_mask` (proof): Quotient comparison excludes every same-time rank pair from past history.
- `same_time` (proof): Separate R×R block includes every current-step input/output rank interaction exactly once.
- `rank_pullback` (proof): Backward sums the same sharing fibers used by forward head/rank broadcasts.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
