# row_column_rescaling@1: iteratively rescaled matrix

## When to consider

Use when a caller requests finite row/column rescaling of4x4 logits; bind iterations and preserve its epsilon policy. Apply stable row softmax plus epsilon, one column normalization and then the remaining row/column rounds. For iterations1 return immediately after the first column normalization; this output need not have exact unit row or column sums.

## Computation before and after

Stable row softmax plus epsilon then finite alternating column and row rescaling.

Start from4x4 logits. For each row subtract its maximum, exponentiate and divide by that row's exponential sum, then add epsilon to every entry. Normalize each column by its sum plus epsilon. For rounds1 through iterations-1 normalize rows by row sum plus epsilon and columns by column sum plus epsilon. Return this finite state; do not replace it with an assumed converged doubly-stochastic matrix.

Recover the original computation: Read the final matrix as the requested finite row/column rescaling policy applied to the original logits.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| iterations | runtime / boundary | {"lo": 1, "hi": 65536} | Requested number of column-normalization rounds; changing it changes the coarse numerical policy. |

## Coupled constraints and scope

A caller-defined finite rescaling policy on4x4 coefficient logits. The first row operation differs from later row operations and epsilon placement is part of the workload. This record makes no convergence or exact unit-row/unit-column-sum claim. A collective Realisation must preserve row and column lane groups independently.

Scope predicate: `p.extent == 4 and p.finite_rescaling_policy_requested`.

- Precondition `pre_domain` (semantic): `p.finite_logits and p.epsilon >= 0 and p.positive_finite_denominators` — Row softmax and every requested normalization have finite positive denominators.
- Precondition `pre_contract` (semantic): `p.finite_rescaling_policy_requested and p.numeric_contract_admits_realisation` — The caller requests this finite policy and admits the selected exponential/reciprocal/reduction arithmetic.
- Constraint `c_policy` (semantic): `iterations == p.requested_iterations and p.extent == 4` — Preserve the requested iteration count and this version's square4x4 geometry.

## Local example and misuse

For a local instance of `map`, identify the object `iteratively_rescaled_matrix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/row_column_rescaling@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/row_column_rescaling@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One evolving4x4 coefficient matrix.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `group_identity` (exhaustive): Every row and column reduction uses exactly that row or column's four matrix entries with no inactive-lane contamination.
- `stage_order` (test): The first softmax-plus-epsilon stage and all subsequent row/column stages occur in the requested order and count.
- `finite_arithmetic` (test): Exponential and reciprocal implementations and reduction rounding satisfy the requested numerical contract over the admitted input range.

Fast exponential/reciprocal and finite-precision reductions affect coefficients; extreme logits and small denominators need numerical checks.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
