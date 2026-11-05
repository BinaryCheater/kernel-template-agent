# blocked_unit_lower_inverse@1: block lower inverse

## When to consider

When a short unit lower-triangular solve lies between contractions, invert its small diagonal blocks locally and form off-diagonal inverse blocks by ordered matrix products. This exposes GEMM work while retaining the block dependency order. The pinned binding is four 16-token blocks; solving arbitrary dense or ill-conditioned systems requires a different contract.

## Computation before and after

Block forward substitution to assemble a unit lower-triangular inverse.

Partition A into four-by-four equal blocks with zero upper blocks and unit lower-triangular diagonal blocks. Compute D_ii=inverse(A_ii) by forward substitution. For each block column j and increasing i>j compute D_ij = -D_ii * sum_(k=j..i-1) A_ik * D_kj. Set upper blocks to zero. Products may use staged matrix contractions, but block dependencies and sum completeness are retained. The source forms adjacent blocks 10,21,32 first, then 20,31, then 30 and stores the assembled lower inverse for its later W/U contractions.

Recover the original computation: Multiply the assembled inverse by the original unit lower-triangular matrix; the block equations recover the identity.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| block_order | compile / software | [16] | Diagonal block order in the pinned four-block source binding; larger menus require new binding evidence. |

## Coupled constraints and scope

Four-block unit lower-triangular inverse with source block order 16 and matrix order 64. TileLang KDA computes gate-weighted off-diagonal blocks and combines them with diagonal blocks produced separately. Tail, singular/general dense systems and a guarantee of numerical stability for arbitrary coefficients are outside this version. Gate factorization and score construction are separate upstream transformations.

Scope predicate: `p.order == 4 * block_order and p.unit_lower_matrix and p.complete_matrix_and_sequence_binding and p.inverse_residual_policy_defined and p.products_and_casts_admitted and p.diagonal_solve_valid and p.dependencies_completed_before_read and p.scratch_aliases_valid`.

- Precondition `pre_matrix` (semantic): `p.unit_lower_matrix and p.complete_matrix_and_sequence_binding` — The matrix is I plus a strictly lower-triangular part; sequence and chunk boundaries do not mix rows.
- Precondition `pre_numeric` (implementation): `p.inverse_residual_policy_defined and p.products_and_casts_admitted and p.diagonal_solve_valid` — The diagonal solves, products and stores satisfy the inverse residual and numerical contract.
- Precondition `pre_order` (implementation): `p.dependencies_completed_before_read and p.scratch_aliases_valid` — Every inverse block is complete before dependent products read it; scratch lifetimes do not overlap.
- Constraint `c_order` (semantic): `p.order == 4 * block_order` — This version instantiates the four-block lower-triangular decomposition.

## Local example and misuse

For a local instance of `map`, identify the object `block_lower_inverse` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tilelang/delta/blocked_unit_lower_inverse@1/template.yaml`. Original outline: `knowledge/studies/tilelang/delta/blocked_unit_lower_inverse@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Diagonal inverses and completed lower blocks retained for dependent matrix products.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `block_inverse` (proof): Every diagonal block solves its local triangular system and each lower block satisfies the identity block equation.
- `dependency_order` (test): No product reads an unfinished D_kj and every k term is included once.
- `residual` (test): Native precision and cast choices meet the chosen inverse residual policy.

Triangular invertibility in exact arithmetic does not bound conditioning or low-precision residuals.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
