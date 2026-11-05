# blocked_unit_lower_rhs_solve@1: completed triangular rhs block

## When to consider

When a unit-lower triangular system has many RHS columns, solve block rows in order: use matrix products for completed off-diagonal couplings and forward substitution inside each small diagonal block. This trades row-level serialization for parallel RHS work without materializing a full inverse. Choose only the admitted16-row source block and retain packed completed rows through later couplings.

## Computation before and after

Block forward substitution with matrix off-diagonal couplings and serial in-block triangular solves.

For block rowb in order0..3, startRb=Bb. Compute cross-block couplingRb-=sum(j<b)Abj*Xj using completed blocks. Solve(I+Abb)Xb=Rb by forward substitution within the16-row block independently per RHS column. PublishXb and any narrow packed transpose required by later coupling products. Do not construct or store the full inverse. Retain completed blocks until their final coupling consumer finishes.

Recover the original computation: For each completed blockb, rearrangeXb+AbbXb+sum(j<b)AbjXj=Bb; together the four equations recover(I+A)X=B.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| block_order | compile / software | [16] | Diagonal block rows; pinned64-row source uses four16-row blocks. Other orders require a realization. |

## Coupled constraints and scope

Pinned four16-row unit-lower RHS solve. KDA source64x256 RHS uses FP32 local substitution and three BF16 cross productshi*hi+lo*hi+hi*lo, excludinglo*lo, then storesP BF16 andu0 FP32. Generic exact algebra does not certify those numerical approximations or the SEQ0 inverse-block variant.

Scope predicate: `p.unit_lower_matrix and p.order == 4 * block_order and p.products_and_casts_admitted`.

- Precondition `pre_matrix` (semantic): `p.unit_lower_matrix and p.coefficients_rhs_same_generation and p.products_and_casts_admitted` — Solve(I+A)X=B whereA is strictly lower; cross-block products and diagonal serial substitutions have a declared numerical policy.
- Precondition `pre_storage` (implementation): `p.completed_blocks_ready_and_alive and p.workspaces_fit and p.all_rhs_columns_covered` — Every earlier row block is visible and retained for all later couplings; workspace and RHS column ownership are valid.
- Constraint `c_shape` (implementation): `p.order == 4 * block_order and p.rhs_columns >= 1` — Four-block source binding with one or more RHS columns.

## Local example and misuse

For a local instance of `contract`, identify the object `completed_triangular_rhs_block` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/blocked_unit_lower_rhs_solve@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/blocked_unit_lower_rhs_solve@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Completed RHS rows and optional transposed narrow coupling operands; no full inverse.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `rhs_equations` (exhaustive): Each block solves its diagonal unit-lower system after subtracting every preceding block coupling once.
- `dependency` (test): Every coupling reads a completed matching-generationXj and all RHS columns have unique owners.
- `precision` (test): Narrow operand encodings and dropped component products satisfy the declared solve residual policy.

Three-product BF16 expansion omits low*low and changes rounding; compare residual and end-to-end application error separately.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
