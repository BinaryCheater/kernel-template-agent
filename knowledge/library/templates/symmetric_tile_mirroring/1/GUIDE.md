# symmetric_tile_mirroring@1: symmetric tile mirroring

## When to consider

When the whole output contract is symmetric under transpose, compute one triangular tile representative and write its transpose partner instead of computing both contractions.

## Computation before and after

When the whole output contract is symmetric under transpose, compute one triangular tile representative and write its transpose partner instead of computing both contractions.

For n square tile groups, enumerate triangular work indices0..n*(n+1)/2-1. Decode row r with r*(r+1)/2 <= idx < (r+1)*(r+2)/2, and c=idx-r*(r+1)/2. Compute tile(r,c), store it, and for r!=c also store its transpose at(c,r). Suppress the second store on diagonal groups. Apply a full-output symmetry premise including epilogue and preexisting beta*C. A non-square CTA decomposition may form square cluster groups; mirror only at that group granularity and preserve internal transpose layout. Approximate sqrt decoding must be checked at integer triangular boundaries or corrected.

Recover the original computation: Symmetric output maps, not arbitrary GEMM. Erasing triangular scheduling and mirrored stores recovers a full symmetric matrix under the complete epilogue contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_tiles | compile / software | {"lo": 1, "hi": 1048576} | n: derived square output group count. |

## Coupled constraints and scope

Symmetric output maps, not arbitrary GEMM. Erasing triangular scheduling and mirrored stores recovers a full symmetric matrix under the complete epilogue contract.

Scope predicate: `(p.output_contract_symmetric and p.group_square and p.mirror_stores_disjoint and p.tail_stores_guarded)`.

- Precondition `pre0` (semantic): `p.output_contract_symmetric and p.group_square and p.mirror_stores_disjoint and p.tail_stores_guarded` — Contraction, epilogue and any beta*C together are symmetric.

## Local example and misuse

For a local instance of `map`, identify the object `symmetric_tile_mirroring` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/symmetric_tile_mirroring@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/symmetric_tile_mirroring@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `triangle_bijection` (proof): Triangular decode hits each lower triangle group exactly once.
- `mirror_coverage` (proof): Off-diagonal transpose stores fill the other half once; diagonal is written once.
- `numeric_symmetry` (proof): Reusing one computed value for its partner is admitted by the numeric contract.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
