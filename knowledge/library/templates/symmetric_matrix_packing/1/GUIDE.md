# symmetric_matrix_packing@1: symmetric matrix packing

## When to consider

When every consumer accepts a symmetric matrix representation, store only one triangular set of independent values and retain the matching symmetric derivative convention. QuatScaleToCovarCUDA.cu:35-115,205-285 uses3x3 upper order00,01,02,11,12,22; backward halves off-diagonal v_covar/v_preci before symmetric VJP.

## Computation before and after

When every consumer accepts a symmetric matrix representation, store only one triangular set of independent values and retain the matching symmetric derivative convention.

Store upper-triangle coordinates in lexicographic row order (i,j),i<=j. Index=sum_{a<i}(D-a)+(j-i); packed size D*(D+1)/2. Read lower entries from their upper mirror. For a gradient on packed values, choose a symmetric full-matrix gradient representative: diagonal unchanged, each off-diagonal packed gradient split equally between mirrored positions. This representative has the same inner product with every symmetric matrix perturbation. For a full-matrix gradient contracted back to independent packed variables, sum mirrored off-diagonal gradients. Do not use this packing for nonsymmetric data or independent off-diagonal parameters.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| dimension | compile / software | {"lo": 1, "hi": 1024} | Symmetric square matrix dimension; source3. |

## Coupled constraints and scope

Real symmetric matrices and symmetric perturbations; half splitting requires admitted arithmetic. Complex Hermitian conjugation is outside this version. Resource effects: D*(D+1)/2 scalar slots versus D², plus triangular address decoding.

Scope predicate: `p.matrix_exactly_symmetric and p.all_consumers_accept_triangular_representation and p.packed_order_declared and p.symmetric_tangent_gradient_contract and p.capacity_valid`.

- Precondition `pre_0` (semantic): `p.matrix_exactly_symmetric and p.all_consumers_accept_triangular_representation` — Off-diagonal mirrored entries are one logical value, not independent inputs.
- Precondition `pre_1` (semantic): `p.packed_order_declared and p.symmetric_tangent_gradient_contract and p.capacity_valid` — Packed order and backward interpretation are shared; gradient is tested only on symmetric perturbations.

## Local example and misuse

For a local instance of `map`, identify the object `symmetric_matrix_packing` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/symmetric_matrix_packing@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/symmetric_matrix_packing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: D*(D+1)/2 scalar slots versus D², plus triangular address decoding.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `value` (proof): Unpacking mirrors exactly the original symmetric matrix at every coordinate.
- `gradient` (proof): Packed directional derivative equals the full-gradient inner product on the symmetric tangent space.
- `order` (proof): Every consumer uses same triangular ordering and forward/backward representation generation.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
