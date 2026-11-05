# piecewise_affine_hessian_elision@1: fixed region derivative graph

## When to consider

When a second reverse pass enters a field that is affine in all differentiated coordinates within its fixed support region, skip the zero pure-coordinate Hessian. Keep the upstream-gradient and parameter-mixed paths. Simplex interpolation has this property; bilinear f(x,y)=x*y does not, since its mixed coordinate derivative is1. Even y=theta*x has zero pure-input Hessian but a nonzero parameter/input derivative, so zeroing all second-pass outputs is invalid.

## Computation before and after

Elide only the pure-coordinate second derivative of a fixed affine piece.

For y(x,theta)=A(theta)*x+b(theta) inside the selected region, the first reverse input derivative is g_x=A(theta)^T*g_y. When differentiating this expression with incoming vector v, emit zero for its pure-coordinate path d(g_x)/dx. Still compute dL/dg_y=A(theta)*v and the requested parameter path obtained by contracting v and g_y with dA/dtheta. Do not zero the Hessian of an enclosing nonlinear loss or eliminate the derivatives of moving support, masks, coordinate transforms or region boundaries without a separate contract.

Recover the original computation: Reinsert the omitted zero pure-coordinate Hessian contraction. The retained mixed and upstream paths reproduce the full local second reverse derivative graph.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Local differentiation of affine pieces such as simplex barycentric fields. Multilinear Cartesian interpolation is not jointly affine in several coordinates, even when each coordinate separately enters linearly. Changed support, soft level weights differentiated as inputs, nonlinear coordinate maps and the Hessian of an enclosing nonlinear objective are outside scope.

Scope predicate: `p.field_affine_in_coordinates_within_region and p.region_selection_frozen and p.interior_query_or_selected_piece_derivative_contract and p.upstream_gradient_held_independent and p.mixed_parameter_and_upstream_paths_retained and p.affine_model_and_derivative_rounding_admitted`.

- Precondition `pre_affine` (semantic): `p.field_affine_in_coordinates_within_region and p.region_selection_frozen` — At fixed parameters, the selected field is A(theta)*x+b(theta) locally; support, ranks, level mask and shifts are frozen during differentiation.
- Precondition `pre_boundary` (semantic): `p.interior_query_or_selected_piece_derivative_contract` — Queries are interior to a region or the workload explicitly requests its selected-piece derivative; no distributional boundary Hessian is claimed.
- Precondition `pre_graph` (semantic): `p.upstream_gradient_held_independent and p.mixed_parameter_and_upstream_paths_retained` — This local derivative treats the incoming gradient as independent, while later chain-rule terms and requested mixed derivatives remain in the graph.
- Precondition `pre_numeric` (implementation): `p.affine_model_and_derivative_rounding_admitted` — The differentiation contract admits the affine real model and its actual cast/MMA/interpolation arithmetic.

## Local example and misuse

For a local instance of `map`, identify the object `fixed_region_derivative_graph` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tiny-cuda-nn/delta/piecewise_affine_hessian_elision@1/template.yaml`. Original outline: `knowledge/studies/tiny-cuda-nn/delta/piecewise_affine_hessian_elision@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: No pure-coordinate Hessian tensor or vertex reads for that zero path.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `local_affineness` (proof): On the declared region the Jacobian is independent of coordinates at fixed parameters, including all coordinate transforms differentiated here.
- `derivative_graph_partition` (proof): Only the pure-coordinate path is removed; requested upstream and mixed-parameter contractions remain and their ownership is correct.
- `region_and_numerics` (test): Support/rank ties, masks, casts and rounding satisfy the selected-piece derivative and numerical contract.

Differentiating an admitted affine real model is distinct from differentiating quantized or rounded machine arithmetic as a discontinuous function.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
