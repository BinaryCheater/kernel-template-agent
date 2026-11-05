# anisotropic_residual_coordinate_update@1: anisotropic residual coordinate update

## When to consider

When quantization error parallel to a vector matters more than perpendicular error, refine one coordinate at a time while maintaining the shared residual projection. eta3,h1,delta_h=-0.2,delta_norm0.1 gives delta_cost=0.1+2*0.04+4*(-0.2)*1=-0.62, so accept and set h0.8. A subsequent proposal must use0.8 rather than1.

## Computation before and after

When quantization error parallel to a vector matters more than perpendicular error, refine one coordinate at a time while maintaining the shared residual projection.

Start from the requested low-precision representation. Let residual r=x-q, u=x/||x|| and h=r dot u. For a proposed one-coordinate change delta_r at i, compute delta_h=delta_r*u_i and delta_norm=r_new_i^2-r_old_i^2. Precompute a=delta_norm+(eta-1)*delta_h^2 and b=2*(eta-1)*delta_h. Serialize coordinates, accept exactly when a+b*h<0, update q_i and h, then broadcast h before the next coordinate. Repeat until no changes or max_rounds; retain all nonaccepted coordinates.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| max_rounds | compile / software | [10] | Observed finite coordinate-descent round cap; a search budget, not convergence guarantee. |

## Coupled constraints and scope

NVIDIA ScaNN noise-shaped BF16 conversion. Sequential update order is part of the heuristic; parallel acceptance from stale h changes the algorithm. Zero norms and nonfinite/representation endpoint cases require explicit handling before use. Resource effects: One scalar residual projection per warp, local proposal coefficients and O(dim) codes; avoids recomputing a full vector objective for each proposal, at cost of ordered warp broadcasts.

Scope predicate: `p.nonzero_finite_vector and p.eta_defined and p.quantized_neighbor_valid and p.residual_projection_consistent and p.coordinate_updates_serialized and p.numeric_policy_admitted`.

- Precondition `pre_0` (semantic): `p.nonzero_finite_vector and p.eta_defined` — Vector norm is finite positive and anisotropic weight eta is defined.
- Precondition `pre_1` (semantic): `p.quantized_neighbor_valid and p.residual_projection_consistent` — Proposed adjacent representable value is legal and maintained projection matches all accepted coordinate changes.
- Precondition `pre_2` (semantic): `p.coordinate_updates_serialized and p.numeric_policy_admitted` — Each coordinate decision sees the latest shared projection, with agreed rounding.

## Local example and misuse

For a local instance of `map`, identify the object `anisotropic_residual_coordinate_update` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/anisotropic_residual_coordinate_update@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/anisotropic_residual_coordinate_update@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One scalar residual projection per warp, local proposal coefficients and O(dim) codes; avoids recomputing a full vector objective for each proposal, at cost of ordered warp broadcasts.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `cost_delta` (proof): a+b*h equals the change of ||r||^2+(eta-1)*(r dot u)^2 under the single-coordinate proposal.
- `latest_projection` (proof): Each decision uses projection after all earlier accepted proposals.
- `representation` (proof): Neighbor-step direction and sign handling produce a valid adjacent requested code, including representation boundaries.
- `termination` (proof): Finite cap terminates; no claim of global minimum or exact convergence.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
