# blurred_gaussian_opacity_compensation@1: blurred gaussian opacity compensation

## When to consider

When a projected Gaussian is broadened by an isotropic pixel filter, pair covariance blur with a determinant-ratio opacity correction and keep any numerical floor explicit. Sigma=diag1,1 and eps1 gives det1->4,compensation0.5. Applying0.5 twice changes continuous mass; clamping a very small ratio to0.005 changes the mass intentionally.

## Computation before and after

When a projected Gaussian is broadened by an isotropic pixel filter, pair covariance blur with a determinant-ratio opacity correction and keep any numerical floor explicit.

Replace Sigma by Sigma_blur=Sigma+eps2d*I. Compute raw compensation=sqrt(det(Sigma)/det(Sigma_blur)); source uses sqrt(max(MIN_COMPENSATION^2,ratio)). Multiply opacity by compensation exactly once when the selected antialiasing path requests it, and invert the blurred covariance to obtain the conic. Continuous unnormalized2D Gaussian mass is preserved only when the compensation floor is inactive; the floored path is an explicitly modified numerical contract. Radius/opacity culling must use the same compensated-opacity convention as downstream rendering.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA EWA/UT3DGS antialiasing metadata. Source MIN_COMPENSATION0.005 bounds opacity scaling. Pixel sampling, support truncation and nonlinear camera approximations remain separate from continuous mass identity. Resource effects: Adds determinant ratio/sqrt and conic inversion to projection; broadens screen support while adjusting amplitude, potentially increasing covered tiles.

Scope predicate: `p.covariance_positive_definite and p.blur_nonnegative and p.compensation_policy_explicit and p.blur_consumers_consistent and p.finite_positive_determinants`.

- Precondition `pre_0` (semantic): `p.covariance_positive_definite and p.blur_nonnegative` — Original projected covariance is positive definite and eps2d>=0.
- Precondition `pre_1` (semantic): `p.compensation_policy_explicit and p.blur_consumers_consistent` — Projection/culling/rasterization agree whether compensation is applied and on its floor.
- Precondition `pre_2` (semantic): `p.finite_positive_determinants` — Original/blurred determinants and ratio are finite positive for the admitted arithmetic.

## Local example and misuse

For a local instance of `map`, identify the object `blurred_gaussian_opacity_compensation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/blurred_gaussian_opacity_compensation@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/blurred_gaussian_opacity_compensation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Adds determinant ratio/sqrt and conic inversion to projection; broadens screen support while adjusting amplitude, potentially increasing covered tiles.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `mass_relation` (proof): Without floor activation, opacity_new*sqrt(det_blur)=opacity_old*sqrt(det_orig).
- `single_application` (proof): Compensation is applied once across projection/consumer boundaries, never both omitted and doubled.
- `floor_boundary` (proof): Floor activation is visible in the contract and is not described as exact mass preservation.
- `shape_metadata` (proof): Conic/radius and opacity correspond to the same blurred covariance generation.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
