# nonlinear_sigma_point_moments@1: nonlinear sigma point moments

## When to consider

When camera projection is nonlinear over a Gaussian support, approximate projected moments from symmetric sigma points instead of a single local Jacobian, under an explicit quadrature contract. At an affine camera the weighted sigma mean/covariance recover affine Gaussian moments subject to rounding. Nonlinear fisheye projection can differ from J*cov*J^T. With alpha0.1 the center covariance weight can be negative, so determinant>0 alone must not certify a negative-definite2x2 matrix.

## Computation before and after

When camera projection is nonlinear over a Gaussian support, approximate projected moments from symmetric sigma points instead of a single local Jacobian, under an explicit quadrature contract.

For D3 let lambda=alpha2*(D+kappa)-D. Use points mean and mean+/-sqrt(D+lambda)*scale_i*R_column_i. Project all7 through the selected camera and shutter pose. Mean weights are w0=lambda/(D+lambda), wi=1/(2*(D+lambda)); center covariance weight is w0+1-alpha2+beta and remaining covariance weights wi. Sum projected points to mean, then weighted outer products about that mean to covariance. Keep camera interpolation constants shared over points when they are identical. Reject invalid output covariance before inverse/conic use.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| sigma_points | compile / software | [7] | D3 mean plus/minus three covariance-axis perturbations; fixed quadrature rather than tunable thread count. |

## Coupled constraints and scope

NVIDIA UT3DGS camera projection. This is an approximation of nonlinear image moments, not exact projected Gaussian distribution. Source any-valid mode still accumulates all projected numerical values and needs a separate defined-invalid-point contract; it is not the all-valid instance. Camera-specific loop-unroll/slerp policy is a realization choice. Resource effects: Seven camera projections and moment accumulation per Gaussian/camera versus one Jacobian projection; shared interpolation invariants may reduce repeated setup, with increased register work.

Scope predicate: `p.gaussian_factor_valid and p.lambda_plus_dimension_positive and p.all_sigma_projections_valid_and_finite and p.camera_state_consistent and p.quadrature_numeric_contract and p.projected_covariance_admissible`.

- Precondition `pre_0` (semantic): `p.gaussian_factor_valid and p.lambda_plus_dimension_positive` — Quaternion/scale factor is valid and alpha2*(D+kappa)>0 so sigma displacement and weights are defined.
- Precondition `pre_1` (semantic): `p.all_sigma_projections_valid_and_finite and p.camera_state_consistent` — This version requires all7 projected points finite/valid under the same camera/shutter model.
- Precondition `pre_2` (semantic): `p.quadrature_numeric_contract and p.projected_covariance_admissible` — Caller requests this approximate moment rule and rejects or explicitly handles indefinite projected covariance.

## Local example and misuse

For a local instance of `map`, identify the object `nonlinear_sigma_point_moments` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/nonlinear_sigma_point_moments@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/nonlinear_sigma_point_moments@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Seven camera projections and moment accumulation per Gaussian/camera versus one Jacobian projection; shared interpolation invariants may reduce repeated setup, with increased register work.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `quadrature` (proof): Sigma coordinates and separate mean/covariance weights match the requested3D quadrature exactly.
- `projection` (proof): Each point uses the same model parameters and its declared shutter-pose evaluation.
- `validity` (proof): Invalid sigma points are not silently dropped/renormalized under the all-valid version.
- `covariance` (proof): Negative center weights do not imply positive covariance; inversion follows explicit determinant/diagonal validity policy.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
