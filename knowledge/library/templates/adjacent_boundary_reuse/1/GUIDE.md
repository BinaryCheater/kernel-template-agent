# adjacent_boundary_reuse@1: shared adjacent boundary samples

## When to consider

When many outputs subtract adjacent samples of the same expensive function, evaluate the shared endpoints once. Carry the right endpoint into the next interval, or let aligned bin lanes shuffle their next neighbor's value. OneBlob with16 bins uses16 periodic-CDF boundary evaluations plus a final +1 seam correction instead of32 evaluations. Without a valid seam identity or with bin groups crossing shuffle boundaries, use the serial form or reject the subgroup binding. Serial evaluation uses17 boundaries for16 bins.

## Computation before and after

Reuse identical interval endpoints by a serial carry or subgroup exchange.

Replace independently evaluating two boundaries per interval. For serial exchange evaluate F(0), then for b=0..bins-1 evaluate F(b+1), emit their difference, and retain F(b+1) as the next left endpoint. For subwarp exchange, aligned lane b evaluates F(b), obtains F((b+1) modulo bins) from the next lane in the same subgroup, adds the declared endpoint_jump only for the final lane, and subtracts its own F(b). Subtract/cast at the same logical output stage. The boundary function may itself sum translated CDFs; its computation and admitted coordinate domain are inputs to this rule.

Recover the original computation: Replace every retained or shuffled boundary by its defining F(b). The seam premise replaces F(0)+endpoint_jump by F(bins), recovering each independent interval difference.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| exchange | compile / software | ["serial", "subwarp"] | Carry the previous boundary in one thread or exchange neighboring boundary values across a lane subgroup. |
| bins | compile / boundary | {"lo": 1, "hi": 1024} | Number of adjacent intervals belonging to one logical input. |

## Coupled constraints and scope

Adjacent finite differences of one boundary function. Serial uses bins+1 evaluations instead of2*bins; subwarp uses bins evaluations plus exchange, but requires an aligned full-participation power-of-two group up to32 lanes and a valid endpoint jump. No arbitrary-input periodicity or derivative seam jump is inferred from a forward CDF example.

Scope predicate: `p.outputs_are_adjacent_boundary_differences and p.boundary_function_same_input_and_parameters and p.reused_boundary_rounding_admitted and p.seam_reconstruction_error_admitted`.

- Precondition `pre_difference` (semantic): `p.outputs_are_adjacent_boundary_differences and p.boundary_function_same_input_and_parameters` — Output b is F(b+1)-F(b) for one common input/parameter binding, so adjacent endpoints are identical computations.
- Precondition `pre_exchange` (implementation): `exchange != 'subwarp' or (p.subgroups_aligned_with_bins and p.all_required_lanes_participate and p.endpoint_equals_first_plus_jump)` — Lane neighbors belong to the same bin group, all shuffle participants are active, and the last endpoint is reconstructible from the first plus the declared jump.
- Precondition `pre_numeric` (implementation): `p.reused_boundary_rounding_admitted and p.seam_reconstruction_error_admitted` — Sharing rounded evaluations and reconstructing the seam satisfy the output numerical contract.
- Constraint `c_shuffle_width` (implementation): `exchange != 'subwarp' or (bins <= 32 and is_pow2(bins))` — The pinned shuffle implementation uses a power-of-two subgroup that fits one warp.

## Local example and misuse

For a local instance of `map`, identify the object `shared_adjacent_boundary_samples` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tiny-cuda-nn/delta/adjacent_boundary_reuse@1/template.yaml`. Original outline: `knowledge/studies/tiny-cuda-nn/delta/adjacent_boundary_reuse@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Each interior boundary supplies two adjacent interval outputs.. Storage: One retained boundary per thread or lane.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `common_boundary` (proof): Each reused value is exactly the endpoint required by the adjacent interval of the same logical input.
- `subgroup_and_seam` (proof): No shuffle crosses input groups, all required lanes participate, and the final endpoint identity holds on the declared domain.
- `numeric_equivalence` (test): Reused evaluations, seam correction and output casts satisfy the declared numerical policy.

Reconstructing a periodic endpoint by a jump may round differently from separately evaluating it; derivative boundary functions can have a different jump.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
