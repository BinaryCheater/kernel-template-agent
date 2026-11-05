# selected_group_residual_encoding@1: base residual group pair

## When to consider

When a low-bit representation spends extra storage on selected reordered groups, retain actual base quantization error in independently scaled residual groups. Choose repair_groups; base/residual scales and coordinate identities must stay paired. Bind16-element groups and the ACT suffix layout; do not substitute the WEIGHT duplicate path.

## Computation before and after

Selected suffix groups get independently scaled residual codes after base quantization.

Reorder the input and round to the declared intermediate type. For group g compute a base scale from max absolute value, round/store that scale, encode base codes using its decoded value and declared approximate reciprocal. Form residuals from the same rounded input minus the actual decoded base code times the stored base scale, then round residuals to the intermediate type. For the last repair_groups groups independently choose/store a residual scale and encode residual codes. Map unselected group g to physical group g. Selected group g maps to base position2*g-(groups-repair_groups) and residual positionbase+1. Reconstruction decodes base and residual with their own scales and adds them; unselected groups decode base only.

Recover the original computation: Erase representation groups by decoding one base and optional independently scaled residual per original reordered group under the declared approximation.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| repair_groups | compile / software | {"lo": 0, "hi": 4096} | Number of trailing groups after a fixed reorder that receive a residual; source KE/16. |

## Coupled constraints and scope

Encoding-stage refinement only. Selected suffix groups after a fixed reorder; independently scaled E2M1 residual. Source ACT path on SM100+. WEIGHT duplication is a distinct consumer-facing representation and is not covered by residual reconstruction. No claim of exact recovery or downstream GEMM speed.

Scope predicate: `p.group_size == 16 and p.reorder_is_valid_permutation and p.approximation_admitted`.

- Precondition `pre_format` (implementation): `p.group_size == 16 and p.reorder_is_valid_permutation and p.quantizer_decoder_policy_explicit` — NVIDIA source uses16 elements/group, BF16-rounded reordered values, rounded E4M3 scales and E2M1 payloads.
- Precondition `pre_decode` (semantic): `p.base_and_residual_scales_positive_finite and p.consumer_understands_pair_layout and p.approximation_admitted` — Consumer distinguishes base and repair groups and pairs both scales with the correct generation.
- Constraint `c_budget` (semantic): `repair_groups <= p.groups` — Each selected group is unique and already has a base group.

## Local example and misuse

For a local instance of `store`, identify the object `base_residual_group_pair` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tensorrt-llm/delta/selected_group_residual_encoding@1/template.yaml`. Original outline: `knowledge/studies/tensorrt-llm/delta/selected_group_residual_encoding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: groups+repair_groups physical code groups and scale entries. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `mapping` (exhaustive): Each original group owns one base and selected groups exactly one adjacent residual, without collisions.
- `residual_generation` (test): Residual uses actual decoded stored base code and scale after intermediate rounding; independent residual scale stays paired.
- `numeric` (test): Scale conversion, reciprocal, saturation and both payload conversions satisfy the admitted approximation and range.

Two BF16 rounding steps, E4M3 scales, approximate reciprocal and E2M1 saturation contribute error; zero/subnormal scales require explicit guards.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
