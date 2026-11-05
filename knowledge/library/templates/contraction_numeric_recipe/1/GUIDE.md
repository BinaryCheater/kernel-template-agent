# contraction_numeric_recipe@1: contraction numeric recipe

## When to consider

When changing persistence or operand layouts could hide a numeric conversion, expose the operand preparation and accumulator policy first. Widen mixed E4M3 operands before layout choice, or select the target-correct TF32 rounding explicitly. Bind the intrinsic accumulation limit independently from tile scheduling and verify the requested numeric contract.

## Computation before and after

A contraction explicitly selects operand preparation and intrinsic accumulation policy before layout selection.

For each affected input tile either preserve its representation; widen E4M3 to the other operand FP16/BF16 type before dot lowering; or explicitly round FP32 values to TF32 with nearest-even on Hopper+ and nearest-away on the source Ampere path. Preserve unmodified operands. Initialize accumulator as FP64 only for the both-FP64 route, otherwise FP32. Pass the declared allow_tf32 and max_num_imprecise_acc policy to dot; do not infer the emitted instruction or its internal reduction schedule from those flags alone. Apply identical policy in regular and persistent routes so a scheduling choice does not silently change the requested numeric preparation.

Recover the original computation: Erase layout and execution assignment while retaining the selected prepared-input and accumulator recipe; this recovers the requested contraction, not an unrounded real-valued sum.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| operand_preparation | compile / software | ["identity", "widen_e4m3_to_peer16", "fp32_to_tf32_rn", "fp32_to_tf32_rna"] | Numeric preparation of each affected operand, with source/target legality guards. |
| accumulator | compile / software | ["f32", "f64"] | FP64 only when both source operands and selected backend support it. |
| imprecise_accumulation | compile / software | ["backend_default", "bounded"] | Pass through the requested intrinsic policy; this is not a promise about a native instruction count. |
| imprecise_limit | compile / software | {"lo": 1, "hi": 65536} | Requested maximum imprecise accumulation interval when bounded. |

## Coupled constraints and scope

Explicit numeric recipe for mixed E4M3/16-bit and FP32-to-TF32 NVIDIA contraction paths. Source mode is architecture-gated, not universally interchangeable; native instruction details and error bounds require separate lowering/assurance.

Scope predicate: `(p.requested_numeric_recipe and p.error_and_exception_policy_admitted) and (p.operand_formats_and_backend_supported and p.rounding_mode_matches_target) and (p.preparation_before_layout_selection_and_exactly_once and p.product_coordinates_unchanged)`.

- Precondition `pre0` (semantic): `p.requested_numeric_recipe and p.error_and_exception_policy_admitted` — The coarse operation requests the selected preparation/accumulation recipe; no equivalence to a higher-precision contraction is inferred.
- Precondition `pre1` (semantic): `p.operand_formats_and_backend_supported and p.rounding_mode_matches_target` — Widening and TF32 rounding match the selected operand types and architecture.
- Precondition `pre2` (semantic): `p.preparation_before_layout_selection_and_exactly_once and p.product_coordinates_unchanged` — Preparation is exposed before backend chooses fragments, preserving logical contraction coordinates.

## Local example and misuse

For a local instance of `contract`, identify the object `contraction_numeric_recipe` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/contraction_numeric_recipe@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/contraction_numeric_recipe@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `preparation_identity` (test): Every input coordinate receives the requested cast/rounding once before fragment lowering.
- `native_policy` (test): Selected intrinsic/backend implements the requested numeric and accumulation policy over the supported range and exceptions.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
