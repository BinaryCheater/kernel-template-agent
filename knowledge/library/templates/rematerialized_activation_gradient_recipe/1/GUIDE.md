# rematerialized_activation_gradient_recipe@1: activation backward rematerialization and precision recipe

## When to consider

When backward rematerializes an activation, bind its complete derivative recipe rather than a single dtype label. Preserve the original rounded or unrounded checkpoint convention, local partial format and full final reduction/cast chain; wide final merging is distinct from wide local arithmetic.

## Computation before and after

Rematerialize activation derivatives under a selected checkpoint/arithmetic recipe, then merge stored partials at an explicit precision and cast chain.

Retain original forward operands and document metadata; do not retain a large preactivation tensor. On backward rematerialize each four-tap preactivation with the selected program arithmetic order and bias-last stage. For rounded_compatibility, round preactivation to BF16 and widen to FP32 before applying the declared approximate SiLU derivative; for unrounded_wide_merge, evaluate at the unrounded FP32 preactivation using its native FMA recipe. Apply the same selected derivative to all local/halo rows. Accumulate dx and tap/bias partials with declared FP32 order; write dx through its BF16 cast and exclusive parameter partials as FP32. After producer completion, merge all live partials with the selected fixed tree in FP32 for rounded_compatibility or FP64 after widening each stored FP32 partial for unrounded_wide_merge. Convert final parameter totals to FP32, then either store FP32 or convert that FP32 total to BF16 according to the actual output tensor dtype. Keep parameter load and output store flags independent. Compare against the selected admitted gradient convention, not against an assumed exact derivative of the quantized forward.

Recover the original computation: Expand every stored partial into the selected rematerialized derivative and local products under its fixed arithmetic recipe. This recovers the chosen native backward convention with unchanged coordinates/generation and declared casts; no equivalence between the two conventions or recovery of discarded FP32 information is asserted.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| gradient_recipe | compile / software | ["rounded_compatibility", "unrounded_wide_merge"] | Source rounded compatibility evaluates SiLU derivative at BF16-rounded preactivation and merges FP32 partials in FP32. Unrounded recipe uses its FP32 preactivation/FMA program and merges the same FP32 partial format in FP64 before a mandatory FP32 total checkpoint. These are coupled source routes, not freely interchangeable flags. |

## Coupled constraints and scope

FlashInfer deterministic zero-history Cake width4 depthwise convolution training on SM100/103/107. BF16 activations/dy/dx, BF16 or FP32 taps and optional bias; FP32 or BF16 parameter gradient destinations. Source silu_grad_preactivation bf16/fp32 selects rounded compatibility versus unrounded/FMA plus FP64-final-merge native routes. Both retain FP32 local parameter partials and final FP32 total checkpoint. Native exp2/reciprocal approximations, input finite/range policy and error acceptance belong to the binding; this rule does not claim exact autodiff through a cast or universal error improvement.

Scope predicate: `(p.selected_gradient_recipe_explicitly_admitted_by_training_numeric_contract) and (p.original_inputs_parameters_document_masks_and_forward_generation_retained) and (p.whole_preactivation_arithmetic_recipe_and_derivative_checkpoint_bound) and (p.local_input_gradient_and_parameter_partial_stages_explicit) and (p.final_reducer_precision_tree_and_all_output_casts_explicit) and (p.load_types_and_actual_gradient_destination_types_match_flags) and (p.exclusive_partial_rows_publish_before_reducer_and_live_through_reads) and (p.selected_named_recipe_bias_dtype_route_and_target_native_programs_exist)`.

- Precondition `pre_contract` (semantic): `p.selected_gradient_recipe_explicitly_admitted_by_training_numeric_contract` — Caller selects a named backward numerical recipe, including where the activation derivative is evaluated. This is an approximation/convention through the forward cast, not the mathematical derivative of a discrete BF16 quantizer or an algebraic identity between recipes.
- Precondition `pre_replay` (implementation): `p.original_inputs_parameters_document_masks_and_forward_generation_retained` — Backward rematerializes the same stencil inputs, parameters, optional bias and document support from the original forward generation. Halo derivative rows follow the same recipe as interior rows; no hidden state or changed weights enter reconstruction.
- Precondition `pre_preactivation` (implementation): `p.whole_preactivation_arithmetic_recipe_and_derivative_checkpoint_bound` — Specify tap product/add/FMA order, optional bias added after taps, preactivation FP32 operations, optional BF16 round-and-widen, sigmoid approximation/FTZ and derivative grouping. Merely selecting FP32 versus BF16 does not fix the arithmetic recipe.
- Precondition `pre_partials` (implementation): `p.local_input_gradient_and_parameter_partial_stages_explicit` — Local dx accumulates in declared FP32 order then casts to BF16. Tap/bias partials are stored in FP32 under the actual token/grid partition; widening those stored values later cannot recover information already lost in their local computation.
- Precondition `pre_merge` (implementation): `p.final_reducer_precision_tree_and_all_output_casts_explicit` — Reducer specifies FP32 or widened FP64 additions and its lane/halving tree, then converts the final sum to FP32. BF16 gradient destinations apply a further BF16 conversion from that FP32 value; do not substitute direct FP64-to-BF16 rounding.
- Precondition `pre_types` (implementation): `p.load_types_and_actual_gradient_destination_types_match_flags` — BF16/FP32 tap and bias loads widen into the same declared FP32 recipe; gradient stores follow actual dweight/dbias destination dtype, which may differ from parameter dtype. Flag bits and buffer strides must describe those actual tensors.
- Precondition `pre_join` (implementation): `p.exclusive_partial_rows_publish_before_reducer_and_live_through_reads` — Each existing CTA partial has unique ownership, empty streams emit zero and capacity holes are unread. The reducer waits for producer completion, and scratch/output generation remains owned through all consumers.
- Precondition `pre_backend` (implementation): `p.selected_named_recipe_bias_dtype_route_and_target_native_programs_exist` — Concrete backend ships selected route for SM100/103/107, width4 depthwise convolution plus SiLU, contiguous BF16 x/dy/dx and C divisible by8. Optional architecture routes are checked, not inferred from a vocabulary name. Source rounded and unrounded choices select coupled native programs and final merge precision.

## Local example and misuse

For a local instance of `contract`, identify the object `activation_backward_rematerialization_and_precision_recipe` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/rematerialized_activation_gradient_recipe@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/rematerialized_activation_gradient_recipe@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Original operands/metadata and FP32 local parameter partials; no retained preactivation array or hidden full-FP64 local gradient.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `recipe_recovery` (test): Recomputed preactivation, derivative stage and optional bias reproduce the selected backward numerical convention at every stencil coordinate.
- `partial_and_final_precision` (test): FP32 local partials, FP32/FP64 final tree and FP32-then-optional-BF16 store chain preserve their actual boundaries.
- `dtype_binding` (test): Loads and stores follow actual operand and destination formats, including independently provided gradient buffers.
- `publication_and_support` (proof): Every local/halo contribution and live partial is included once after its producer, with document isolation and scratch lifetime.

Changing derivative checkpoint or FMA order changes gradients. FP64 final merge cannot repair local FP32 losses, and FP64-to-FP32-to-BF16 may differ from a direct BF16 rounding.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
