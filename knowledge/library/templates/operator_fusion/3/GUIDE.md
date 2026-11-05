# operator_fusion@3: fused intermediate

## When to consider

When an interleaved feature pair immediately feeds a gated activation, fuse pair extraction and the exact activation at the producer boundary. Preserve gate/linear order, input scale, alpha/limits, exp_ftz and the linear+1 shift; this is not interchangeable with textbook SiLU(g)*l. Existing version2 bindings remain unchanged.

## Computation before and after

Fuse a local producer-consumer edge, optionally saving an external checkpoint.

Apply the immediate consumer to the producer's on-chip intermediate tile. For dense_layer and activated_dense_layer, assign a sample tile and its full intermediate feature axis to the same executor; contract that complete feature axis with the next layer weights, then apply exactly the specified activation and conversion. Repeat pairwise through the chain while each dependency remains local. If checkpoint=materialize, additionally write the named logical stage for later consumers without making the immediate consumer reload it. Other fused_ops retain their version1 transformations. Never erase an externally required value or replace an activation input checkpoint with its output unless the backward contract permits that substitution. Version3 interleaved_swiglu: pair source features (2j,2j+1) as gate g and linear l; decode input scale before activation. If limit is present, clamp g only above at limit and l to [-limit,limit]. Compute s=g/(1+exp_ftz(-alpha*g)), then fma(s,l,s), preserving the source linear+1 shift. Consume this pair locally in registers instead of materializing separate gate and linear tensors. Output/observed quantization scale is a separate refinement.

Recover the original computation: Insert a logical materialization at every fused edge and run producer then consumer with identical stage precision and activation order. Keep each optional retained checkpoint in its declared stage/layout. Erasing on-chip scheduling and duplicate checkpoint copies recovers the original dataflow.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| consumer_kind | compile / software | ["elementwise", "row_reduce_then_map", "contract"] | what the fused consumer does with the intermediate |
| intermediate_site | compile / software | ["registers", "smem"] | where the intermediate tile lives |
| fused_ops | compile / software | ["linear_combination", "scale", "scale_and_cast", "routed_weight", "softmax_then_matmul", "norm_scale", "dense_layer", "activated_dense_layer", "interleaved_swiglu"] | the consumer computation absorbed into the producer |
| checkpoint | compile / software | ["none", "materialize"] | Whether a later external consumer also retains the fused intermediate; this does not force the immediate consumer to reload it. |

## Coupled constraints and scope

Version1 fusion scope plus complete-feature dense chains, optionally retaining intermediate checkpoints. A checkpoint increases traffic and storage; fusion does not erase training state. Dense fusion is invalid when the next consumer needs features owned by other independent executors without an explicit exchange. Performance benefit is workload/target dependent. Version3 adds only interleaved pair gating with the exact clipped/shifted source recipe; all prior fused_ops retain their old domain.

Scope predicate: `(True) and (fused_ops != 'interleaved_swiglu' or (consumer_kind == 'elementwise' and p.interleaved_pair_coordinates_exact and p.clipped_shifted_gate_recipe_preserved))`.

- Precondition `pre_tile_local` (semantic): `consumer_kind == 'elementwise' or p.consumer_needs_full_row == False or p.consumer_reduce_rescalable or p.consumer_full_dependency_local` — The consumer dependency is local, or a valid rescalable combination is supplied; full-feature dense layers require the local-dependency clause.
- Precondition `pre_dense` (semantic): `fused_ops not in ['dense_layer', 'activated_dense_layer'] or (consumer_kind == 'contract' and p.complete_feature_axis_owned and p.layer_numeric_steps_preserved)` — Dense consumers have every required feature of each owned sample locally and preserve admitted MMA accumulation, activation and cast steps.
- Precondition `pre_checkpoint` (semantic): `checkpoint == 'none' or (p.checkpoint_matches_intermediate and p.checkpoint_visible_before_external_read and p.checkpoint_storage_survives)` — A checkpoint saves the exact logical stage required by later consumers and survives until their last read.
- Precondition `pre_external` (semantic): `p.no_omitted_side_effects_or_consumers and p.onchip_lifetime_safe` — All external consumers and effects are accounted for; destructive reuse follows the immediate consumer last read.
- Precondition `pre_pair` (semantic): `fused_ops != 'interleaved_swiglu' or (consumer_kind == 'elementwise' and p.interleaved_pair_coordinates_exact and p.clipped_shifted_gate_recipe_preserved)` — New pair-fusion binding retains the requested alpha, one-sided/two-sided limits, linear +1 shift and exp/cast recipe.

## Local example and misuse

For a local instance of `map`, identify the object `fused_intermediate` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/operator_fusion@3/template.yaml`. Original outline: `knowledge/studies/triton/delta/operator_fusion@3/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: intermediate reused on chip instead of via global memory. Storage: registers or smem. Structured effects describe work and resource changes, not timings.

```yaml
transactions:
- id: traffic
  pattern: transfer
  kind: dram_stream
  count: '1'
  amounts:
    bytes: p.producer_input_bytes + p.consumer_extra_input_bytes + p.output_bytes
      + p.checkpoint_bytes
storage: []
overlap: []
```

## Obligations and numerical checks

- `same_values` (test): The consumer reads exactly the producer's values for its tile (no partial or stale values).
- `no_cross_tile_dependency` (test): No consumer output depends on intermediate values of another tile, unless an online/rescalable combination step accounts for it.
- `retained_consumers` (proof): Every later consumer sees its required stage, layout and generation; its checkpoint is complete and alive.
- `dense_dependency_closure` (proof): Every dense consumer dependency belongs to the resident feature tile, and per-layer rounding/activation semantics satisfy the declared contract.

Changed accumulation precision, activation placement or intermediate conversion can change network values and gradients; fusion alone does not authorize these changes.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
