# contiguous_group_tile_walk@1: contiguous flat tiles with per group segment slots

## When to consider

When a grouped reduction has many uniform tile slots but an awkward number of row-level CTAs, split the flattened group-by-tile line into contiguous CTA intervals. Workers walk adjacent tiles and may cross groups; each group intersection emits one numbered partial. Derive per-group segment counts and capacity from the interval boundaries. Preserve original group coordinates, reset math at group changes and keep persistent input-ring generations consistent.

## Computation before and after

Balanced contiguous flattened intervals with exact per-group partial numbering.

Let P be group count, T allocated tiles per group and S=P*T. CTA c owns [floor(c*S/workers),floor((c+1)*S/workers)). Within this interval decode group=floor(slot/T) and tile=slot%T. For each intersected group g compute lo=max(0,start-g*T) and hi=min(T,end-g*T), initialize that group accumulator and process [lo,hi) under its original masks. Slot owner(s)=floor(((s+1)*workers-1)/S). Assign the completed group partial to segment=c-owner(g*T), with segment count owner(g*T+T-1)-owner(g*T)+1. Derive maximum_derived_group_segments by taking the maximum of these counts across g; do not treat it as an independently tunable split number. Emit exactly one complete state per nonempty intersection and combine all actual group segments once. Initialize or predicate unused capacity slots to identity. If every group has one segment and the backend supports it, the worker may finalize directly. On group changes reset logical row/accumulator state but preserve any live input pipeline ring generations and retirement rules across the walk.

Recover the original computation: Unroll every CTA interval into its original group/tile coordinates; group intersections partition each original reduction exactly once, and the declared state combiner reconstructs that group result.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| workers | runtime / software | {"lo": 1, "hi": 65535} | CTA interval count; source chooses min(SM count,total allocated tiles) before deriving partial capacity. |

## Coupled constraints and scope

Rectangular independent groups with fixed allocated tiles per group and a valid partial-state combiner. Source persistent QSA uses (row,KV head) groups,64-selection tiles and min(SM count,total slots) workers, with normalized partials and LSE. Varying valid lengths are predicates inside the allocated rectangle. Ragged tile-prefix domains, global atomic stealing and a fixed equal split count per group are outside this version. Native resident-resource and cross-group pipeline behavior remain bindings.

Scope predicate: `p.rectangular_group_tile_enumeration_complete_and_unique and p.masked_tail_tiles_contribute_defined_identity and p.segment_state_and_combiner_preserve_original_group_reduction and p.segment_states_include_matching_normalization_metadata and p.segment_slot_formula_matches_worker_intersections and p.unused_partial_slots_have_defined_identity_and_no_stale_reads and p.group_accumulators_reset_at_each_intersection_and_inputs_keep_original_group_coordinates and p.pipeline_generations_continue_consistently_across_group_changes and p.integer_products_and_inverse_owner_formula_fit_types and p.selected_persistent_body_resources_and_progress_legal and p.all_partial_writes_complete_before_group_combine and p.partial_storage_lives_until_all_readers_retire`.

- Precondition `pre_domain` (semantic): `p.rectangular_group_tile_enumeration_complete_and_unique and p.masked_tail_tiles_contribute_defined_identity` — Each logical group has the same allocated tile extent; validity tails contribute a defined reduction identity.
- Precondition `pre_combine` (semantic): `p.segment_state_and_combiner_preserve_original_group_reduction and p.segment_states_include_matching_normalization_metadata` — Partial states retain the data/frame needed to recover the original group reduction under the admitted arithmetic policy.
- Precondition `pre_slots` (implementation): `p.segment_slot_formula_matches_worker_intersections and p.unused_partial_slots_have_defined_identity_and_no_stale_reads` — Every nonempty group/CTA intersection has a unique per-group slot; merge padding cannot read stale state.
- Precondition `pre_state` (implementation): `p.group_accumulators_reset_at_each_intersection_and_inputs_keep_original_group_coordinates and p.pipeline_generations_continue_consistently_across_group_changes` — Math state resets when changing groups while input rings retain the correct cumulative generations and logical group identity.
- Precondition `pre_resources` (implementation): `p.integer_products_and_inverse_owner_formula_fit_types and p.selected_persistent_body_resources_and_progress_legal` — Flattened indices and inverse products fit their integer types; actual CTA resources/launch are legal.
- Precondition `pre_completion` (implementation): `p.all_partial_writes_complete_before_group_combine and p.partial_storage_lives_until_all_readers_retire` — The combiner sees completed partials and scratch is not reset while readers remain.
- Constraint `c_domain` (semantic): `p.groups >= 1 and p.tiles_per_group >= 1 and workers >= 1 and workers <= p.groups * p.tiles_per_group` — Every worker interval is nonempty in the rectangular tile domain.
- Constraint `c_capacity` (resource): `p.partial_slots_per_group >= p.maximum_derived_group_segments and p.maximum_derived_group_segments >= 1` — Storage capacity covers the maximum actual number of CTA intersections with any group.

## Local example and misuse

For a local instance of `work_assign`, identify the object `contiguous_flat_tiles_with_per_group_segment_slots` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/contiguous_group_tile_walk@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/contiguous_group_tile_walk@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Each worker can process adjacent tiles and cross group boundaries while amortizing resident CTA/pipeline setup.. Storage: Derived maximum segment capacity per group; no global next-task counter or explicit full task list is required by this arithmetic.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: group_partials
  level: dram
  scope: kernel
  bytes: p.groups * p.partial_slots_per_group * p.partial_state_bytes
overlap: []
```

## Obligations and numerical checks

- `flat_cover` (proof): Half-open worker intervals are disjoint and cover every flattened tile exactly once.
- `inverse_and_slots` (proof): The inverse owner formula and group segment index uniquely cover all group intersections with no partial-slot collision.
- `state_and_pipeline` (test): Group math reset, tail identity and cumulative input pipeline generations preserve each original group without stale state.
- `combine_and_capacity` (test): Derived capacity fits every group, unused slots are safe, and all complete partial states precede their consumers.

Changing worker count changes group split boundaries and therefore the partial combine association and rounding. Normalized attention partials require matching LSE/frame metadata, not plain addition.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
