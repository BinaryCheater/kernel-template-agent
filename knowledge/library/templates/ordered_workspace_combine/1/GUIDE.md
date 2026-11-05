# ordered_workspace_combine@1: exclusive partial workspace followed by ordered linear merge

## When to consider

When parallel reduction producers must finish freely but output addition order must stay fixed, materialize their exclusive FP32 partials and merge by logical split index in another kernel. Compute liveness from exact ranges rather than assuming a live prefix, and retain one final output cast.

## Computation before and after

Unique FP32 producer slots plus a dependency-separated ascending-index merge.

Let T=p.reduction_steps and S=split_count. Producer s owns steps[floor(s*T/S),floor((s+1)*T/S)); s is useful exactly when hi>lo. It computes its partial in the fixed inner order and writes a unique FP32 slot[tile*S+s,element]. After all producers finish under the declared dependency, one reducer owner per output element initializes FP32 acc=0 and scans s=0..S-1 ascending. For hi>lo, load the complete partial and add in FP32; never load inactive slots even when their addresses are allocated. Cast once after the ordered sum, then store only the logical output region. Different elements may reduce in parallel; no producer completion schedule can change one element addition sequence. Preserve workspace until all readers finish. Direct unsplit/single-contributor and empty-group output branches must use disjoint ownership and are outside this version.

Recover the original computation: Concatenate the useful floor-partition ranges to recover every reduction step once. The final value is the explicitly ordered FP32 sum of those fixed partials; changing producer completion order leaves that sequence unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| split_count | runtime / software | {"lo": 2, "hi": 2147483647} | Number of nominal floor-partition slots; useful slots may be scattered rather than a prefix. |
| load_elements | compile / software | [1, 4] | Scalar or aligned FP32x4 workspace loads; source loads four vectors per16-element output fragment. Scalar is an inferred compatible lowering. |

## Coupled constraints and scope

Pinned SM100/103/107 deterministic grouped BF16 weight-gradient multi-contributor tail. Floor partitions can have holes when split_count exceeds the row-step count. Native partials are FP32 and reducer output is BF16 or FP32. Fixed-plan reproducibility does not imply equality across split counts, native MMA shapes or low-precision intermediate reductions. Singleton/empty output policies are separately bound.

Scope predicate: `p.reduction_steps >= 2 and p.output_elements >= 1 and (p.partial_inner_order_and_output_coordinates_fixed and p.tail_products_neutral_and_all_steps_covered_once) and (p.partial_slots_exclusive_fp32_and_capacity_strides_exact and p.all_read_partials_published_before_reducer) and (p.producer_and_reducer_use_same_ranges_and_generation and p.inactive_slots_never_loaded) and (p.final_output_elements_owned_once_and_padding_untouched and p.direct_main_tiles_disjoint_from_split_tail_outputs) and (p.finite_fp32_partials_and_intermediate_sums and p.linear_ascending_fp32_add_order_and_final_cast_admitted) and (p.selected_load_width_alignment_and_resources_supported and p.merge_completion_before_scratch_reuse) and p.partition_and_workspace_integer_arithmetic_no_overflow`.

- Precondition `pre_partition` (semantic): `p.partial_inner_order_and_output_coordinates_fixed and p.tail_products_neutral_and_all_steps_covered_once` — Each partial has a fixed arithmetic order and output identity; masks preserve the complete reduction including its ragged tail.
- Precondition `pre_slots` (implementation): `p.partial_slots_exclusive_fp32_and_capacity_strides_exact and p.all_read_partials_published_before_reducer and p.partition_and_workspace_integer_arithmetic_no_overflow` — No atomic shared accumulation occurs between producers; each useful slot contains complete FP32 state before the separate merge starts. Floor-range products and workspace address arithmetic must not overflow.
- Precondition `pre_presence` (implementation): `p.producer_and_reducer_use_same_ranges_and_generation and p.inactive_slots_never_loaded` — Live-slot predicates come from the same partition metadata; inactive slots can remain uninitialized.
- Precondition `pre_owner` (implementation): `p.final_output_elements_owned_once_and_padding_untouched and p.direct_main_tiles_disjoint_from_split_tail_outputs` — Each final element has one merge owner, with no conflicting direct main-kernel writes and no stores into row padding.
- Precondition `pre_numeric` (implementation): `p.finite_fp32_partials_and_intermediate_sums and p.linear_ascending_fp32_add_order_and_final_cast_admitted` — Linear ascending slot order and one final cast define reproducibility; comparison against other partitions/trees requires a separate rounding contract.
- Precondition `pre_backend` (implementation): `p.selected_load_width_alignment_and_resources_supported and p.merge_completion_before_scratch_reuse` — Vector loads respect alignment; workspace remains immutable through all reducer readers and is exclusive across concurrent launches.
- Constraint `c_shape` (semantic): `p.reduction_steps >= 2 and p.output_elements >= 1` — This multi-contributor projection has at least two live reduction steps and two nominal splits; singleton/empty direct output is separate.

## Local example and misuse

For a local instance of `reduce`, identify the object `exclusive_partial_workspace_followed_by_ordered_linear_merge` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/ordered_workspace_combine@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/ordered_workspace_combine@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One FP32 partial per nominal split and output element.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: partial_workspace
  level: dram
  scope: kernel
  bytes: 4 * split_count * p.output_elements
overlap: []
```

## Obligations and numerical checks

- `range_and_presence` (test): Useful ranges cover[0,T) once, and holes in nominal slots never trigger stale loads.
- `reproducible_sequence` (test): Fixed partial values plus ascending split-index addition produce the same result across producer completion permutations.
- `exclusive_ownership` (test): Producer slots and final element owners are disjoint, including main versus tail outputs and row padding.
- `publication_and_precision` (test): Publication precedes loads, every accumulated state stays FP32, one final cast is preserved and scratch is not recycled early.

A fixed addition order is reproducible but can differ from unsplit or tree reductions. Replacing FP32 workspace with BF16, adding in arrival order or changing the partition changes the numerical operator.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
