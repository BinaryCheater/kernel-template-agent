# counter_indexed_mask_replay@1: coordinate indexed random replay checkpoint

## When to consider

When backward needs a forward random mask but storing the full mask is costly, save the invocation descriptor and regenerate each bit from its original logical coordinates. Preserve the generator rounds, counter/component map and threshold comparison exactly. A different tile traversal can replay the same mask; a sequential cursor or fresh generator reservation cannot. Save state before early exits so an empty leading work tile cannot suppress the checkpoint.

## Computation before and after

Regenerate random mask values from a stable logical-coordinate counter and a saved invocation descriptor.

Replace a full stored random mask with an immutable descriptor containing the invocation key/base offset, generator identity, integer/byte convention, threshold/comparison and logical coordinate map. For element x, decode address(x)=(counter,subsequence,component), evaluate the fixed counter generator, select its fixed random component and apply the fixed comparison to obtain mask(x). Later consumers repeat that function from the saved descriptor. Thread/CTA traversal order may change only when their coordinate map still gives each original element the same address/component. Skipped work must not advance a shared sequential RNG cursor. Reserve a new invocation subspace once rather than reacquiring random state for the replay; publish the saved descriptor before any branch that could skip its designated writer.

Recover the original computation: Materialize the regenerated mask at every original logical coordinate; descriptor equality and equal coordinate/component maps recover the original mask exactly.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Stateless counter-addressed masks with a retained invocation descriptor and equal element-address maps. Source FA2 retains seed/base offset and derives batch/head/lane and 16x32 tile counters using its seven-round Philox-like implementation and byte threshold. Changing the generator round count, comparison, lane/component map or random law needs another binding. Native stochastic quality, arbitrary RNG compatibility and dropout derivatives are not implied by exact replay.

Scope predicate: `p.counter_generator_identity_rounds_and_integer_arithmetic_fixed and p.random_word_byte_order_and_comparison_fixed and p.every_logical_element_has_stable_counter_and_component and p.forward_replay_coordinate_maps_equal and p.counter_coordinates_in_range and p.invocation_random_subspace_reserved_once and p.invocation_descriptor_published_before_any_early_exit and p.descriptor_immutable_and_alive_until_replay and p.consumer_uses_same_mask_scaling_and_normalization_domain and p.discrete_mask_law_and_numeric_policy_admitted`.

- Precondition `pre_generator` (semantic): `p.counter_generator_identity_rounds_and_integer_arithmetic_fixed and p.random_word_byte_order_and_comparison_fixed` — Generator family, round count, integer width/wrap rules, byte extraction, threshold quantization and comparison are identical in production and replay.
- Precondition `pre_coordinates` (semantic): `p.every_logical_element_has_stable_counter_and_component and p.forward_replay_coordinate_maps_equal and p.counter_coordinates_in_range` — Each logical element has the same counter/subsequence/component identity across passes and tilings, with no unintended coordinate truncation or overlap.
- Precondition `pre_invocation` (implementation): `p.invocation_random_subspace_reserved_once and p.invocation_descriptor_published_before_any_early_exit and p.descriptor_immutable_and_alive_until_replay` — Reserve a distinct invocation subspace once, publish its descriptor on every supported control-flow path, and retain it for every replay.
- Precondition `pre_semantics` (semantic): `p.consumer_uses_same_mask_scaling_and_normalization_domain and p.discrete_mask_law_and_numeric_policy_admitted` — Replaying a mask preserves its exact discrete law and the consuming computation's scaling/support; replay does not itself establish unbiasedness or a softmax derivative.

## Local example and misuse

For a local instance of `store`, identify the object `coordinate_indexed_random_replay_checkpoint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/counter_indexed_mask_replay@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/counter_indexed_mask_replay@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: The same random address serves forward and every later replay, independent of work traversal order.. Storage: One invocation descriptor instead of one mask value per logical element; random arithmetic is repeated.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: replay_descriptor
  level: dram
  scope: kernel
  bytes: p.replay_descriptor_bytes
overlap: []
```

## Obligations and numerical checks

- `bitwise_replay` (test): Forward and replay select the same random word/component and comparison result for each logical element, including tile/lane remappings and tails.
- `invocation_separation` (test): Invocations reserve nonoverlapping intended random addresses without counter overflow or double reservation, including graph-capture behavior when supported.
- `checkpoint_control_flow` (test): A valid immutable descriptor is published even when ordinary work exits early, and every replay waits for it and retains its lifetime.
- `mask_semantics` (test): Threshold quantization, keep scaling, normalization and any derivative replay preserve the declared discrete stochastic operation.

Identical mask replay does not imply the requested real-valued keep probability is exactly represented by a discrete byte threshold or that scaling is unbiased for that law.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
