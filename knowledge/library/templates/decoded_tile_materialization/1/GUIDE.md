# decoded_tile_materialization@1: reusable decoded operand tile

## When to consider

When many compute fragments repeatedly decode the same packed operand tile, consider decoding it once into shared memory and reusing ordinary 16-bit loads. Keep the decoder and metadata map unchanged, publish the full tile before use, and wait for every reader before reusing its storage. This trades shared capacity and barriers for repeated conversion/shuffle work. A trait flag alone does not establish that the consumer implements the shared path.

## Computation before and after

Materialize a cooperatively decoded operand tile once or decode repeatedly at fragment loads.

Keep the global packed representation and its metadata unchanged. With fragment_registers, each consumer decodes the requested codes and scales into its compute fragment immediately before use. With shared_tile, assign every logical tile element to one cooperative decoder owner, apply the same decoder arithmetic and rounding, and write the result into the shared layout accepted by ordinary 16-bit compute loads. Publish all decoded writes, then reuse that generation for every consumer fragment. A staging buffer may serve K then V only after every K consumer has retired. Preserve separate row/group scales outside the chosen decoder boundary; materialization is not permission to move them through a contraction or softmax. Reject a consumer whose launcher allocates the buffer but has no decode/store/read path, or whose actual shared/register budget cannot support the selection.

Recover the original computation: Replace each decoded shared read by the corresponding fragment-local decoder at its original logical coordinate, recovering the same admitted compute operand and consumer values.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| decode_site | compile / software | ["fragment_registers", "shared_tile"] | Decode when each consumer loads its fragment or materialize one decoded shared tile and reuse its elements. |

## Coupled constraints and scope

Fixed finite-domain local decoders producing FP16/BF16 operands. Source FA2 prefill has FP8 cast and NVFP4 E2M1/group16-scale bindings. The source repack policy requires supported shapes and launcher capability; it is not enabled for every consumer reusing the traits. Other codebooks, special-value laws, intermediate precisions and scale placement require explicit bindings. Costs of repeated decoding versus shared loads/barriers remain unpriced.

Scope predicate: `p.code_and_metadata_coordinate_map_fixed and p.decoder_value_scale_and_rounding_steps_fixed and p.remaining_scale_sites_explicit and p.code_metadata_generation_matches_and_lives_through_decode and p.valid_value_and_scale_domain_admitted and p.tail_decoding_values_defined and p.fragment_and_tile_decoders_produce_same_logical_values and p.compute_load_layout_matches_decoded_tile and (decode_site != 'shared_tile' or (p.all_tile_writes_publish_before_reads and p.all_decoded_readers_retire_before_reuse and p.aliased_decoder_regions_have_disjoint_lifetimes)) and p.selected_consumer_implements_decode_site_and_budget_matches and p.decode_reuse_numeric_policy_admitted`.

- Precondition `pre_decoder` (semantic): `p.code_and_metadata_coordinate_map_fixed and p.decoder_value_scale_and_rounding_steps_fixed and p.remaining_scale_sites_explicit` — The selected local decoder has fixed code order, metadata ownership and arithmetic/casts; scales outside that boundary retain their original sites.
- Precondition `pre_inputs` (semantic): `p.code_metadata_generation_matches_and_lives_through_decode and p.valid_value_and_scale_domain_admitted and p.tail_decoding_values_defined` — Codes and scales belong to the same generation, are valid for the decoder and have defined inactive/tail behavior.
- Precondition `pre_layout` (implementation): `p.fragment_and_tile_decoders_produce_same_logical_values and p.compute_load_layout_matches_decoded_tile` — Both paths produce the same admitted logical operand, including scale-before/after-cast order, and native fragment loads read matching coordinates.
- Precondition `pre_lifetime` (implementation): `decode_site != 'shared_tile' or (p.all_tile_writes_publish_before_reads and p.all_decoded_readers_retire_before_reuse and p.aliased_decoder_regions_have_disjoint_lifetimes)` — The shared decoded generation is complete before use, survives all readers and is not overwritten by another K/V generation or role.
- Precondition `pre_binding` (implementation): `p.selected_consumer_implements_decode_site_and_budget_matches and p.decode_reuse_numeric_policy_admitted` — The actual consumer implements the chosen path and its resource budget matches allocation; source path availability and numerical policy are explicit.
- Constraint `c_shape` (implementation): `p.rows >= 1 and p.features >= 1 and p.decoded_element_bytes == 2` — This binding materializes a nonempty tile in FP16 or BF16.
- Constraint `c_capacity` (resource): `decode_site != 'shared_tile' or p.rows * p.features * p.decoded_element_bytes <= p.decoded_scratch_capacity_bytes` — The decoded tile fits its dedicated or time-shared scratch region.

## Local example and misuse

For a local instance of `transfer`, identify the object `reusable_decoded_operand_tile` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/decoded_tile_materialization@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/decoded_tile_materialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Shared decoded values amortize conversion over several query/fragment consumers.. Storage: Additional two bytes per logical tile element in shared memory, potentially aliased across retired K/V phases.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: decoded_tile
  level: smem
  scope: cta
  bytes: p.rows * p.features * p.decoded_element_bytes
  when: decode_site == 'shared_tile'
- id: decoded_fragment
  level: reg
  scope: thread
  bytes: p.fragment_elements * p.decoded_element_bytes
  when: decode_site == 'fragment_registers'
overlap: []
```

## Obligations and numerical checks

- `decoder_and_metadata` (test): Each logical code uses its own metadata exactly once in the fixed decoder order; both paths yield the same admitted operand including tails.
- `decoded_layout` (test): Cooperative writes and native compute reads compose to the original logical tile coordinates without omission, overlap or scale-group swaps.
- `lifetime_and_publication` (test): Decoded writes publish before reads and every consumer finishes before scratch reuse, including K/V alias transitions.
- `legal_resource_binding` (test): The selected device path exists and its launch/layout/occupancy budget includes the actual decoded buffer and control state.

A decoded buffer can insert rounding or change scale order; preserving the same abstract dequantized real value is insufficient if the source fragment decoder uses a different finite-precision sequence.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
