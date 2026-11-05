# segmented_bit_packing@1: independently byte padded boolean segments

## When to consider

When variable-length boolean masks need independent byte ranges, round each segment length to bytes and prefix-sum those rounded extents. Select the consumer bit order, reset bit position at every segment and zero-fill the last byte. This preserves empty and short segments without letting adjacent masks share a byte; retain original lengths for decoding.

## Computation before and after

Independently byte-align variable boolean segments with reversible within-byte bit order.

Let input starts I and lengths L_s=I[s+1]-I[s]. Construct B[0]=0 and B[s+1]=B[s]+ceil(L_s/8). Segment s writes bytes B[s]..B[s+1]-1. For byte j initialize zero, then for t=0..7 read x[I[s]+8*j+t] only when8*j+t<L_s, otherwise use false. Set bit7-t for bit_order=big or bit t for little. No byte contains values from another segment, including adjacent short/empty segments. Emit packed bytes and B together. A decoder reads the declared bit at B[s]+i//8 for each original local i<L_s; it never interprets byte padding as additional values. A flat pack is the one-segment instance. Zero-segment batches need an explicit no-launch extension outside this version.

Recover the original computation: For every segment/local coordinate read its selected bit from the matching byte prefix; discard zero tail positions and recover the original boolean array exactly.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| bit_order | compile / boundary | ["big", "little"] | Consumer representation contract: first value occupies bit7 or bit0 of its byte; not a free numerical or traversal choice. |

## Coupled constraints and scope

Source quantization/packbits.py prepares per-segment byte prefixes then converts both prefixes to Int32; quantization.cuh uses one CTA per segment,256 threads and8 values per thread, zero-filling incomplete loads. Individual empty segments are admitted. This version does not extend to arbitrary multi-bit codes, 32-bit containers, nonboolean conversion semantics, cross-segment bit concatenation or zero-grid launches.

Scope predicate: `p.segments >= 1 and p.total_input_values >= 0 and p.total_input_values <= 2147483640 and (p.boolean_inputs and p.input_prefix_monotone_and_complete and p.input_and_output_prefixes_immutable) and (p.output_prefix_is_sum_of_per_segment_ceil_length_over_8 and p.prefix_and_length_plus_7_arithmetic_fit) and (p.each_output_byte_has_one_writer and p.allocated_output_covers_packed_prefix and p.segment_grid_supported_and_loop_progress_guaranteed) and (p.partial_final_byte_uses_zero_for_missing_boolean_values and p.consumer_uses_matching_order_lengths_and_prefix) and (p.published_prefixes_and_inputs_precede_pack and p.output_payload_and_prefix_publish_before_readers and p.input_output_alias_safe)`.

- Precondition `pre_segments` (implementation): `p.boolean_inputs and p.input_prefix_monotone_and_complete and p.input_and_output_prefixes_immutable` — Input segments are disjoint slices of one boolean array; their exact nonnegative lengths and prefixes are fixed throughout packing.
- Precondition `pre_prefix` (implementation): `p.output_prefix_is_sum_of_per_segment_ceil_length_over_8 and p.prefix_and_length_plus_7_arithmetic_fit` — Output starts sum independently rounded byte lengths, not global input bit offsets; input/packed prefixes, lengths+7 and loop counters cannot overflow.
- Precondition `pre_ownership` (implementation): `p.each_output_byte_has_one_writer and p.allocated_output_covers_packed_prefix and p.segment_grid_supported_and_loop_progress_guaranteed` — Every required byte has one exclusive producer and fits its output allocation; empty segments issue no payload store. The selected segment grid fits native launch limits and every segment loop terminates.
- Precondition `pre_tail` (implementation): `p.partial_final_byte_uses_zero_for_missing_boolean_values and p.consumer_uses_matching_order_lengths_and_prefix` — Inactive final-byte positions are zero; every reader uses the same bit order and original semantic lengths instead of treating padding as input.
- Precondition `pre_lifetime` (implementation): `p.published_prefixes_and_inputs_precede_pack and p.output_payload_and_prefix_publish_before_readers and p.input_output_alias_safe` — Input and metadata generations remain valid through packing, buffers do not overwrite live input, and consumers see matching output bytes/prefixes.
- Constraint `c_segments` (implementation): `p.segments >= 1 and p.total_input_values >= 0 and p.total_input_values <= 2147483640` — At least one possibly empty segment; pinned Int32 preparation/dispatch and length+7 arithmetic fit.

## Local example and misuse

For a local instance of `store`, identify the object `independently_byte_padded_boolean_segments` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/segmented_bit_packing@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/segmented_bit_packing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One byte per up-to8 local boolean values plus a separate byte-prefix table.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `coordinate_identity` (exhaustive): Each semantic boolean maps injectively to its own segment byte and selected bit; unpack returns its original value.
- `prefix_and_tails` (test): Byte prefixes sum per-segment rounded extents, including empty segments, and every nonsemantic final-byte bit is zero.
- `ownership_and_publication` (test): Exclusive byte writers and immutable prefixes avoid races and publish the same payload generation to consumers.

None for boolean values on the admitted domain. Treating padding as logical values, mixing global and local bit offsets or using the other bit order changes masks.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
