# structured_zero_compression@1: compressed structured zero operand

## When to consider

When an operand already has at most two nonzeros in every aligned quartet, consider value/position compression for sparse MMA. Preserve every nonzero, fill unused retained slots with known zeros and pair the position nibble format with its native consumer. Upstream weight selection and unsupported sparse layouts require separate rules or bindings.

## Computation before and after

Remove only structured zeros and encode retained positions for sparse contraction.

For each row and aligned K quartet, find the positions of its at most two nonzero values in increasing order. With two nonzeros retain those positions. With one nonzero at0 choose positions(0,1); with one atj>0 choose(0,j). With no nonzeros choose(0,1). Store both selected values, including any selected zero, and nibble i0|(i1<<2). Pack consecutive nibbles at bit offsets4*j in the selected metadata word. The sparse MMA interprets missing positions as zero and contracts retained values with the corresponding coordinates of the other operand. This is exact compression of existing structured zeros, not an algorithm for pruning arbitrary nonzero dense weights.

Recover the original computation: Decode each position pair and scatter its two values into a zero-initialized quartet; the original finite logical operand and admitted contraction are recovered.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| metadata_word_bits | compile / software | [16, 32] | Pack four or eight position nibbles into one metadata word; encoder and consumer must select the same format. |

## Coupled constraints and scope

Exact2:4 structured-zero operand encoding, with a paired position nibble and supported16/32-bit metadata words. Source representative uses FP16 operands. Other value types need an explicit supported backend binding. TF32's1:2 form, unaligned groups, arbitrary nonzero pruning, NaN/Inf propagation and observed signed-zero payloads are outside this version. Native encoding/layout assurance is separate from the logical compression identity.

Scope predicate: `p.at_most_two_nonzeros_per_group and p.input_values_finite and p.zero_products_are_neutral and p.position_order_preserved and p.encoder_decoder_match and p.metadata_shape_matches and p.value_type_supported and p.sparse_mma_backend_supported and p.compressed_load_layout_supported and p.padding_and_tail_policy_valid`.

- Precondition `pre_support` (semantic): `p.at_most_two_nonzeros_per_group and p.input_values_finite and p.zero_products_are_neutral` — Only known zero contributions may be omitted under the contraction's numerical policy.
- Precondition `pre_encoding` (implementation): `p.position_order_preserved and p.encoder_decoder_match and p.metadata_shape_matches and p.value_type_supported` — Retained values and position nibbles identify the same logical coordinates and use a supported value/metadata format.
- Precondition `pre_backend` (implementation): `p.sparse_mma_backend_supported and p.compressed_load_layout_supported and p.padding_and_tail_policy_valid` — The selected native sparse operand and copy layouts realize this logical representation.
- Constraint `c_group` (semantic): `p.K % 4 == 0 and p.k_origin % 4 == 0` — Compression groups are complete and aligned to their logical K origin.
- Constraint `c_format` (implementation): `metadata_word_bits == p.metadata_word_bits` — The selected packed metadata word matches the consumer format.

## Local example and misuse

For a local instance of `contract`, identify the object `compressed_structured_zero_operand` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tilelang/delta/structured_zero_compression@1/template.yaml`. Original outline: `knowledge/studies/tilelang/delta/structured_zero_compression@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two values plus a position nibble for each logical quartet.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `support_preserved` (proof): Every original nonzero is retained once and all omitted positions are zero.
- `format_identity` (test): Nibble packing and consumer decoding recover the same ordered coordinate/value pairs.
- `contraction_identity` (proof): Omitted zero products are neutral under the declared sparse contraction policy.
- `native_mapping` (test): Physical metadata/value layouts and target sparse MMA implement the declared logical decoder.

Exact finite logical compression does not authorize lossy upstream sparsification or dense IEEE zero-times-NaN equivalence.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [bitstream_packing@1](../../bitstream_packing/1/GUIDE.md), [bitstream_packing@2](../../bitstream_packing/2/GUIDE.md), [bitstream_packing@3](../../bitstream_packing/3/GUIDE.md); The ordered position nibbles can be packed without changing their logical support.

Structured contract: [template.yaml](template.yaml).
