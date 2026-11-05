# warp_interleaved_code_append@1: warp interleaved code append

## When to consider

When warp lanes scan distinct database vectors at the same feature, store codes dimension-major within fixed warp-sized vector blocks and preserve old sub-byte fields on append. 4-bit vectors v0 and v1 share one byte. If old v0=3 and append v1=10, OR0xa0 gives0xa3; writing0xa0 directly destroys v0. Overwriting v1 later with2 by OR is invalid because old high bits remain.

## Computation before and after

When warp lanes scan distinct database vectors at the same feature, store codes dimension-major within fixed warp-sized vector blocks and preserve old sub-byte fields on append.

Let vector v have block floor(v/32) and lane v%32. For each feature d, store its 32 codes consecutively in code_bits*32/8 bytes; feature planes are contiguous within the vector block. On append starting inside an old block, align start down to32, make lanes before start invalid, pack their values as zero and OR only new fields into existing bytes. Consumer loads each dimension plane cooperatively, shuffles packed fields back to vector lanes and decodes each code for its feature before metric accumulation.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| vector_block | compile / software | [32] | NVIDIA warp vector grouping. |
| code_bits | compile / software | [4, 5, 6, 8, 16, 32] | Observed storage formats, independent of quantizer numerical policy. |

## Coupled constraints and scope

NVIDIA CUDA IVF interleaved Flat/SQ/PQ layout. 4/5/6-bit packers use OR, so this is append into clean new fields, not in-place replacement. Numerical scalar/PQ decoding is separate. Resource effects: Stores vector_block*D*code_bits/8 bytes per complete block; cooperative lane rearrangement trades shuffle work for contiguous same-feature loads. Append may revisit a partial block without rewriting its existing fields.

Scope predicate: `p.codes_in_range and p.append_slots_disjoint and p.new_fields_zero and p.old_fields_immutable and p.producer_consumer_mapping_equal and p.full_warp_participation`.

- Precondition `pre_0` (semantic): `p.codes_in_range and p.append_slots_disjoint` — Codes fit width and append reserves disjoint list positions.
- Precondition `pre_1` (semantic): `p.new_fields_zero and p.old_fields_immutable` — New bit fields begin zero; existing packed fields are preserved, with no concurrent writer to a shared byte.
- Precondition `pre_2` (semantic): `p.producer_consumer_mapping_equal and p.full_warp_participation` — Append and scan share dimension/block/bit mapping, all shuffle lanes participate.

## Local example and misuse

For a local instance of `map`, identify the object `warp_interleaved_code_append` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/faiss/delta/warp_interleaved_code_append@1/template.yaml`. Original outline: `knowledge/studies/faiss/delta/warp_interleaved_code_append@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Stores vector_block*D*code_bits/8 bytes per complete block; cooperative lane rearrangement trades shuffle work for contiguous same-feature loads. Append may revisit a partial block without rewriting its existing fields.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_identity` (proof): Logical (vector,feature,bit) maps injectively to the same physical bit in append and scan.
- `preserve_prefix` (proof): Unaligned append changes no field of vectors before append start.
- `no_shared_byte_race` (proof): One owner writes each byte for a list/block/feature and new bits are zero before OR.
- `tail` (proof): Invalid last-block lanes contribute zero fields but never enter distance selection.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
