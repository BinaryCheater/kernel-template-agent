# fp4_bf16_bit_interleave@1: fp4 bf16 bit interleave

## When to consider

When software FP4 decoding dominates, pre-interleave payload bits to match packed BF16 expansion. Preserve every code bit through an explicit bit permutation and bind the matching decoder. This changes stored representation, not the quantizer or the value/scale grouping.

## Computation before and after

Eight FP4 payloads in four bytes are rearranged into one word optimized for packed BF16 expansion.

For each half h=0,1 take nibble q_j=(byte_j>>(4*h))&15 for j=0..3. Let C(q)=((q&8)<<12)|((q&7)<<6), D(q)=((q&8)<<11)|((q&6)<<9)|((q&1)<<13). Place C(q0), C(q1)>>3, C(q2)>>6, D(q3) in disjoint16-bit positions by OR, then place half h at word bits16*h. Decode the first three nibbles from those disjoint fields and the fourth from its separate positions, retaining signed zero and all payload bits. Hopper consumer expands the prepared fields to BF16 representations; the storage transform itself changes no FP4 codes.

Recover the original computation: Extract each source nibble using the inverse bit incidence map, restoring its original byte and low/high position.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Pinned Hopper MXFP4 four-byte bit layout before its matching BF16 decoder. This is not contiguous bitstream packing or a per-byte payload shift. Complete words and source little-endian representation are required.

Scope predicate: `(p.input_four_bytes_each_two_fp4 and p.complete_word_ownership) and (p.little_endian_byte_view_and_unsigned_shifts and p.decoder_matches_bit_positions)`.

- Precondition `pre0` (semantic): `p.input_four_bytes_each_two_fp4 and p.complete_word_ownership` — Each complete32-bit word owns exactly eight E2M1 codes.
- Precondition `pre1` (semantic): `p.little_endian_byte_view_and_unsigned_shifts and p.decoder_matches_bit_positions` — Encoder and decoder agree on byte order and masked unsigned shifts.

## Local example and misuse

For a local instance of `store`, identify the object `fp4_bf16_bit_interleave` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/fp4_bf16_bit_interleave@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/fp4_bf16_bit_interleave@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_incidence` (test): Every one of32 input bits maps to one distinct output bit, with inverse recovering all256 possible source byte patterns in each position.
- `consumer_decode` (test): Native BF16 expansion interprets those positions as the original E2M1 values.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
