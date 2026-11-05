# cyclic_window_code_decode@1: cyclic window code decode

## When to consider

When adjacent quantized values share code bits and a consumer fragment must decode without a lookup table, assign one cyclic bitstream per consumer sequence and decode overlapping state windows.

## Computation before and after

When adjacent quantized values share code bits and a consumer fragment must decode without a lookup table, assign one cyclic bitstream per consumer sequence and decode overlapping state windows.

Partition logical coordinates into disjoint sequences with a declared bijection F(sequence,j). Stream length B bits is a multiple of32. State j is the w-bit cyclic window starting at j*step mod B, interpreted little endian. Apply the declared deterministic decoder D(state) yielding v consecutive values and place them at F(sequence,v*j+e). For source QTIP B128,w16,step4,v1; D hashes s by (89226354*s+64248484) mod2^32, forms two BF16 halves using (h & 0x81FF81FF) xor0x3E003E00, and adds the halves with BF16 rounding. QTIP2 B256 or128,w16,step8,v2 uses duplicated byte windows and a different hash, so it is a separate decoder binding. Tail-biting makes the final window consume the beginning of its own stream, never another lane. Packing/quantization must solve the overlapping-state constraints; concatenating independent codes is invalid.

Recover the original computation: Correlated cyclic codes with fixed encoder/decoder and coordinate map; this preserves a quantized representation and does not assert equivalence to unquantized weights.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| stream_bits | compile / software | {"lo": 32, "hi": 256} | B fixed by consumer fragment grouping. |
| window_bits | compile / software | {"lo": 1, "hi": 32} | w: decoder state width. |
| step_bits | compile / software | {"lo": 1, "hi": 16} | step: newly introduced bits per state. |
| values_per_state | compile / software | {"lo": 1, "hi": 2} | v: decoder output arity. |

## Coupled constraints and scope

Correlated cyclic codes with fixed encoder/decoder and coordinate map; this preserves a quantized representation and does not assert equivalence to unquantized weights.

Scope predicate: `(stream_bits % 32 == 0 and stream_bits % step_bits == 0 and window_bits <= stream_bits) and (p.coordinate_map_bijective and p.decoder_matches_encoder and p.decoder_rounding_admitted and p.stream_allocated)`.

- Precondition `pre0` (semantic): `p.coordinate_map_bijective and p.decoder_matches_encoder and p.decoder_rounding_admitted and p.stream_allocated` — Decoder, layout and stream representation are shared by producer and consumer.
- Constraint `c0` (implementation): `stream_bits % 32 == 0 and stream_bits % step_bits == 0 and window_bits <= stream_bits` — Word-addressable complete cyclic stream.

## Local example and misuse

For a local instance of `decode`, identify the object `cyclic_window_code_decode` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/cyclic_window_code_decode@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/cyclic_window_code_decode@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `window_wrap` (proof): Each state consumes exactly w bits modulo B within its own sequence.
- `coordinate_agreement` (proof): Each decoded value reaches the same logical coordinate used by the encoder.
- `decoder_identity` (proof): Integer hash and BF16 rounding match the declared D, not another codebook.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
