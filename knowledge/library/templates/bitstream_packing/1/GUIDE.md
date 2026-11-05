# bitstream_packing@1: contiguous code bitstream

## When to consider

Consider packing when many fixed-width integer codes leave unused bits in byte-sized storage and the consumer can extract bit fields. Assign each code a disjoint interval and use the same bit offsets on readback. Humming binds the width to the input encoding: 32 three-bit codes occupy three 32-bit words. This implementation supports widths 1..8 and complete 32-code groups. A short final group needs a separate tail path.

## Computation before and after

Fixed-width code concatenation and inverse extraction across 32-bit words.

Initialize code_bits words to zero for each group of 32 codes. For code i set j=(i*code_bits)//32 and r=(i*code_bits)%32. OR its low bits at word j offset r, and, when r+code_bits>32, its remaining bits into word j+1. Unpack extracts min(32-r,code_bits) bits from j and the rest from j+1. Thread t in block b owns codes [1024*b+32*t,1024*b+32*t+32); store only groups within length.

Recover the original computation: Extract code i from its code_bits consecutive bit positions; discard the containers.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| code_bits | compile / software | {"lo": 1, "hi": 8} | Independent stored width of one code; interpretation as signed or floating point belongs to its consumer. |
| word_bits | compile / software | [32] | Container width fixed by this version's source witness and shift arithmetic. |

## Coupled constraints and scope

Lossless storage refinement of already-defined 1..8-bit codes in contiguous int32 tensors. Source witness is Humming pack/unpack on NVIDIA, independent of a matrix fragment layout. Matrix prepacking and quantization are separate refinements. Traffic is owned by the surrounding transfer chain, so this record introduces no duplicate DRAM transaction.

Scope predicate: `p.length >= 32 and p.length % 32 == 0`.

- Precondition `pre_code_range` (semantic): `p.min_code >= 0 and p.max_code < 2 ** code_bits` — Codes fit without loss; outside this premise the implementation retains only their low bits.
- Constraint `c_width` (implementation): `code_bits >= 1 and code_bits <= 8 and word_bits == 32` — The observed width domain fits the two-word extractor.
- Constraint `c_whole_groups` (implementation): `p.length >= 32 and p.length % 32 == 0` — Complete groups of 32 codes; callers exclude a partial code group.

## Local example and misuse

For a local instance of `store`, identify the object `contiguous_code_bitstream` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/humming/delta/bitstream_packing@1/template.yaml`. Original outline: `knowledge/studies/humming/delta/bitstream_packing@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Packed contiguous global-memory words, register-local groups during conversion.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `bit_identity` (exhaustive): Each logical bit (i,b) maps to physical bit i*code_bits+b, injectively and with the same value after unpacking.
- `group_ownership` (exhaustive): Each complete 32-code group has exactly one writer; an inactive thread in the final CTA writes no word.
- `memory_safety` (test): All visible reads and writes are in range; final-CTA inactive local arrays require separate CUDA language and sanitizer review.

None for in-range codes; truncating out-of-range input is outside the lossless premise.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
