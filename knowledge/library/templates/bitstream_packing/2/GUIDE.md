# bitstream_packing@2: contiguous code bitstream

## When to consider

When fixed-width codes must feed a byte- or word-based consumer, concatenate their disjoint logical bit intervals with the same inverse extractor; source QuACK packs four6-bit codes into three bytes. Container width and complete-stream extent are explicit.

## Computation before and after

Fixed-width code concatenation and inverse extraction across 32-bit words.

Initialize length*code_bits/word_bits containers to0. Logical bit b of code i maps to physical bit i*code_bits+b, container floor((i*code_bits+b)/word_bits), offset modulo word_bits. Packing ORs those disjoint bit intervals; extraction concatenates the matching source bits. Container ownership must prevent read-modify-write races. Source QuACK6-bit codes use4 codes per3 bytes; Humming32-code groups preserve their original owner mapping and32-bit extractor. The common algorithm does not require every code fit within one container.

Recover the original computation: Extract code i from its code_bits consecutive bit positions; discard the containers.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| code_bits | compile / software | {"lo": 1, "hi": 8} | Independent stored width of one code; interpretation as signed or floating point belongs to its consumer. |
| word_bits | compile / software | [8, 32] | Byte containers for source QuACK or32-bit containers for Humming; extraction uses matching word width. |

## Coupled constraints and scope

Version1 complete32-code/int32 domain retained. Version2 includes byte-container complete bitstreams from QuACK FP4/FP6 packers. Partial final physical container is excluded; native write ownership remains realization-specific.

Scope predicate: `p.length >= 1 and (p.length * code_bits) % word_bits == 0 and p.min_code >= 0 and p.max_code < 2 ** code_bits`.

- Precondition `pre_code_range` (semantic): `p.min_code >= 0 and p.max_code < 2 ** code_bits` — Codes fit without loss; outside this premise the implementation retains only their low bits.
- Constraint `c_width` (implementation): `code_bits >= 1 and code_bits <= 8 and (word_bits == 8 or word_bits == 32)` — Supported logical code/container widths.
- Constraint `c_whole_groups` (implementation): `p.length >= 1 and (p.length * code_bits) % word_bits == 0` — The finite code bitstream ends on a whole physical container.

## Local example and misuse

For a local instance of `store`, identify the object `contiguous_code_bitstream` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/bitstream_packing@2/template.yaml`. Original outline: `knowledge/studies/quack/delta/bitstream_packing@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
