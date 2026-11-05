# operand_bitfield_placement@1: instruction operand payload position

## When to consider

When an unpacking load and its arithmetic consumer use different bit positions for the same floating payload, consider an explicit register bitfield placement. Preserve each coordinate and all code bits, and apply the placement once to newly unpacked bytes. The mixed SM120 FP4 operand moves low E2M1 bits to bits2..5; the FP8 side and same-type FP4 path stay unchanged. This is not integer widening or quantization. Destination position is dictated by the consumer and native layout/exact-once preparation must be verified separately.

## Computation before and after

Preserve floating code bits while relocating them to the instruction payload field.

Consume each zero-extended E2M1 code q in register bits0..3 of one byte. Replace that byte by q<<destination_lsb, retaining its coordinate and applying the transformation exactly once after unpacking and before the matching MMA. The consumer extracts four bits at destination_lsb and interprets the original E2M1 code, preserving all code bits including sign and signed-zero encoding. No numerical conversion, scale multiplication, quantization or global-memory repacking occurs. A different code format, nonzero unused bits or another load/consumer layout needs a verified binding or variant.

Recover the original computation: Extract (prepared_byte>>destination_lsb)&15 to recover the exact original floating code and coordinate.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| destination_lsb | compile / boundary | {"lo": 0, "hi": 4} | Least-significant bit expected by the consumer; the selected mixed-FP4 MMA binding fixes2, not a source tuning knob. |

## Coupled constraints and scope

Code-preserving register reencoding of four-bit E2M1 payloads in eight-bit containers. SM120 mixed FP4xFP8 uses b4x16_p64 unpack then destination_lsb2; same-type FP4 and FP8 operands do not receive this shift. Source code uses Int8 views but the payload is floating encoding, not signed integer value. Native lane/register layout and exact-once load/shift ordering remain backend obligations. Other positions describe the logical map only until a consumer binding establishes support; placement does not authorize another source tile.

Scope predicate: `p.input_code_e2m1 and p.input_payload_in_low_bits and p.unused_source_bits_zero and p.coordinate_order_preserved and p.exactly_once_after_unpack and p.unpack_backend_supported and p.consumer_payload_decode_supported`.

- Precondition `pre_input` (semantic): `p.input_code_e2m1 and p.input_payload_in_low_bits and p.unused_source_bits_zero` — The load produced an unsigned four-bit E2M1 payload in the low bits of a zero-extended byte.
- Precondition `pre_identity` (semantic): `p.coordinate_order_preserved and p.exactly_once_after_unpack` — Each newly unpacked code is shifted once without changing its logical coordinate or re-shifting a prepared fragment.
- Precondition `pre_backend` (implementation): `p.unpack_backend_supported and p.consumer_payload_decode_supported` — The selected load and MMA agree on the code identity and register-byte layout.
- Constraint `c_consumer` (implementation): `destination_lsb == p.consumer_payload_lsb and p.code_bits == 4 and p.container_bits == 8` — Four payload bits fit one byte and match the exact consumer interpretation.

## Local example and misuse

For a local instance of `map`, identify the object `instruction_operand_payload_position` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/operand_bitfield_placement@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/operand_bitfield_placement@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Existing per-byte register containers after unpack.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `code_identity` (exhaustive): All16 E2M1 code patterns and every coordinate survive placement and extraction exactly.
- `fragment_identity` (test): No code is shifted twice and no neighboring register-byte payload or padding bit is consumed as data.
- `backend_agreement` (test): The selected unpack operation and consumer instruction implement the stated byte positions and coordinate order.

No code change under the premises; omitting or repeating the shift changes the consumed floating code.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
