# block_scaled_instruction_operands@1: block scaled instruction operands

## When to consider

When hardware accepts block-scaled operands directly, preserve code and scale as separate instruction operands. Bind format/group size and logical scale coordinates before packing. E2M1 byte offset k//2 and scale offset k//16 are different maps; using byte offsets for scales silently doubles scale coverage. Native scaled MMA is an alternative realization to software dequantization or explicit raw-partial promotion.

## Computation before and after

A scaled MMA consumes quantized operands plus independently indexed scale tensors instead of software-dequantized operands.

Keep encoded A/B values and scale tensors separate through transfer. For logical product (m,n,k), select A scale at (m,k//scale_group_k) and B scale at (n,k//scale_group_k), independent of each operand packed-byte coordinate. Feed those scale fragments to the scaled MMA instruction with the encoded operands and the admitted accumulator. NVFP4 uses E2M1 data/E4M3 scale and groups16; MX formats use E8M0 scales/groups32. Convert packed element K to descriptor bytes using each operand own elements_per_byte; scales use logical K/group, never byte K/group. Scale TMEM has its own TensorMemoryScalesLayout, distinct from accumulator TMEM.

Recover the original computation: Replace the intrinsic by its specified contraction over decoded scaled operand values at each logical (m,n,k), retaining the admitted native rounding contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| format | compile / software | ["nvfp4", "mxfp4", "mxfp8", "mixed_e4m3_e2m1"] | Requested operand/code format pair from the source instruction menu. |
| scale_group_k | compile / software | {"lo": 1, "hi": 65536} | Logical consecutive K values sharing each operand scale, source16 or32. |

## Coupled constraints and scope

NVIDIA Blackwell supported scaled-MMA format menu with explicit16/32 K scale groups and matching scale layout. Does not assert that hardware internally forms raw partial sums and promotes them exactly as block_scaled_partial_accumulation@1.

Scope predicate: `(p.code_and_scale_formats_match_instruction and p.group_membership_and_scale_coordinates_exact) and (p.packed_byte_offsets_match_logical_K and p.scale_storage_layout_legal) and (p.input_and_scale_transfers_complete_before_mma and p.accumulation_rounding_admitted)`.

- Precondition `pre0` (semantic): `p.code_and_scale_formats_match_instruction and p.group_membership_and_scale_coordinates_exact` — Code format, scale format and K grouping match both operands and the instruction.
- Precondition `pre1` (semantic): `p.packed_byte_offsets_match_logical_K and p.scale_storage_layout_legal` — Sub-byte operand descriptors and TMEM scale layouts preserve logical coordinates.
- Precondition `pre2` (semantic): `p.input_and_scale_transfers_complete_before_mma and p.accumulation_rounding_admitted` — Operands/scales are ready and hardware accumulation meets the requested numerical contract.

## Local example and misuse

For a local instance of `contract`, identify the object `block_scaled_instruction_operands` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/block_scaled_instruction_operands@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/block_scaled_instruction_operands@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `scale_identity` (proof): Every encoded operand product receives the corresponding logical scale pair.
- `packing_and_layout` (test): Packed bytes and physical scale fragments decode to the same logical K coordinates.
- `native_numeric` (test): Scaled instruction, accumulator and output casts meet the declared numerical recipe.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
