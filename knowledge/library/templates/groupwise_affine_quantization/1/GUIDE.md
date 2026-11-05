# groupwise_affine_quantization@1: affine quantization group

## When to consider

Use when the requested operation quantizes finite values in complete contiguous groups. Bind group_size, code_bits and integer or floating zero point to the requested format. Compute each group’s range and metadata, then round and clamp each value to its code. Emit codes, scales and zero points. For a group of 128 with unsigned 4-bit codes, pass the resulting codes to a separate packer if the consumer expects packed storage. The numerical contract determines which format choices may vary.

## Computation before and after

Group extrema then affine nearest-even quantization with clamping.

For each group reduce min and max, with source floors min<=-1e-30 and max>=1e-30. Compute scale=(max-min)/(2**code_bits-1), z=-min/scale. For integer zero point, round z to nearest-even and reset min=-z*scale. Convert each (x-min)/scale to a nearest-even integer, clamp to [0,2**code_bits-1], and store the group scale and z. Floating zero point retains the unrounded z. Each group is handled by one warp in the source; code packing into a bitstream is a separate operation.

Recover the original computation: Read the code and metadata as the output of the requested affine quantizer; this is not an inverse recovering arbitrary original real values.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_size | compile / boundary | {"lo": 8, "hi": 65536, "step": 8} | Quantization group extent required by the coarse numerical contract. |
| code_bits | compile / boundary | {"lo": 2, "hi": 8} | Unsigned target code width required by the contract. |
| zero_point | compile / boundary | ["integer", "floating"] | Whether range-derived zero point is rounded to an integer before code generation. |

## Coupled constraints and scope

Unsigned integer affine quantization with a floating scale and integer or floating zero point. Group size and bit width are numerical boundary parameters, not free optimization knobs for an unchanged GEMM. Symmetric negative scales, one-bit and low-float quantizers are separate observed branches retained as gaps. No error bound for an original GEMM is implied by implementing the requested quantizer.

Scope predicate: `p.affine_quantizer_requested and p.finite_input and p.scale_nonzero_finite`.

- Precondition `pre_finite` (semantic): `p.finite_input and p.affine_quantizer_requested` — The coarse operation already requests finite affine quantization rather than exact real-valued storage.
- Precondition `pre_range` (semantic): `p.scale_nonzero_finite` — Range calculation and reciprocal must be finite and nonzero in the selected arithmetic.
- Constraint `c_length` (implementation): `p.length % group_size == 0` — Every group is complete.
- Constraint `c_contract` (semantic): `p.group_size == group_size and p.code_bits == code_bits and p.zero_point == zero_point` — Grouping and code range agree with the requested quantizer.

## Local example and misuse

For a local instance of `map`, identify the object `affine_quantization_group` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/humming/delta/groupwise_affine_quantization@1/template.yaml`. Original outline: `knowledge/studies/humming/delta/groupwise_affine_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Group scale and zero point plus one integer code per input value.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `group_range` (exhaustive): Each group's extrema include exactly its input values and the documented nonzero scale floors.
- `quantizer_contract` (test): Scale and zero-point arithmetic and nearest-even clamping implement the chosen numerical policy.
- `metadata_alignment` (exhaustive): Each code uses and produces the metadata of its own group exactly once.

FP32 extrema floors and reciprocal/rounding boundaries are part of the policy; quantization is intentionally lossy.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [bitstream_packing@1](../../bitstream_packing/1/GUIDE.md), [bitstream_packing@2](../../bitstream_packing/2/GUIDE.md), [bitstream_packing@3](../../bitstream_packing/3/GUIDE.md); Integer codes can subsequently use compact storage without changing their values.

Structured contract: [template.yaml](template.yaml).
