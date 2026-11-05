# power_of_two_scale_encoding@1: exponent scale code

## When to consider

Use when the consumer supports UE8M0 and scale metadata is already positive normal FP32 powers of two; choose producer or layout_transform encoding and preserve each scale's owner. Extract the exponent byte and let the consumer decode it with bias127, for example FP32 scale8 becomes code130 and decodes to8. Zero padding is a storage convention and must remain outside the live scale domain.

## Computation before and after

Extract the biased exponent of an already-normalized power-of-two scale.

For each live scale s, reinterpret its FP32 bits u, emit code=(u>>23)&255, and preserve the logical metadata coordinate. The consumer decodes code e to 2**(e-127). Words of four bytes, MN transposes, per-group word boundaries and shared-memory UTCCP permutations are separate storage refinements. A layout implementation may emit code0 for padding; it is a finite positive UE8M0 value rather than numerical zero.

Recover the original computation: Decode each live byte with the declared biased-exponent representation at its original logical metadata coordinate.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| site | compile / software | ["producer", "layout_transform"] | Encode when producing metadata or in a separate layout transform before consumption. |

## Coupled constraints and scope

Value-preserving storage for already-defined positive normal FP32 power-of-two metadata, exponents -126..127. Rounding arbitrary scales upward to powers of two is a different numerical policy. The source's sign/mantissa guard is weaker than this exactness domain: it permits zero and positive infinity, whose UE8M0 codes mean a positive minimum and NaN. Operand quantization and byte packing are separate steps.

Scope predicate: `p.scales_positive_normal_pow2 and p.consumer_supports_ue8m0`.

- Precondition `pre_exact` (semantic): `p.min_scale_exponent >= -126 and p.max_scale_exponent <= 127 and p.scales_positive_normal_pow2` — Extraction is lossless on live scales; source guard alone also accepts zero and infinity.
- Precondition `pre_padding` (semantic): `p.padding_not_logical_scale` — Filled padding code0 is never interpreted as a live zero scale.
- Constraint `c_exact_domain` (semantic): `p.min_scale_exponent >= -126 and p.max_scale_exponent <= 127 and p.scales_positive_normal_pow2` — Every live input scale has zero sign/mantissa and exponent byte1..254.
- Constraint `c_consumer` (implementation): `p.consumer_supports_ue8m0` — The consuming scale decoder interprets byte e as 2**(e-127).

## Local example and misuse

For a local instance of `store`, identify the object `exponent_scale_code` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/power_of_two_scale_encoding@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/power_of_two_scale_encoding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One byte per live scale before independent word packing and layout.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `value_identity` (exhaustive): Decoding each live exponent byte recovers the original FP32 scale exactly over the admitted domain.
- `metadata_identity` (exhaustive): Every live scale keeps its logical owner through encoding and any separately composed physical layout.
- `padding_exclusion` (test): Padding code0 and reserved code255 are never substituted for a required live scale.

No change on the admitted domain; applying the extraction to zero or infinity changes values and is outside the contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
