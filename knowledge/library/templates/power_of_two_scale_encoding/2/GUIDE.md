# power_of_two_scale_encoding@2: exponent scale code

## When to consider

When an already-defined positive power-of-two scale feeds UE8M0 consumers, emit its biased exponent. Normal scales use FP32 exponent bits; the minimum2^-127 needs explicit live code0. Keep padding separate and leave rounding/clamping raw scales to the numerical quantizer.

## Computation before and after

Extract the biased exponent of an already-normalized power-of-two scale.

For each live positive finite power-of-two scale s in[2^-127,2^127], emit code0 when s==2^-127; otherwise reinterpret its normal FP32 bits and emit(u>>23)&255. Decode e as2**(e-127) at the same logical metadata coordinate. Code0 therefore represents a positive minimum, not numerical zero. Preserve padding identity separately; code255 remains excluded. Upward rounding arbitrary scales into this domain is a separate numerical step; byte packing and metadata layout are independent.

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

Version1 positive normal exponent-126..127 storage domain retained. Version2 adds exactly2^-127, a representable subnormal FP32 scale requiring special code0 handling. Other subnormals, zero,negative scales,infinity,non-powers and reserved255 remain outside. E8M0 native conversion may produce live0 for tiny/zero raw scales; only its resulting positive scale, not the raw zero, is encoded by this rule.

Scope predicate: `p.min_scale_exponent >= -127 and p.max_scale_exponent <= 127 and p.scales_positive_pow2_and_representable and p.consumer_supports_ue8m0 and p.padding_not_substituted_for_live_scale`.

- Precondition `pre_exact` (semantic): `p.min_scale_exponent >= -127 and p.max_scale_exponent <= 127 and p.scales_positive_pow2_and_representable` — Positive finite FP32 powers of two with exponent-127..127; exactly2^-127 maps to live code0, normal scales use exponent byte1..254.
- Precondition `pre_padding` (semantic): `p.padding_not_substituted_for_live_scale` — Any padding byte, including0, stays outside required logical coordinates; a live byte0 is allowed only when its actual scale is2^-127.
- Constraint `c_exact_domain` (semantic): `p.min_scale_exponent >= -127 and p.max_scale_exponent <= 127 and p.scales_positive_pow2_and_representable` — Positive finite FP32 powers of two with exponent-127..127; exactly2^-127 maps to live code0, normal scales use exponent byte1..254.
- Constraint `c_consumer` (implementation): `p.consumer_supports_ue8m0` — The consuming scale decoder interprets byte e as 2**(e-127).

## Local example and misuse

For a local instance of `store`, identify the object `exponent_scale_code` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/power_of_two_scale_encoding@2/template.yaml`. Original outline: `knowledge/studies/sglang/delta/power_of_two_scale_encoding@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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
- `padding_exclusion` (test): Padding bytes are not substituted for required live coordinates; reserved255 is never a live scale. Live0 is admitted only for exactly2^-127.

No change on the admitted domain; applying the extraction to zero or infinity changes values and is outside the contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
