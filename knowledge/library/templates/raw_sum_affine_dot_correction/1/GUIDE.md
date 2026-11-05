# raw_sum_affine_dot_correction@1: raw activation sum for affine weight offset

## When to consider

When weights have a group-constant affine offset, save the raw activation group sum during activation quantization and apply the offset through that statistic. Reuse it across weight rows; distribute the correction only across a known exact contributor count. This preserves a higher-fidelity offset term than recomputing it from quantized activation codes, while the code-dependent term remains an admitted integer-dot approximation.

## Computation before and after

Separate affine weight offset using a reusable sum of raw activations while quantizing only the code-dependent dot term.

While quantizing a32-value activation groupx to signedq with scaledx, also reduces_raw=sum(x) before quantization and store it in its declared metadata dtype. For affine weight groupw=d_w*q_w+m_w compute the code termd_w*d_x*integer_dot(q_w,q_x) under the source scale-product arithmetic. Addm_w*stored(s_raw), either once or equally across exactlycorrection_owners final partials. For centered weight formd_w*(q_w-z), use offsetm_w=-d_w*z. Reuse the same raw activation sum for every weight row sharing this activation group; retain scale/sum generation pairing.

Recover the original computation: The original affine weight dot splits exactly into d_w*sum(q_w*x)+m_w*sum(x). Only the first term is approximated by the declared activation quantizer; the offset term uses the retained raw sum with its explicit metadata rounding.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| correction_owners | compile / software | {"lo": 1, "hi": 32} | Number of contributing partial-dot owners that share one offset correction equally; source derives this fromVDR/QI/QR. |

## Coupled constraints and scope

Pinned GGUF Q4_0/Q4_1/Q5 affine weights withQ8_1 activation groups32, stored half scale and half raw sum. Numerical recipe includes source half2 scale products and per-owner division. Different superblock/subgroup intersections need an explicit binding; this does not certify arbitrary codebook offsets.

Scope predicate: `p.weight_decode_is_gain_times_code_plus_offset and p.affine_offset_constant_on_group and p.quantized_activation_and_raw_sum_same_group_generation`.

- Precondition `pre_inputs` (semantic): `p.weight_decode_is_gain_times_code_plus_offset and p.quantized_activation_and_raw_sum_same_group_generation` — Weight representsd*q+m; activation quantizer also retains sum of original input values for the same32 coordinates.
- Precondition `pre_numeric` (implementation): `p.raw_sum_metadata_cast_and_integer_dot_policy_admitted and p.integer_dot_no_overflow` — The hybrid approximation uses stored activation scale and rounded raw sum explicitly; it is not assumed equal to a dot of two decoded quantized operands.
- Precondition `pre_distribution` (semantic): `p.correction_contributors_exact and p.final_combine_includes_each_partial_once` — Exactlycorrection_owners partials add1/correction_owners of the same correction, with an admitted rounding order.
- Constraint `c_group` (implementation): `p.activation_group_length == 32 and p.affine_offset_constant_on_group` — Pinned32-value activation statistic and a constant weight offset over that group.

## Local example and misuse

For a local instance of `contract`, identify the object `raw_activation_sum_for_affine_weight_offset` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/raw_sum_affine_dot_correction@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/raw_sum_affine_dot_correction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One raw sum alongside activation scale per32-element group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `sum_domain` (exhaustive): Retained statistic sums original activation values of the exact group, including only neutral padded zeros.
- `correction_once` (test): Distributed correction coefficients sum toone and every required partial is combined exactly once.
- `hybrid_numeric` (test): Scale products, raw-sum metadata cast and correction distribution meet the declared hybrid-dot error policy.

Half raw-sum storage, half2 scale products and divided correction parts add rounding. Replacing raw sum byscale*sum(q) changes the approximation, especially at rounding boundaries.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
