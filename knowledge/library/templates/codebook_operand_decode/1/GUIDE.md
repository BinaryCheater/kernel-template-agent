# codebook_operand_decode@1: indexed vector with scale sign offset

## When to consider

When low-bit weights index a nonuniform scalar/vector codebook, retain the table and sign-map semantics through decode. Choose standalone expansion or an admitted fused codebook dot while preserving coordinate, scale and offset identity. This exposes codebook reuse and materialization choices without pretending the index bits are uniformly spaced signed integers.

## Computation before and after

Index-table vector expansion with original sign/scale/offset metadata, optionally consumed inside a dot.

Unpack each indexI for its original vector coordinate. Fetch table vectorT[I,j],j=0..vector_width-1. Obtain sign_j from the exact explicit or encoded sign-map table, or use+1 when signs are embedded inT. Decodevalue_j=gain_group*sign_j*T[I,j]+offset_group with the original scale/cast order. In standalone_output write the declared output dtype. In at_contraction form dot products over exactly the same logical components using the admitted codebook/integer accumulation recipe and preserve every required scale/offset term; do not treatI as a uniformly spaced numeric code.

Recover the original computation: Expand each codebook index and metadata record into its vector values at original coordinates; the contraction consumes that declared representation under its numerical policy.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| vector_width | compile / boundary | [1, 4, 8] | Logical values per codebook index, fixed by storage format. |
| decode_site | compile / software | ["standalone_output", "at_contraction"] | Materialize reconstructed values or consume the same codebook interpretation inside a dot product with an explicitly admitted arithmetic policy. |

## Coupled constraints and scope

Pinned GGUF nonuniform scalar IQ4 and vector IQ1/IQ2/IQ3 operand formats, with immutable format codebooks and explicit sign/metadata maps. Source IQ2_XXS uses256 eight-byte vectors and128 sign masks; IQ4_NL uses16 signed scalar entries. This rule decodes existing representations; it does not train codebooks or prove quantization error against original weights.

Scope predicate: `p.indices_in_codebook_range and p.table_and_sign_decoder_fixed and p.coordinate_maps_complete and p.selected_decode_contraction_recipe_supported`.

- Precondition `pre_format` (semantic): `p.indices_in_codebook_range and p.table_and_sign_decoder_fixed and p.coordinate_maps_complete` — Code indices, table byte/component order, sign expansion and output coordinates are format-defined; no linear code approximation is substituted.
- Precondition `pre_meta` (implementation): `p.scale_offset_groups_match_indices and p.metadata_generation_immutable_and_visible` — Every decoded component uses the matching table, sign, local/global scale and optional offset generation.
- Precondition `pre_numeric` (implementation): `p.selected_decode_contraction_recipe_supported and p.scale_offset_cast_and_integer_accumulation_policy_admitted` — Standalone half casts and fused integer/scaled accumulation need their own numerical admission; neither is automatically bitwise equivalent.

## Local example and misuse

For a local instance of `contract`, identify the object `indexed_vector_with_scale_sign_offset` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/sglang/delta/codebook_operand_decode@1/template.yaml`. Original outline: `knowledge/studies/sglang/delta/codebook_operand_decode@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Packed indices and immutable format codebook; decoded intermediate optional.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `table_identity` (exhaustive): Indices, byte/component order and sign-map expansion reconstruct the exact format-defined codebook vectors.
- `metadata_owner` (test): All vector components use matching gain/offset and original coordinates with no repeated or missing contribution.
- `numeric_boundary` (test): Standalone and fused decode recipes implement their explicitly declared scaling, rounding and accumulation policy.

Nonuniform table indices are not affine levels. Fused integer and standalone half decode differ in intermediate rounding and scale application; preserve each recipe explicitly.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
