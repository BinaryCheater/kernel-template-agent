# bitplane_integer_dot_reconstruction@1: bitplane integer dot reconstruction

## When to consider

When small unsigned codes are stored in bit planes or packed nibbles, compute the integer dot in that representation and apply row metadata only after the exact code reduction. Codes A=[1,2],B=[3,1] give raw dot5. Plane intersections weighted by2**(i+j) must give5 as well. Binary bit1 expands to nibble1, not15 or signed-1; swapping MSB-first coordinate order changes asymmetric products.

## Computation before and after

When small unsigned codes are stored in bit planes or packed nibbles, compute the integer dot in that representation and apply row metadata only after the exact code reduction.

For bit-sliced codes, sum over plane pairs(i,j) of popcount(A_plane_i AND B_plane_j)*2**(i+j). For packed4b words, separately mask low/high nibble byte lanes and perform two unsigned dp4a operations. A packed1b operand can expand each MSB-first bit into a corresponding unsigned nibble0/1 for the same u4 dot engine, preserving coordinate order. After the integer dot, combine per-row lower/scale and code-sum terms to recover the declared affine-decoded dot; then apply L2/IP/cosine correction and clamp/normalization policy separately.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| word_bits | compile / software | [32] | Native packed/popcount word width. |
| max_code_bits | compile / software | [4] | Narrow source witness: binary or nibble unsigned codes; wider variants require their own range bound. |

## Coupled constraints and scope

NVIDIA BBQ local-join code inner products, binary/packed4b/bit-sliced representations. Integer dot exactness does not imply original-vector distance exactness; quantization corrections and original norm metadata define the final approximate metric. Resource effects: Bit-plane dot costs code_bits_A*code_bits_B popcount streams; packed4b uses two dp4a per word; u4 promotion increases binary operand storage4x before MMA. Target engine determines tradeoff.

Scope predicate: `p.codes_unsigned_in_range and p.bit_order_consistent and p.integer_accumulator_no_overflow and p.tail_bits_zero and p.metadata_rows_match and p.decoded_metric_policy_defined`.

- Precondition `pre_0` (semantic): `p.codes_unsigned_in_range and p.bit_order_consistent` — Producer/consumer agree on per-row code/bit mapping; codes are unsigned, not signed bipolar values.
- Precondition `pre_1` (semantic): `p.integer_accumulator_no_overflow and p.tail_bits_zero` — Exact integer reduction fits its accumulator and padded codes contribute zero.
- Precondition `pre_2` (semantic): `p.metadata_rows_match and p.decoded_metric_policy_defined` — Each operand keeps its own lower/scale/sum/correction metadata and final metric convention is explicit.

## Local example and misuse

For a local instance of `map`, identify the object `bitplane_integer_dot_reconstruction` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cuvs/delta/bitplane_integer_dot_reconstruction@1/template.yaml`. Original outline: `knowledge/studies/cuvs/delta/bitplane_integer_dot_reconstruction@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Bit-plane dot costs code_bits_A*code_bits_B popcount streams; packed4b uses two dp4a per word; u4 promotion increases binary operand storage4x before MMA. Target engine determines tradeoff.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `integer_dot` (proof): Every original coordinate product contributes exactly once with its correct bit weight, independent of storage path.
- `promotion` (proof): Packed1b to4b preserves0/1 values and source coordinate order, never maps bits to +/-1 implicitly.
- `metadata` (proof): Affine cross terms use document/query row metadata and retained code sums from the same rows.
- `range` (proof): All plane shifts, integer sums and tail reads remain within declared bounds.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
