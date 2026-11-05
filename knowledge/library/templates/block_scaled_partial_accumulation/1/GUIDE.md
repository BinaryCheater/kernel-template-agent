# block_scaled_partial_accumulation@1: scaled dot partial

## When to consider

Use for block-scaled contractions when both scales are constant within a dot partial; bind the input scale granularities and choose partial_k dividing both. Reset and compute an unscaled partial, multiply by that partial's A/B scales, then add it once to the final accumulator; for K=256 with per-128 scales, promote two 128-product partials with their own scales. Varying partial_k requires numerical admission, while changing the input scale format changes the workload.

## Computation before and after

Unscaled dot partial followed by multiplicative scale promotion into the final sum.

For each output (m,n), set final=0. For t from 0 to ceil_div(K,partial_k)-1, reset raw=0 and accumulate qA[m,k]*qB[n,k] over this partial's valid k indices. Look up sA[floor(m/gran_m),floor(t*partial_k/gran_k_a)] and sB[floor(n/gran_n),floor(t*partial_k/gran_k_b)]. Promote final += (sA*sB)*raw. Convert final to the requested output type after all partials. Select metadata separately for each output even when the physical output tile spans multiple MN scale groups. Instruction-specific raw and final rounding belong to the Realisation.

Recover the original computation: Sum the scaled partials to recover the contraction of decoded operands, under the declared numerical contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| partial_k | compile / software | {"lo": 1, "hi": 65536} | Reduction extent before each scale promotion; it may vary only within the admitted numerical grouping contract. |
| gran_k_a | compile / boundary | {"lo": 1, "hi": 65536} | A scale extent along K fixed by the input representation. |
| gran_k_b | compile / boundary | {"lo": 1, "hi": 65536} | B scale extent along K fixed by the input representation. |
| gran_m | compile / boundary | {"lo": 1, "hi": 65536} | Number of A rows sharing one scale fixed by the workload. |
| gran_n | compile / boundary | {"lo": 1, "hi": 65536} | Number of B output columns sharing one scale fixed by the workload. |

## Coupled constraints and scope

Separable multiplicative operand scales, no affine zero point. Scale groups start at k_origin and partial_k divides both K granularities. Metadata layout is a separate refinement. The SM90 source witness uses FP8 operands, FP32 raw/final accumulators, per-row A scales, per-128-column B scales, and partial_k=128. Broader divisors are algebraic choices whose numerical admission and backend support need separate checks. Transaction/resource amounts remain unpriced until a Realisation specifies sharing.

Scope predicate: `p.block_scaled_inputs and p.numeric_contract_allows_partial_scaling`.

- Precondition `pre_origin` (semantic): `p.k_origin % gran_k_a == 0 and p.k_origin % gran_k_b == 0` — Each independently scheduled contraction begins at the scale-group origin.
- Precondition `pre_scale_contract` (semantic): `p.block_scaled_inputs and p.numeric_contract_allows_partial_scaling` — The requested decoded contraction admits this accumulation and rounding policy.
- Precondition `pre_tail` (semantic): `p.K % partial_k == 0 or p.neutral_tail_padding` — A short final K partial has zero padded operands and valid metadata.
- Constraint `c_scale_constant` (semantic): `gran_k_a % partial_k == 0 and gran_k_b % partial_k == 0` — Origin-aligned K partials cannot straddle either operand's scale boundary.
- Constraint `c_format` (semantic): `gran_k_a == p.gran_k_a and gran_k_b == p.gran_k_b and gran_m == p.gran_m and gran_n == p.gran_n` — Metadata belongs to the requested input representation.

## Local example and misuse

For a local instance of `contract`, identify the object `scaled_dot_partial` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/block_scaled_partial_accumulation@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/block_scaled_partial_accumulation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Separate raw and final accumulators; scale metadata consumed by logical ownership.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `partition` (exhaustive): Each valid K index belongs to exactly one partial; tail fills contribute zero.
- `scale_owner` (exhaustive): Each raw partial uses the same A and B scale as every decoded product it represents, including MN boundaries.
- `reset_and_promote` (test): The raw accumulator is reset for every partial and each completed partial is promoted once before reuse.
- `numerical_policy` (test): Raw dot rounding, scale product, final accumulation and output conversion satisfy the requested tolerance for the admitted data range.

Moving scales after a dot and restarting raw accumulators changes floating-point rounding; unequal scales, cancellation and large_dynamic_range are required numerical witnesses.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- requires: [reduction_partition@1](../../reduction_partition/1/GUIDE.md), [reduction_partition@2](../../reduction_partition/2/GUIDE.md); Scaling can move outside a partial only because its K indices form a partition with constant scales.

Structured contract: [template.yaml](template.yaml).
