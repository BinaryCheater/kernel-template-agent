# chunked_decayed_outer_product_state@1: chunk boundary state

## When to consider

When a causal weighted outer-product recurrence has a state much smaller than its token-pair matrix, compute intra-chunk interactions by triangular contractions and carry only the boundary state between chunks. Preserve temporal order and the exact boundary convention; chunk width trades local pair work against state traffic. Scalar log-decay gates are supported; feature-dependent delta rules need a separate factorization.

## Computation before and after

Factor causal recurrence into local triangular contractions and sequential boundary states.

For one chunk let c_i=sum_(j<=i) g_j, measured from zero at its input boundary, and c_end its final prefix. With incoming H, output row i is exp(c_i)*q_i*H + sum_(j<=i) exp(c_i-c_j)*(q_i dot k_j)*v_j. Compute the second term as a lower-triangular weighted QK contraction followed by multiplication with V. After all outputs that need the incoming state have captured it, replace H by exp(c_end)*H + sum_j exp(c_end-c_j)*(k_j outer v_j). The next chunk consumes this new state. With decay_mode=none all c are zero. Output/state feature tiling is separate; reducing key-dimension shards must combine every partial once.

Recover the original computation: Expand the chunk contractions in token order; each prior token contribution carries exactly the product of decays between its insertion and the queried token.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_tokens | compile / software | [64, 256] | Chunk width; the pinned linear/retention representative uses 64 and Mamba examples use 256. Other widths need a source binding. |
| decay_mode | compile / software | ["none", "scalar_log_prefix"] | No decay, or one cumulative scalar log-decay per token and head within the chunk. |

## Coupled constraints and scope

Complete chunks of a linear outer-product state with scalar per-token/head decays. Source linear/retention kernels use 64-token chunks and FP32 carried state with lower-precision contraction copies. Mamba chunk-state/scan witnesses factor scalar cumulative gates, insertion dt and readout C. The inter-chunk state propagation for separately launched Mamba chunks is an external input, not supplied by those two kernels. Featurewise gates, delta corrections, arbitrary tails and generic nonlinear recurrence are outside this version.

Scope predicate: `p.sequence_length > 0 and p.sequence_length % chunk_tokens == 0 and p.scalar_decay_outer_product_recurrence and p.sequence_origins_match and p.initial_state_defined and (decay_mode != 'none' or p.all_log_decays_zero) and p.exponents_in_range and p.reassociation_and_state_casts_admitted and p.tail_and_output_ownership_valid`.

- Precondition `pre_recurrence` (semantic): `p.scalar_decay_outer_product_recurrence and p.sequence_origins_match and p.initial_state_defined` — The recurrence is H_i=exp(g_i)*H_(i-1)+k_i outer v_i, output q_i times H_i; sequence origin and initial state are fixed.
- Precondition `pre_frames` (semantic): `decay_mode != 'none' or p.all_log_decays_zero` — The no-decay specialization binds every gate to zero.
- Precondition `pre_numeric` (implementation): `p.exponents_in_range and p.reassociation_and_state_casts_admitted and p.tail_and_output_ownership_valid` — Gate exponentiation, contraction regrouping, state casts and output shard combination satisfy the numerical and ownership policy.
- Constraint `c_chunks` (semantic): `p.sequence_length > 0 and p.sequence_length % chunk_tokens == 0` — This first version covers complete chunks only.

## Local example and misuse

For a local instance of `map`, identify the object `chunk_boundary_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tilelang/delta/chunked_decayed_outer_product_state@1/template.yaml`. Original outline: `knowledge/studies/tilelang/delta/chunked_decayed_outer_product_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Key-by-value state plus chunk tiles instead of a full token-pair matrix.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `prefix_identity` (proof): Local and incoming-state terms partition causal history without overlap and use matching decay prefixes.
- `state_boundary` (test): Every output uses the incoming state before it is overwritten; the next chunk receives the completed outgoing state.
- `numeric_binding` (test): Chosen state precision and factored exponent arithmetic meet the numerical contract.

Changes summation and state conversion order; factored exp(c_i-c_j) must remain in range. No bitwise equivalence is claimed.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
