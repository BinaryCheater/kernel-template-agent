# chunked_delta_feedback@1: causal delta feedback factorization

## When to consider

When a scalar-gated recurrent update subtracts a key prediction before its rank-one insertion, form the chunk feedback solve instead of treating inserts as independent. Choose corrected-value residuals first, or fold feedback into query coefficients for frozen output computation. Keep beta placement, decay frames and after-update readout explicit.

## Computation before and after

Weighted lower-triangular feedback solve followed by residual-first or query-coefficient-first contractions.

Use K-major H (transpose a V-major source state). For a chunk define E_i=product_(u=0..i) alpha_u and D_ij=product_(u=j+1..i) alpha_u, so D_ii=1. Set L_ij=beta_i*D_ij*(k_i dot k_j) for j<i and zero otherwise. Solve (I+L)*T=diag(beta). Define EK_i=E_i*k_i, EQ_i=E_i*q_i and causal C_ij=D_ij*(q_i dot k_j) for j<=i. In residual_first form R=T*(V-EK*H_in), then Y=EQ*H_in+C*R. In coefficient_first form P=C*T and A=EQ-P*EK, then Y=A*H_in+P*V; P and A depend on Q/K/gates but not V or H_in. If outgoing state is requested, obtain R (also in coefficient_first) and form H_out=E_end*H_in+sum_j D_end,j*k_j outer R_j after all H_in readers finish. Output-only mode omits this state work and retains the frozen checkpoint. Apply resolved query scale consistently. Solve strategy, feature tiling, checkpoint retention and matrix instruction layout are separate choices.

Recover the original computation: Expand each state contribution within the chunk. Residuals satisfy R_i+sum_(j<i) beta_i*D_ij*(k_i dot k_j)*R_j=beta_i*(v_i-E_i*k_i^T*H_in); forward uniqueness gives the sequential delta residuals. Substitution yields both output forms and the inclusive outgoing state.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_tokens | compile / software | [4, 8, 16, 64] | Short MTP native/staged widths4/8/16 or Hopper prefill width64. Other widths require a new binding. |
| evaluation_order | compile / software | ["residual_first", "coefficient_first"] | Residual-first exposes corrected value rows and outgoing state; coefficient-first reuses state/value-independent query coefficients and can omit state update in frozen verification. |

## Coupled constraints and scope

Scalar positive-decay GDN chunks. Hopper SM90a prefill uses64 rows, FP16 inverse storage with BF16/FP16 operands and FP32 accumulation/carried state options. Frozen BF16 no-prepack MTP uses a16-row working tile with4/8/16 active specialization, K=V128 and mandatory in-kernel Q/K normalization on SM90+ TMA-capable paths. Coefficient-first outgoing state is a composition requiring an implemented residual/state path, not a claim about that output-only kernel. Featurewise gates, multiple Householder substeps per token, generic nonlinear feedback and native bitwise equivalence are outside this version.

Scope predicate: `p.logical_tokens >= 1 and p.logical_tokens <= chunk_tokens and (p.scalar_decay_delta_recurrence_and_after_update_readout) and (p.positive_scalar_decays_and_complete_chunk_frames) and (p.unit_lower_weighted_solve_implemented_and_residual_admitted) and (p.gate_difference_and_all_contraction_rounding_stages_admitted) and (p.incoming_state_readers_complete_before_state_replacement) and (p.complete_chunks_or_neutral_padding_and_unique_output_ownership) and (p.selected_order_shape_dtype_and_target_supported)`.

- Precondition `pre_recurrence` (implementation): `p.scalar_decay_delta_recurrence_and_after_update_readout` — K-major state follows H_i=alpha_i*H_(i-1)+k_i*r_i with r_i=beta_i*(v_i-k_i^T*(alpha_i*H_(i-1))). Output is q_i^T*H_i. Query scaling and key normalization are resolved consistently before this factorization.
- Precondition `pre_frames` (implementation): `p.positive_scalar_decays_and_complete_chunk_frames` — Each alpha is positive; local E_i is the product through token i from this chunk boundary, and D_ij is the product from j+1 through i. Chunks cannot mix sequences or head mappings.
- Precondition `pre_solve` (implementation): `p.unit_lower_weighted_solve_implemented_and_residual_admitted` — A=I+L is unit lower triangular, L_ij=beta_i*D_ij*(k_i dot k_j) strictly below the diagonal. T solves A*T=diag(beta), including zero beta without division. The selected solve/casts meet the declared residual policy.
- Precondition `pre_numeric` (implementation): `p.gate_difference_and_all_contraction_rounding_stages_admitted` — Compute decay ratios from local log-prefix differences where needed, without overflowing separate reciprocals. KKT, QKT, T, residual/coefficient products, state and output casts have an explicit admitted error policy; real equivalence does not imply bitwise equality.
- Precondition `pre_boundary` (implementation): `p.incoming_state_readers_complete_before_state_replacement` — All outputs/residuals needing H_in capture that generation before H_out overwrites it. Optional frozen output-only execution leaves H_in unchanged. The next chunk waits for complete H_out.
- Precondition `pre_scope` (implementation): `p.complete_chunks_or_neutral_padding_and_unique_output_ownership` — Complete chunk coordinates, or a validated neutral extension alpha=1,beta=0,k=q=v=0 and masked stores, preserve valid rows. Each output/state shard has one owner and reductions include every key shard.
- Precondition `pre_backend` (implementation): `p.selected_order_shape_dtype_and_target_supported` — The selected computation order, width, dtype, state layout and target have an implemented binding. Output-only coefficient-first does not by itself produce an outgoing state.
- Constraint `c_shape` (implementation): `p.logical_tokens >= 1 and p.logical_tokens <= chunk_tokens` — One nonempty chunk with complete or neutral-padded valid-token extent.

## Local example and misuse

For a local instance of `map`, identify the object `causal_delta_feedback_factorization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/chunked_delta_feedback@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/chunked_delta_feedback@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Chunk KKT/QKT/weighted solve and residual or query coefficients plus incoming recurrent state.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `feedback_system` (proof): The strictly lower feedback system has the same unique residuals as the sequential recurrence, including beta0 and nonzero incoming state.
- `contraction_order` (test): Both output orders preserve causal support and after-update readout, and outgoing state includes every residual once.
- `numeric_and_lifetime` (test): Native stage precision and decay arithmetic meet the selected error policy; incoming state and aliased scratch remain live through their last reader.

Small triangular systems are always invertible in exact arithmetic but can be ill-conditioned. Coefficient cancellation, BF16/FP16 solve storage, gate underflow and reassociation change native error.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
