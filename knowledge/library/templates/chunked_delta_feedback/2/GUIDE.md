# chunked_delta_feedback@2: causal delta feedback factorization

## When to consider

When state key channels have different decays, keep their cumulative gate frames inside Q/K pairwise dots and the outgoing insertion. The same causal weighted feedback solve then supports residual-first or query-coefficient-first contraction order. Scalar gates are the equal-channel specialization, not an approximation chosen for arbitrary KDA input.

## Computation before and after

Weighted lower-triangular feedback solve followed by residual-first or query-coefficient-first contractions.

Use K-major H. Bind gate_geometry from the workload: scalar alpha repeated across key channels, or key_channel alpha_i,c. Define E_i,c=product_(u=0..i) alpha_u,c and D_ij,c=E_i,c/E_j,c. Form EK_i=k_i elementwise E_i, EQ_i=q_i elementwise E_i, KI_j=k_j elementwise reciprocal(E_j). For j<i set L_ij=beta_i*(EK_i dot KI_j); set its diagonal/upper entries zero. C_ij=EQ_i dot KI_j for j<=i, zero above. Solve (I+L)*T=diag(beta). residual_first: R=T*(V-EK*H_in), Y=EQ*H_in+C*R. coefficient_first: P=C*T, A=EQ-P*EK, Y=A*H_in+P*V. If outgoing state is requested, also obtain R and form H_out=diag(E_end)*H_in+sum_j (k_j elementwise D_end,j) outer R_j after all H_in readers finish. Frozen output-only omits state writes. Direct channelwise log differences can compute each pairwise coefficient; a factored GEMM binding stores separate E and reciprocal-E operands and must admit their intermediate range/rounding. Neutral padded tokens have alpha=ones, beta=0, k=q=v=0 and masked stores. Solver schedule, factor preparation and instruction layouts remain separate choices.

Recover the original computation: Expanding diag(alpha_i)*H_(i-1) gives channelwise D_ij on each prior insertion. Dotting by k_i yields the strictly lower feedback equation; its unique residuals are the sequential ones. Substituting those residuals gives both output orders and inclusive outgoing state. Repeating one scalar gate across all key channels recovers chunked_delta_feedback@1 exactly in real arithmetic.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| chunk_tokens | compile / software | [4, 8, 16, 64] | Short MTP native/staged widths4/8/16 or Hopper prefill width64. Other widths require a new binding. |
| evaluation_order | compile / software | ["residual_first", "coefficient_first"] | Residual-first exposes corrected value rows and outgoing state; coefficient-first reuses state/value-independent query coefficients and can omit state update in frozen verification. |
| gate_geometry | compile / boundary | ["scalar", "key_channel"] | Workload gate geometry: a scalar per token/head or one decay per state key channel. This is not a free optimizer replacement of channel gates by their mean. Bind from the workload; only equal channel values permit the scalar specialization. |

## Coupled constraints and scope

Extension of version1 retaining all scalar bindings. Concrete key-channel witness is KDA BT16, DK=DV128, SM100-family tcgen05 path, residual-first with BF16/FP16 rounded decay operands, FP32 prefix exchange and declared carried-state conversion. Other width/order/target combinations require implemented backend support rather than a cross-product claim. Featurewise scalar replacement, non-diagonal state transforms, multiple GDP substeps per real token and unrestricted reciprocal range are outside this version.

Scope predicate: `p.logical_tokens >= 1 and p.logical_tokens <= chunk_tokens and (p.delta_recurrence_matches_bound_gate_geometry_and_after_update_readout) and (p.positive_diagonal_decays_and_complete_chunk_frames) and (p.unit_lower_weighted_solve_implemented_and_residual_admitted) and (p.gate_frame_operands_and_all_contraction_rounding_stages_admitted) and (p.incoming_state_readers_complete_before_state_replacement) and (p.complete_chunks_or_neutral_padding_and_unique_output_ownership) and (p.selected_order_shape_dtype_and_target_supported)`.

- Precondition `pre_recurrence` (implementation): `p.delta_recurrence_matches_bound_gate_geometry_and_after_update_readout` — K-major Hbar_i=diag(alpha_i)*H_(i-1), r_i=beta_i*(v_i-k_i^T*Hbar_i), H_i=Hbar_i+k_i outer r_i; output q_i^T*H_i. Scalar geometry repeats one alpha over key channels; key_channel preserves every channel. Normalization/query scale resolve consistently before factorization.
- Precondition `pre_frames` (implementation): `p.positive_diagonal_decays_and_complete_chunk_frames` — E_i is the elementwise cumulative product of positive alpha through token i; D_ij=E_i/E_j channelwise. No sequence/head mixing. Scalar mode uses identical channel entries.
- Precondition `pre_solve` (implementation): `p.unit_lower_weighted_solve_implemented_and_residual_admitted` — A=I+L is unit lower triangular with L_ij=beta_i*sum_c k_i,c*D_ij,c*k_j,c strictly below diagonal. Solve A*T=diag(beta), including zero beta without division. Solve/casts meet the declared residual policy.
- Precondition `pre_numeric` (implementation): `p.gate_frame_operands_and_all_contraction_rounding_stages_admitted` — The chosen direct-difference or separately factored E/reciprocal-E operands remain in range at every stored precision. Pairwise dots, weighted solve, residual/coefficient products, state and output casts have an explicit admitted error policy. Finite final ratios do not certify intermediate reciprocal safety.
- Precondition `pre_boundary` (implementation): `p.incoming_state_readers_complete_before_state_replacement` — All outputs/residuals needing H_in capture that generation before H_out overwrites it. Optional frozen output-only execution leaves H_in unchanged. The next chunk waits for complete H_out.
- Precondition `pre_scope` (implementation): `p.complete_chunks_or_neutral_padding_and_unique_output_ownership` — Complete chunk coordinates, or a validated neutral extension alpha=1,beta=0,k=q=v=0 and masked stores, preserve valid rows. Each output/state shard has one owner and reductions include every key shard.
- Precondition `pre_backend` (implementation): `p.selected_order_shape_dtype_and_target_supported` — The selected computation order, width, dtype, state layout and target have an implemented binding. Output-only coefficient-first does not by itself produce an outgoing state.
- Constraint `c_shape` (implementation): `p.logical_tokens >= 1 and p.logical_tokens <= chunk_tokens` — One nonempty chunk with complete or neutral-padded valid-token extent.

## Local example and misuse

For a local instance of `map`, identify the object `causal_delta_feedback_factorization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/chunked_delta_feedback@2/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/chunked_delta_feedback@2/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

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

Small triangular systems are always invertible in exact arithmetic but can be ill-conditioned. Coefficient cancellation, BF16/FP16 solve storage, gate underflow and reassociation change native error. Key-channel inverse-prefix operands can overflow or underflow even when the mathematical pairwise ratio is finite; safe-gate clipping changes the gate contract and must be explicit.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
