# Scalar delta feedback: residuals or query coefficients

When each insertion subtracts a prediction from the evolving state, its value
is not an independent outer-product input. A short chunk can expose matrix
contractions by solving its causal feedback system first. The new
[chunked_delta_feedback@1](delta/chunked_delta_feedback@1/template.yaml) and
[outline](delta/chunked_delta_feedback@1/proof.yaml) retain that system and choose
residual-first or query-coefficient-first evaluation.

Paths below refer to the retained [source](inputs/sources.json). Use K-major
H here; transpose the source's V-major storage. The recurrence decays first:
Hbar_i=alpha_i*H_(i-1), r_i=beta_i*(v_i-k_i^T*Hbar_i),
H_i=Hbar_i+k_i outer r_i. Readout is after the update, q_i^T*H_i.

For local E_i=product(alpha_0..alpha_i), D_ij=product(alpha_(j+1)..alpha_i),
the residual equation is
r_i+sum_(j<i) beta_i*D_ij*(k_i dot k_j)*r_j
=beta_i*(v_i-E_i*k_i^T*H_in).
Its coefficient matrix I+L is unit lower triangular. T solves
(I+L)*T=diag(beta); zero beta is legal and never requires inverse-beta division.
Define EK_i=E_i*k_i, EQ_i=E_i*q_i and causal
C_ij=D_ij*(q_i dot k_j), including the diagonal.

| Source decision / concrete binding | Rule field or gap | Evidence |
|---|---|---|
| `gdn_decode_bf16_wy_output_only.py:1260-1285`: M=-L below diagonal, causal QKT C including diagonal | `pre_recurrence`, `pre_solve`; beta scales the feedback equation row | Log-prefix differences form D directly, avoiding separate exp(G_i)*exp(-G_j) overflow; KKT/QKT and M are explicitly rounded at different stages |
| `:1298-1489`: solve T00 and T11 independently, Y=M10*T00, then solve T10=(I-M11)^-1*Y; final T is BF16 | Solver Realisation boundary, not a claimed full inverse | Two8-row blocks with diagonal-beta RHS. Eight lanes each retain one column in FP32 registers. Source header/Phase2 Neumann descriptions are stale |
| `:1493-1567`: P=C*T, A=EQ-P*EK; `:1860-1880`: output=A*H_in+P*V | `evaluation_order=coefficient_first`, chunk_tokens4/8/16 working specializations | P/A are independent of V and H_in. The native path is frozen output-only, K=V128, mandatory Q/K normalization, BF16 state/operands, SM90+ TMA path. It does not supply H_out |
| `delta_rule_dsl/delta_rule_sm90.py:300-343`: inverse then beta multiply by column | T=(I+L)^-1*diag(beta), not diag(beta)*(I+L)^-1 | Inverse storage is FP16, converted to selected operand type. Beta placement is part of the feedback equation, not an interchangeable scaling |
| Same file`:1095-1190`: incoming-state output, SK, V-SK, NewV=(V-SK)*T^T and output correction | `evaluation_order=residual_first`, chunk_tokens64 | Source V-major orientation transposes R=T*(V-EK*H_in). The first chunk without initial state omits its zero-state products |
| Same file`:1203-1235`: decay carried state by chunk coefficient then insert decayed NewV*K | `pre_boundary`: inclusive H_out | H_out=E_end*H_in+sum_j D_end,j*k_j outer r_j. Output/residual readers finish using old state before replacement |
| No-prepack wrapper`:2140-2160` and body`:1303-1388`: neutral short rows, native T4/8, staged other T, preserved sync topology | `pre_scope`, backend menu distinct from logical extent | Short rows must be a validated neutral extension with masked stores. The source's claim that arbitrary garbage times zero is harmless is not adopted: nonfinite garbage can contaminate an MMA |
| Output-only body`:1569-1573`: Q aliases later V scratch | `pre_boundary` plus scratch lifetime obligation | All Q readers finish before cp.async writes V into those bytes; old input lifetime extends beyond the thread issuing its own read |

The outgoing state formula is part of the recurrence factorization. Selecting
coefficient-first does not make the output-only source emit it: an implemented
state path must also form R or an equivalent state transform. A complete
coefficient-first plus outgoing-state binding is a proposed composition, not a
native claim for that file. Frozen output can compose with
[deferred_prefix_state_commit@1](delta/deferred_prefix_state_commit@1/template.yaml)
when the verify recipe, retained raw inputs and later accepted-prefix replay
have an admitted common state/readout policy.

The [exact check](checks/check_delta_feedback.json) compares an independent
serial decay/predict/correct recurrence against both contraction orders and
outgoing state in96 one-/two-chunk cases. It checks48 neutral-padded extents and
48 two8-row weighted solves, including zero beta, signed values, nonzero incoming
state, varying decay and tails. It does not establish native residual bounds,
scratch synchronization or arbitrary conditioning. A scalar counterexample
shows why omitting feedback changes final state from3 to6.

## Existing rules and solver reconciliation

[chunked_decayed_outer_product_state@1](../tilelang/delta/chunked_decayed_outer_product_state@1/template.yaml)
has independent insertion values and explicitly excludes feedback. Its local
causal contraction/carry pattern is reused conceptually, but its actual
`pre_recurrence` and program cannot supply this triangular producer.

[blocked_unit_lower_inverse@1](../tilelang/delta/blocked_unit_lower_inverse@1/template.yaml)
fixes four16-row blocks and a full64-row inverse. The SM90 prefill producer
`collective_inverse_hmma.py:378-410` uses local8-row inverses, then recursively
combines8→16→32→64 with completed-level barriers. It satisfies the same full
unit-lower inverse contract, while the recursive schedule/leaf ownership is an
additional Realisation choice. The output-only two8-row weighted solve has a
different RHS/result contract and cannot recover a full inverse by dividing
zero beta columns. Proposed extension: keep the old version addressable; add
explicit block hierarchy and RHS kind (identity versus diagonal weights), plus
per-level completion/cast obligations. Do not overwrite the other study's rule.
For now this study records exact bindings/checks and leaves solver schedule as
a named pool gap beneath the new feedback rule.

Featurewise gates put channel weights inside key contractions; multiple GDP
Householder substeps change token/update coordinates. Neither is silently
covered by the scalar, one-update-per-token candidate. Continue those designs
through the [recurrent inventory](recurrent-inventory.md).

## Key-channel extension: preserve the gate inside the dot

[chunked_delta_feedback@2](delta/chunked_delta_feedback@2/template.yaml) and
[outline](delta/chunked_delta_feedback@2/proof.yaml) retain version1 and add
`gate_geometry=key_channel`, bound from the workload. Now E_i and D_ij are
vectors over state key channels. EK_i=k_i elementwise E_i,
EQ_i=q_i elementwise E_i, KI_j=k_j elementwise reciprocal(E_j).
L_ij=beta_i*(EK_i dot KI_j) below the diagonal and
C_ij=EQ_i dot KI_j on/below it. The same weighted solve and two output equations
then apply. Outgoing state uses diag(E_end)*H_in and insertion vectors
k_j elementwise(E_end/E_j). Scalar gates are exactly equal-channel vectors.

| Concrete KDA source decision | Binding / consequence |
|---|---|
| `kda_kernels/kda_chunked_bt16.py:1635-1669`: per-key-channel log prefix, exp2 and FP32 exchange; tail gates contribute zero | `gate_geometry=key_channel`; chunk-local frame and neutral tail, not one per-token mean gate |
| `:1700-1819,1883-1928`: complete Q/K L2 normalization, K*E, K/E, K*(E_end/E), Q*E with operand casts | `pre_numeric`; individual transformed operands must remain representable. Computing a bounded final ratio alone does not certify the separate reciprocal operand |
| `:2139-2270`: ldmatrix loads SW128 decay LHS and token-major inverse-key RHS; two m16n8k16 N fragments accumulate KK over key blocks | Concrete NVIDIA producer of EK_i dot KI_j; not only the Python reference. First half-DK consumption waits for the matching early publication, while the second half is still staged |
| `:2048-2100` beta/strict-lower construction and `super_mma_qk_causal_value` inclusive mask | Same beta row and causal-diagonal conventions as version1; solve/inverse precision remains a separate obligation |
| `:1238-1278` reference documents native numerical stages | R from beta*(V-EK*rounded(H)), inverse, rounded update, EQ*rounded(H)+C*rounded(R), then channelwise outgoing state. Real algebra does not erase these checkpoints |

The extended [exact check](checks/check_delta_feedback.json) adds16 independent
channelwise recurrence comparisons and16 equal-channel reductions to version1.
It also demonstrates that replacing gates(1/2,1/4) with scalar1/2 changes a
readout from3/4 to1 even with no insertion. Coefficient-first feature-gated
algebra is inferred from the same system; the concrete BT16 witness here uses
residual-first. Version2 does not claim an implemented cross-product of every
gate geometry, chunk width, order and target.

The same source`:6340-6378` additionally exposes a distinct preparation/chain
schedule: independent chunks prepare W=K_restore^T*T and P=C*T, while a later
serial state chain computes X=EK*rounded(H), updates
H_next=diag(E_end)*H+W*(V-X), and emits EQ*rounded(H)+P*(V-X).
Workspace stores permuted factors to match the later TMA images and retains
original V rather than materializing value-dependent U/O factors. This is not
full parallel state propagation: chunk preparation is independent, but the
state chain remains ordered. Its concrete publication/layout and amortization
choices are the next scheduling trace; the factor equations alone do not close
that design.
