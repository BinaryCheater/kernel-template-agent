# GDP: real-token boundaries and ordered state substeps

When a real token applies several successive key corrections before emitting
one output, keep the real-token boundary distinct from the update coordinate.
The retained [source](inputs/sources.json) contains a concrete two-update
representative, not only a provider signature. The coordinate adapter and concrete decode/prefill bindings below are recorded;
its approximate-history split has a separate linked disposition.

`gdp_prefill.py:45-72` defines one scalar decay per real token, n ordered
beta*(v-S*k) rank-one updates, then one readout. Packed cu_seqlens count real
tokens, while K/V/beta rows count n times as many updates. n=1 recovers scalar
GDN. This is workload semantics; changing n or moving decay between updates is
not an optimizer freedom.

`cake_gdp_backend/cake_jit.py` MODULES selects a concrete `lanerow` runtime-token
route with num_householder2, max_t_steps16, variant l4r4 and fused Q/K L2 norm.
Its body is
`cake_gdp_backend/csrc/cake_gdp_3342c82179fa758b0570_kernel.cu`:

| Source decision | Concrete mechanism / extraction consequence |
|---|---|
| `:176-200`: CTA index resolves batch/head; u=tid%4, group=tid/4, row0=4*group; steps=2*num_tokens | One64-thread CTA owns a64x128 FP32 state. Four adjacent lanes jointly own four value rows; each lane retains32 keys per row,128 state scalars. Sixteen groups cover all64 rows |
| `:746-750,1202`: one alpha at real token t, then first update index2*t | Decay applies once before the first correction, not once per expanded update |
| `:1277,1450-1485`: beta for first update, lane-local state/key prediction, XOR1/2 fold over each four-lane group, residual then rank-one FMA | Complete key prediction must precede its correction. This is evolving-state feedback, not two independent inserts |
| `:2793,2868,3067`: second update index2*t+1 and its residual from the already corrected state | Substeps retain order. Coalescing into a chunk solve must include feedback from substep0 to1 |
| `:4647-5255`: query readout/output after both corrections; `:5256-5267`: optional snapshot slot from sSsm[t] | Outputs/checkpoints are indexed by real token and observe its final substep state |
| `:6354-6356,6810,8401,10255`: no-snapshot loop repeats same real-token/first-update/second-update/readout organization | Snapshot selection changes stores, not the state recurrence or real-token readout boundary |
| `:10866-10867`: valid destination and disable_state_update control final state write | Frozen output and per-token snapshots have separate write policies. Caller slot-generation/nonalias obligations still require host binding |

A useful composition expands update coordinate m=t*n+j, sets alpha_m=alpha_t
only at j0 and1 thereafter, and samples the expanded recurrence only after
m=t*n+n-1. This would reuse the weighted feedback equations from
[chunked_delta_feedback@2](delta/chunked_delta_feedback@2/template.yaml) on
expanded coordinates, but its current real-token/readout and compiled-width
scope do not supply this adapter automatically. Required extraction: explicit
real/update prefix maps, readout/checkpoint gather boundaries, neutral substeps,
and acceptance lengths in real-token units. The selected implementation has
n2; registry relatives admit n1..4 with target/variant menus, not arbitrary n.

The following sections supply the coordinate adapter, proof and concrete
prefill binding. Their scope remains separate from approximate-history splitting.

## Extracted coordinate adapter and concrete prefill

[grouped_state_substeps@1](delta/grouped_state_substeps@1/template.yaml) and
[outline](delta/grouped_state_substeps@1/proof.yaml) now supply the adapter.
`substeps_per_token` is workload-fixed; `update_grouping` chooses the direct
token loop or expanded chunks with unchanged real-token observations.

The selected plain, zero-seed SM90 V6 prefill body is
`cake_gdp_backend/csrc/cake_gdp_0de8fd0ee57885709a1c_kernel.cu`, registered as
`body=prefill_sm90_v6` in cake_jit.MODULES. Its initial-state/pool relatives add
seed/slot contracts; they do not change substep coordinates.

| Source decision | Binding / consequence |
|---|---|
| `:387-398`: persistent tile walk over sequence/head, expanded length=L*n, ceil(expanded_length/64) chunks | `update_grouping=expanded_chunks`; one sequential state chain per sequence/head,64 internal updates per chunk. This is not64 real tokens when n>1 |
| `:543-558`: chunk index decomposes into groups of n chunks sharing64 real query rows; cp_* maps the real-token endpoints in the current expanded chunk | `pre_coordinates`, `pre_readouts`; a chunk can end inside a token when n3 |
| `:2474-2494`: rr=n*(qrow+1)-1-bp_roff and causal keep col<=rr; log-prefix difference scales QK | Rectangular real-Q versus expanded-K support includes all corrections before the real readout. The implementation reuses64 real Q rows across n expanded chunks instead of materializing an expanded query tensor |
| `:2520-2553`: strictly lower KKT*beta_row*exp(prefix_row-prefix_col) stored FP16; `:2562-2610`: unit diagonal/zero upper,8-row elimination and inverse stores | Scalar weighted-feedback producer from chunked_delta_feedback@2 after the new coordinate adapter. The recursive solve/inverse schedule remains the previously named solver Realisation extension, not supplied by a four16-row binding unchanged |
| `:1027-1047,1080-1115`: incoming query-state contribution, V-SK residual and per-chunk gate coefficient | Match original state dependency and declared residual/state arithmetic. BF16 subtract and FP16 solve storage are separate numeric checkpoints |
| `:3430-3468`: subtract the prefix before the current expanded row, then repeat real-token cumulative gate values at their substeps | `pre_frames`; a mid-token chunk start does not apply its real-token decay again. Gate construction uses log2(alpha+1e-10), approximate exp2 and native rounding, which require their own admitted recipe |

The [focused check](checks/check_grouped_state_substeps.json) compares56 nested
token/substep recurrences against an independent affine expanded stream and
1960 accepted-prefix selections. It covers14 chunk starts inside tokens,
1586 source cp_* chunk maps and78128 rectangular query-mask rows. It uses exact
rationals and labeled coordinates, with explicit repeated-decay, early-readout
and accepted-token/update-count counterexamples. Those resolve the adapter's
boundary uncertainty; native solve precision and snapshot-slot retirement
remain their source/Realisation obligations.

Composition: grouped_state_substeps@1 gives the expanded update/gate frames and
real readout coordinates; chunked_delta_feedback@2 supplies the strictly lower
feedback equations and residual/output/state contractions; the source-specific
64-update inverse and rectangular Q schedule implement those equations. This
adds a concrete scalar K128,V64 GDP binding and does not claim that every
previous GDN compiled-width/target option accepts GDP tensors directly.

The GDP substep/serial-decode and V6 expanded prefill designs are now traced.
Its merged-body gate-saturation split is a separate approximate-history design:
independent items replay a short warm-up from zero and write only their owned
interval, skipping planned cuts whose recent gate product does not pass the
threshold. Its control/error contract is now recorded in the
[bounded warm-up trace](trace-bounded-history-warmup.md).
