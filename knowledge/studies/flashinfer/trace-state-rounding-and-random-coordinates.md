# Checkpoint rounding and stable random coordinates

When folding a recurrent history into a narrow checkpoint, separate three
contracts: the FP32 value being converted, the destination's random-carry law,
and each element's random address/epoch. Identical counters do not make two
state-fold algorithms bitwise equal, and a scalar stochastic mean does not
establish independent dithers across a state tensor.

[random_discarded_bit_rounding@2](delta/random_discarded_bit_rounding@2/template.yaml)
and [outline](delta/random_discarded_bit_rounding@2/proof.yaml) extend the
[retained QuACK version1](inputs/reuse-additions/random_discarded_bit_rounding@1/template.yaml)
with [outline](inputs/reuse-additions/random_discarded_bit_rounding@1/proof.yaml).
The BF16 branch remains16-bit carry; destination now determines discarded_bits.
FP16 adds13-bit carry plus exponent/subnormal/special handling. The old
carry/special/RNG obligations retain their purpose; new format/input/pair/native
conditions prevent treating FP16 as direct BF16 truncation. Owner files remain
unchanged; [reuse additions](inputs/reuse-additions.json) retains this input.

The concrete [retained source](inputs/sources.json) path is
`gdn_kernels/gdn_replay_mtp_umma.py`, helpers in `_umma_helpers.py`, plus
`include/flashinfer/mamba/conversion.cuh`. The UMMA body uses an FP16 checkpoint,
BF16 q/k/v/k-ring/output and BF16 or FP16 U ring, with FP32 accumulation. The
common checkpoint+rank-one journal and cursor math reuse
[low_rank_state_journal@1](delta/low_rank_state_journal@1/template.yaml) on its
T4/8 intersection. Native UMMA T1..8 is an additional explicit backend menu,
not a claim that version1's draft_tokens domain already includes those values.

| Concrete decision | Rule field or supported binding | Evidence |
|---|---|---|
| `gdn_replay_mtp_umma.py:1100-1205`: flush-only seed load, half-state FP32 RMW and packed final store | pre_input/pre_backend; destination=fp16, discarded_bits13 | Verify rows do not read the seed and do not write a folded checkpoint. State conversion happens after egp*S0+fold_F, not on every draft update |
| `_umma_helpers.py:303-326`: PTX cvt.rs.f16x2 invoked with hi then lo; low result uses bits12:0, high uses28:16 | pre_pair | A high/low operand swap without matching random fields changes individual outputs even though the word's marginal law may be unchanged |
| `conversion.cuh:184-251`: source software exponent preparation, random add/truncate, original Inf/NaN classification | pre_format, fp16_exponent_recipe | Normal and subnormal input domains use different integer preparations; guard the original input, not noise-perturbed classification |
| UMMA`:1122-1181`: offset=((slot*HV+head)*128+row)*128+half*64; Philox stream counters offset+8*s, four output words feed four pairs | pre_rng, rng_identity |128-bit counter=(offset_lo,offset_hi,0,0),64-bit seed key; component and low/high field select each element. Offset arithmetic is64-bit. Load16/32 regrouping preserves8-element logical chunks |
| `:1140-1186`: row/half LCG seed hash, four jump-ahead chains and word fold s^(s>>16) | declared joint law, not a Philox substitution | Four chains emit serial LCG word indices4*chunk+i. Correlations remain across words; a uniform marginal premise alone does not establish joint/conditional independence |
| `:1399-1420,1486-1524`: mode/target/seed validation and caller seed advancement | invocation epoch belongs to caller | Source .rs target menu is SM100a/103a/107a. Reusing the same seed and slot repeats bits; use a separate layer/step epoch when freshness is required. Unseeded LCG reads the clock and has no replay identity |

The authoritative [PTX conversion documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
defines stochastic conversion through random-bit addition to discarded mantissa
bits and documents the packed FP16 random-bit layout. The source helper's
operand reversal preserves low/high element identity. Native subnormal/special
byte equality with the software fallback remains a separate binding obligation.

## Exact software law and its finite domain

For finite normal FP16-range values,13 discarded FP32 significand bits give a
uniform random carry with exact away-from-zero probability residual/8192.
Overflow/special behavior is separate. In the software subnormal branch,
values are first truncated toward zero on a2^-37 magnitude grid, then rounded
stochastically to the FP16 subnormal grid2^-24. Therefore its mean equals that
prepared value. Preparation loses less than2^-37 magnitude; an FP32 value2^-38
always becomes zero. Do not claim unconditional unbiasedness for all FP32
inputs from this source algorithm or from the wrapper's informal comments.

The [focused check](checks/check_fp16_random_carry_and_rng_map.json) uses an
independent exact-rational FP16 grid oracle and all8192 random fields for82
normal mean cases and84 subnormal prepared-mean cases, plus20 special cases.
It checks16384 regrouped element addresses,64 row/half origins above32-bit
counter range, six CTA-head/request-order traversals,15 peeled-two-round
Philox cases and32 LCG jump-ahead cases. It proves the stated software law
and integer/address identities, not GPU conversion equality or RNG quality.
The helper's peeled path unconditionally prepares two rounds; host accepts
positive round counts, so native one-round support remains unestablished here
rather than being inferred from that host guard.

## Head grouping and pipeline binding

UMMA`:128-170` binds heads_per_cta to a divisor of value_heads/key_head, with
ratios1..4. It processes that group serially after a shared normalized-key/query
prologue; splitting to1/2 heads duplicates setup across more CTAs but shortens
the per-CTA head chain. This is a concrete instance of seed output_tiling@2:
executor=cta, index_map=grouped_heads, tile_m=heads_per_cta,tile_n1 over complete
head-subproblems. Independent head-state/output ownership supplies pre_indep;
only subgroup0 writes the shared key ring. Instruction/register/TMEM budgets
and backend support remain required. The source's batch thresholds select a
cost policy; this pass does not reuse its performance numbers as measurements.

The concrete async_transfer@1 binding uses mechanism=controller,
issue_point=after_slot_released,complete_point=barrier_wait,16384-byte state
halves and the supported128-byte TMA layout. Its readiness and
no_overwrite_in_flight obligations are realized by the following source events.
The fold overlaps TMA halves and next-head readout. `_rmw` publishes a completed
shared half only after TMEM load completion, state conversion and shared-view
fence; its mbarrier lets the TMA producer store/reuse that half. Source producer`:493-526` waits f_gdone for the old half, issues TMA store,
commits and waits for its shared-source reads before reloading the next head
into that half. Math `_head`:1202-1225 waits fold completion plus old readout
retirement (rd0 for half0,mma_done for half1) before in-place RMW. `_rmw`
fence+arrival publishes the new half for its store. This gives explicit
read-complete→modify-complete→store-read-complete→reload order. Native reader
layout/proxy visibility remain obligations. multi_buffer_rotation@2's actual
scope uses CTA-wide barriers, so it is not claimed as a direct binding of
these independently participating warp/mbarrier roles.

counter_indexed_mask_replay@1 supplies useful generator/coordinate/epoch
obligations but its program produces mask predicates and scaling, not FP16
conversions. This is an obligation-level reuse, not a full mask-Template binding.
The source trace thus records conversion law, RNG correlations/epochs, head
ownership and numeric/protocol boundaries separately before continuing to state
locality and training representatives.
