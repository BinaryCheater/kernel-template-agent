# VC attention: tile means or selected value residual replays

Source paths below are relative to the retained [tree](inputs/sources.json).
Route: `flashinfer.attention.prims_ts.BatchPrefillTSWrapper` -> context task
builder `flashinfer/attention/prims_ts/kernels/fmha_context/fmha_tasks.py` ->
resources/matrix instructions in `fmha_resources.py`. Operand preparation is
retained CuTe DSL in `flashinfer/attention/prims_ts/vc_attention/prepare.py`;
it is source material, not an installed construction dependency.

The retrieval cue is a dense attention V tensor whose large common offsets or
few large quantization errors dominate low-precision PV. The source offers two
mutually exclusive correction representations, each with a distinct mass rule:

| Need | Candidate | Choice and invariant |
|---|---|---|
| Reduce residual range within each128-key V tile | [tile_value_center_restoration@1](delta/tile_value_center_restoration@1/template.yaml), [proof](delta/tile_value_center_restoration@1/proof.yaml) | Store centered narrow V; add each tile center times its **same cast-P mass**. Carry pending masses in the live online reference until restoration. |
| Correct only a small high-error subset of V rows | [selected_value_residual_replay@1](delta/selected_value_residual_replay@1/template.yaml), [proof](delta/selected_value_residual_replay@1/proof.yaml) | Replay copied key identities with value residual codes; include these extra PV terms but exclude their mass from normalization. |

These are value-side corrections. softmax_key_centering@1 removes a common key
translation before score formation; query_center_residual_scores@1 recovers
key-dependent query-center scores before softmax. Neither supplies weighted
V centers or numerator-only replay with original-support mass.

## Tile centers: pending statistics are part of the online state

Source package README:610-673 scopes VC to dense fixed-length D128 MHA,
BF16 or E4M3 Q/K, E4M3 V and FP16/BF16 output; LSE and sinks are unsupported.
`VCAttentionConfig` forbids positive repair_tiles together with demean.

| Source decision | Candidate field / obligation | Binding |
|---|---|---|
| `vc_attention/prepare.py:470-595` | `pre_center`, `pre_scale` | Re-read permuted V, subtract its128-token tile mean, divide residuals by per-channel FP32 V scale and encode E4M3. The mean operand is BF16 in the same code units. This is a requested quantized reconstruction, not a lossless V representation. |
| `kernels/fmha_context/fmha_resources.py:5666-5707` | `restore_schedule=grouped16`, `pending_frame` | When reference changes, scale every pending tile mass and the previous tile sum by exp2(old-new). Insert the preceding tile's mass into its modulo16 slot; publish full groups and clear retired slots. Row mass follows the same reference change. |
| Same file :4832-4866; `vc_resources.py:50-95` | `cast_and_scale`, group operand identity | Split each mass into BF16 hi/lo at K slots2i/2i+1, duplicate the tile center for those slots. Two K16 operands cover16 tiles: operand o covers tiles8o..8o+7. Hi/lo splitting and MMA error are admitted arithmetic, not exact FP32 recovery. |
| Same file :6864-6904; `fmha_tasks.py:1474-1498,1680-1718` | `group_protocol`, delayed restoration | Mean MMA follows the PV of the tile after a completed group. The published masses and centers must share that issue point's reference. Correction must not treat newly inserted mean terms as old-frame contributions. |
| `fmha_resources.py:5742-5763`; `fmha_tasks.py:2786-2807` | Final partial-group flush | The final group has no following loop tile; a separate mass publication and final mean MMA recover it before output. A multiple-of16 steady-state example does not cover this tail. |

For a fixed set of cast weights w, sum_j w_j*(R_j+mu_tile(j)) equals
sum_j w_j*R_j + sum_t mass_t*mu_t. Native center/residual casts replace V
with an admitted reconstructed V. The denominator and center mass must use
the same declared cast-P convention as residual PV; substituting the ideal
precast softmax mass is another operator. Per-tile restoration in the candidate
is an algebraic alternative, while grouped16 is the source realization.

## Selected residual rows: duplicate scores do not duplicate mass

| Source decision | Candidate field / obligation | Binding |
|---|---|---|
| `vc_attention/prepare.py:570-612` | `repair_tokens`, `pre_residual` | Rank per-token sum_d((V/s-base_E4M3)*s)^2, using the actual stored base quantizer. Selection ignores query weights; no per-query optimality or monotone accuracy guarantee is implied. |
| `VcSelectPass` :938-1076 | `pre_selection` | Select high-energy original token IDs. The selected list must be unique and in range; detailed radix/tie order remains a native selector binding rather than a new attention identity. |
| `VcRepairRowsPass` :1078-1184 | `residual_identity`, `pre_score` | Copy each selected K and encode E4M3(V/s-decode(base_code)), zero-filling unused repair slots. Repair values are residuals, not a second copy of V. |
| `_Shape` :265-295 and base V pass :570-578 | `mass_roles` | Full original tiles stay first, repair tiles occupy the next full128-row regions, and the original partial tail shifts by128*repair_tiles. Original tail rows still contribute mass once. |
| `VcQKPass` :733-814,870-905 | `weight_replay` | QK8 independently rotates/quantizes repair-key blocks with their own block scales. A copied original K does not prove identical quantized scores or ExpCast weights. Candidate requires admitted replay error; exact common-weight identity is conditional. |
| `fmha_resources.py:5711-5738`; `fmha_tasks.py:2798-2800` | `pre_mass`, `pre_frame` | Carry both all-row and kept-original mass, scale both on reference changes, and choose kept mass for final output. Non-tail replay tiles add zero kept mass; the separately handled moved original tail retains its mass. |

In an ideal common frame, one selected key contributes w*b + w*r to the
numerator but only w to the denominator. Counting repair mass would divide
by a different support distribution. Using residuals from an old base-code
or scale generation also changes the reconstructed value.

## Probability and preprocessing boundaries

`helpers.py:130-185` contains ExpCast: derive approximate E4M3 code coordinates
from scaled logits, clamp/round through packed FP16 arithmetic, gather bytes,
then sum the **decoded E4M3 values**. `vc_resources.py` fixes gain256 and code
bias with beta=-0.35. This is not native exp2 followed by an ordinary E4M3 cast.
The two new candidates consume its declared weights; instruction-level code
range, packed half-sum rounding and approximation error are separate realization
obligations. No generic approximate-math rule is assumed to prove this codec.

`vc_attention/quantization.py:132-214` groups similar V rows with a per-head
k-means and stable label sort. K and V share the resulting permutation;
normalization/output is invariant to that common permutation under dense
support and admitted arithmetic. A tile center is useful only insofar as this
partition reduces residual range; clustering quality is a source heuristic.
QK8 at :277-328 additionally centers K and applies the same normalized
Hadamard128 to Q/K before per-block E4M3 encoding (`prepare.py:_hadamard128_row`).
The real dot-product invariance of a common orthogonal transform does not make
its subsequent block quantization exact. A selectable orthogonal-preconditioning
Template remains a named pool gap; it is not hidden inside value restoration.

The V-Smooth cadence in the preprocessor is a model schedule: update grouping
and demeaning early, retain permutation later. A stale permutation remains a
valid dense permutation, but stale codes/scales/means do not represent new V.
Prepared static geometry and task rings do not authorize reusing value-dependent
statistics from another generation.

## Logical evidence and continuation

The [logical check](checks/check_value_compensation.json) covers mean restoration
for1/15/16/17/33 tiles, reference changes while means are pending, final-group
flush, cast-P versus ideal mass, residual generation, duplicate repair mass,
repeated selection, score-weight mismatch, and a130-row original domain with
one repair tile before its shifted two-row tail. It uses dyadic weights and
simulated narrow value codebooks to isolate these invariants; native ExpCast,
BF16 split operands, UMMA and task publication remain separate obligations.

Continue prims-TS runtime split/CLC and direct sparse semantic-block/page mapping,
then reconcile B12x/model and training relatives. The full library inventory
remains active; this pass does not close attention or the library.
