# B12x dual-code probability contraction and selection relatives

Source paths refer to the retained [tree](inputs/sources.json), primarily
`flashinfer/experimental/b12x/attention/`. This pass resolves the dual-code
probability gap from [stream compression](trace-b12x-stream-compression.md)
and gives concrete dispositions for selector/contiguous-attention relatives.

## A residual code pays for a second contraction

Retrieval cue: a single E4M3 P operand loses important PV accuracy, while a
second narrow matrix product is affordable. New
[residual_code_contraction@1](delta/residual_code_contraction@1/template.yaml)
and its [proof](delta/residual_code_contraction@1/proof.yaml) expose that choice.

| Source choice | Candidate field / obligation | Binding |
|---|---|---|
| `dense_mla/_forward.py:255-276` | `pre_mass`, common online frame | Apply returned warp-to-row factors to all four local probabilities before passing them to PV. Encoding probabilities in their separate warp frames would mismatch the row accumulator. |
| `dense_mla/_math.py:465-526` | `pre_scale` | Reduce a per-head maximum across the64-key chunk, floor it at1e-10, form s=max/448 and an FP32 reciprocal. Scale choice belongs to this row/tile and its current online frame, not a fixed tensor gain. |
| Same file :383-394 | `encoding=high_and_residual`, `residual_generation` | Clamp scaled input to±448; high=E4M3(x), residual=FP32(x-decode(high)), low=E4M3(clamped residual). Low may be **negative** although the original probabilities are nonnegative. |
| Same file :527-564 | `component_coverage`, paired storage | High occupies logical matrix rows0..7 and low rows8..15, with weight stride64+16. Both represent the same head/key positions. Publication barrier precedes matrix reads; the storage rows are components, not extra attention heads/support. |
| Same file :570-606 | `arithmetic_order` | m16n8k32 contractions accumulate FP32 high and low products; output adds x0+x2 or x1+x3, then multiplies their shared scale once. There is no reconstruction into E4M3 before MMA. Exact ldmatrix/native fragment identity remains a realization obligation. |
| Same file :263-379,710-800 | `mass_frame` | Online mass is formed before P encoding. Final output normalizes by that original mass and applies V descale. Split partials store normalized O and base2 LSE; unsplit final LSE is natural-log. Empty rows use the declared sentinel/zero inverse. |

In real arithmetic s*(H*V+L*V) equals (s*(H+L))*V. Native FP32 accumulation
association is part of the numerical contract; neither form proves equality
to original unquantized P. The single-code candidate projection removes the
low contraction as an inferred work/accuracy choice, requiring its own backend.

rounding_residual_checkpoint@1 already defines a paired arithmetic checkpoint,
but its `pre_consumer` reconstructs in the accumulator **before** downstream
multiplication. It does not supply the two independent contraction streams,
component-row combination or separate original-mass denominator. This new rule
preserves that distinction. selected_value_residual_replay@1 instead repairs a
chosen subset of V rows; tile_anchor_probability_encoding@1 instead broadcasts
one local anchor scale across K16 E2M1 groups. No candidate is silently rebound.

The [logical check](checks/check_residual_code_contraction.json) uses a finite
E4M3 RN-even codebook and signed residual witnesses. It distinguishes high-only
projection, wrong low sign, component mispairing, code-derived/duplicate mass,
residual underflow and rounding a reconstructed pair back to E4M3. A scalar
FP32 witness shows separate versus reconstructed-first contraction association
can differ. It does not assert native MMA association or a universal accuracy
improvement; those are the rule's explicit arithmetic obligations.

## Stable QSA winners and DSA overflow paths

| Representative | Existing rule / boundary | Disposition |
|---|---|---|
| `qsa/_stable_select_cute.py:25-86` | Threshold certificate premise of radix_threshold_selection@2 | Recover cutoff as the minimum of provisional top values. Exact final membership requires that provisional set to provide the actual kth threshold. This source is a finalizer, not independent proof of the upstream threshold producer. |
| Same file :89-153 | radix_threshold_selection@2 `tie_policy=smallest_index` / ordered emission obligation | Eight warps own consecutive input spans. Per-warp greater/equal counts plus lane prefix ranks emit all greater scores and exactly the remaining equal-score quota. Ties choose **earlier concatenated candidate columns**. Later ID remapping does not turn this into lowest-final-ID selection. |
| Same file :117-151 | Candidate identity map | Previous winners occupy a carry prefix; they recover prior_ids. New candidates map by group_offset+column-carry. Logical candidate position, persistent output identity and presentation order remain separate. |
| `dsa_indexer/tiled_topk.py:45-68,317-350,1030,1190-1209` | Coarse radix narrowing plus complete-key refinement; count_certified_topk_filter@1's no-drop principle | Topk512/1024/2048 have bounded candidate buffers. A coarse10-bit histogram is a preliminary narrowing, not one legal digit_bits binding of radix_threshold_selection@2, whose refinement menu is4/8. On overflow, exact rescan avoids dropped winners; output buffers are initialized before fallback. |

The QSA finalizer supplies a concrete ordered threshold-tie emitter missing
from the earlier arbitrary-tie examples. Its comparison domain excludes NaNs
under the bound score contract; score arithmetic and sign/zero key classes
must match the threshold producer. It does not inherit radix_boundary_topk@1's
**larger original ID** tie order merely because both routes use radix selection.
The DSA comments identify a source lineage, not evidence that an uninspected
sglang body has already been studied in this catalog.

## Contiguous/paged target specialization

`_shared/contiguous/forward.py` retains the SM120 organization: a160-thread CTA
with four compute warps and one TMA producer, TMA outer pipeline and warp-level
mma.sync math. `warp_mma_gemm`/:94-158 supplies shared/register operand lowering;
`_get_tiled_mma`/:420-431 selects QK/PV warp MMA atoms. This is a concrete
producer-consumer/tiling realization, not an SM100 TMEM/UMMA path selected by
architecture brand alone. Fixed and packed-varlen APIs route through that
shared computation with their own sequence/mask metadata.

`paged/forward_paged.py` additionally has page-aware TMA source/plane helpers,
while query/head grouping and complete-row masks remain source bindings.
Existing indexed_gather@1, grouped_head_row_packing@1, boundary_predication@1
and online_rescaled_reduction@2 supply the logical composition. Exact TMA plane
packing, stage release frontiers and provider/model-specific composite paths
remain named realization work; no duplicate Template is added for a namespace.

## Next full-library work

This resolves the dual-P contraction choice and records concrete stable-tie,
metadata-ID and target-specialization reuse. B12x/model attention retains named
native maps and common orthogonal-preconditioning/ExpCast obligations in the
inventory. Move next to GEMM/grouped GEMM for full-library design coverage,
then return to final attention/training reconciliation with those gaps visible.
The library remains active until every distinct family has a source-backed
disposition; these examples do not close the catalog or the library.
