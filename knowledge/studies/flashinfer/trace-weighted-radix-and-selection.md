# Weighted radix boundaries and exact selection reuse

When a cutoff is defined by probability mass, radix bins need **mass** as well as
counts. AIR top-p introduces
[mass_rank_radix_boundary@1](delta/mass_rank_radix_boundary@1/template.yaml) and
its [outline](delta/mass_rank_radix_boundary@1/proof.yaml). Keep that weighted
crossing separate from count-ranked top-k and from identity selection inside a
tied boundary value. Paths below refer to the retained [tree](inputs/sources.json).

Route: `flashinfer/sampling.py:2090-2176` -> `csrc/renorm.cu:49-56` ->
`include/flashinfer/air_top_p.cuh:493-528`. Registration instantiates FP32 only,
with float-atomic or deterministic integer-histogram paths. The generic header's
16-bit key schedule is an inferred logical binding, not registered native support.

| Decision and concrete source binding | Rule field / reuse boundary | Evidence |
|---|---|---|
| `air_top_p.cuh:78-108`: inverted CUB sortable key,11-bit digits | `key_bits=32`:11/11/10. Inferred16-bit binding:11/5 | Descending values become ascending bucket IDs; representation bits, not optimizer choice, fix the last digit width |
| `:229-282`: matching-prefix occurrence adds its value and count to separate histograms | `pre_partition`, `pre_mass` | Count describes presence/next active length/capacity; it cannot substitute for weighted cumulative mass |
| `:284-405`: first pass target=p*full_mass; later choose first occupied cumulative>= residual, subtract earlier-bin mass, append digit | New weighted `program` | Residual target travels within the chosen prefix. All unresolved occurrences retain original value/multiplicity |
| `:110-136,284-318`: two bounded value banks; omit compaction when incoming active count exceeds capacity; reread original row when no complete bank exists | `candidate_access=compact_if_fits`, `pre_capacity`, `pre_cache` | First pass has no compact output. A saved bank contains the incoming prefix before that pass extends it, so later readers still filter the newer prefix. Atomic order may change storage order, not multiplicity |
| `:434-455`: rescan original coordinates, keep value>=decoded threshold, normalize complete retained mass | `pre_apply` | Entire equal-value cohort stays. Source uses norm1 when retained mass<=1e-8; that branch is outside this normalized candidate |
| `:137-194,326-349`: deterministic mode adds mantissa bits plus companion counts, reconstructs bucket mass from one shared exponent | Native sum codec is a separate pending extraction/binding | Initial11-bit FP32 buckets share the exponent; later prefixes retain it. Integer accumulation/reconstruction is reproducible only with exact counts/no overflow and its specified conversion policy. It is not proof of correctly rounded real summation |
| `:318-405`: last producer CTA folds completed histograms; host launches dependent passes on one stream | `pre_lifetime` | Global histogram visibility, last-block counter census/reset and scratch retirement remain native obligations; no spinning grid barrier is introduced |

The [exact-rational check](checks/check_mass_rank_radix.json) compares13440 radix
runs with a sorted weighted reference, plus32 low-mantissa cases that exercise
the later digits. Both schedules and capacities0/1/2/4 preserve exact thresholds,
full cohorts and normalized output through compact/rescan transitions. These are
logical access counts, not performance measurements. Values[8,1,1] at half mass
have boundary8; a second-by-count selector has boundary1. At values
[5/8,1/8,1/8,1/8] and fraction3/4, the whole1/8 boundary cohort is retained.

Fixed and variable-length top-k have concrete existing-rule dispositions:

| Source design | Actual reuse / gap | Evidence |
|---|---|---|
| `topk.cuh:2288-2349`: FP32 coarse key rounds to FP16 then takes its high byte, but refinement reads full FP32 sortable keys | Monotone coarse filtering, followed by full-key radix; not quantizing the actual selected scores | Coarse collisions/FP16 overflow widen a bucket rather than authorize dropping true winners.16-bit source types use high-byte coarse plus low-byte refine |
| `:2470-2555,2840-2916`: complete scans, threshold refinements, overflow reset and full-scan rebuild | `count_certified_topk_filter@1` supplies completeness/capacity/fallback obligations, but its one DRAM list is not this staged SMEM resource layout | On overflow source resets old partial winner state before rebuilding, avoiding stale list fragments. Native tiers/maps remain explicit |
| `:2616-2650`: Small uses ascending contiguous tie collection, Large descending, None thread-strided | `radix_threshold_selection@2(tie_policy=smallest_index,digit_bits=8,key_bits=16/32)` supplies Small's threshold/tie contract after its equal-value/key premise; `radix_boundary_topk@1` supplies largest-ID membership for Large, not its native contiguous-collect implementation | Source sortable keys distinguish signed zero unless separately canonicalized. Whole value cohorts, exact-k ID cuts and unordered deterministic presentation are different contracts |
| `flashinfer/topk.py:1046-1090`: optional selected-value sort then gather selected IDs; cluster and radix paths have distinct sort implementations | Output presentation is separate from exact membership | Stable=deterministic is requested only in the relevant fallback. `sorted=True` alone does not establish one universal equal-value ID order |
| `topk_varlen/kernels/filtered_topk_util.py:306-350`: REREAD_ALWAYS, REREAD, full spill, bounded spill then reread | Existing full-count/fallback obligations apply to exact modes; SMEM/GMEM tier layout needs native binding | TRUNCATE explicitly drops overflow candidates and may emit fewer than k; it is outside an exact candidate certificate, even though refinement remains internally consistent |
| `filtered_topk_decode.py:504-543`: row length from seqlen/next_n, clamped physical width; whole-row-fit path; per-row dynamic CTA counts | Preserve semantic row/length/generation before thresholding | Invalid/padded output positions and independently replicated early-exit CTAs must not be treated as live score candidates |

No extra rule is needed for exact byte-radix threshold/tie reuse on its admitted
key domain. Coarse FP16 bucketing, deterministic native mass reconstruction,
SMEM/spill layout and cluster/varlen role maps remain named refinements rather
than silently extending those versions. Finish the remaining histogram codec,
softmax and logit-processing rows in the [sampling inventory](sampling-inventory.md),
then continue the recurrent and distributed families.
