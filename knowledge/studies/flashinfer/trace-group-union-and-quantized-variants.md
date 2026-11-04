# Group union masks and quantized attention relatives

Source paths below refer to the retained [tree](inputs/sources.json).
The new [grouped_sparse_union_masks@1](delta/grouped_sparse_union_masks@1/template.yaml)
and its [proof](delta/grouped_sparse_union_masks@1/proof.yaml) capture a
shared-support choice, distinct from occurrence-preserving training compaction.

## Neighboring rows share payloads, not their normalizers

`flashinfer/token_sparse.py:166-279` implements a three-step group transform:

| Source decision | Candidate field / binding | Evidence and boundary |
|---|---|---|
| :207-221: OR every valid logical key into a shared universe bitmap | `pre_set`, `pre_payload`, `pre_valid` | Group rows share one request/page table. Uint32 bound comparison skips negative/out-of-range positions. Bitmap OR collapses duplicate occurrences; this is set support, not multiplicity-weighted attention. |
| :223-246: prefix-scan popcounts and emit union slots by logical-key rank | `union_rank`, `payload_identity` | The page translation is PAGE64; physical cache slots need not be ascending after logical keys are sorted. Store the union size, not just a capacity bound. |
| :248-278: each row maps selected keys to union columns and sets its bitmask | `membership`, `pre_masks` | Prefix counts below each128-key tile plus preceding-word/lower-bit popcounts define the same rank. Mask each row before its own attention normalization. |
| Same loop processes team+4*i rows and clears staging after writes | `pre_lifetime`, `staging_and_consumer` | Four128-thread teams share one512-thread CTA; group rows can exceed four. Team barriers join writers/readers and reset the stage before its next row. |
| :294-354: equal consecutive groups, key capacity, output union/mask capacity and shared-memory checks | `group_rows`, `c_output`, `c_scratch`, `pre_maps` | Support width is multiple512, mask stores use complete128-column tiles. Only words through ceil(U/128) and union slots below U are written; other capacity tails remain outside the valid output region. Negative output IDs skip stores while participating in synchronization. |

The [logical check](checks/check_group_union_masks.json) uses five rows with
four reusable teams, union keys[1,2,33,65] and swapped pages[9,3], giving
physical slots[577,578,609,193]. Masks[9,10,0,4,2] recover each original set.
Sorting physical slots without permuting masks breaks identity; retaining
team0's previous mask contaminates its fifth row. A row selecting only33
returns value6 while normalizing over the whole union returns4.5.

Set versus occurrence semantics is consequential: row[65,1,1] with values
[9,0,0] has occurrence mean3 but set mean4.5. Therefore this new rule requires
set semantics; it does not relax stable_sparse_support_compaction@1's duplicate
preservation. shared_index_union@1 serves two sorted unique lists and returns
split-local incidence slots, not arbitrary row membership masks for a dense
attention consumer. Native bitmap/scan mapping, publication and the selected
masked consumer remain implementation obligations.

`token_sparse.py:35-89` has a separate direct slot-list builder for token-sparse
TRTLLM attention: it copies each token's mapped list into K and V request rows,
with requests_per_token copies and valid-prefix counts. That path preserves
list occurrences and requires valid entries first; union-mask preparation is
an alternative contract, not a transparent replacement for every such caller.

## Deterministic DSA indexer is a score-arithmetic contract

`experimental/cake_dsa_indexer_det/README.md:61-95` fixes score evaluation:
BF16 products accumulated in a fixed K-step order, four interleaved FP32 FMA
head-weight chains and a fixed final combination. Positive logit scale moves
into signed head weights under nonnegative_scale_hoisting@1 with explicitly
admitted rounding. Fixed arithmetic across candidate, fallback and program
variants is the premise needed by count_certified_topk_filter@1's unchanged
score requirement. Deterministic **membership** is not obtained merely by using
an exact local selector if the score producer changes its reduction order.

The source :115-125 describes an exact certified candidate gate with exact
streaming retry when its threshold is too aggressive. It shares the certificate
concept, but the SM120 candidate-count buffer implementation is not inferred
from this package's description. Its finalize implementations use bitmap prefix
ranks or CUB radix sorting to present selected IDs ascending; presentation
remains separate from larger-ID tie ranking. Detailed in-kernel retry admission,
score-bit equality and native collective order remain backend obligations. No
new Template is added just for a fixed numeric evaluation recipe.

## Quantized dense and gated sparse relatives

| Route and source evidence | Existing choice / remaining gap | Disposition |
|---|---|---|
| `flashinfer/qsa.py:19-28,39-95`: packed BF16 QSA delegates to cudnn.ops, bitmap or indexed support | Provider boundary; logical support and row-mask identity still apply | Retained wrapper declares head256, block4, 12Q/1KV or24Q/2KV and no implicit causal tail. cuDNN device code is not retained here; do not invent its instruction or pipeline implementation. |
| `experimental/cake_fp8_attention/cake_backend.py:172-186,421-443` and README launch table :63-76 | Four-stage statistics -> finalize -> quantize -> attention chain | Every launch re-reads changing operand values; prepared shapes/strides/buffers are fixed. It must not reuse old quantization statistics just because a runner is reusable. Device dependent-launch waits and scratch lifetime are implementation bindings. |
| Same dense route: INT8 Q and mean-centered K, per-channel scaled transposed E4M3 V, FP32 block/half scales | Mean-centering before quantization gap; existing block-scale and axis-permutation concepts are partial | K channel mean is common over tokens of a batch/head. In exact attention it induces one constant score shift per query; quantizing the centered values introduces a new approximation contract. Follow quantizer/consumer scale and correction sites before extracting the rule. |
| `experimental/cake_qsa_nvfp4/README.md:1-70,265`: BF16 Q over E2M1/group16 E4M3 KV with global K/V scales and optional gate | decoder/scale binding, split_combine_kernel@1, explicit gated-output rounding | This route has global scales in addition to group scales, unlike the earlier sparse-MLA NVFP4 cache. Output is bf16(bf16(merged)*sigmoid(gate)), so fusing the gate before the first BF16 cast changes the numeric checkpoint. |
| Same package: mma.sync for8.x/9.0/12.0, tcgen05 for10.0/10.3/10.7 | Architecture legality and distinct realization menus | One target brand cannot select instructions: SM120 route does not inherit TMEM from SM100. Source auto selection is capability-exact and does not admit all nearby versions. No measured benefit is imported from its README tables. |

The union/mask helper is now extracted; deterministic score recipes and provider
boundaries have explicit dispositions. Continue dense mean-centered quantization
and quantized sparse consumers, then reconcile remaining attention families
before moving to GEMM/recurrent/MoE and auxiliary groups.
