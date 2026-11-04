# Operator studies

**Parallel work:** follow [exclusive worker assignments](assignments.md) before selecting a library. Workers update their own study pages; the coordinator maintains this shared index.

A continuing worker resumes an active library or starts the next outstanding entry in the [NVIDIA catalog](../../docs/runbooks/learndemo-1/operator-source-catalog.md). Humming is complete as a source study. DeepGEMM is also complete as a source study. Marlin is complete as a source study. CUTLASS is partially studied, paused by user direction; do not resume in this pass unless asked. TileLang is complete as a source-design pass. TransformerEngine is complete as a source-design pass. tiny-cuda-nn is complete as a source-design pass. FlashAttention is complete as a source-design pass. Workers A and B are checkpointing partial studies at user request; no new libraries are to be started. Workers C, D and E have finished their assigned source-study passes. The local source cache inventory is at `/Users/dingyi/kta-data/dev/operator-sources/catalog-sources.json`.

| Library | Upstream revision | Study status | Retained source | Useful rules |
|---|---|---|---|---|
| Humming | 145bed14896b | Extracted; pending independent admission | [Manifest](humming/inputs/sources.json) | [Four candidates](humming/review.md) |
| DeepGEMM | 5081f8513af5 | Extracted; pending independent admission | [Manifest](deepgemm/inputs/sources.json) | [Nine candidates](deepgemm/review.md) |
| Marlin | 1f25790bdd49 | Extracted; conditional reuse, pending independent admission | [Manifest](marlin/inputs/sources.json) | [Conditional reuse](marlin/review.md) |
| CUTLASS | e13f5cd4142c | Partial; paused by user direction.23 validated provisional pairs plus saved mask draft; attention/C++ coverage incomplete | [Manifest](cutlass/inputs/sources.json) | [Current trace](cutlass/review.md) |
| TileLang | 2123e412ce22 | Extracted by distinct design; six provisional pairs, pending independent admission | [Manifest](tilelang/inputs/sources.json) | [Design notes](tilelang/review.md) |
| TransformerEngine | b021980343e3 | Extracted by distinct design; eight provisional pairs, pending independent admission | [Manifest](transformerengine/inputs/sources.json) | [Design notes](transformerengine/review.md) |
| tiny-cuda-nn | 3daa6e5b8b8e | Extracted by distinct design; five provisional pairs, pending independent admission | [Manifest](tiny-cuda-nn/inputs/sources.json) | [Current rules](tiny-cuda-nn/review.md) |
| FlashAttention | 1dc8232fe209 | Extracted by distinct design; fifteen provisional pairs, pending independent admission | [Manifest](flash-attention/inputs/sources.json) | [Current rules](flash-attention/review.md) |
| FlashInfer | a1b4ed390c56 | Active: classic/quantized prefill, decode/cascade and POD/Hopper traced; thirty-six provisional pairs including hybrid cluster/CTA packing and row-relative fixed-point masses, segmented mask packing and replicated cluster statistics, MX floor-anchor quantization and ordered workspace merge; routed GEMM, SVDQuant/W4A16 and grouped promotion traced; remaining attention and other families open | [Manifest](flashinfer/inputs/sources.json) | [Current rules](flashinfer/review.md) |

## Local run state

`<study>/inputs/` (except `sources.json`) and the `.json`/`.txt` results in `<study>/checks/` are run state and stay out of Git (`.gitignore`). They are not disposable: they hold the versions of the rules a study consulted, which rules it used and what its checks found, so rerunning a check or tracing a judgement needs them. A fresh clone does not have them, and some checks fail there until they are restored; keep them when cleaning this directory. All extracted Templates are committed.

## Remaining work across the catalog

User direction: stop expanding Workers A/B now, preserve results, and record resume points. CUTLASS also remains paused. Listed gaps are unfinished work, not completed coverage. Completed source studies still await kernel-reconstruction testing.

| Library | Status / remaining scope | Resume record |
|---|---|---|
| CUTLASS | Paused: attention and remaining DSL/C++ families incomplete | [Inventory](cutlass/inventory.md) |
| FlashInfer | Checkpointing: recurrent training/codecs, MoE, model-block/distributed families and remaining variants | [Study](flashinfer/study.md) |
| FlashMLA | Not started | NVIDIA source catalog |
| ThunderKittens | Not started | NVIDIA source catalog |
| sglang | Checkpointing: MoE expert scheduling, attention, collectives, KV publication and residual utilities | [Study](sglang/study.md) |
| DeepEP | Preliminary proposals only; full representative-design pass outstanding | [Study](deepep/study.md) |
| TensorRT-LLM | Preliminary source inventory/proposal; representative-design pass and source acquisition incomplete | [Study](tensorrt-llm/study.md) |

### Worker A remaining work

FlashInfer is partial and paused by user direction; all sources and artifacts
are preserved. The [checkpoint](flashinfer/study.md) records56 indexed pairs,
51 existing logical check scripts and one additional schema-validated draft.
Resume that draft, [rematerialized_activation_gradient_recipe@1](flashinfer/delta/rematerialized_activation_gradient_recipe@1/template.yaml): finish its zero-history convolution backward precision trace/check before registration. No closure checks or new library work were started for this checkpoint.

Materially unfinished: recurrent GDN/KDA/training variants; Mamba selective-state
codecs/checkpoint/materialization and SSD scheduler variants; remaining causal
convolution backward bias/dtype/grid-trim and inference state paths; MoE;
hyperconnection/composite model blocks; communication/distributed fusion;
vocabulary loss/LM head; remaining attention, GEMM, quantization/norm,
rotary/cache and sampling variants; provider boundaries and final file
reconciliation. Focused inventories and the checkpoint retain the exact gaps.

FlashMLA: not started. ThunderKittens: not started. Do not advance either queue
without new instructions. CUTLASS remains separately user-paused.

### Worker B remaining work

Paused by user direction; all three studies are partial. **sglang** preserves35 proposed pairs; the last recorded schema check covers34 and the latest operator_fusion@4 is unchecked. Resume the current MoE pass at moe_finalize_fuse_shared.cu's vectorized finalizer around line200, then reconcile the already-read routing/packing/grouped/quantization/paged-expert bodies. Remaining major families are attention variants, collectives, KV transfer/publication and residual operators, followed by full inventory reconciliation. Exact paths and pending native/dependency obligations are in [sglang study](sglang/study.md) and [checkpoint](sglang/inputs/checkpoint.md).

**DeepEP** remains preliminary with two proposals: dispatch locality/final issue observations do not cover full combine, forwarding publication/channel capacity, deferred epilogues, redundant expert weights/gradients, bucket transport schedules, Engram waits/reset, PP or indexing/barriers. Resume hybrid_dispatch.cuh around515-547, then combine and the remaining [inventory](deepep/inventory.md); see [study](deepep/study.md).

**TensorRT-LLM** remains preliminary with one ARCQuant proposal, limited dispatcher/sampling reads and only seven retained source blobs plus a complete tree listing. The full archive failed and remains unused. Resume ARCQuant consumer pairing and complete sampling merge/rejection paths, then acquire concrete representatives for attention, GEMM/MoE, quantization, normalization, collectives, recurrent/cache and remaining Python/native families. See [study](tensorrt-llm/study.md) and [inventory](tensorrt-llm/inventory.md). No DeepEP or TensorRT-LLM continuation, new closure checks or commits were made for this checkpoint.
