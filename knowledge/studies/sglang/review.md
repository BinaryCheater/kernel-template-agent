# sglang: reusable choices and active coverage

Status: partial; paused by user direction. Thirty-five paired candidate records are preserved. The complete NVIDIA study remains unfinished; see the exact [resume points](study.md) and [checkpoint](inputs/checkpoint.md). [Rule menu](delta/README.md) links every candidate and proof; [inventory](inventory.md) names the remaining scope.

| Kernel situation | Useful choice | Evidence |
|---|---|---|
| Small complete state reused by an iterative solve | Register/shared residence with the same recurrence and termination | [Small-row trace](trace-expansion-and-small-row.md) |
| Long recurrence can be summarized by affine pieces, triangular blocks or retained update logs | Choose piece prefix, blocked solve, delayed materialization or accepted-prefix replay with explicit state ownership | [Prefill](trace-kda-prefill.md), [replay](trace-recurrent-replay.md) |
| Blackwell recurrent chunk consumer | Six-input descriptor binding, resident tensor state, exact channel participant counts and inverse realization | [Blackwell trace](trace-blackwell-recurrence.md) |
| Quantized contraction changes representation or metadata ownership | Symmetric quantizer, regular rotation, codebook decode, raw-sum affine correction, zero-pair load gate or rematerialization | [Matrix/quantization trace](trace-matrix-quantization.md) |
| Large-row top-k with small output | Running threshold compaction or attained-tile-maximum cover; repair boundary ties only when deterministic identity is required | [Selection trace](trace-selection-sampling.md) |
| Repeated weighted selection under changing launch partition | Key/context/original-token score identity and the same score/ID comparator at both reduction levels | [Selection trace](trace-selection-sampling.md) |
| Small-expert routing distributed over lanes | Sort each contiguous lane run once, merge heads, then recover original expert IDs under the same tie order | [MoE trace](trace-moe.md) |
| Tree histories, dependent walks and leading-match acceptance | Exact tree views, resident state with required output tape, first-failure reduction and explicit packed-row ownership | [Navigation trace](trace-speculative-navigation.md) |
| Expensive residual mass needed only for a scalar acceptance decision | Cheap conservative upper bound guards exact refinement | [Block verifier](trace-selection-sampling.md) |

One composition: LiLiCorr computes every tile's maximum and full-row normalization partials together, selects the highest-maximum tiles, gathers their complete valid contents, then selects k values. The maxima provide actual rank witnesses; normalization still combines every group. A finite padding value must not win, and tied identities may be arbitrary. [Evidence and checks](trace-selection-sampling.md).

The last recorded schema/identity check covers34 pairs. operator_fusion@4 was added afterward and remains unchecked; no new closure checks were run. Focused host checks replay coordinates, arithmetic or rank invariants; no GPU timing or native implementation admission was run. Retained source is in [sources](inputs/sources.json). Unfinished work includes MoE scheduling/finalization, remaining attention, collectives, KV publication and utilities. DeepEP and TensorRT-LLM are preliminary and paused by user direction.
