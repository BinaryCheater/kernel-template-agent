# TensorRT-LLM NVIDIA study

Status: partial; paused by user direction. Preliminary inventory and one proposal only. Do not continue source acquisition or extraction without a new request. This remains the last library in Worker B's original queue; no commits.

The full pinned Git tree is retained in inputs/tree-inventory.txt. Only seven representative source files are retained and hashed in [sources](inputs/sources.json). The full archive transfer failed with curl exit92; partial bytes remain outside Git, unverified and unused. There is no running full-download claim. Preserve these inputs rather than treating the partial archive as a usable tree.

The saved selected_group_residual_encoding@1 pair comes from ARCQuant activation residual expansion. [Trace](trace.md) also records limited attention eligibility and single/multiCTA sampling observations. Existing logical replay checks expanded coordinate identity, not native quantization, consumer reconstruction or full-family coverage.

## Materially unfinished families and exact resume points

Resume only after sglang and DeepEP are completed and the user explicitly resumes this queue. Begin with the retained cpp/tensorrt_llm/kernels/arcquantFP4.cu around230-287: finish the downstream consumer's independent base/residual-scale pairing, separate WEIGHT duplication semantics and native conversion/layout obligations. For sampling, resume fusedSampling/fusedSamplingKernelsCommon.cuh around745-788,942-997 and its complete single/multiCTA launch/merge path; the capped-rejection fallback is not an exact-law realization. The legacy top-k/top-p/speculative families remain unread.

fmhaDispatcher.cpp and decoderMaskedMultiheadAttention/decoderXQARunner.cpp were read only for eligibility/dispatch. Context/decode generated device bodies, cache layouts, split/cascade and architecture paths still need concrete witnesses. Other materially unfinished families are remaining FP4/FP8/quantization and fused activation; dense/grouped/low-latency GEMM; MoE/router/transport; norm/RoPE/diffusion; collectives/fabric/GEMM overlap; recurrent/convolution/SSM; KV/compression/index selection; embedding/loss/masks/preparation and Python/JIT surfaces. [Inventory](inventory.md) and inputs/design-paths.json identify source locations for selective acquisition at the pinned revision.

Do not infer coverage from the complete tree listing or external library studies. No further source acquisition, GPU measurement or closure checks were performed for this pause checkpoint.
