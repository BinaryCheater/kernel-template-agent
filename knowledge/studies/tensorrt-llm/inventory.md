# NVIDIA design inventory

Baseline: complete pinned Git tree in inputs/tree-inventory.txt. Main native build registration is cpp/tensorrt_llm/kernels/CMakeLists.txt with nested CMake files; native launch wrappers, TensorRT plugins and Python/JIT kernels are distinct surfaces. Generated cubin catalog and public dispatcher are not the same as readable device bodies.

| Design group | Representative / variants | Disposition |
|---|---|---|
| Context attention | fmhaDispatcher.cpp, contextFusedMultiHeadAttention runners/cubins, generated tllm-gen | Dispatch traced; generated/device mapping gap |
| Decode attention | decoderXQARunner.cpp; MMHA, XQA JIT, cascade, flashMLA | Eligibility traced; cache layout, split and architecture bodies open |
| Sampling | fusedSampling single/multiCTA and common body; samplingTopK/TopP and speculative siblings | Representative output specialization, radix, rejection cap and split statistics traced; legacy/speculative paths open |
| Quantization | arcquantFP4; quantization, FP8PerTokenQuant, fusedActivationQuant and FP4/MX families | Selected residual encoding proposed; other codebooks/layouts and error-selection designs open |
| Dense/grouped GEMM | cutlass_kernels FP4/FP8/int8/fpA_intB, fused gated and low-latency; cuda_graph_grouped_gemm | Tiling/scale rules are retrieval leads; exact fragment and scheduling designs open |
| MoE | moe kernels, expert transport and fused router/GEMM; dsv3MinLatency and glmMoeDsa | Named unexamined design group; cannot inherit CUTLASS/DeepEP proof |
| Norm/activation/RoPE/diffusion | fusedLayernorm, fusedGatedRMSNormQuant, DiT fusion and split normalization | operator_fusion@2 retrieval lead; layout/numeric details open |
| Collective/fabric communication | customAllReduce, communicationKernels, dwdpFabricGather/Tma and allreduce_gemm | Native readiness, low-precision sum and multicast lifetimes open |
| Recurrent/convolution/SSM | causalConv1d, selectiveScan, gdnVerify and Mamba kernels | State/chunk composition open; independent assigned libraries remain owned by other workers |
| KV cache/compression/index selection | KV gather/scatter, IndexerTopK, compressor, blockCausalLayout | Selection, page mapping and publication open |
| Embedding, loss, decoding utilities, masks and preparation | Native root files and Python kernel directories in retained tree | Residual scope named explicitly; not deep-read |

This is a representative source pass, not a claim that every family is explained. inputs/design-paths.json enumerates directory-level source surfaces to support subsequent review against the full tree, including paths outside the native kernel root. Missing dependencies and generated bodies remain extraction gaps.
