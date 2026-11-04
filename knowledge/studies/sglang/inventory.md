# NVIDIA design inventory

Inventory baseline: [retained file list](inputs/file-inventory.txt). Entry surfaces are `python/sglang/kernels/ops`, AOT `csrc/common_extension.cc` and `aot/CMakeLists.txt`; JIT implementations are under `jit/csrc`. Source paths below are relative to the pinned superproject. AMD/MUSA/CPU/Metal paths are excluded. This is a representative extraction pass with explicit open groups, not full design closure.

| Group / entry surface | Representative and variant differences | Disposition |
|---|---|---|
| Attention decode/extend/merge | ops/attention/decode_attention.py; dense and grouped heads, paged KV, split partials, MLA extra RoPE dot; HIP lean scheduling excluded | Partial reuse of reduction_partition@1 and online_rescaled_reduction@2; sparse/verify/window and CuTe GDN separate gaps |
| Attention adapters | flash_attention_v3/v4, flash_mla_interface, sparse_mla_q8kv8_prefill_sm90 | External dependencies; exact fetched revisions differ from other study baselines; no inherited proof |
| Row norm / activation / RoPE / diffusion | jit/csrc/elementwise/hc_combine.cuh; AOT fused_add_rms_norm; diffusion epilogues | Row reduction + operator_fusion@2; model-specific layouts and rotary conventions open |
| MoE routing | Hash support versus score/group route; route_warp local sort/head merge; route_radix/fused front pending | Hash routing is workload-defined. Concrete sorted_run_topk@1 binding, ties, weight/scoring differences and conditional bank map in trace-moe.md; additional radix/fused front variants active |
| MoE packing | aot/csrc/moe/moe_align_kernel.cu; prepare_moe_input, sum, grouped GEMM | segmented_aligned_storage@1 reuse; atomic order is not stable sorting; GEMM expert scheduling open |
| Quantization | aot/csrc/gemm/per_token_group_quant_8bit.cu and v2/JIT variants | Concrete quantizer, MX exponent/layout, Hadamard rotation and GGUF/Marlin operand witnesses in trace-matrix-quantization.md; MoE fused scheduling remains separate |
| GEMM/GEMV | AOT fp8/int8; JIT Marlin/AWQ, SM120 NVFP4, GGUF, expert_specialization | Concrete SM89/SM120 FP8, GEMV, warp/cluster split-K, GGUF and Marlin representatives traced in trace-matrix-quantization.md; fused routing/expert scheduling remains open |
| Selection and sampling | DeepSelect scalar/cluster; LiLiCorr and TextSeal; local chain/target-only tree; DFlash, EAGLE/ngram topology and DSpark split residuals | Concrete field bindings, seven additional navigation/representation candidate pairs and checks in trace-selection-sampling.md and trace-speculative-navigation.md. KV staging publication remains assigned to cache family; non-NVIDIA and external sampler bodies remain separate |
| Communication | AOT custom/quick allreduce; JIT custom_all_reduce, beam_all_reduce, ipc_a2a | Beam cluster normalization inspected; epoch/flag encoding, reusable readiness protocol open |
| KV transfer and metadata | kvcacheio/transfer.cu; JIT kvcache, memory, DSA/store, page tables | Address/publish contract and overlapping copy paths open |
| Recurrent/linear attention | CUDA KDA prefill/packed/fused decode, CuTe MTP/ring, FLA chunk, linear/segmented attention | Concrete KDA, FLA chunk, linear/segmented, replay/log, standalone conv and Blackwell persistent representatives in trace-kda-prefill.md, trace-recurrent-replay.md and trace-blackwell-recurrence.md; reconcile remaining repetitive routes during final inventory pass |
| Embedding, loss, grammar, reduction, optimizer and utilities | Remaining ops directories in inputs/file-inventory.txt | Named residual scope; not deep-read or explained in this pass |

Build reconciliation: AOT CMake fetches CUTLASS, FlashInfer, Triton and sgl-attn; those bodies are not in this archive. Native AOT compilation sources and Python ops/JIT are separate entry surfaces, so the former alone cannot establish library coverage. The retained list deliberately includes wrappers and tests; it is navigation, not a count of NVIDIA implementations.
