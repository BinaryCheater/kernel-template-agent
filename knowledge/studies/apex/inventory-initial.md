> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# NVIDIA family inventory

Reconciled setup.py CUDAExtension source lists and glob registrations against retained csrc/**/*.cu and apex/contrib/csrc/**/*.{cu,cuh}. Generated group_norm one_pass_*.cu and group_norm_v2 gn_cuda_inst_*.cu are channel/shape instantiations of shared bodies, not separate designs.

| Design | Representative / variants | Disposition |
|---|---|---|
| Multi-tensor optimizers and transforms | csrc/multi_tensor_apply.cuh; multi_tensor_adam.cu, sgd, lamb stage1/2/mp, novograd, adagrad, axpby, scale, l2norm | New bounded_chunk_launch_table@1; arithmetic, master weights, norm/trust-ratio staging vary; reduction_partition@1 and selective_parameter_state_update@1 provisional reuse. |
| Normalization | csrc/layer_norm_cuda_kernel.cu, welford.cu; contrib layer_norm/, group_norm/, group_norm_v2/, groupbn/ | Raw moments vs centered statistics; one-kernel inter-CTA synchronization vs two-kernel workspace. Centered_moment_merge@1 only for actual centered-stat paths. GroupBN peer protocol and v2 specialized layouts open. |
| Softmax / losses | csrc/megatron/*softmax*, contrib/xentropy/, focal_loss/ | Row partition and fused masks/scales; focal sigmoid weighting and xentropy semantic contracts remain separate. |
| Dense and MLP | csrc/mlp_cuda.cu, fused_dense_cuda.cu; megatron/fused_weight_gradient* | GEMM library calls plus fused activation/gradient accumulation; operator_fusion@2 provisional; vendor GEMM internals excluded. |
| Positional / indexed maps | fused_rotary_positional_embedding_cuda.cu, contrib/index_mul_2d/ | Rotary paired-coordinate mapping and indexed gradient accumulation; indexed_additive_scatter@1 provisional. |
| Transducer | contrib/transducer/{transducer_joint,transducer_loss}_kernel.cu, philox.cuh | Joint/dropout and dynamic-programming loss have distinct designs; DP wavefront and packed-input mappings remain open. |
| Communication / fused CNN | contrib/{peer_memory,nccl_p2p,groupbn,bottleneck,conv_bias_relu} | Peer exchange, NCCL and cuDNN wrappers; custom synchronization and vendor backend boundary open. |
| Sparsity search | contrib/sparsity/permutation_search_kernels/ | Permutation-search scoring; unread mapping gap. |
| Loss-scale state | csrc/update_scale_hysteresis.cu | Stateful overflow/backoff/growth policy, not stateless quantization scale selection. Source-only disposition; no new rule. |
