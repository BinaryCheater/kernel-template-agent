# Representative decision ledger

Target: NVIDIA CUDA C++; installed build options in setup.py select extension families, not one global architecture.

| Binding / decision | Rule field or gap | Source evidence |
|---|---|---|
| depth4 Adam operands, capacity36 tensors /320 blocks; chunk address base+c*C | bounded_chunk_launch_table@1 capacities, coverage/addressing | csrc/multi_tensor_apply.cuh:12–100; csrc/multi_tensor_adam.cu:25–104. Tensor slot is batch-local; chunk index remains tensor-global. |
| Raw sum and sum-square per (sample,group), two atomics into zeroed buffer | reduction_partition@1 combine_complete, physical_storage@1; raw moment instance | group_norm_nhwc_fwd_two_pass.cu:20–105. Normalize by hw*channels_per_group then variance=E[x²]-mean² at262–264. Centered_moment_merge@1 is not a substitute for this raw-state contract. |
| Channels/group128 instantiates common forward/backward body | output_tiling@2 binding; generated menu | group_norm_nhwc_one_pass_128.cu includes shared headers; one_pass_fwd body:94–169 reduces, publishes scratch, spin-waits and normalizes. Unlike two pass it retains the loaded values across completion. |
| One-pass inter-CTA scratch slot reused across steps | Gap: phase/barrier completion and concurrent residency | group_norm_nhwc_fwd_one_pass_kernel.cuh:117–158. One-kernel closure requires progress as well as visibility; no claim that arbitrary grid sizes are safe. |
| one/two pass buffers allocated and branch launch | Instance branch selected by passes, not proof of speed | group_norm_nhwc_op.cpp:113–142. Both paths allocate their required buffers; separate kernels order complete statistics before scale. |
| AdamW versus coupled decay, capturable versus ordinary | selective_parameter_state_update@1 provisional; local scalar arithmetic | multi_tensor_adam.cu:79–100,112–192; skip flag is checked in capturable path and is not inferred in ordinary path where check is commented out. |

Worked tensor binding: lengths[3,9,1], C4 and descriptor capacity2 flush into [(0,0),(1,0)], [(0,1),(0,2)], [(0,0)] after rebasing; flatten each batch back to original tensor identities. Final chunks guard respectively 3,1,1 values. Retaining tensor metadata must preserve the original identity across the middle flush.
