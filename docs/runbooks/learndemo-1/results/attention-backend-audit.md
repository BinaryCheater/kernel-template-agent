# Actual attention comparator backend

The evaluator calls `torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)` without forcing a backend. The explicit FP64 matmul/softmax/matmul computation is the correctness oracle, not the timed comparator.

On 2026-11-05, a post-hoc dispatch audit reused the original pinned image, GB10, retained evaluator code, timing inputs and three original attention shapes. CPU operator traces show `aten::_scaled_dot_product_flash_attention` and `aten::_flash_attention_forward` for every shape. Actual CUDA traces show `pytorch_flash::flash_fwd_kernel` for all three. This is PyTorch's integrated fused FlashAttention implementation. The timed comparator was not explicitly forced to the unfused math backend. An ordinary eager API call can dispatch to a fused kernel.

The audit does not establish identity with the separately pinned FlashAttention source studied for extraction. It also does not turn the generic implementation campaign into a source-linked reconstruction experiment. See [scope correction](scope-correction.md).

Evidence: `/Users/dingyi/kta-data/dev/template-replay/environment/sdpa-backend-audit-2026-11-05/receipt.json` and its `remote-output/retry-user-env/` directory contain actual operator/kernel names and three Chrome traces. Original batches had no profiler trace; this evidence is a later reproduction, not a retrospective claim that a trace existed. The first probe failed during profiler import because the container UID had no username; its logs remain. The retry supplied `USER=kta` for that lookup and did not override SDPA selection. No performance comparison was made in this audit.

The [PyTorch 2.11 SDPA documentation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html) explains automatic backend selection and explicit backend controls.
