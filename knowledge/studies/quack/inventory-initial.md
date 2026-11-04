> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# NVIDIA design inventory

Scope reconciliation: public exports in quack/__init__.py are narrower than the module API. Inventory also includes gemm_interface.py, linear.py, mlp.py, linear_cross_entropy.py, distributed/, blockscaled/, operand_transform/, epilogue/, transform/ and sort/. Utility/DSL/cache/spec files supply these implementations rather than separate operators. Full retained tree is in inputs/sources.json.

| Design group | Source and variants | Disposition |
|---|---|---|
| Row norm / softmax / cross entropy, forward and backward | rmsnorm.py, softmax.py, cross_entropy.py, reduction_base.py, reduce.py | Partition/storage seeds; online_rescaled_reduction@2 and centered_moment_merge@1 provisional reuse; cluster threshold tables are dispatch menus, not legality theorems. |
| Distributed vector reduction | reduce.py:371–458; sum_swap_shuffle | New distributed_slice_reduce@1; multiple coupled planes and surplus-lane duplicates included. |
| Dense GEMM architecture paths | gemm_sm80.py, gemm_sm90.py, gemm_sm100.py, gemm_sm120.py; gemm_base.py | Seeds for tile/storage/async/rotation/instruction choices; SM80 register MMA, SM90 WGMMA, SM100 TMEM one/two CTA, SM120 warp MMA are distinct realizations. Exact backend layouts remain open. |
| Split-K | gemm_config.py:9–33, split_k_reduce.py | reduction_partition@1; serial turnstile, arrival-order atomic, separate workspace differ in ordering and finalizer; epilogue exactly once. |
| Scheduling and irregular operands | tile_scheduler.py, varlen_utils.py, tensormap_manager.py, gemm_symmetric.py | Static/dynamic/persistent/CLC, triangular symmetry, varlen and gather inventories; launch_cancellation_work_reuse@1 provisional for CLC; triangular index/pointer shifting needs follow-up mapping. |
| Epilogue and operand fusion | epilogue/, operand_transform/, gemm_w4.py, activation.py, rotary.py | Operator fusion and butterfly_linear_transform@1 reuse; QTIP/QTIP2 tail-biting format and rounding policies recorded as open mappings, not a generic affine decoder. |
| Low-bit/blockscaled | blockscaled/, gemm_w4.py | Scale layout and quantization target-specific; generic group scaling overlap, exact NVFP4/MX/QTIP mappings open. |
| Linear CE / MLP checkpointing | linear_cross_entropy.py, mlp.py | Vocabulary partition and rematerialization reuse. Scaled-exp pipeline has integer anchor and target correction, traced separately. |
| Top-k / sort | topk.py, sort/ | Bitonic value/index pair ordering then optional selected-set softmax; tie/NaN total-order extension open. |
| Distributed all-gather GEMM | distributed/all_gather_gemm.py, tile_scheduler.py:124 | chunk_ready_collective_pipeline@1 provisional; resident shards and monotonic epoch flags; transport progress must not depend on blocked SM capacity. |
