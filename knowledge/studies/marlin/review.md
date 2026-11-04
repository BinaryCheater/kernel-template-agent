# Marlin reusable rules

| Kernel need | Pinned rule | Useful choice |
|---|---|---|
| Prepare constant low-bit weights for MMA reads | [offline_operand_prepack@1](../../seed/references/marlin_int4_gemm/templates/offline_operand_prepack.yaml) | Bind16×64 tiles, four bits, fragment order and the matching scale permutation. |
| Spread small GEMM work across K and output columns | [linearized_work_partition@1](../../seed/references/marlin_int4_gemm/templates/linearized_work_partition.yaml) | Bind K-major stripes; preserve clipped first/last contributors. |
| Decode immediately before MMA | [fused_dequantization@1](../../seed/references/marlin_int4_gemm/templates/fused_dequantization.yaml) | Fold zero point8 into magic-number decode; group128 scales before MMA or channel scales after accumulation. |
| Merge K-splitting warp groups | [smem_partial_combine@1](../../seed/references/marlin_int4_gemm/templates/smem_partial_combine.yaml) | FP32 log tree, four participants for N128 or two for N256; keep source rounding order. |
| Complete cross-CTA partials without numerical atomics | [ordered_global_combine@1](../../seed/references/marlin_int4_gemm/templates/ordered_global_combine.yaml) | Source reverse-K order, FP16 partials and last-contributor final output. |

For M16/K1024/N512 with K128/N128 tiles and grid5, the32 K-major work
units form stripes of seven. The first output column has contributors K[7,8)
then K[0,7) in commit order. Prepack fixes their B coordinates, decode fixes
q−8 values, the CTA tree merges local partials, and the last contributor
performs the output permutation. [Partial Chain](delta/chain.yaml) binds the
four cross-rule interfaces; shared pipeline/tree/output glue is traced separately.

Source extraction is complete across all18 files and ten compiled variants.
No new candidate is needed. [Evidence and six passing commands](checks/README.md)
cover unique coordinate maps, decode, stripes, rounding counterexamples, scale
ownership, M tails and the partial Chain. [Inventory](inventory.md) and
[decision trace](trace.md) locate each disposition.

Reuse retains these premises: valid contiguous typed operands, symmetric scale-only
quantization, positive max_par, initialized and invocation-exclusive workspace,
reverse-K FP16 partial rounding, and cast-to-half before per-channel scaling. The
lock resets before final output stores, so another invocation must wait for actual
completion. GPU visibility/progress, emitted instructions and numerical acceptance
remain admission obligations. These are provisional source bindings; no GPU run or
Store admission occurred.
