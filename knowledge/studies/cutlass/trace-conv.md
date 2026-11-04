# Blackwell implicit-GEMM convolution

The two entries are in `examples/python/CuTeDSL/cute/blackwell/kernel/conv/`.
`dense_implicit_gemm_fprop.py` is source-extracted below. The block-scaled companion
is source-extracted in [its input](trace-conv-blockscaled-inputs.md) and
[output](trace-conv-blockscaled-output.md) traces; [bindings](inputs/conv-blockscaled-open.json)
retain methods and outstanding backend obligations. Both entries now have source
dispositions; native assurance remains separate.
All source paths/lines refer to the pinned tree in inputs/sources.json.

## Dense contract and witness

Input is NDHWC, filter is KTRSC, and output is NZPQK, all channel-contiguous.
The witness uses batch1, input C8/D5/H5/W5, output channels16, filter3×3×3,
padding1 on both sides, stride/dilation1, FP16 inputs/output and FP32 accumulation.
MMA128×128, one-CTA instructions and equal preferred/fallback clusters1×1 pass
the retained frontend; the neighboring C7 fails alignment. Output has125 spatial
rows and216 reduction elements, so this is also a logical M/K-tail case.
[Binding](inputs/conv-dense-binding.json) retains exact source and method spans.
No CUDA compilation, target launch or device tail behavior was checked.

## Ordered decisions and rule ownership

| Source decision | Concrete mapping or binding | Rule field, evidence or gap |
|---|---|---|
| Dense:2130–2148 computes Z/P/Q;433–526 constructs A descriptor | Output length is floor((input+lower+upper-dilation*(filter-1)-1)/stride)+1. The leading side is lower padding. Helper `python/CuTeDSL/cutlass/cute/nvgpu/helpers.py:583–599` passes WHD corners (-lowerW,-lowerH,-lowerD), upper-(tap-1)*dilation and tap strides=dilation. | New [implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml), patch_identity and geometry. Asymmetric padding is required to distinguish reversed-side explanations; misleading before/after prose is not substituted for the equations. |
| Dense:463–467,565–569,609–636 transforms views | A descriptor sees ((W,H,D,batch),C); B is (outputChannel,(C,S,R,T)); D is ((Q,P,Z,batch),outputChannel). Output-Q and reduction-C are fastest; dummy GEMM batch is1. | New rule coordinate_bijections/program. Original convolution batches are part of GEMM-M rather than independent GEMM-L. tensor_axis_permutation@1 is a3-axis view and does not supply rank expansion, patch replication or zero fill. |
| Dense:478–519,618–623 selects im2col copy atoms | patch_storage=on_load. TMA constructs input windows during G2S; output S2G is an im2col-shaped store without patch expansion. B uses ordinary tiled TMA. | New rule supplies the virtual operand values; async_transfer@1 supplies abstract input readiness after that mapping. Its controller name does not by itself establish im2col coordinate generation. Full patch materialization is an algebraic alternative only. |
| Dense:167–238 validates descriptor geometry | Source guard admits corners[-16,15], largest tap offsets<=31 and stride<=8. Alignment helper:137–164 checks16-byte C/K channel alignment. | New rule implementation_bounds;720 retained guard domains pass. The field-width helper alone accepts stride0, so positive stride/dilation and legal extents remain explicit rule/caller premises. Native descriptor encoding is not executed. |
| Dense:375–402,900–965;755–897 prepares launch | Parent storage/layout setup is sized from fallback; preferred is a componentwise multiple, both clusters legal and enough CTA tiles exist per preferred dimension. Both descriptors and scheduler parameters are retained; actual runtime cluster dimensions choose the matching branch at1054–1105. Shared branch allocations may merge. | Source dispatch glue, not an independent optimizer knob or proof of occupancy. cluster_multicast@1 describes sharing receivers after branch selection; it does not choose which runtime cluster forms. Per-region storage lifetime can explain mutually exclusive branch allocation, not measured capacity savings. |
| Dense:651–752 derives MMA/SMEM stages, and1267 onward creates pipelines | Four instruction-K extents per producer tile; stage/layout choices follow dtype and memory budgets. TMA/MMA/epilogue/scheduler use inherited fixed roles. | Reuse async_transfer@1, multi_buffer_rotation@2 and per_slot_handshake@1 only inside their declared stage/arrival domains. Existing role-granularity and concrete backend-count obligations remain; no actual S4 binding is inferred from the old example alone. |
| Dense:1376–1387,1528–1657 uses CLC scheduling | Response-reader count32*(1+cluster_size*(1+4+1)), i.e.224 for one CTA,416 for two. One response slot; initial work is native launch identity. | Reuse launch_cancellation_work_reuse@1 participants field. Existing unguarded coordinate-query IR in clc.py remains the same implementation gap; convolution does not discharge the successful-response premise. |
| Dense:792–803,847–863,1833–1925 stages optional bias | Bias dtype must match D; channel row broadcast has zero M strides.32-bit vector per lane, with multiple passes when N-active exceeds128 epilogue threads; FP32 N256 needs two passes. Predicated loads zero-fill complete tail vectors. | operator_fusion@1 consumer_kind=elementwise, intermediate_site=registers, fused_ops=linear_combination supplies on-chip bias addition. Boundary predication applies to aligned bias vectors; exact generated copy/lane mapping remains a backend obligation. |
| Dense:1942–1959 orders epilogue arithmetic | Add bias in accumulator type, convert to D type, then call epilogue_op. This is not activation in accumulator precision followed by narrowing. | Old fusion record does not name arbitrary activation or cast placement; retain those as source numeric-order glue/gap. No free algebraic reordering is claimed. The source reference at2424–2430 follows the same order. |
| Dense:1961–1994 stages/stores output | Derived shared D ring, one committed store group per subtile and source-read wait before reuse, with producer_tail at completion. Descriptor maps flattened rows back into NZPQK and clips logical tails. | Reuse staged_async_output_store@1 after new patch rule's output map. Exact TMA clipping/completion remains an implementation premise; a scalar domain check is not a device store proof. |

## Worked composition and checks

The witness maps m124 to batch0/z4/p4/q4 and reduction k215 to t2/r2/s2/c7.
That pair addresses input (5,5,5), outside D/H/W5, so the patch value is zero;
interior coordinates retain their distinctive input value. For each output channel,
contract all216 patch/filter products, add its bias in FP32 if present, narrow once,
then activate and stage the output. CLC can move ownership of whole flattened output
tiles without changing this coordinate map.

[Coordinate/frontend checks](checks/conv-contracts-result.json) execute the retained
pure output-size, descriptor, alignment and cluster guards with symbolic dtype
identities and real selected parent predicates. They compare nested spatial sums
with independently flattened patch matrices for16 geometries,16032 patch elements
and2004 output values; swapped padding and swapped tap-axis failures are preserved.
48 bias-vector domains cover tails and the FP32 second pass. These are source-equation
checks, not lowered im2col descriptors or hardware arithmetic. The proposed pair
and executable scope guards pass schema checks. Its materialized alternative is
not an existing source backend or a measured performance claim.

## Block-scaled companion

The companion inherits this coordinate/dual-cluster route but changes much more
than dtype. [Input continuation](trace-conv-blockscaled-inputs.md) now extracts its
joint byte/arrival completion rule, shifted remote arrivals/tails, SFA spatial
recurrence and SFB per-filter K padding. Native counter visibility and exact
multicast incidence remain backend obligations. [Output continuation](trace-conv-blockscaled-output.md) now accounts for SFB
N-window variants, frontend and output epilogue/quantizer. Native group/lane
mapping and numeric extremes remain explicit backend/extension gaps.

The companion constructor:476–499 adds four SFA cp.async warps and an optional
residual warp. Frontend:4034–4061 excludes bias/residual/named non-identity activations
for narrow block-scaled outputs. The output path emits SFD metadata (3061 onward;
reference4967 onward). Humming's groupwise_affine_quantization@1 is an unsigned
integer range/zero-point quantizer and cannot supply this low-float output scale
policy. The output trace proposes groupwise_low_float_quantization@1 with an explicit
finite range and metadata/code agreement witness. [Bindings](inputs/conv-blockscaled-open.json)
record the retained instance and remaining implementation obligations.
