# Block-scaled convolution: input readiness and scale ownership

Source is `examples/python/CuTeDSL/cute/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py`
in the pinned tree. The companion is now source-extracted together with [its output trace](trace-conv-blockscaled-output.md).
This slice covers the mixed completion protocol and SFA/SFB K traversal; the
output trace accounts for N-window variants, frontend and quantizer.
[Continuation](inputs/conv-blockscaled-open.json) retains source and method spans.

## Two readiness channels, one full wait

| Source decision | Actual binding | Ownership and evidence |
|---|---|---|
| Local PipelineTmaCpAsyncUmma:113–218, caller:1790–1809 | Full barrier joins A/B/SFB TMA bytes and SFA-copy thread arrivals. Initial arrivals129 for one CTA or257 for two:128/256 copy threads plus one reserved registration owner. | New [byte_arrival_completion_join@1](delta/byte_arrival_completion_join@1/template.yaml), transaction_bytes and copy_arrivals. per_slot_handshake@1's full_signal selects one kind and cannot represent both counters or the registration race. |
| Local helper:220–247; `python/CuTeDSL/cutlass/pipeline/helpers.py:458–469` | Leader performs elected arrive_and_expect_tx, installing the exact bytes while consuming the reserved arrival. Helper sm100.py:59–92 selects MbarrierArray for AsyncLoad. | New rule registration_race. If all128 copy threads arrive first, one arrival remains; dropping the reservation would expose a completed generation before any TMA byte expectation existed. Native atomicity/visibility is a backend premise, not proved by the scalar model. |
| Caller:1797–1801,1187–1194 | Bytes=sum(A-layout bytes,B-layout bytes,SFB-layout bytes), each with the instruction's CTA multiplier. SFA is not included as transaction bytes. | New rule c_incidence. The retained illustrative one-CTA binding A8192+B8192+SFB1024 uses17408 bytes; actual replay must substitute sizes from selected DSL layouts. This is not a compiled source configuration. |
| Local helper:269–282,169–173 | Two-CTA copy arrivals target rank&~1, the group's leader; one-CTA arrivals stay local. Both cooperating CTAs contribute128 copy-thread arrivals. | New rule incidence_and_target. Exact source mask/operand-layout incidence and remote write visibility remain target obligations; source comments about peer SMEM are not emitted-address evidence. |
| SFA loop:2408–2552 | Each iteration commits one copy group, waits until at most two remain, then signals the stage from two iterations earlier. Two pending stage indices are retained; tail wait0 emits the last one or two arrivals. | Source binding of progress_and_reuse. Scalar replay covers T0..32/S1..8. With immediate consumers, capacity must be at least min(T,3); T3/S2 blocks before reaching the first signal, while T1/S1 and T2/S2 drain correctly. This is a progress premise, not a claim that a public source configuration necessarily selects S2. |
| _compute_stages:3794–3963 | Construct exact SharedStorage sizes and scan upward for the largest fitting AB/D stage counts. Includes per-array1024 alignment and staged barrier arrays, including the separately allocated sfa_full array. The AB lower bound can be1. | Derived resource heuristic; not a general performance model. The source does not express the two-group delay's minimum-stage premise as an explicit check. Actual selected stage capacity and hardware progress need implementation replay. Full/empty and extra allocations must not be omitted merely because one full barrier joins readiness. |

[Event checks](checks/conv-blockscaled-inputs-result.json) explore612 counter states
across1..4 copy arrivals and1..4 transaction chunks. They preserve the omitted-owner
race and264 delay/capacity domains. This establishes the scalar incidence/progress
argument under stated signal assumptions; it does not execute a native mbarrier.

## Scale coordinates and traversal

The logical convolution map still uses the virtual patch values of
[implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml).
The block-scaled mainloop changes execution order: S, R, T vary before C_tile
(source2153–2179), and channels inside a tile remain fastest. This is a traversal
binding, not a silent change to logical patch coordinates. Matrix arithmetic and
changed reduction grouping must satisfy the numerical contract.

| Source decision | Concrete mapping | Field, source glue or gap |
|---|---|---|
| sf_k_tile_channels:4201–4209; setup:745–762 | SF span=max(cta_tile_k,4*vector). NVFP4 vector16 supports K64/128/256; MXFP8 vector32 supports K64/128. K64 MXFP8 therefore shares one K128 SF atom over two AB tiles. | Input representation fixes vector/format; tile_k is a compile choice inside the source menu. Span/cadence derive from them. Physical scale atom packing remains a separate layout refinement. |
| SFA:2260–2274,2355–2385 |128 threads biject to128 output rows using8*(tid//32)+32*((tid%32)//8)+tid%8, plus the instruction CTA row offset. Decode output row into batch/z/p/q, apply lower padding and stride, then add tap dilation. | Same spatial patch_identity, now also used to select A's scale row. Spatial and output-M predicates gate copies; invalid pointers use a safe clamped base before predicated zero fill. Physical shared addresses/TMEM copies remain backend edges. |
| SFA:2311–2326,2498–2519 | Carry recurrence visits S→R→T→C_tile. Delta strides are dilW*SFextent, (dilH*W-S*dilW)*SFextent and (dilD*HW-R*dilH*W)*SFextent. | checks/conv_blockscaled_inputs.py executes the retained source recurrence directly and compares it to independent tap coordinates for403 positions. The source execution order is distinct from the dense entry's default reduction traversal. |
| SFA:2441–2494 | Each copy reads four SF bytes. Every thread visits SI=((tid%32)//8+i)%num_groups; clamp a trailing group's start to SFextent-4. C must be a multiple of4*vector so starts remain aligned. |30 input domains cover both format menus and partial C tiles. Padding SF duplication is safe only when the padded operand arithmetic is neutral under the admitted value/metadata domain; zero operands alone do not justify arbitrary NaN metadata. Scale tail initialization/numeric behavior remains an implementation premise. |
| SFB:2161–2163,2205–2218; sfb_per_position_channels:4212–4222 | Each filter position gets ceil(C/SFspan)*SFspan channels in its physical scale stream. SF chunk index=ctile//cadence+(s+S*(r+R*t))*ceil(Ctiles/cadence). | Per-filter physical padding is essential. C192 with SF span256 must start the next position at256, not192. This source layout binding does not change the logical valid convolution channels. Scalar check preserves that counterexample. |
| SFB:2134–2138 and setup/TMEM consumers | N64 shares one128-channel chunk; N192 has additional overlapping-window/TMEM handling. | N mappings are source-extracted in the output trace. The K-index check alone does not establish them or native cluster multicast incidence. |

## Output and backend continuation

The [main convolution trace](trace-conv.md) preserves the dense path; the
[output trace](trace-conv-blockscaled-output.md) accounts for SFD quantization,
residual/bias/activation, SFB N windows and the full frontend. Humming's affine
integer quantizer cannot implement the low-float policy, so the latter has its own
candidate. Native visibility, progress, fragment mapping and numerical extremes
remain implementation/extension obligations rather than source coverage claims.
