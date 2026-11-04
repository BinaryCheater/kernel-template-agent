# Complete-feature ownership in dense chains

For a small dense chain, consider keeping a batch tile's **entire hidden feature
axis** with one executor. Every next dense layer then consumes local values;
tiling that axis across independent CTAs would require an exchange or a new
reduction. This ownership is the enabling condition for layer fusion.

Pinned sources: [manifest](inputs/sources.json). Representative: offline FP16,
width 64, three hidden layers, input 64, output 8 padded to 16, batch 256,
column-major API input (sample-contiguous feature vectors), ReLU hidden activation.
`src/network.cu:53–126` selects `FullyFusedMLP`; architecture must exceed SM70
and width must be one of 16/32/64/128. The root CMake conditional agrees.

| Source choice and concrete binding | Rule field or gap | Evidence / scope |
|---|---|---|
| `fully_fused_mlp.cu:499–557,585–632`: CTA b owns samples `[128b,128b+128)` and all 64 hidden features; 4 warps, eight 16-sample iterations | `output_tiling@2`: tile 128×64, executor CTA; `reduction_partition@1` for each full dot-product partition | Batch must be a multiple of 128. Source comments saying one 16-element chunk omit `N_ITERS`; executable indexing controls. No masked batch tail in this path. |
| `:47–129`: warp w owns 16 output features; each WMMA result covers 16 samples×16 features, iterating all four K fragments | `compute_instruction_selection@1` supplies FP16 accumulator and fragment obligation; **offline WMMA frontend Realisation gap** | The rule's direct `mma_sync_m16n8k16_f16` label is not evidence for a particular lowering of the C++ WMMA call. |
| `:60–124`: complete hidden layer stored at `sample*(WIDTH+8)+feature`; next layer reads it after CTA barrier | `physical_storage@1`: smem, shared, ordered layer occupancies; `shared_resource_binding@1`: padding | All consumers finish loading before replacement: results stay in registers until the pre-store barrier. Next layer's initial barrier waits for complete publication. Padding is a source choice, not a measured conflict-free claim. |
| `:499–557`: input layer then two hidden matrix multiplies and output layer in one launch | `operator_fusion@2`: activated_dense_layer, smem, checkpoint materialize/none; v1 lacks these choices | Complete-feature ownership establishes locality. Training side saves also fall outside the old object's “only on chip” wording; do not claim direct reuse of that contract. |
| `:315–419,517–535`: dynamic first layer when width/layout differs; stages full first weights and 16 samples for sample-contiguous input, reads feature-major input directly | `physical_storage@1`, `output_tiling@2`; layout/transfer bindings | API layouts are transposed at the launcher. API CM maps to kernel row-major; API RM maps to kernel col-major. |
| `:421–476,544–556,673–708`: fuse final output only through padded width 16; larger output materializes final hidden layer then calls `fc_multiply` | Fusion boundary and local lifetime binding; vendor GEMM implementation unavailable | Representative output 8 is padded to 16 and writes only the API's declared/padded output storage. Width 17 uses a separate last GEMM. |
| `:606–615`: reserve max of hidden+last-weight and dynamic-first scratch | `physical_storage@1` capacity and non-overlapping lifetimes | Width64: `2*(16+128)*(64+8)=20736` bytes. Dynamic first scratch for input64 is `2*(64+16)*(64+8)=11520`; no sum of these disjoint phases. Shared-memory opt-in must succeed. |
| `:710–741,894–909`: training writes every hidden layer to contiguous global context | Separate global occupancies under `physical_storage@1`; fusion v2 retains side consumers | Saves are `layers*batch*width*sizeof(half)` = 98304 bytes here. Inference avoids them, except one final-hidden save when output is wide. |
| `:150–259,744–859`: reverse hidden chain uses transposed weights, saved activations and materializes activation gradients | Same ownership argument, reversed feature chain; training numeric/activation contract | Output activation derivative is done separately because both activation and last-weight gradient consume it. Input derivative fuses only matching width and API CM layout. |
| `:785–853`: weight gradients reduce over the batch in independent GEMMs | `reduction_partition@1` + existing split-combine concepts, vendor split-K realization boundary | Split factor = batch/min(4096,batch), not a freely selected source knob. Per-layer side streams start after readiness and join through `SyncedMultiStream`; no inter-layer gradient dependence is invented. |
| `:862–894`: non-JIT second derivative | Explicit implementation scope | Only one hidden ReLU layer, no output activation, no requested input Hessian. The public method does not imply arbitrary second-derivative support. |

Witness of ownership: warp 2 computes hidden features 32–47 for each of sample
chunks 0–15,16–31,…,112–127. The next layer's warp 0 reads those features too;
they must be visible in shared memory before its K-fragment load. A hypothetical
CTA partition of features 0–31 and 32–63 would break this locality even though
both CTAs independently compute valid first-layer outputs.

Related widths change warp count and scratch, not the mechanism. Activation
menus are compile-time branches; the offline forward/backward switches do not
include Sine even though initialization has a Sine branch. No timings or native
WMMA lowering were obtained. Generated warp-network and layerwise paths remain
to be compared before the network group closes.

## Generated warp and layerwise relatives

| Concrete design / source | Binding or delta | Difference that matters |
|---|---|---|
| `src/network.cu:218–313`: generated FP16 forward converts each lane's vector to cooperative `mma_vec<width>`, keeps hidden chain in registers, emits final per-lane output | [operator_fusion@2](delta/operator_fusion@2/template.yaml): contract, registers, activated_dense_layer, checkpoint selected by fwd_ctx; `output_tiling@2`: warp/sample32/full feature axis | This is a different executor from offline CTA/sample128. Both `CutlassMLP` and `FullyFusedMLP` generate the same network body. No materialization is needed at the offline width16 output boundary in this generated path. FP32 code generation throws. |
| `mma.h:93–141,446–465`: SM80+ emits two m16n8k16 FP16 MMA instructions per16×16 fragment, earlier target has WMMA PTX fallback | `compute_instruction_selection@1`: mma_sync_m16n8k16_f16, rr, f16; older fallback is an unregistered realization | For sample32,width64, each layer covers eight16×16 output fragments, four K fragments each, two native MMA statements/fragment update =64 issued warp MMA statements. FP16 accumulation order is part of the binding, not an FP32 substitute. |
| `mma.h:171–218,267–285`, `network.cu:156–216`: linear/native serialization and parameter conversion | Explicit implementation binding, `physical_storage@1`; generic fragment serialization realization remains unregistered | Native address `(fragment*32+lane)*8+2*register_within_fragment+half` differs from linear address. Conversion reads a whole layer into warp registers before writing it in place. `object.h:725–748` nests layout guards and restores linear layout on exit. Concurrent users must agree on layout lifetime. |
| `mma.h:374–438`: shuffle vector↔MMA layout | Fragment ownership binding under compute_instruction_selection | Every warp lane must participate; full shuffle masks and matrix collectives forbid a branch skipping arbitrary individual samples. The source's generated wrapper has no i<num_elements predicate; public API requires batch multiples256 (`common.h:246`, `object.h` shape checks). Enclosing custom callers need equivalent bounds/participation. |
| `network.cu:321–410`, `mma.h:233–260`: outer products reduce sample gradients within warp through MMA, then power-of-two tree over CTA warps, then one warp's packed global atomics | `reduction_partition@1` for warp/CTA partial grouping; exact tree/atomic realization binding | Selected thread menu256→128→64→32 chosen by shared-memory capacity (`object.h:401–407`), not a measured tuning search. Eight-warps tree merges2,4,8. Shared writes/reads have CTA barriers; final atomics require initialized global epoch. Non-power-of-two counts are not this tree's established scope. |
| `network.h:85–105`: forward context and gradient-reduction workspace | `physical_storage@1` occupancy and maximum of disjoint layer scratch | Representative per-sample context bytes `(64+3*64)*2=512`, grouped as warp-native matrices. With256 threads, gradient scratch `2*max(64²,64²,64*16)*4=32768` bytes. The same scratch serves successive weight-gradient stages; it is not multiplied by layer count. |
| `cutlass_mlp.cu:95–158`: per-layer inference GEMM with activation epilogue, two global ping-pong hidden buffers | `operator_fusion@2` binds activated elementwise edge only; `physical_storage@1`: two gmem occupancies, not multi_buffer_rotation@2's pipeline synchronization contract | Inter-layer hidden activations still travel through global memory. Weight GEMM/backend tiling remains external. No inter-layer network fusion is implied by an activation epilogue. |
| `cutlass_mlp.cu:95–115,167–205,276–300`: training separates Sine/SiLU activation so both pre- and post-activation buffers exist; other admitted activations may derive derivative from output | operator_fusion@2: retained_consumers obligation; explicit stage choice | A saved post-activation alone cannot replace a required pre-activation. Inference can fuse because no backward state is needed; aliased output/activation_output call sites intentionally relax retention. Source branch alone does not establish validity of arbitrary non-invertible derivatives. |
| `cutlass_matmul.h:56–110,143–180`: tensor/SIMT dispatch and activation epilogue | `output_tiling@2`, compute selection and fusion bindings; underlying CUTLASS body unavailable | FP16 full-layer CTA128×128×32, warp64×64×32; weight-gradient CTA64×64×32, warp32×32×32. SIMT128×128×8, warp32×64×8. Declared accumulator is FP16 for FP16 networks and FP32 for FP32 networks. Descriptor shapes do not prove the final device instruction sequence. |

The [address check](checks/check_fragment_serialization.py) ran locally with
`.venv/bin/python knowledge/studies/tiny-cuda-nn/checks/check_fragment_serialization.py`;
[results](checks/check_fragment_serialization.json) cover ten rectangular/layout
cases with unique coordinate identities, bijection and roundtrip. Treating native
serialization as ordinary row/column-major would fail those identities. This
checks source address arithmetic only; shuffle/MMA interpretation, numeric
precision and concurrent layout lifetime remain obligations.

Network designs are accounted for at source level, with the offline WMMA/vendor
realization and unsupported derivative/activation domains named above. The new
fusion revision preserves historical bindings and adds dense dependency closure
plus retained checkpoints; [migration](delta/operator_fusion@2/migration.md) gives
the version1 projection. It is a proposal, not Store admission.
