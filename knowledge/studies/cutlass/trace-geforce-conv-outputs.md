# SM120 convolution side streams and output ownership

When two math groups alternate output tiles while sharing input, bias, residual
and output buffers, retrieve
[interleaved_consumer_streams@1](delta/interleaved_consumer_streams@1/template.yaml)
for **each** fixed-length producer stream. Preserve its own global cursor and
use the same local tile owner across streams. Retrieve
[staged_async_output_store@1](delta/staged_async_output_store@1/template.yaml)
for output source-read retirement. A handed-off MMA turn does not grant permission
to overwrite a still-live output staging slot.

This follows the [input trace](trace-geforce-conv-inputs.md). Source paths refer
to `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/` in the
[retained tree](inputs/sources.json). `dense` and `scaled` denote the two files in
the [inventory](inputs/geforce-conv-variants.json). Their math/output source paths
are now traced; host scale preparation, native layout composition and resource
policy reconciliation remain before closing this family.

## Decisions and exact old-field matches

| Source binding | Decision and invariant | Existing owner or boundary |
|---|---|---|
| Dense:1100–1158,1432–1450; scaled:1986–2040,2475–2508 | Both groups see the entire CLC stream; parity starts0 and flips after each next-work lookup. Group0 owns local ordinals0,2,…; group1 owns1,3,… . A skipped item advances all applicable consumer cursors and never waits, reads or releases its input stages. | `interleaved_consumer_streams@1.pre_stream`, `pre_protocol` and `program`. Ownership is stream ordinal, not the cancelled grid tile's coordinate. Fixed positive input length is per launch. Resetting local K count preserves ring index and wrap phase. |
| Dense:1110–1114,1269–1296,1438–1441; scaled:1515–1523,2300–2332,2498–2500 | Bias is one producer stage per tile in a two-stage ring. The owner waits, captures its tile-local N values into registers, releases, then adds them. The non-owner advances one stage without releasing it. | A second old-rule binding with `steps_per_item=1`; `physical_storage@1` supplies last shared read before next write. Register arithmetic can continue after capture/release. The old input K-stage count does not determine bias phase. |
| Scaled setup:187–213; residual producer:2533–2584; consumer:2393–2428 and skip:2484–2496 | Residual uses its own G2S ring, one stage per epilogue subtile, in row-major subtile order. Output and residual partition the same tile and use congruent fragment coordinates. The owner captures to registers before release; the peer skips the full subtile count. | Third old-rule binding with `steps_per_item=(tileM/epiM)*(tileN/epiN)`. Every division must be exact. Native congruent TV maps and descriptor coverage remain implementation obligations. Source residual ring depth is1; scalar checks also exercise other depths without claiming a compiled menu. |
| Scaled `_setup_attributes`:187–213 | No residual: clamp epi tile from128×128. With residual: start64×64, clamp and halve N until it divides CTA N. For N96 this yields64×32 and six subtiles; N64 gives two, N128 four. | Derived legal tiling policy, not an independently proven performance optimum. This aligns producer count, consumer count and skip count. `epi_stage=min(subtile_count,4)` and a separate residual stage1 do not imply identical phases. |
| Dense:1224–1261,1376–1431; scaled:2280–2293,2339–2343,2430–2473 | Four order slots arbitrate math and epilogue turns. Dense hands off math after final shared-to-register capture/release, before final register MMA; scaled hands it off after final MMA. Each waits for epilogue ownership before using shared D, and drains TMA source reads before epilogue handoff. | Old stream rule requires `order_protocol_valid` and `shared_output_lifetime_valid`; it does not implement order arbitration. Dense's early handoff depends on independent live register fragments, not shared input remaining live. Both declarations explicitly have four slots, unlike the earlier standalone GEMM declaration. |
| Both output loops | Write slot `subtile mod epi_stage`, publish shared writes, issue S2G, commit, acquire with source-read wait, then `producer_tail` before order arrive. | Exact `staged_async_output_store@1.program` binding. At most stages−1 source-unread groups remain before the next slot use. Kernel/global output observation still requires destination-write completion; a source-read wait alone does not supply that permission. |
| Dense:1049–1067,1444–1450,1571–1606; scaled:1483–1505,2504–2532 | CLC stage1 broadcasts next work to all participating roles. Dense counts320 lanes; scaled448 without residual,480 with residual. Residual's extra32 readers are gated by the same compile-time condition as its work loop. Unused round-up warps do not participate. | `launch_cancellation_work_reuse@1` conditional incidence binding, within its4096 reader bound. Failed-response coordinate query in the unchanged CLC wrapper still needs guarded lowering; lane visibility and progress are native obligations. |

The hardest reused decision is keeping the residual cursor aligned when N96
produces six subtiles while bias advances once and input advances K tiles. The
old stream rule's `steps_per_item` and global-generation `program` supply this
independently for each stream. One shared "pingpong" label or resetting all phases
at the start of a tile cannot supply that mapping.

The owner reads bias/residual values into registers before releasing their shared
slots. Later arithmetic is not a later shared read. The source fences and copy
primitives still need native ordering validation; the lifetime argument does not
replace their backend contracts. CLC terminal handling and all-producer tail
progress remain conditional on a valid next-work response protocol.

## Numerical order and menu boundaries

Scaled device order is `activation(alpha*acc + bias + beta*residual)`, with runtime
FP32 alpha, compile-time beta and activation, FP32 accumulator, and final f16/bf16
cast. Bias and residual enter with output dtype. Beta0 compiles out the residual
stream and its CLC readers. Applying alpha after bias changes acc2/bias1/alpha3
from7 to9; applying ReLU before residual changes acc−2/residual3/beta1 from1 to3.
The checks execute the retained arithmetic expression trees on scalar values to
distinguish these orderings. Native fusion/rounding and approximation policies
remain separate from this expression order.

Dense is `activation(acc + bias)` but accepts FP16 as well as FP32 accumulation.
The source casts bias to `self.acc_dtype`; its fragment allocation and activation
input also use the selected accumulator type. The comments saying “in FP32” do
not prove promotion for FP16. A source-menu-accepted f16 input/f16 accumulator/f32
output/f32 bias instance with bias1.0001000166 converts that bias to1 before the
addition under binary16 conversion, whereas an FP32 add retains1.0001000166.
This is a scalar conversion witness, not an observed GPU discrepancy or a verdict
on the example's tolerance. Reuse must retain `matrix_arithmetic_admitted` and the
actual accumulator precision rather than substituting a universal FP32 epilogue.

Both mainloops clear accumulators, but still preload fragments and execute a
hoisted final MMA. Frontend C0 acceptance therefore does not establish an empty-K
identity implementation. Positive input-stage length remains required by the
stream rule and patch binding.

## Evidence and remaining scope

[Output check](checks/geforce_conv_outputs.py) verifies retained source jump,
epilogue tile and CLC-count arithmetic; source AST also checks four-slot order
arrays, no release in skip branches, and tail-before-handoff in both owner paths.
It replays6720 stream domains/143416 owner releases with odd/empty final stripes,
and224 abstract output-ring domains/4032 source reads. Six actual epilogue choices
cover each CTA coordinate once, including N96's six-subtile residual branch.
Old stream scope/constraints and rejecting premise mutations are checked.
[Result](checks/geforce-conv-outputs-result.json) retains the domains and FP16 bias
witness. The order-model evidence from the
[earlier schedule trace](trace-geforce-schedules.md) remains conditional on native
barrier semantics; these scalar checks are not a native progress proof.

No new Template is needed for these side-stream and output choices. Remaining
source work: host random/scale preparation and references, exact shared layouts
and resource/stage policy reconciliation, then a family correction pass. Native
MMA/TV maps, encoded zero/finite padding, fixed-width address ranges, TMA output
coverage and failed-query/progress obligations remain explicit.
