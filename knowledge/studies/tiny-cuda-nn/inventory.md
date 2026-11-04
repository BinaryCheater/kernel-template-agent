# Design inventory

Revision and retained paths: [inputs](inputs/sources.json). Paths below are relative
to that tree. Dispositions close this source-design pass; explicit extraction and native-evidence gaps remain named.

| Design group | Entry/registration and representative | Differences to account for | Disposition |
|---|---|---|---|
| Offline fused dense network | `src/network.cu:53–126`, `src/fully_fused_mlp.cu` | Widths 16/32/64/128; FP16; first/last layer boundaries; training activation saves; backward and restricted second derivative | [Traced](trace-networks.md); fusion v2 candidate |
| Layerwise dense network | `src/cutlass_mlp.cu`, `include/tiny-cuda-nn/cutlass_matmul.h` | GEMM epilogue activation, split-K weight gradients, ping-pong inference buffers, stream overlap | [Traced](trace-networks.md); vendor device body unavailable |
| Generated warp network | `src/network.cu`, `include/tiny-cuda-nn/mma.h`, `include/tiny-cuda-nn/networks/*.h` | JIT layout conversion, all-lane participation, fused enclosing caller, forward/backward generated code | [Traced](trace-networks.md); fragment check passed, native/activation domains remain guarded |
| Multiresolution Cartesian grids | `src/encoding.cu:69–75`, `encodings/grid.h` | Dense/tiled/hash lookup; nearest/linear/smoothstep; stochastic gradients; first/second derivatives; LoD | [Traced](trace-grid.md); lookup candidate, explicit stencil and coordinate variants pending |
| Simplex lattice encoding | `src/encoding.cu:78–80`, `encodings/internal/permuto.h` | Simplex corners, hash storage, derivatives vs Cartesian grid | [Traced](trace-simplex-and-composition.md); mapping/adapter gaps named, Hessian candidate |
| Fixed basis encoding | `src/encoding.cu:65–67,87–104`, `encodings/{frequency,triangle_wave,oneblob,spherical_harmonics,identity,empty}.h` | Periodic phase/bins, finite basis, padding and derivative storage | [Traced](trace-basis-encodings.md); adjacent-boundary candidate and padding differences recorded |
| Encoding composition | `src/encoding.cu:57–59,83–85,106–131`, `encodings/{composite,multi_level_lod}.h` and `network_with_input_encoding.h` | Nested dimension slicing/reduction, level gating, encoded output layout and network alignment | [Traced](trace-simplex-and-composition.md); mapping/adapter gaps named, Hessian candidate |
| Pointwise training updates and wrappers | `src/optimizer.cu`, `optimizers/{adam,sgd,novograd,ema,average,lookahead,exponential_decay,batched,composite}.h` | Master/cast parameter representations, moment state, sparse zero-gradient semantics, wrapper update epochs | [Traced](trace-training.md); selective-state candidate, wrapper clocks explicit |
| Matrix preconditioned update | `optimizers/shampoo.h` | Statistics, inverse matrix roots, update frequencies, external solver boundary | [Traced](trace-training.md); inverse-root and refresh abstractions remain gaps |
| Loss and reduction | `src/loss.cu:57–65`, `losses/*.h`, `src/reduce_sum.cu` | Elementwise losses/gradients, PDF normalization, partial reduction | [Traced](trace-runtime-and-loss.md); native/runtime gaps explicit |
| Runtime training and code fusion | `include/tiny-cuda-nn/{object,trainer,cuda_graph,multi_stream}.h`, `src/{object,rtc_kernel,cpp_api,common_host}.cu` | JIT entry generation, loss/backward/update boundaries, CUDA graph and stream lifetime | [Traced](trace-runtime-and-loss.md); native/runtime gaps explicit |
| Common implementation primitives | `include/tiny-cuda-nn/{common_device,mma,random,gpu_matrix}.h` | Fragment loads/stores, activations, layout, RNG; bind consumers instead of cataloging helpers independently | [Traced](trace-runtime-and-loss.md); native/runtime gaps explicit |

The active build authority is root `CMakeLists.txt:324–341,390`, which registers
11 translation units (including the conditional fused MLP) and NVRTC support.
`src/CMakeLists.txt` is a stale alternative list naming absent `common.cu` and
`common_device.cu`; it does not define the root target. Header device bodies are
part of the scope, including generated source strings. Samples, benchmarks, tests
and PyTorch bindings supply caller evidence, not additional optimization families.
Reconcile these groups against all registrations and device-body locations before
closure. External CUTLASS/fmt/cmrc submodules are absent from the archive.

## Final reconciliation

[Source map](inputs/source-map.json) records the eleven top-level CUDA units,
fifteen encoding names/aliases, nine registered losses, ten optimizer names and
45 device-body/header locations. Network aliases map to two offline designs and
one shared generated design. Encoding aliases Grid/HashGrid/TiledGrid/DenseGrid
share grid kernels; NRC/OneBlobFrequency share a Composite. Permuto and LoD are
excluded when offline forward/backward support is disabled. Shampoo is conditional
on its build gate; the unsupported factory branch throws. ConstantGradientLoss is
an unregistered header operation, covered with external-gradient loss handling.

All device locations map to the table above: encoding/loss/optimizer headers map
to their families; mma and network units to network trace; common_device math,
vec/ministd/common type helpers to their consumers; common_host parallel-for,
object/trainer/reduction/random helpers to runtime trace. grid_interface and
multi_level_interface contain the position-copy/transpose helpers already bound
by the grid/layout adapters. The former is a compatibility copy, not a new grid
algorithm. No CUDA device family was found outside these groups in the top-level
library; tests/samples/benchmarks contain caller/check kernels, not independent
public optimization families. External submodule implementations remain absent.

Source-design coverage is complete with named gaps: Cartesian/simplex stencil
and coordinate construction, BaseConvert lookup extension, iterative inverse root
and its convergence/refresh abstraction, whole-training fusion, host launch graph
reuse, unsupported generated composite derivatives and legacy reduction. These
do not silently inherit proofs from similarly named existing rules. Native
fragment/numeric/concurrency evidence and end-to-end replay remain deferred.
