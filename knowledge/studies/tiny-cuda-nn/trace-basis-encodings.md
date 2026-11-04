# Fixed basis encodings

These encodings expand independent input coordinates or a direction into a fixed
basis, without trainable lookup tables. Basis type/count is part of the requested
representation; scheduling, derivative retention and repeated evaluation sharing
are separate choices. Source paths are relative to the [pinned tree](inputs/sources.json).

| Group / concrete source | Reused rule or delta | Distinct choice and limit |
|---|---|---|
| `encodings/frequency.h:46–103,207–275`: sin(pi*2^f*x+phase), phase0 orpi/2 | `output_tiling@2` on sample×expanded feature; `physical_storage@1` for optional FP32 derivative array; `reduction_partition@1` for input-gradient dot product | Offline path saves derivative if requested; generated backward saves input then recomputes cosine. Example x=.5,f0 gives sin(pi/2)=1 and cosine channel near0, not two arbitrary frequencies. Approximate intrinsics and cast precision remain numerical bindings. |
| `encodings/triangle_wave.h:46–111,212–285`: triangle with val=2^(f-1)*x+f/4, output4*abs(fract(val)-.5)-1 | Same scheduling/storage/reduction bindings as Frequency, different workload basis | Only one output/frequency; derivative uses the source parity convention at creases. Source phase shift distinguishes levels; a sine-basis Template is not reused as mathematical equivalence. |
| `encodings/oneblob.h:46–126,169–247,300–402`: differences of translated quartic CDF sums over bins | [adjacent_boundary_reuse@1](delta/adjacent_boundary_reuse@1/template.yaml) | AoS: aligned bin lanes shuffle next boundary; final bin adds1. SoA and generated code: one thread carries right boundary into next interval. Subwarp scope requires power-of-two bins<=32 and proper groups; constructor's power-of-two check alone does not prove this stronger launch condition for every configuration. |
| `encodings/oneblob.h:130–165,350–390`: input gradient uses upstream weights times left-density minus right-density | `reduction_partition@1` + serial boundary reuse | Forward endpoint jump1 is not a derivative endpoint jump1. The unwrapped `one_blob` helper at:71–87 differs from the periodic public paths; no unconditional wrap/nonwrap equivalence. |
| `encodings/spherical_harmonics.h:46–118,206–247`, `common_device.h:476` onward: polynomial basis of direction2*input-1 | `output_tiling@2`: one sample/thread; `operator_fusion@2` when generated into enclosing network | Three inputs, degree1..8, degree² outputs. Common polynomial evaluator reuses x²,y²,z²,xy,xz,yz across basis terms; formulas are the representation contract, not a new freely interchangeable algorithm. Degree2 output has constant and linear components; direction rescaling gives a factor2 in input gradient. No hidden vector normalization is introduced. |
| `encodings/identity.h:45–111`: scale*x+offset and scale derivative; `empty.h` | Independent pointwise map binding; no new Template | Identity has constant padding and zero derivatives for padding; Empty supplies the requested empty/padded output and zero input derivative. These are workload/adapter semantics rather than kernel design deltas. |

OneBlob witness: for16 bins, AoS evaluates16 left boundaries and performs neighbor
exchange plus the endpoint correction. Serial paths evaluate17 boundaries, compared
with32 when independently evaluating every interval's two endpoints. This is an
operation-count substitution, not a speed measurement. For the periodic function
F(t)=Q(t-x)+Q(t-x-1)+Q(t-x+1), quartic CDF Q saturates below/above its support
(`common_device.h:1080–1097`), so for x in[0,1], F(1)-F(0)=1. Arbitrary external
coordinates and floating seam error need their own domain/error binding.

Padding is observable and not uniform across encodings. Frequency, TriangleWave,
OneBlob and Identity use constant1 padding; grid uses0. Offline SH prepends padding
before the basis, while its shown generated path appends it. Layout equivalence
must preserve feature indices or transform corresponding network weights; record
this adapter difference without pursuing unrelated bugs. Base polynomial/basis
coefficients and transcendental errors are not independently numerically certified
in this source pass. These fixed basis families introduce no pipeline or atomics
beyond the recorded fusion and derivative storage/recomputation choices.
