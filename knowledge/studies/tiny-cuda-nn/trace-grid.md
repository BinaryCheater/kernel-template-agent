# Multiresolution lattice tables and interpolation

When encoding coordinates at several resolutions, separate the field's
parameter-sharing map from its interpolation stencil and from the GPU schedule.
Only the first determines which nominal grid vertices are the same trainable
parameter. Sources are pinned in the [manifest](inputs/sources.json).

Representative: 2D CoherentPrime HashGrid, two FP16 features/level, two levels,
base resolution4, per-level scale2, table cap8, linear interpolation, ordinary
floating coordinates, exact corner-gradient path, overwrite gradient mode.
`encodings/grid.h:1730–1771,1830–1860` compiles 2/3/4 position dimensions and
1/2/4/8 features per level; only CoherentPrime is enabled. The dormant
BaseConvert branch is not an available higher-dimensional path.

| Source binding / design choice | Rule and field, or gap | Evidence / remaining obligation |
|---|---|---|
| `common_device.h:886–895`: scale `base*2^(level*log2_step)-1`, resolution `ceil(scale)+1` | Instance geometry; not a performance knob independent of encoding semantics | Levels have scales3,7 and resolutions4,8. Base resolution counts vertices, not cells. |
| `grid.h:694–730`: each level gets rounded-up-to-8 dense vertex count, then Dense/no cap, Tiled/base-grid cap, Hash/hash-budget cap; prefix offsets in vector entries | `segmented_aligned_storage@1` is only a storage analogue: use an explicit prefix allocation binding here; new lookup rule takes scalar offsets | Representative vector offsets0,8,16; scalar offsets0,16,32. Padding can be reached by modulo; it is not necessarily dead storage. |
| `common_device.h:848–883`: linear map with overflow threshold, conditional hash only if table smaller than stride, modulo final size | [bounded_lattice_parameter_lookup@1](delta/bounded_lattice_parameter_lookup@1/template.yaml): address_mode/hash, table_entries8, dimensions2, features2 | At R4,H16 the hash is bypassed; at H8 it is active. Tiled mode uses linear-index modulo, not per-coordinate wrapping. |
| `common_device.h:1001–1049`, `grid.h:100–118`: `u=fma(scale,x,.5)`, floor cell, interpolate fractional part | Coordinate/stencil binding; **explicit stencil Template pending** | Half-cell staggering is part of the encoded field. Coordinate range, signed-to-unsigned wrap and boundary conventions must match all consumers. |
| `grid.h:121–165`: Nearest reads lower shifted cell; Linear/Smoothstep enumerate 2^D corners with product weights | Workload algorithm boundary; `reduction_partition@1` explains only summation after weights/corners are defined | Smoothstep uses `t*t*(3-2*t)` per axis. Result uses T-valued weights and T-valued FMA; do not replace with an FP32 weighted sum without numerical admission. |
| `grid.h:175–208`: input derivative as opposite-corner differences, multiplied by scale, other-axis weights and interpolation derivative | Finite tensor-product derivative **gap**; no claim that reduction partition alone supplies it | Each gradient axis reads 2^(D-1) pairs; linear derivative is1, smoothstep derivative is6t(1-t). Piece boundaries need the declared derivative convention. |
| `grid.h:246–320`: exact parameter gradient sends each corner's weighted upstream derivative to its mapped parameter | New lookup rule's gradient_pullback obligation + atomics | Colliding corners cannot overwrite or deduplicate contributions. FP16 feature pairs may use packed atomicAdd; feature count1 uses FP32 scratch under the architecture branch at650–670. |
| `grid.h:284–299`: optional stochastic update selects a single corner using one shared sample across all dimensions | Approximate path; excluded from exact stencil-gradient binding | It does not preserve the multilinear per-corner distribution in general. See separating witness below; no unrelated implementation investigation. |
| `grid.h:323–350,775–801`: optionally save FP32 Jacobian, later contract over all features for input gradient | `physical_storage@1`: retained Jacobian occupancy; `reduction_partition@1`: feature sum | Saves `samples*features*D*4` bytes; source input-gradient call needs matching forward context. |
| `grid.h:354–657,915–1043`: second derivative paths reuse corner differences, smoothstep diagonal second derivatives and cross-axis terms | Higher-derivative binding remains partial until explicit stencil rule | Three consumers: table gradient, upstream-output gradient via saved Jacobian, and input Hessian-vector contribution. Input accumulation over levels is atomic and initialized to zero. |
| `grid.h:775–815`: CTA y selects level, x selects samples, 512 threads forward, feature-major output; AoS requests add transpose | `output_tiling@2` custom map; `physical_storage@1` for transpose scratch | One level per CTA is a locality choice. The source comment's cache rationale is a hypothesis, not a timing or a guarantee that only one level is resident globally. |
| `grid.h:865–898`: backward assigns up to two features/thread,256 threads, level in y | Output ownership and lookup pullback binding | Exact collision-safe accumulation and gradient epoch initialization are required. Existing alternate-FP32-scratch accumulate mode does not visibly import the prior gradient; record this host-mode issue without chasing it. Overwrite representative supplies initialized storage. |
| `grid.h:74–92`: per-sample or global hard max level suppresses outputs and derivatives | Explicit mask binding | Level >= normalized_limit*level_count+.001 is zero. This differs from a continuous fade. |
| `grid.h:1138–1218,1230–1348,1592–1695`: generated per-sample lookup unrolls levels, fuses into caller and computes derivatives locally | Lookup map reusable; schedule differs; explicit stencil/coordinate variant pending | Generated lookup supports fixed-point position bit patterns; offline shown kernel uses floating `pos_fract`. Generated shown bodies enumerate all levels and exact corners; offline max-level and stochastic flags cannot be presumed equivalent. |

## Separating source substitutions

Take level0 x=(1/4,1/12). In exact arithmetic shifted coordinates are (1.25,.75),
cell=(1,0), fractions=(.25,.75). Linear corner weights in x-bit-first order are
`.1875,.0625,.5625,.1875` for offsets00,10,01,11. With H8 coherent hashing,
coordinates(1,0) and(0,1) both map to1; coordinate(1,1) maps to0, and(2,1)
maps to3. At R4,H8, coordinate(1,2) maps to3 under hash but1 under tiled linear
modulo, separating those address explanations. Level1 starts at scalar16;
feature1 of mapped entry3 is scalar23, not vector offset8+3+1.

For stochastic mode and fractions(.25,.75), a uniform shared sample U produces
11 on U<.25, 01 on .25<=U<.75, and00 otherwise. Its probabilities are
(.25,0,.5,.25) in order00,10,01,11, rather than the four multilinear weights.
In particular corner10 has positive exact weight .0625 but is never selected.
This settles the relevant abstraction boundary: that source branch cannot
instantiate an exact or independently sampled unbiased tensor-product gradient
rule. RNG quality and downstream training quality are not investigated.

The generated fixed-point mode, high-resolution overflow branch and Hessian
numeric details remain named gaps. No native converter, atomic or interpolation
run was performed. The source substitutions establish the map/ownership choices;
they do not establish GPU numerical error or measured cache benefit.
