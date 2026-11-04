# SM120 convolution scale storage, host reference and stage policy

When scale metadata has a coarser tile than the data operand, preserve its logical
group independently of host blocking, shared layout and MMA lane partitioning.
When choosing a stage count, size the **actual** staged arrays and aligned fixed
fields rather than treating data bytes as the whole allocation. Existing patch,
transfer and stream rules apply only with those implementation bindings supplied.
These mappings and resource heuristics add source instances, not new rule identities.

This completes source extraction of both GeForce convolution entry files after
the [frontend](trace-geforce-conv-frontend.md), [input](trace-geforce-conv-inputs.md)
and [output](trace-geforce-conv-outputs.md) traces. The
[inventory](inputs/geforce-conv-variants.json) retains every function/method range.
Paths below are relative to `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/`
in the [retained tree](inputs/sources.json). Source closure retains the listed
native encoding/layout/progress obligations; it is not backend admission.

## Physical and resource choices

| Source and choice | Recovered binding | Rule boundary or remaining obligation |
|---|---|---|
| Scaled `_setup_attributes`:213–267, `_make_smem_layouts`:2873–3003 | SF N rounds to128 even for data N64/96; SF K=max(data K,4*vector). All scale stages have full-chunk geometry. A/B use the selected Hopper shared atom, source dtype width and1024-byte array alignment; D/residual use the same per-subtile atom. | Explicit physical implementation of the logical scale map, separate from data tiling. Residual congruent fragments and native TMA/LDSM compositions remain required. Mixed internal Int8 allocation branches are not selected by equal-type host inputs. |
| Scaled `_sm120_make_smem_layout_sfb`:3177–3277 | N atom(32,min(4,N/32)), strides(16,4); scale-group chunks4, stride512 per N chunk; stage stride=ceil(N/128)*512*(SFK/vector/4), for legal N≥128 multiples or smaller helper N. | Source helper metadata checked at15 domains/9216 distinct scales. Nine helper-only narrow-N domains have cosize smaller than stage stride: append must use whole chunk stride. Actual host setup supplies SF N128, so those narrower helper branches are not connected convolution instances. |
| Scaled SFA shared helper, selected by `_make_smem_layouts` | Standard `sm120_make_smem_layout_sfa` with SF tile(128,128,SFK). Repeats a scale within its vector group by zero stride; each logical(row,group) has a distinct storage address. | Same logical/physical distinction as the [GeForce GEMM layout trace](trace-geforce-layouts.md); its native composition obligations remain. Input-trace checks establish the new scalar spatial address binding, not native lane equivalence. |
| Scaled `_compute_stages`:2698–2870 | Build `ProbeStorage` from the same six layouts and13 fields as runtime `SharedStorage`; reject if one stage exceeds capacity. Compute a conservative lower bound using raw per-stage slope plus two8-byte barriers; reserve four1024-byte alignment losses, then scan upward until the next size exceeds capacity. | All13 probe/runtime field annotations match after exact variable substitution. The retained search returns the maximal fitting stage count on32 abstract aligned monotone cost domains. Native `cute.cosize`, struct ABI/alignment, monotonic growth and capacity still need actual compilation; this is not a measured throughput optimum. The error message says `can_implement` should reject oversized tiles, but its actual predicates do not evaluate this exact size. |
| Dense `_compute_stages`:1613–1660 and `_setup_attributes`:544–563 | Fixed output stages4; estimate A/B stages from logical operand bytes, output bytes and fixed helper/reserve terms; reject a nonpositive result. Actual struct also contains aligned A/B/D, bias ring and pipeline fields. | Source estimate, not the scaled path's field-for-field probe. Native capacity fit for admitted dense tile/type/bias combinations remains an explicit resource obligation. Do not transfer the scaled size-proof claim to dense. |
| Dense `_compute_grid`:1725–1749; both host compilers | Flatten output into CTA tile counts; cluster1 CLC scheduler. Query active clusters on host, pass a symbolic compile stream and real launch stream separately. Scaled allows query failure→active1 only when `skip_ref_check` enables that explicit option; `no_cache=True` requests recompilation. | Launch/JIT policy, not a demonstrated hardware capability. Compile-only symbolic stream does not stand in for kernel execution. Runtime geometry is boxed consistently at compile and launch; selected types/tile/activation/beta remain compile choices. |

The stage choice here is a deterministic capacity heuristic. The source comments
about deeper A/B rings or residual-stage1 winning performance do not establish a
general performance ordering. A later optimizer may vary these choices only after
providing a cost model and the same lifetime/resource constraints.

## Host values, layouts and references

Scaled `create_scale_factor_tensor_swizzled`:3313–3390 allocates SFB with original
shape `(L,ceil(N/128),ceil(SFgroups/4),32,4,4)` and permutes axes(3,4,1,5,2,0).
The converter groups the row and scale modes, copies each valid logical coordinate,
then the reference expands one scale over `vector` channels. Host padding outside
the valid N/group shape starts at float zero; its eventual low-float encoding is
a native obligation. In contrast, padding **inside each filter position's C span**
is part of the valid allocated scale tensor and receives the finite random1/2
values. The reference reshapes by that padded span, then crops each filter position
back to C. Cropping only a single flattened K suffix would select wrong later taps.

The [layout check](checks/geforce_conv_layouts.py) executes retained shape and
permutation constants for64 domains/117504 uniquely identified scale coordinates.
It recovers host storage from logical nested coordinates without repeated low-bit
labels. These are host-layout arithmetic checks; native CuTe layout and device
conversion were not executed. SFA is naturally `(N*D*H*W,C/vector,1)`, with dynamic
channel-group stride. It needs no row padding because the host gate requires whole
four-factor atoms.

Both scale generators use integer random1/2, exactly representable as E4M3 and
E8M0. Swizzled SFB explicitly round-trips the reference through the storage dtype;
SFA's integer fixtures are already exact. Both entry activation menus contain identity and ReLU; the retained device ReLU
selects x only when x>0 and otherwise zero. Its scalar NaN comparison therefore
selects zero; a requested NaN/signed-zero policy needs explicit numerical
admission. The general E8M0 rounding/zero/NaN policy
is not tested by these values. This is input-scale preparation, not
`groupwise_low_float_quantization@1` output quantization or a trained-tensor
quantizer. Source helper comments claiming a theorem forbids TMA for all SFA are
not adopted: the observed implementation uses software cp.async for its spatial
gather; no universal impossibility follows from this source trace.

Shared `prepare_tensors`:2019–2051 creates input/filter float values−1/0/1. They
remain exact in the selected input formats. `create_cute_tensor`:1977–2016 packs
FP4 two elements per byte with an even trailing dimension, then converts from the
source tensor; other types use `cutlass/torch.py:273–338`, which initializes the
allocated tensor by conversion or copy. This is not an uninitialized-input path.
These small fixtures do not validate general low-float range, overflow or rounding.

Scaled `run_conv`:3745–3807 multiplies the input/filter by their expanded scales,
uses ordinary conv3d for symmetric padding or explicit F.pad followed by conv3d
for asymmetric padding, then applies alpha, stored-dtype bias, stored-dtype
residual and activation. It converts device D to FP32 and compares with supplied
absolute tolerance and relative tolerance1e−2. This matches expression order;
matrix reduction grouping, fused arithmetic and native conversion still require
numerical admission. Dense uses the corresponding unscaled reference and retains
the FP16 accumulator conversion boundary from the output trace.

The timing helpers rotate only A/B/D for cold-L2 workspaces; scales, alpha, bias
and residual remain shared. No whole-working-set cold-cache or performance result
is inferred. This study ran no example timing or native reference comparison.

## Correction pass and closure

- Distinguish `can_implement` from the scaled `run_conv` wrapper: the wrapper
  separately rejects malformed/nonpositive strides and dilations, nonpositive TRS
  and nonpositive **individual** output extents before allocation. The earlier
  predicate-only acceptance witnesses remain valid at that narrower boundary;
  scaled wrapper does not pass those malformed geometries through unchanged.
  C0 and nonnegative-padding premises still are not fully established by it.
- H32769 remains accepted by the scaled wrapper's geometry checks. Its NVFP4
  signed16 H/W base-packing discrepancy remains an implementation boundary.
- N96's custom shared helper supports narrow metadata, but actual setup rounds
  SF N to128. MXFP8 C192 is a helper-only span example, excluded by its C128
  atom-alignment gate. Neither is counted as an admitted new host branch.
- Both order arrays reserve four slots; both output paths drain source reads.
  Existing standalone GEMM two-slot/missing-tail findings do not apply here.
- Dense FP16 accumulator/bias casts, C0 hoisted final MMA, native zero/finite scale
  encoding, fixed-width address bounds and failed-CLC-query lowering remain open.
  Source closure preserves these obligations rather than admitting the backend.

[Check result](checks/geforce-conv-layouts-result.json) retains layout counts,
13-field agreement and bounded search domains. No new candidate or version is
added: the existing patch, joint completion, striped stream, lifetime and output
rules express the reused choices; source-specific hoisting/packing and resource
heuristics remain explicit implementation bindings/gaps. Both entry files now
have source-backed dispositions. Next: the remaining GeForce dense GEMM entry,
then the outstanding DSL/C++ families in the full CUTLASS inventory.
