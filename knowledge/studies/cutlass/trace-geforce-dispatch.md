# SM120 block-scaled GEMM: dispatch and register payload placement

When a load unpacks a narrow floating payload into a byte but the arithmetic
instruction reads a different bit field, preserve the code and move its position
explicitly. That transformation is neither integer widening nor a new numeric
quantizer. The new
[operand_bitfield_placement@1](delta/operand_bitfield_placement@1/template.yaml)
records that load-to-consumer interface and exact-once preparation requirement.

Scope is shared `blockscaled_gemm_dispatch.py` plus the selected mixed-format call
sites in both persistent kernels under
`examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/`.
[All three files](inputs/geforce-blockscaled-variants.json) are inventoried at the
[pinned revision](inputs/sources.json). This slice source-extracts shared dispatch
and code placement. The [schedule](trace-geforce-schedules.md) and [layout/host](trace-geforce-layouts.md)
companions complete those source routes with explicit native obligations.

## Selected branch and decision table

Use A=Float4E2M1FN, B=Float8E4M3FN, SF=Float8E8M0FNU, sf_vec_size32,
FP32 accumulation and tile128×128×128 on SM120. Both CLI menus admit this mixed
combination. Reversing A/B also is supported. The corresponding warp MMA constructor
is MmaMXF8F6F4Op, with K32 permutation mode. Native target/constructor legality,
register lane maps and complete tensor alignment remain implementation obligations.

| Decision and source | Actual selection | Rule field or gap |
|---|---|---|
| Dispatch:110–202 | Same-type FP4+E4M3/vector16 selects MmaMXF4NVF4Op; same-type FP4+E8M0/vector32 selects MmaMXF4Op, K64 mode. Same-type E4M3 or E5M2+E8M0/vector32 selects MmaMXF8Op, K32 mode. | Workload format and backend dispatch. Source header saying only FP4 is supported is stale; executable branches are authoritative. Factory replay records symbolic constructor calls, not successful native construction. |
| Dispatch:173–202; `cute/nvgpu/warp/mma.py:813–820` | Four ordered FP4×FP8 pairs select MmaMXF8F6F4Op; sf32/E8M0 required. Unequal FP8 types, FP6 and other pairs are rejected. | Shared selected pair set is checked against the actual imported declaration. A broader instruction family name does not admit another source dtype pair. |
| Validator:205–301; cooperative CLI:2006–2009; pingpong:2052–2055 | Both FP4 menus contain128×128×128 and128×128×256; MXFP4 still requires tileK128. FP8/mixed menu is only128×128×128. | Actual compiled source menus, not a measured winner or a complete backend capability table. NVFP4 is the only listed K256 case. |
| Helper:67–81; validator:252–253 | The helper recognizes same-type MXFP8 tile128×256×64, but neither actual CLI menu includes it. Mixed mode excludes it even if a caller supplies a wider menu. | Unselected helper declaration only. Do not infer a connected K64/SF128 pipeline or propose a verified scale-window binding from that helper's description. Direct construction/lowering of this special tile remains open. |
| Cooperative setup:193–216; pingpong:200–223 | mixed_mode means A type differs from B; only the FP4 side uses Int8 shared allocation/TMA internal type. Permutation mode follows FP8/mixed K32 versus same-type FP4 K64. | Physical reinterpretation and backend layout glue. Int8 is a storage/view type, not permission to reinterpret FP4 as a signed integer numerical value. Full TMA and shared-size implications await schedule tracing. |
| Shared load helper:84–107 | Mixed FP4 selects LdMatrix8x16x8bOp, unpack_bits4, no transpose, Int8 copy element. Other paths use LdMatrix8x8x16bOp. | Unpack and consumer must agree on payload order. The selected copy leaves each floating code in low byte bits; exact native lane/register mapping remains an obligation. |
| Cooperative:848–866, corresponding later fragment sites; pingpong:1011–1030 and later sites | Only mixed-mode side with width<8 shifts each Int8 register view by FP4_SHIFT_BITS=2 before GEMM. Eight static shift expressions exist across both kernels, covering A/B and their two prepared-fragment sites. | New operand_bitfield_placement@1 destination_lsb2, four code bits/eight-bit container. Exact-once preparation after each unpack is required; eight static sites do not prove a dynamic fragment lifetime. The schedule trace must check those edges. |
| `include/cute/atom/mma_traits_sm120.hpp:211–217,241–257` | Independent C++ traits describe low four bits moving to bits2..5 for the mixed block-scaled instruction; FP6/FP8 need no such shift. | Reference corroborates the representation contract. It is not a selected C++ backend for this Python witness or native proof of the DSL path. |

## Meaning and nearest old fields

For every code0<=q<16 and destination position0<=d<=4, the byte q<<d fits and
extracting `(byte>>d)&15` recovers exactly q. The source fixes d2. This preserves
E2M1 sign and signed-zero **code patterns**, without doing a floating conversion.
For example, E2M1 code0xA denotes-1; its prepared byte is0x28, whose consuming
four-bit field remains0xA. Omitting or repeating placement changes that field.
Other destination positions are logical maps only until a load/consumer binding
supports them; they are not advertised source tuning options.

`integer_operand_widening@1` instead decodes signed4/signed8/unsigned8 into FP16 or
BF16 numerical values. A signed4 code0xA denotes-6, so that old input format cannot
supply this FP4 code identity. Its compute-format menu also does not describe an
instruction's middle-byte payload. Marlin fused_dequantization and prepacking do
not own this register-local representation bridge either.

The retained reconciled Humming `bitstream_packing@1` permits opaque code
interpretation but fixes contiguous bit coordinates i×code_bits+b in32-bit words.
This instance instead places payload bits at8×i+2+b after unpacking, leaving gaps.
Neither changing a width label nor choosing a word format supplies that shift.
The Humming record is compared here, not reused; its old unavailable body is not
assumed equivalent to the retained body. Original and reconciled snapshots remain
addressable.

## Evidence and continuation

[Retained checks](checks/geforce-dispatch-result.json) execute the actual shared
validator and symbolic constructor branches after dtype-identity substitution.
Across1296 CLI cases,18 are accepted and1278 rejected. Constructor calls are
returned as argument descriptors; no fake MMA/load implementation executes.
The K64 helper accepts its declared same-type witness while both real menus reject
it. Mixed FP4 transposition is rejected by the source helper.

Eighty code/position cases,256 distinctive four-byte packets and all eight retained
shift expressions check logical code identity and byte separation. Omission and
double-placement are discriminating counterexamples. Native LDSM behavior, unused
bit guarantees, coordinate mapping and dynamic exact-once preparation remain open.
The proposed rule and its guard mutations pass schema/expression validation.

Both kernels are source-extracted across the companion traces; native obligations remain.
No GPU, CUDA/DSL compilation, numerical GEMM comparison, performance measurement,
Realisation or Store admission occurred. The companion traces retain cooperative/pingpong lifetime and layout obligations.
