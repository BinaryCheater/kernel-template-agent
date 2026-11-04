# Native host variants, configuration and coverage closure

All paths refer to [the retained source](inputs/sources.json). The source-selected
policies below are not measured performance results. Exact instruction legality,
compiler/resource behavior and device execution retain the obligations in the prior traces.

## Configuration decisions

| Source | Independent input, derived quantity or policy | Rule / disposition |
|---|---|---|
| `csrc/apis/config.hpp`, `heuristics/runtime.hpp` | SM budget, tensor-core utilization, PDL, deterministic flag, compile-dimension suppression and block multiples are runtime control inputs | Backend/dispatch settings, not new Template variables. Settings are not universally data-validated; invalid multiples can leave no candidates or invalid arithmetic. |
| `heuristics/config.hpp:14–68` | GemmDesc records actual shapes separately from expected shapes/groups; expected values fall back to actual when nonpositive | Expected values score configurations, actual device metadata owns work. BF16 vs mixed FP8/FP4 vs pure FP4 determines element bits and shared pack factor. FP8 output is restricted to nonaccumulating Batched descriptors. |
| `heuristics/common.hpp:19–40` | Enumerate layouts, select winner, then infer storage, pipeline and launch | Layout is the primary choice. Stages/swizzles/launch fields are derived and cannot be independently rebound while claiming this source policy. Source assertions reject empty menus. |
| `heuristics/sm90.hpp:17–114` | Normal/batched/K-group M64/128 plus small M16/32 and BF16-output M256; grouped M uses alignment or64/128; N menu depends on kernel kind | Existing output tiling/rasterization/physical-storage instances. Some filters encode performance preferences (large swizzle, enough stages), others shape/resource legality. K-group count>4 and batched suppress multicast by policy. |
| SM90 `:154–247` | Shared capacity232448, max16 input stages, extra SF/descriptor bytes; launch producer128 and math128/256 | Resource formulas and provisional warp/barrier bindings. Score is an L1/L2 byte model adjusted by wave efficiency, with embedded bandwidth constants and single-wave multicast penalty. These are upstream estimates, not KTA hardware calibration. |
| `heuristics/sm100.hpp:31–166` | Grouped-M fixes swap-AB,N128,M=alignment and optional pair along N; other families enumerate swap, M/N and cluster | Normal swapped M steps lcm(16,user multiple) to256, fixed N128; ordinary M32/64/128 by actual M and N16 or steps lcm(32,user multiple). K<=256 caps ordinary N128. Major/swizzle, multicast divisibility, SF/TMEM and complete output scale groups constrain the menu. |
| SM100 `:169–262` | Shared capacity232448, max32 stages; two TMA store stages except selected non-swapped scaled K-group cases | One-store-stage thresholds are16 K blocks for non-FP32,24 for accumulated FP32,32 otherwise. Average group K drives that heuristic, not every group's runtime K. BF16 and swapped branches always use two. |
| SM100 `:264–302` | Rank expected waves, cluster size, last-wave utilization, tile sum and area | Single-wave wins first; otherwise prefer cluster, fewer waves, fuller tail, smaller sum then area. Cycle estimate remains zero here. A performance optimum is not asserted. |

[config_policy.py](checks/config_policy.py) replays the SM100 BF16/default-multiple
menu and comparator over288 M/N/K/major cases on the132-SM source witness. It checks
chosen multicast/shape guards and independently bounds derived shared bytes/stages.
For M1,N256,K1024,K-major inputs, source policy selects swap,BM16,BN128,cluster1×2,
12 input stages. This is a bounded policy substitution; other type/custom-multiple
menus retain their source constraints and implementation obligations. The selected
source12-stage ring is not admitted into a seed rule capped at8 stages.

## Host/build variant reconciliation

| Native family | Dispatch/build path and distinguishing contract | Source-backed evidence |
|---|---|---|
| BF16 Normal and NT/NN/TN/TT aliases | Architecture9/10 native path unless nondeterministic plain alpha-store dispatch permits cuBLAS; aliases transpose views | Native/epilogue and cuBLAS traces; actual shape/major goes into TMA descriptors and kernel specialization. |
| BF16 grouped M | Contiguous row-ID or prefix-end, masked capacity; SM100 optional expected prefix M and ensure_zero_padding | Scheduler ownership/padding trace; source uses expected M for scoring only. SM90 custom epilogues rejected; SM100 resolves one class. |
| BF16 grouped K | MN-major operands; SM90 CPU aligned lengths+C, no prefix; SM100 row/prefix layouts and direct/accumulated output | Empty-group initialization/preservation and physical K vs output group offsets in native trace. |
| Scaled Normal and grouped M | SM90 FP32 SF selects1D2D or per-column1D1D by B scale granularity; SM100 packed SF selects scaled UMMA | Scale storage, promotion and native traces. Recipe/type gates are distinct from public FP4 naming aliases. |
| Scaled grouped K | SM90 K-major+K128 SF and mandatory C; SM100 matching K-grain constraint, packed metadata and row/prefix offsets | SM90 descriptor mutation and SM100 SF selector traces. Direct-empty behavior follows selected backend. |
| Batched BF16/fixed BF16 einsums | BF16 runtime constructs Batched descriptors; two expressions support native or plain cuBLAS epilogues; third uses cuBLAS | Einsum/native/cuBLAS traces, including FP8 output metadata columns flattened over batch,N. |
| SM100 FP8/FP4 mixed and pure FP4 | Mixed shared unpack vs pure packed storage, separate K32/K128 scale selectors and accumulator/SF interval ownership | Existing scale/SM100 trace; no post-partial SM90 numerical equivalence is inferred. |
| Compiled vs runtime dimensions | Host emits specialization constants or zero; device substitutes runtime value only for zero | JIT/backend binding. Changing ignored compile dimensions does not change shape/stride validity or runtime metadata ownership. |

These rows account for each remaining host variant; no new arithmetic mechanism is
introduced by a duplicate alias or a different specialization string. Existing rule
identities remain pinned to the starting snapshot, with explicit domain gaps rather
than silent parameter expansion.

## Shared helpers and final boundary

The implementation headers' common math/packing, descriptor/MMA/PTX wrappers, layouts,
schedulers and epilogues are followed through their consuming traces. `RingPipeline`
requires steps no larger than one full ring; its non-power-of-two branch subtracts
once, so arbitrary multi-ring steps are outside its contract. `packing.cuh` distinguishes
packed FP4 (two elements/byte) from the unpacked shared type. Compilation/exception
macros and tuple utilities add no runtime operator. Optional `common/probe.cuh`
instrumentation is disabled unless DG_PROBE_ENABLE, writes fixed-capacity8-byte clock
records, and requires an aligned buffer within one4GiB window because only the cursor's
low32 bits advance. No probe timing is used as source-study evidence.

Runtime/JIT/build and package files are infrastructure, with selected CUTLASS/DeepJIT
bodies retained separately. External cuBLAS, Triton/TileLang lowering, CUDA driver/ISA,
PyTorch symmetric memory and MLOPart execution are implementation boundaries, not
new superproject kernels. The IDE include target is already accounted separately.
Tests/generators document caller premises and unsupported cases, not proof that those
cases were executed in this study.

The coverage manifest maps every discovered API and device body to these traces.
Its mechanical check detects omissions/duplicates and missing links; semantic quality
still depends on the trace arguments and independent review. Full-library source
extraction may close with these declared proposals and explicit implementation gaps;
no Store admission, GPU correctness, performance model or WP delivery is claimed.
