# Replays for the traced DeepGEMM slices

Run from the primary KTA checkout with the existing locked environment:

```sh
.venv/bin/python knowledge/studies/deepgemm/checks/inventory.py
.venv/bin/python knowledge/studies/deepgemm/checks/grouped_ownership.py
.venv/bin/python knowledge/studies/deepgemm/checks/scaled_partials.py
.venv/bin/python knowledge/studies/deepgemm/checks/scale_storage.py
.venv/bin/python knowledge/studies/deepgemm/checks/sm100_schedule.py
.venv/bin/python knowledge/studies/deepgemm/checks/einsum_views.py
.venv/bin/python knowledge/studies/deepgemm/checks/attention_metadata.py
.venv/bin/python knowledge/studies/deepgemm/checks/attention_values.py
.venv/bin/python knowledge/studies/deepgemm/checks/mhc_schedule.py
.venv/bin/python knowledge/studies/deepgemm/checks/mhc_values.py
.venv/bin/python knowledge/studies/deepgemm/checks/mega_gate.py
.venv/bin/python knowledge/studies/deepgemm/checks/moe_layout_schedule.py
.venv/bin/python knowledge/studies/deepgemm/checks/moe_quant_locality.py
.venv/bin/python knowledge/studies/deepgemm/checks/moe_capacity_localized.py
.venv/bin/python knowledge/studies/deepgemm/checks/legacy_auxiliary.py
.venv/bin/python knowledge/studies/deepgemm/checks/cublas_layouts.py
.venv/bin/python knowledge/studies/deepgemm/checks/native_epilogues.py
.venv/bin/python knowledge/studies/deepgemm/checks/config_policy.py
.venv/bin/python knowledge/studies/deepgemm/checks/coverage.py
.venv/bin/python knowledge/studies/deepgemm/checks/validate.py
.venv/bin/ruff check knowledge/studies/deepgemm/checks
.venv/bin/ruff format --check knowledge/studies/deepgemm/checks
```

All twenty-two commands pass. Retained results:

| Result | Domain | What was checked |
|---|---|---|
| [Inventory](inventory-result.json) | 160 superproject files,71 API registrations,28 device source files | File set and Git blob identities against the pinned Git tree; both direct submodule revisions retained and used file hashes checked. |
| [Grouped ownership](grouped-ownership-result.json) | 240 swizzle domains,864 masked schedules, explicit M/K prefix examples | Source index equations versus independent tile enumerations; distinguishes valid rows from whole-tile store extent. |
| [Scaled partials](scaled-partials-result.json) | 8 distinct row/column scale-owner witnesses,42 divisor/tail cases | Exact rational decoded products equal scaled partials; wrong final scale, wrong B owner, stale raw and scale-crossing partial rejected. |
| [Schemas](schema-result.json) | Nine Template/outline pairs, eight Realisations and five partial Chains | Current Pydantic record bodies and expression syntax. |

The scalar replays are transcriptions of source equations, not compiled source or GPU execution. They establish the stated arithmetic domains; instruction semantics, memory safety, synchronization execution and floating-point tolerance remain open. No ProofResult or measured assurance is produced. No package, fixture, gate, commit or Store admission is added by these source-study checks.

First correction pass: fixed flow-YAML quoting; checked the raw accumulator reset through `mma/sm90.cuh`'s ScaleOut::Zero selection; retained the short-group bitstream mismatch as a gap rather than reuse Humming's complete-32-code domain. The metadata documentation's no-gap-read wording is also left open because the device path first loads its SF tile before masking gap values. Full-library correction and review follow the remaining traces.

| Additional result | Domain | Boundary |
|---|---|---|
| [Scale storage](scale-storage-result.json) |254 exact exponents,216 grouped packings,128 uniquely tagged UTCCP pre-transpose words | Normal live-scale identity, endpoint counterexamples, physical packing/transpose equations. |
| [SM100 schedule](sm100-schedule-result.json) | Four mixed-grain pairs, pure-FP4 K64 pairs,15 alias domains | Scale-row selection and interval lifetime conditions; no instruction or race execution. |
| [Einsum views](einsum-views-result.json) |180 strided view addresses, three canonical contractions, one batch/K split boundary | Address and integer contraction identity; prepacked-SF reshape rejected. |
| [Premises](premise-result.json) | Nine valid bindings and twenty-seven invalid variants | Actual Template constraint/precondition expressions evaluate and reject the intended invalid choices. |

Continuation correction pass: fixed seed multi_buffer_rotation references to version2;
retained direct submodule sources and followed the selected SM90 WGMMA body; distinguished
SM100 instruction scaling from SM90 raw-partial promotion. No byte-packing domain expansion
was silently applied to Humming bitstream@1. The einsum trace preserves original-SFB-major /
transformed-SFB-pointer consistency and missing-C FP32 access as explicit open cases.

| Attention result | Domain | Boundary |
|---|---|---|
| [Metadata](attention-metadata-result.json) |1024 source merge-path pairs,1280 dense schedule domains, S3/8 full-ring reuse | Independent set union and slot recovery; padded absent entries produce no writes; all-empty SM90 metadata cursor is retained as an open unsafe-refresh case. |
| [Values/addresses](attention-values-result.json) |256 exact scale/ReLU cases, BF16 tie/cancellation witness, page crossing and compressed columns | Distinguishes scale sign, scale ownership, activation order, rounding and KV/SF byte layout; no GPU execution. |

Attention correction pass: repaired YAML flow-string quoting; bound the sparse split menu to
FP8=512/FP4=640 tokens; kept the BF16 head reduction distinct from a final-only cast.
The schema replay now validates all five partial Chains, Template identities, complete variable
bindings, variable domains, dependency order and Realisation constraint syntax. It checks
referenced baseline bodies against their pinned hashes. Sorted uniqueness is an explicit
premise for the sparse full-contiguous endpoint shortcut; duplicate rows are a counterexample.

| MHC result | Domain | Boundary |
|---|---|---|
| [Schedule/publication](mhc-schedule-result.json) | 256 split policies, 25 epoch/count domains, 97 publication orders, 28 task maps | Exact partition and stale-state rejection; source specialization mismatch and empty standalone split retained; no device synchronization execution. |
| [Values/layout](mhc-values-result.json) | Nine coefficient policies, 32512 BF16 significands and scale cases, 36 SF row domains | Routing orientation, required BF16 rounds, independent lane groups, scale-byte packing and padded backing addresses; no TF32/FP8 instruction emulation. |

MHC correction pass: bound producer count at compile time, retained the release/acquire and
resident-grid progress obligations, and kept the finite iteration policy distinct from
convergence. The partial chain leaves split combination, RMS and affine edges external.
Standalone SM100 N40 violates the device swizzle condition after host admission; K64/S2
has an empty split. These are source counterexamples pending compilation/device evidence.

| MegaGate result | Domain | Boundary |
|---|---|---|
| [Selection/routing](mega-gate-result.json) | 81 sort-network inputs, 168 top-k domains, 540 token schedules, 48 duplicate selectors and 1904 distinct-draw domains | Global-sort oracle checks exact identity order including ties/padding; distinguishes ranking bias from weights, fixed duplicate slots and random physical-ID premise. |

MegaGate correction pass: preserved uint32 wrap in duplicate selection and three-bit
cursor exhaustion, separated grid-id/64-wide publication from MHC's shared epoch/128-wide
rule, and retained rounded-activation tie collapse as a numerical boundary. No device
collective, RNG distribution, graph replay or configuration performance claim is made.

| Partial MegaMoE result | Domain | Boundary |
|---|---|---|
| [Layout/schedule](moe-layout-schedule-result.json) | 33792 tagged weight/SF coordinates, 262 rank-count domains, eight M widths, 3492 publication orders across generations and 36 symmetric mappings | Independent row/rank enumerations; repeated-toggle, generation-lap, metadata-alias, duplicate-routing and weighting-round counterexamples. |

MoE correction pass: kept the partial family status, distinguished full logical source
metadata from reusable payload rows, and retained ring capacity/progress as obligations
rather than consequences of parity. The independent pointer/slot checks distinguish
remote allocation bases and show that readiness padding is not a valid-row count.

| Quantized MoE / locality result | Domain | Boundary |
|---|---|---|
| [SF/prefetch/locality](moe-quant-locality-result.json) | 16 warp-pair row groups, 5376 SF byte writes over six tile sizes, 463 prefetch/tail domains, 340 locality label maps | K32 amax ownership, SF address bijection and exact last-load release; 98 initially balanced maps become unbalanced, with a no-steal queue coverage counterexample. |

Continuation correction pass: used XOR4/8/16 for intergroup amax instead of an adjacent
four-lane reduction; kept FP4 TMA expansion at its unexecuted datatype/ISA boundary.
Locality queue evidence is scalar source arithmetic and does not claim a measured hang.

| MoE capacity / localized descriptor result | Domain | Boundary |
|---|---|---|
| [Capacity/addresses](moe-capacity-localized-result.json) | 640 synchronous-wave geometries, 32 padded descriptor domains and 7168 tagged coordinates | Source warmup/live-block bound covers the bounded wave model; padding-aware descriptor indices agree with direct localized coordinates. No asynchronous capacity proof, allocator or IPC execution. |

Closure correction pass for these groups: retained the synchronous-model boundary,
followed the external allocation handle across IPC import/service shutdown/tensor
release, and distinguished driver/resource constraints from runtime evidence. Both
source groups now have variant dispositions; no blanket correctness or admission follows.

| Legacy / auxiliary result | Domain | Boundary |
|---|---|---|
| [Raster/gather/shapes](legacy-auxiliary-result.json) | 363 raster domains, 120 integer gather outputs and 120 actual SF shape-helper inputs | Exact tile coverage, operand ownership, unmasked group-tail witness, row-major return-shape mismatch and per-hidden-block scale-group mismatch. |

Correction pass: distinguished IDE header indexing from a runtime operator and kept
source-policy descriptions separate from executed checks. The shape helper is executed
from the pinned AST; Triton/TileLang lowering and CUDA behavior remain unexecuted.

| cuBLASLt descriptor result | Domain | Boundary |
|---|---|---|
| [Layouts](cublas-layouts-result.json) | 64 major/batch geometries, 896 integer outputs and 630 einsum contraction coordinates | Swapped operands, independent padded batch strides and asymmetric scale extents; no external algorithm, scale decoder or floating-point execution. |

cuBLAS correction pass: followed early_return's distinct-C copy before interpreting
beta, kept native UE8M0 separate from NVFP4 UE4M3, and preserved the direct third
einsum's unresolved deterministic behavior. No new candidate identity is introduced.

| Native / epilogue result | Domain | Boundary |
|---|---|---|
| [Fragments/numerics](native-epilogues-result.json) | 256 swapped fragment/hash positions, 27 batched SF domains with 3024 bytes, 128 merged-stage cases and 15 per-column scale outputs | Same coordinate/hash identity across layouts; BF16 count/accumulation round boundaries and existing promotion rule's gran_n1 binding. No random-distribution, cvt.rs or GPU correctness claim. |

Native correction pass: followed mandatory C for SM90 empty-group skips, kept SM100's
dummy initializing MMA distinct, and preserved BF16 rounding before FP8 amax/count.
Hash-coordinate consistency is separate from stochastic-rounding statistical assurance.

| Final configuration / coverage results | Domain | Boundary |
|---|---|---|
| [Configuration](config-policy-result.json) | 288 SM100 BF16 default-policy cases and scaled K-group store thresholds | Source menu/ranking and shared-byte limits; no compiled selector or measured optimum. |
| [Coverage](coverage-result.json) | 71 API registrations and 28 device sources | Explicit trace assignment completeness and link existence against the pinned inventory; semantic review remains independent. |

Final correction pass: kept expected shape separate from runtime ownership, bounded
policy replay to BF16/default settings, reconciled all source-family rows and preserved
implementation gaps. The entire listed check batch was rerun for source-study closure.
