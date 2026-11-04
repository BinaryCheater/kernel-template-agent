# SM120 convolution geometry and entry menus

When a channel-last cross-correlation needs a GEMM implementation, retrieve
[implicit_patch_contraction@1](delta/implicit_patch_contraction@1/template.yaml).
Its `patch_storage=on_load` choice and `program` supply the virtual patch matrix:
output coordinates flatten Q/P/Z/batch, reduction coordinates flatten C/S/R/T,
and an input coordinate subtracts **lower** padding before adding the dilated
filter tap. Preserve that map while selecting a transfer implementation. Positive
extents, strides and dilations are separate premises; an entry-menu acceptance
does not supply them.

This slice inventories both SM120 convolution entries and extracts their frontend
only. [Inventory](inputs/geforce-conv-variants.json) retains all methods and helpers;
device schedules, scale copy coordinates, output arithmetic and host layouts are
still active. Paths below are relative to the [retained source tree](inputs/sources.json),
under `examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/`.

## Decisions and conditional reuse

| Source decision | Concrete binding | Existing owner or remaining gap |
|---|---|---|
| Dense `_setup_conv_input_attrs`:361–393 | A NDHWC and B KTRSC have contiguous C; D NZPQK has contiguous K. A/B types match. | `implicit_patch_contraction@1.program` supplies the logical indexing. This is not a general grouped-channel or transposed convolution. Native strides, descriptor lowering and tails remain `pre_backend` obligations. |
| Dense `_setup_conv_tma`:395–504 | A select(3,2,1,0,4), group first four modes; B select(0,4,3,2,1), group last four; D select(3,2,1,0,4), group first four. Add a dummy batch1 to GEMM tensors. A uses im2col G2S, B tiled G2S, D im2col S2G. | Exact old `program` flattening and `patch_storage=on_load`, conditionally. Output S2G belongs to the staged output rule rather than `async_transfer@1`'s G2S direction. Device visitation/order is not yet established here. |
| Dense `_compute_im2col_params`:235–283 | Reverse DHW into WHD; lower corner=-lower padding; upper corner=upper padding−(filter−1)*dilation; traversal stride=stride, tap stride=dilation. Filter extents and pad/stride/dilation feed runtime descriptor operands. | Old `patch_identity` and `implementation_bounds` obligations. Runtime geometry versus compile-time tile/type choices is explicit. Distinct-axis replay detects reversed axes and swapped padding sides. |
| Dense `_check_im2col_descriptor_limits`:143–232 | Each corner in[-16,15], each maximum tap offset≤31, each traversal stride≤8. | Source implementation guard for this 5D encoding; not an optimizer knob. Positive stride/dilation, nonnegative padding and positive filter sizes still come from `c_geometry`, not this upper-bound helper. |
| Dense `can_implement`:1751–1902 | A/B f16,bf16,E4M3,E5M2; accumulator f16/f32; D those four input floats or f32. M divisible32 and, above64, divisible64; N divisible32; tile K in32,64,128. Bias type matches D. C/D contiguous axes aligned16 bytes; at least one cluster's flattened M/N tiles. Constructor fixes cluster1 and atom layout(2,2,1). | Compiled menu and legal alignment/ownership predicates, not measured winners. Predicate does not filter accumulator/input cross-pairs beyond these independent lists. Native MMA construction remains required. Comments recommend minimizing channel-tail waste and preferring wider ties, but `run`:2187–2245 forwards the supplied tile unchanged; this is not an implemented automatic selector. |
| Scaled `can_implement`:3006–3159 | NVFP4 E2M1 +E4M3 scales/vector16; MXFP8 E4M3 +E8M0 scales/vector32. M128; N64/96/128; NVFP4 tileK64/128, MXFP8 tileK64; accumulator f32; D f16/bf16. N96 requires one N tile. C is divisible by4*vector; D alignment16 bytes. | Actual narrower menu supersedes the shared GEMM dispatch helper's broader capabilities. Mixed FP4/FP8 and E5M2 are not connected convolution entry branches. No new Template follows from these menu constants. |
| Scaled `sf_k_tile_channels`:3280–3289, `sfb_per_position_channels`:3292–3310 | SF K span=max(tileK,4*vector); pad **each filter position's** C to that span. MXFP8 C192/tileK64/vector32 gives span128 and padded256, despite C being divisible by tileK. | Independent SF transfer granularity and per-position physical padding. A/B zero-fill and actual SFA/SFB consumer addresses must be bound before declaring scale correctness. Old scale window rules cannot infer these addresses from ordinary data tile width. |
| Scaled `__call__`:352–430 | Pad/stride/dilation must feed both im2col data and software SFA coordinate reconstruction. Alpha runtime; beta and activation compile-time; bias/residual opt-in guards. | Coupled data/scale logical coordinates are a required implementation invariant. Device scale map and fused arithmetic remain for the next slice; this row records the interface, not a proved numerical composition. |

The hardest reuse is the asymmetric-padding coordinate map. It is supplied by the
old rule's `program`, `c_geometry`, `pre_mapping` and `pre_backend`, not by a generic
"convolution" tag. Existing [coordinate checks](checks/conv-contracts-result.json)
separate lower/upper padding and T/S order using distinctive values; the present
check executes the new entry's actual descriptor parameter helper. We keep the
old rule and proof unchanged. SM120 native lowering remains a new conditional
binding rather than a silently admitted backend.

## Recovered frontend instance and rejecting neighbors

Use input N1/C128/D5/H5/W5, filter output channels128/T3/R3/S3, stride1,
lower/upper padding1 and dilation1. The shared `compute_zpq` gives Z=P=Q=5,
GEMM M125/N128/logical reduction3456. Dense tile128×64×64 with f16/f32/f16
passes; scaled tile128×64×64 with NVFP4/E4M3/vector16/f32/f16 also passes its
frontend. These are source-menu witnesses, not native launches or scale-layout
proofs. The scaled entry's padded reduction traversal must still be reconciled
with that logical contraction.

[Frontend check](checks/geforce_conv_frontend.py) executes the retained AST after
replacing only dtype identity/width and the unsupported exception symbol. It checks
16464 dense cases (720 accepted),6912 scaled cases (100 accepted),49 scaled output
type cases,648 descriptor cases across all three axes,20 SF-span cases and all11
old-rule scope premise mutations. Both source files are checked against retained
blob hashes. [Result](checks/geforce-conv-frontend-result.json) records the domains.

Three distinguishing boundaries remain:

- Both `can_implement` predicates accept C0. The old rule rejects this through
  `extents_positive`; it is not an established empty-reduction implementation.
- D=H=W=1 with T=R=3,S=1,padding0,stride/dilation1 produces output(-1,-1,1).
  Its flattened product is positive, and dense `can_implement` accepts it. Each
  output extent must be checked separately before binding the rule. No allocation
  or native launch with this invalid geometry is claimed.
- The descriptor upper-bound helper accepts stride0. `compute_zpq` would divide
  by zero; the rule's positive-stride premise is essential.

No new candidate is needed for this frontend slice. Next: reconcile both device
mainloops and their CLC/parity ownership, then SFA/SFB copy maps, output lifetimes,
fused arithmetic and host reference branches before closing these two files.

Continued in the [input trace](trace-geforce-conv-inputs.md), which distinguishes
connected format domains from helper-only examples and tests the signed packed
coordinate boundary. Math/output/host extraction remains active.
