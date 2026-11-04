# SM100 block-scaled GEMM: contract, scale windows and TMEM lifetime

Paths are relative to the [retained source](inputs/sources.json). Let `kernel` mean
`examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/`
`dense_blockscaled_gemm_persistent_prefetch.py`. [The initial trace](trace.md)
separates cache hints from actual copies; this trace follows a frontend-valid tail
instance and the neighboring N256 accumulator branch.

## Caller and selected instance

Use M144,N208,K320,L2, A/B Float8E4M3FN, scale Float8E8M0FNU with vector32,
C Float32, A/B K-major, C N-major, MMA tiler128×192, cluster1×1 and identity
epilogue. Source static `can_implement` accepts this configuration. The helper
`python/CuTeDSL/cutlass/utils/blackwell_helpers.py:1420–1465` chooses
MmaMXF8F6F4Op with K32; four instruction K blocks make one K128 mainloop tile.
There are three K tiles, two M tiles and two N tiles per batch, with M16/N16/K64
final fragments. This is a source/frontend instance, not a compiled kernel result.

`kernel:1941–2196` checks five legal (AB,SF,vector) tuples: FP4 with E8M0/16,
E8M0/32 or E4M3/16, and either FP8E5M2 or FP8E4M3 with E8M0/32. Output types are
FP32,FP16,BF16,FP8E5M2,FP8E4M3. Only FP4 requires both operand major modes K in
that predicate; dtype/shape menu is separate from actual atom construction. MMA
M is128/256 and N64/128/192/256. Cluster extents belong to1/2/4, with at most16
CTAs, and two-CTA MMA requires even cluster M. The alignment predicate requires
16-byte contiguous extents for A/B/C, not full MMA-tile divisibility. It does not
validate arbitrary major-mode strings, positive dimensions or every storage/target
premise. Tensor construction and lower-level APIs supply further requirements.

## Scale and output decisions

| Source choice | Rule / binding or explicit boundary | Checked evidence |
|---|---|---|
| `kernel:2347–2421,2199–2210`: host conversion into padded SF shape `(L,ceil(MN/128),ceil(SFK/4),32,4,4)` | Metadata layout interface; BlockScaledBasicChunk is ISA-constrained, not a freely tuned quantization group | 55680 unique logical scale coordinates compare contiguous host storage with expanded atom strides. Groups16/32 broadcast each stored scale across the corresponding K vector. |
| `python/CuTeDSL/cutlass/utils/blockscaled_layout.py:24–91`: atom `((32,4),(vector,4))`, strides `((16,4),(0,1))` | Existing scale-owner obligation; zero-stride K dimension is value replication, not missing storage | Live SF groups preserve distinct tags. Padded rows/groups are outside logical numerical scope. |
| `kernel:591–615,1059–1068,1285–1310`: SFB N64 and N192 alignment | Fixed scale-window glue; no new optimizer variable inferred | N64 shares a128-column load window per pair of output tiles, shifting odd tiles64 columns. N192 uses overlapping256-column windows, then the same odd shift. 10880 window coordinates match output columns. |
| `kernel:1195–1240,1321–1377,1617–1644`: shared SF → TMEM → MMA | `async_transfer@1` for global/shared stage plus target-specific scale-copy boundary; do not claim generic shared-memory readiness proves TMEM use | Source filters broadcast-zero modes, uses Cp4x32x128bOp, copies SFA/SFB after AB consumer_wait, and supplies per-kblock TMEM iterators to tcgen05 MMA. Exact TMEM intrinsic mapping remains below the retained Python implementation boundary. |
| `kernel:1314–1377`: reset accumulate at every output tile, enable after first kblock | Original GEMM reduction contract | FP32 accumulator uses scaled block MMA. A/B scale ownership is per K vector; algebraic scale motion or bitwise equality with a PyTorch reference is not inferred. |
| `kernel:417–430,992–1016,1264–1270,1498–1542`: N256 accumulator overlap | `physical_storage@1` lifetime/alias obligation plus source phase and traversal glue | Two logical256-column slots start at0 and208; SF occupies464..512. Slot0 reads N subtiles backwards, slot1 forwards. With N32 epilogue subtiles, consuming two subtiles removes the overlap before release.12 interval witnesses cover output widths and CTA modes. |
| `kernel:1544–1549`: convert before epilogue_op | Numerical caller contract, not a tunable equivalence | A supplied elementwise lambda receives C-typed values. Identity is used in the witness. Applying an arbitrary FP32 op before conversion is a different contract. |
| `kernel:1552–1599`: register → shared C → TMA global store | `staged_async_output_store@1` proposal plus shared ring/visibility obligations | Ring index counts all previous output subtiles. Source proxy fence, epilogue barrier, bulk store commit/acquire and final tail remain required. This selected implementation always uses the TMA store path. |
| `kernel:1903–1938,882–906,1515–1576`: ceiling tiles and final C domain | `output_tiling@1` with shape/padding and descriptor-domain glue | 884736 live output-domain coordinates partition uniquely under full and short extents; this is a descriptor-bound oracle, not a replay of generated TMA clipping or lane ownership. |

For N192, let j be output tile index. The source changes the SFB descriptor's
outer128-column layout from uniform strides to nested `((2,2),rest)` strides
`((x,x),3x)`. Selecting two outer groups per256-column fetch starts at
128*floor(3j/2). Odd j selects a TMEM pointer two32-bit words later, accounting
for64 scale rows/columns. Thus base+64*(j%2)=192j. With j1 the fetch covers
[128,384) and the selected logical output columns are[192,384). Only[192,208)
is live in the tail witness. N64 instead uses128*floor(j/2) with the same parity
shift. Confusing these windows would attach neighboring column scales to an MMA.

For the N256 branch, accumulator stage count is1 as a handshake state; source
phase still alternates two physically overlapping accumulator regions. The logical
region sizes are256 each, with overlap48 because SFA16+SFB32 TMEM columns also
need space. The early release occurs after reading the overlap, accompanied by a
TMEM-load fence. Releasing at subtile0 before both N32 subtiles were read would
leave16 columns exposed to the next producer. The interval replay covers this
storage hazard; asynchronous device ordering and concrete TMEM layout still need
implementation evidence.

## Evidence boundary and next route

[Checks](checks/README.md) retain the actual static-predicate extraction on symbolic
dtype identities, source host-layout equations, scale windows, pure-Python epilogue
helper and region checks. No CUDA/DSL modules were imported or executed. The test
runner's PyTorch reference scales A/B then contracts K, but its tolerance and GPU
result have not been run. The source's make_tmem_layout_sfa/sfb wrappers delegate
to `_cute_nvgpu_ir`; final intrinsic mapping/emitted instructions remain a declared
boundary rather than a fabricated lowering.

The distinct cache-hint choice is now proposed as [cache_prefetch@1](delta/cache_prefetch@1/template.yaml)
with [ProofOutline](delta/cache_prefetch@1/proof.yaml). It has one distance knob,
keeps actual waits and introduces no shared slot. `cache_eviction_hint@1` instead
chooses eviction policy; `multi_buffer_rotation@2` chooses destination-slot
lookahead. None of those identities is replaced. Next: reconcile the remaining
block-scaled variants and cluster/scheduler branches, then advance through the
whole-library inventory. Full CUTLASS extraction remains active.
