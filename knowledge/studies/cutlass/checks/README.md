# CUTLASS source-study checks

Run from the primary checkout:

```sh
.venv/bin/python knowledge/studies/cutlass/checks/retention.py
.venv/bin/python knowledge/studies/cutlass/checks/prefetch_schedule.py
.venv/bin/python knowledge/studies/cutlass/checks/frontend_contract.py
.venv/bin/python knowledge/studies/cutlass/checks/scale_windows.py
.venv/bin/python knowledge/studies/cutlass/checks/epilogue_overlap.py
.venv/bin/python knowledge/studies/cutlass/checks/validate.py
.venv/bin/python knowledge/studies/cutlass/checks/variant_contracts.py
.venv/bin/python knowledge/studies/cutlass/checks/cluster_scheduler.py
.venv/bin/python knowledge/studies/cutlass/checks/amax.py
.venv/bin/python knowledge/studies/cutlass/checks/circular_windows.py
.venv/bin/python knowledge/studies/cutlass/checks/sm103_layout_output.py
.venv/bin/python knowledge/studies/cutlass/checks/dsl_inventory.py
.venv/bin/python knowledge/studies/cutlass/checks/dense_contracts.py
.venv/bin/python knowledge/studies/cutlass/checks/dense_scheduling.py
.venv/bin/python knowledge/studies/cutlass/checks/grouped_work_descriptors.py
.venv/bin/python knowledge/studies/cutlass/checks/blockwise_contracts.py
.venv/bin/python knowledge/studies/cutlass/checks/conv_contracts.py
.venv/bin/python knowledge/studies/cutlass/checks/conv_blockscaled_inputs.py
.venv/bin/python knowledge/studies/cutlass/checks/conv_blockscaled_output.py
.venv/bin/python knowledge/studies/cutlass/checks/grouped_dense.py
.venv/bin/python knowledge/studies/cutlass/checks/mixed_input_widening.py
.venv/bin/python knowledge/studies/cutlass/checks/grouped_mixed_input.py
.venv/bin/python knowledge/studies/cutlass/checks/moe_dense.py
.venv/bin/python knowledge/studies/cutlass/checks/moe_scaled_metadata.py
.venv/bin/python knowledge/studies/cutlass/checks/moe_scaled_device.py
.venv/bin/python knowledge/studies/cutlass/checks/row_reduction.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_dispatch.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_schedules.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_layouts.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_conv_frontend.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_conv_inputs.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_conv_outputs.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_conv_layouts.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_dense.py
.venv/bin/python knowledge/studies/cutlass/checks/geforce_dense_layouts.py
.venv/bin/python knowledge/studies/cutlass/checks/fmha_contracts.py
.venv/bin/python knowledge/studies/cutlass/checks/fmha_frames.py
.venv/bin/python knowledge/studies/cutlass/checks/fmha_pruning.py
.venv/bin/python knowledge/studies/cutlass/checks/fmha_masks.py
.venv/bin/ruff check knowledge/studies/cutlass/checks
.venv/bin/ruff format --check knowledge/studies/cutlass/checks
```

All forty-one pass. [Retention result](../inputs/retention-check.json) verifies the archive
and all7977 Git blobs. [Prefetch result](prefetch-result.json) checks source index
equations for T0..64,S1..8 and D=auto,−2,−1,0..16, with three persistent output tiles
per domain. Positive distances hint every logical K tile once for each of four
operands; nonpositive distances hint none. Actual load indices and ring slot/phase
progression are unchanged. This is a scalar source-equation replay, with no DSL
compilation, GPU execution, cache observation or performance conclusion.


[Frontend result](frontend-result.json) evaluates the retained five static
predicates with symbolic dtype names/widths:1372 dtype and1620 tile/cluster cases,
plus the aligned M144/N208/K320/L2 witness and a misaligned neighbor. AST translation
replaces only dtype symbols/widths and annotations; it does not import a fake DSL.
This proves the stated frontend branch, not backend acceptance.

[Scale results](scale-windows-result.json) compare55680 unique host-packed scale
coordinates to the expanded atom formula and10880 N-window coordinates for widths
64/128/192/256. K16/K32 broadcast ownership is checked. The TMEM intrinsic layout
is not executed.

[Epilogue results](epilogue-overlap-result.json) use the retained pure-Python integer
subtile helper and check12 early-release region cases and884736 live output-domain
coordinates. Tail clipping is an independent descriptor-domain oracle; generated
TMA boundary behavior and exact lane mapping remain open. [Validation](validation-result.json)
checks twenty-three Template/ProofOutline pairs, duplicate identity absence in the pinned pool,
expressions and rejection of missing address/immutability/backend premises.

Correction pass: split two long lint lines; bounded AST extraction removes type
annotations before symbolic evaluation. No source file was changed. Twenty-three candidate
outlines are proposed, with no registered proof model or independent admission.


[Variant predicates](variant-contracts-result.json) cover750 baseline dtype cases,
25 UNPACK pairs and aligned/misaligned FP6/mixed witnesses; the prefetch/amax dtype
menus agree on the sampled symbols. The SM103 FP4/vector32/K768 witness passes
its separate frontend and an N192 neighbor fails. The ordinary baseline is wider
than its companions; this correction prevents a hint-only equivalence claim.

[Cluster/scheduler result](cluster-scheduler-result.json) covers2916 non-mixed
static-stride/raster/swizzle domains, retaining87948 padded CTA-tile visits, and15
multicast geometries with70 row/column/peer release images. These are coordinate
sets and arrival counts, not emitted mask bits, GPU progress or signal timing.

[Amax result](amax-result.json) checks6174 non-NaN value/update-order cases and
preserves signed-versus-unsigned initial -inf, pre-versus-post FP16 conversion and
unreset-next-invocation witnesses. NaN/padding/lane behavior remains unverified.

[Circular-window result](circular-windows-result.json) extracts all eight SM103
current/next descriptor pairs and release positions2/5/7 from source AST, checks1536
scale-owner coordinates and10560 bounded window domains. It distinguishes the
K256 operand chunks from K192/K384 scale chunks. Hardware descriptor encoding and
asynchronous completion remain open. Candidate validation also rejects W96/C64,
which is outside this version's declared W<=C domain.

Correction pass for this slice: preserved baseline mixed/FP6 differences rather
than treating files as identical; matched amax's elected-lane release count; quoted
a YAML flow value containing a comma. The source-equation checks are retained
without changing source, installing packages or invoking a GPU.


[SM103 layout/output result](sm103-layout-output-result.json) uses the verified
retained pure-Python PyCuTe package to compare24576 atom/host addresses and73728
staged SF positions; it also checks200 complete-vector tail domains, eight
constant-rotation output-ring domains and264 source-read/store streams. One
weakened-wait S1 overwrite witness is preserved. This does not invoke CuTe DSL or
execute descriptors. The third candidate's capacity/backend/observation expressions
are checked; read completion alone is rejected as an output-observation premise.

[DSL declaration inventory](dsl-inventory-result.json) parses all249 retained Python
CuTe DSL example files, locating281 explicit kernel declarations in210 files.
[dsl-family-index.md](../dsl-family-index.md) and inputs/dsl-kernels.json link their
paths/lines. This discovers entry declarations, not semantic coverage; notebook,
generated and C++ routes remain separately open.

Correction pass: bound the inventory visitor's output list, fixed long lint lines,
kept source-read and destination-write completion separate, and replaced an overly
broad output-transfer reuse statement with its own direction/lifetime candidate.
Existing checks are rerun only for the added candidate and changed composition.


[Dense contracts](dense-contracts-result.json) evaluate363 dtype cases for each of
six entry classes, their common128×128 one-CTA geometry, direct-output tail rejection,
and dynamic split-K policy restrictions. The real retained integer target guard is
bound to SM100 and separately rejects SM107.264 manual prologue/rolling index cases
and the affine-before-final-cast witness are retained. Compiler pipeline lowering,
full mixed-pair atom legality and concrete tensor execution remain separate.

[Dense scheduling](dense-scheduling-result.json) explores cancellation ownership
states N1..5 (3,11,45,201,963 states) and153 split-K incidence domains with1340416
mailbox coordinates. It retains the source's three unguarded coordinate-query IR
operations and69 domains containing empty K partitions. Those observations are
implementation gaps, not successful GPU runs or valid empty-partial arithmetic.
The T5/S4 intervals are recorded for later replay. The proposed shard rule rejects
undefined partials, non-FP32 accumulation and incompatible fragment maps.

[Reuse bindings](../inputs/reuse-bindings.json) are checked against uniquely resolved
async_transfer@1 and multi_buffer_rotation@2 bodies, including variable domains,
constraints and scope. S9 is rejected by the old rotation domain. The [partial
proposed composition](../delta/chain.yaml) validates two local identities and ordering;
its CLC backend needs a successful-response guard, and its K-partition/MMA/fragment
edges are explicitly external. No safe source execution is inferred from Chain shape.

The [current reuse audit](../reuse-audit.md) records structural retrieval cues,
implementation boundaries and the hardest old field/Realisation match for current
candidates. Scope guards remain executable. End-to-end agent replay experiments
are deferred as requested.

Pool reconciliation: one external Humming bitstream_packing@1 body changed during
this slice. The original manifest remains untouched; inputs/pool-reconciled-snapshot.json
records the observed current view and retains the changed body under pool-updates/.
It is compared with register payload placement but is not a dependency of the current twenty-three proposals or reused input-pipeline fields.
No semantic equivalence with the unavailable old body is asserted. Candidate identity
checks now use this reconciled view; all other pool bodies retain their original hashes.


[Grouped work/descriptors](grouped-work-descriptors-result.json) compares window32
prefix/cursor decoding against independent bisect for1200 worker domains and23112
queries. It checks81920 logical output coordinates for the K768/K0/K1536 witness,
private publisher tuples, same-key skips, and eight abstract reader-retirement
domains. Backwards-cursor and K0-stale-C counterexamples are retained. Variant
predicates accept both grouped frontends and reject the unselected SM100 FP6/FP4
output branches. Circular-window extraction now compares all eight SM103 descriptor
pairs and release positions in both grouped/non-grouped bodies. Validation checks
the two new pairs and rejects missing metadata/order/publication/lifetime guards.
No device descriptor encoding or transfer-engine retirement is established.

[Blockwise contracts](blockwise-contracts-result.json) checks1029 retained dtype
triples and positive/alignment neighbors for all three entries,20 exact rational
scale-owner cases, aligned contiguous groups and864 cooperative prefix work domains.
Wrong final scaling, per-CTA peer disagreement and terminal metadata index3 for
three groups remain counterexamples. Template validation checks the eighth pair
and the old block_scaled_partial_accumulation@1 variable/constraint/scope binding.
The source terminal validity guard is missing; no backend discharge is inferred.

[Dense convolution](conv-contracts-result.json) checks720 descriptor-field guard
domains,16 spatial geometries/16032 patch elements/2004 output values and48
bias-vector domains. Retained pure parent/frontend predicates accept the aligned
FP16 witness and reject C7. Swapped padding and tap-axis counterexamples remain.
The width guard alone accepts zero stride; the proposed rule separately requires
positive geometry. Validation checks the ninth pair, its guards and the concrete
operator_fusion@1 bias-only binding. Descriptor encoding, TMA clipping and target
arithmetic remain implementation obligations; the companion continuation is recorded separately.

[Block-scaled convolution inputs](conv-blockscaled-inputs-result.json) explores612
byte/arrival counter states,264 delayed-arrival capacity domains and30 scale-index
domains using403 executions of the retained SFA carry recurrence. It preserves
the omitted-registration race, T3/S2 capacity failure and C192/SFspan256 per-position
padding counterexample. The tenth candidate's scope rejects missing atomic
registration, copy completion, target generation and progress premises. Hardware
counter/visibility/multicast semantics, SFB N-window variants and the output
quantizer remain open; the full companion is now source-extracted with those backend obligations in its output trace.

[Block-scaled convolution output](conv-blockscaled-output-result.json) evaluates
the retained full frontend for five input-format/tile bindings and distinct
rejected neighbors,10880 SFB window coordinates,20 retained residual phase/order
expression domains and33 rational quantizer cases plus midpoint ties. Stored-scale
5/4 yields code3/2 where unrounded37/30 incorrectly yields2. A custom nonidentity
callable passes the source registry guard; semantic identity remains required.
A source-admitted cluster16 case needs5152 CLC readers beyond the old4096 domain.
The eleventh pair rejects unsupported format/group pairing, uninitialized partial
groups, changed normalization, missing group identity and unsupported finite code
range. Native conversions/reciprocal, logical fragment ownership and extreme
quantization ranges remain open. The dense frontend loader is now shared; its
original convolution replay still passes after that refactor.

[Dense grouped](grouped-dense-result.json) evaluates12 dtype combinations,0/384
descriptor-shadow bytes in GMEM/SMEM modes, five version1 projection counts and ten
worker/mode visits of the three-group K64/K0/K128 witness. Actual source ASTs retain
zero-K initialization partners in dense and blockscaled grouped entries; the
missing-zero-branch interpretation is rejected. The host prefix accepts a first
misaligned group when the last is aligned, retaining that source validation gap.
Version2 keeps old fields/constraints/premises/obligations and shadow costs under
its shared projection. Early direct field mutation changes an old reader's group;
retirement/proxy ordering and native progress remain backend obligations.

[Mixed-input widening](mixed-input-widening-result.json) checks1056 code/type pairs
and128 coordinate-distinct permutation packets, replays retained shape/scale/
alignment predicates and symbolically selects CUDA>=13.1 shuffle gates. It preserves
the raw0 signed0 versus offset-binary-8 mismatch, accepts TMA tails but rejects
direct tails, and records negative-scale and incomplete capability guards. K0
passes the frontend but retained get_divisibility(0) raises log2(0). Upstream host
scales are randomized after range computation, so they are test data rather than
quantizer evidence. The thirteenth pair rejects mismatched shuffle format/path,
incomplete chunks and missing format/order/backend premises. Native converters,
fragment maps, pipelines and scale-reassociation error remain open.

[Grouped mixed-input](grouped-mixed-input-result.json) executes the retained scalar
cursor and frontend predicates:1800 domains/22506 queries, including repeated
M-inner queries. Nine segmented-output domains cover full TMA, partial SIMT and50
inactive peers. Separately rounded group counts give5 rather than4 global tiles;
removing the group-end mask duplicates writes.924 ring-credit domains check pair
and odd-tail waits; out-of-order retirement supplies a counterexample. Nine
rational scale partitions and an explicit BF16 1/128-versus0 witness separate exact
factoring from admissible numeric relocation. New pair schemas, premise mutations,
negative item count and batch2/stage1 rejection are checked by validate.py. Native
prefix retirement, lane maps, group clipping and numeric tolerance remain open.

[Dense MoE](moe-dense-result.json) executes the retained scheduler's scalar methods:
360 domains/7704 queries, both expert count constructions and both shortest-axis
orders.1176 backward K0 CTA-output instances remain present; the actual epilogue
zero/load branch supplies the empty-reduction identity.1080 retained coordinate
expression checks and304 key/slot addresses distinguish encoded descriptor bases
from global-descriptor coordinate offsets. Source E0 initialization, duplicate
initial-invalid sentinels and17 quiescent tail-state counterexamples remain gaps.
The three-step partial composition, version1 projection and new executable scope
guards validate. No GPU hang, native descriptor visibility or performance result
is inferred from these scalar/protocol models. Scaled MoE remains open.

[Scaled MoE metadata](moe-scaled-metadata-result.json) checks512 retained shape
helper domains and128 prefix-scan domains/11808 segment queries, including257
experts across128-wide chunks.9216 distinct labels recover the actual host
permutation constants;28 descriptor-base expressions check token-equivalent scale
origins. Separate-versus-global padding gives512/256; local-versus-global row-block
pitch gives512/1024. The new padding pair, two-step partial composition and old
descriptor-table tuple4 binding validate. Native scale encoding/padding neutrality,
CuTe layout, descriptor visibility and the scaled device body remain open.

[Scaled MoE device](moe-scaled-device-result.json) checks768 retained shape cases,
120 overlap/read domains and243 K0/positive-K clock streams. Source zero/load/scale
selection, N64 window halves and the old per-region lifetime inequality are
replayed. Accepted MXFP8 K768 produces SF288/stride-32/release9 outside8 subtiles.
A pre-scaled FP32 dot differs by2^-21 from final scaling and survives BF16; K0 with
infinite alpha yields NaN. These are scalar/resource/numeric boundaries, not native
execution results. The five-file MoE source extraction is closed with its explicit
scheduler, native-layout/visibility, resource and numeric obligations.

[Row reduction/RMSNorm](row-reduction-result.json) executes108 source configurations
and eight TV maps/69632 distinct coordinates.38 local/cluster incidence domains,
15342 scalar deliveries and738 replicated reader results check sender slots,
byte counts and add/max identities; three subgroup domains check warp separation.
N4097/32769 miss a column, N32767 selects a vector extending past N, and N20000
requires480 initialized-neutral scheduled columns. The new pair's guards and
four-step partial chain validate; old alignment and approximation premises remain
conditional. No native RMSNorm output or remote PTX run is claimed.

[GeForce dispatch](geforce-dispatch-result.json) executes1296 actual CLI validation
cases (18 accepted/1278 rejected), symbolic native-constructor selection and the
mixed-FP4 unpack helper. Eighty code/position cases,256 distinctive packets and
eight selected source shifts check bit placement; omitted/doubled shifts fail.
The declared K64 helper is absent from both actual menus. New pair guards validate;
native constructors, LDSM layout and persistent kernel schedules remain open.

[GeForce schedules](geforce-schedules-result.json) executes1896 source cursor jumps,
1296 stripe/phase domains and68 register-capture traces. A correctly backed abstract
order protocol explores235 states with66 MMA/epilogue overlap choices. Source order
storage declares2 slots but indexes4; physical padding backing remains unverified,
not an observed out-of-bounds fault. K0 hoisted-MMA, reverse mixed alignment and
cooperative missing output-tail findings remain explicit boundaries. New rule
schema/guards and the two-step partial composition validate; native progress is open.

[GeForce layouts/host](geforce-layouts-result.json) executes retained scale metadata
for24 domains/884736 coordinates and checks72 host layouts/161568 unique scales.
The N permutation is bijective.16 scalar struct cases show four order slots fit
in the declared alignment padding for positive stages1..16, conditional on native
field layout; no native overflow was observed or proved. Fixture inputs/scales,
reference casts, TMA internal types and budget/native obligations are documented.
The three-file GeForce blockscaled source extraction is closed with those gaps.

[GeForce convolution frontend](geforce-conv-frontend-result.json) executes16464
dense and6912 scaled menu cases,49 scaled output-type cases,648 descriptor domains,
20 scale-span domains and11 existing-rule premise mutations. Distinct-axis
parameters recover WHD reversal and leading-side padding. Both predicates accept
C0; the dense predicate accepts output(-1,-1,1) through its positive flat product.
The old patch rule's positive extents remain required. Both source files are
inventoried; their device bodies and host scale layouts remain active.

[GeForce convolution inputs](geforce-conv-inputs-result.json) executes16 spatial
geometries/16384 row-tap cases,100 signed32 packing cases and26 connected scale
selection domains/38272 channel selections. Retained SFA direct/hoisted expressions,
filter/channel carry, SFB tile indices and SF-base expressions are recovered.
An accepted H32769 NVFP4 geometry rebuilds H32768 as negative32768 after a filter
carry. Signed16 base bounds and representable offsets remain implementation
premises. Native copy/scale encoding, layouts and progress are not established.

[GeForce convolution outputs](geforce-conv-outputs-result.json) checks six actual
epilogue choices,6720 side-stream domains/143416 owned releases and224 abstract
output-ring domains/4032 source reads. Retained jump/tile/CLC arithmetic, four-slot
order declarations, skip branches and tail-before-handoff are checked. Dense's
f16-accumulator/f32-bias menu permits bias conversion to1 from1.0001000166; scaled
source expression order keeps alpha before bias and residual before activation.
Native precision, TV maps, CLC failure lowering and progress remain premises.

[GeForce convolution layouts/host](geforce-conv-layouts-result.json) executes15
shared metadata domains/9216 coordinates and64 host layouts/117504 unique scales.
Thirteen probe/runtime storage fields match symbolically;32 abstract monotone
storage costs replay the actual upward stage search. Both identity/ReLU menus
and runtime geometry boxing order are checked. Narrow helper N is separate from
actual SF N128. Source extraction of the two-file family is closed with native
encoding/layout/capacity/progress and numerical boundaries explicitly retained.

[GeForce dense entry](geforce-dense-result.json) executes80 CLI cases,343 dtype
triples,126 resource cases and448 abstract output-ring domains. Width gates
accept77 dtype triples, selected f16/bf16 op validation only3. Negative stage
results remain outside setup's equality-to-zero guard. Source output ring has
post-acquire synchronization and per-tile tail. Scheduler/layout/host correction
is active; this file is not counted as fully extracted yet.

[GeForce dense layouts/scheduler](geforce-dense-layouts-result.json) executes144
static scheduling domains/6600 coordinates and28 epilogue domains/475136 CTA
coordinates. Actual worker index/stride use grid.z. Eight symbolic store choices
separate16-bit StMatrix from universal stores; the geometric StMatrix typed
partition remains a native obligation for non16-bit outputs. Signed fixture
defaults are recovered. The single-entry source extraction is closed with named
native layout, positive-stage/K, capacity and numerical boundaries retained.

[FMHA contracts](fmha-contracts-result.json) executes16807 dtype/1440 shape
cases,216 row-skip decisions,16 four-warp vote combinations and24 finite-reference
freeze cases. Six cardinality/value witnesses distinguish nonidentity pruning
from an exact reference representation change. Zero sequences/negative tuple
entries pass the predicate; KV heads0 escapes as division by zero. Full forward
recurrence/protocol and backward remain active; no new candidate is asserted yet.

[FMHA frames](fmha-frames-result.json) checks192 complete recurrences/384 prefixes,
12 packed denominator cases and14 rejecting scope mutations. Retained setup and
finalizer expressions recover lag/gain policies and a sink mismatch: gain256
returns256/257 and LSE log2(257)-8 rather than1/2 and1. New version2 schema and
finite positive-mass projection preserve all five old variable domains; native
TMEM/order/casts and value-based pruning remain open.22 candidate pairs validate.

[FMHA pruning](fmha-pruning-result.json) verifies10449 exact rational normalized
weighted domains,2484 tight bounds,31347 shared-frame transformations and16
rejecting scope mutations. Retained decisions expose different host/device masks
with frozen references on maxima0/4/-4. Skip-path publication and unconditional
P-release/O-commit are checked structurally; native incidence/progress and
conservative certificate rounding remain open. The new certificate pair is
conditional; source tau alone supplies no mass/value-range bound.

[FMHA masks/channels](fmha-masks-result.json) executes2742 mask domains/440716
score coordinates, including CTA256x128. Selected live domains have no omitted
nonidentity or unmasked-invalid tile. Direct unmasked-helper tails are outside
the host-selected residual branch. The actual host guard accepts empty rows in
Q3/K1 bottom-right causal and rejects the nonempty Q3/K3 diagonal window. A
per-work offset-channel model explores351 states over13 lengths; removing tail
strands a wait and removing prologue misidentifies generations. Native TMEM
visibility/cross-work reuse and zero-mass output remain open.
