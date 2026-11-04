# CUTLASS NVIDIA source study

Assignment: continue the NVIDIA catalog after Humming, DeepGEMM and Marlin. Primary
output is `knowledge/studies/cutlass/`; one library at a time. Scope is the full
NVIDIA kernel family in NVIDIA/cutlass at the catalog revision, including CuTe DSL,
C++ templates, architecture atoms, examples, generated registration and test routes.
The catalog's initial block-scaled persistent/prefetch pair is not the library boundary.

Pinned revision: e13f5cd4142c43efc37c2f6339923eb4a57afb99. Complete7977-file archive
and extracted tree are retained outside Git; all blobs match the Git tree. No
gitlinks occur and `.gitmodules` is empty. BSD-3-Clause with additional notices in
EULA.txt and source headers. [Manifest](inputs/sources.json) records paths and hashes.

Starting pool: [pinned snapshot](inputs/pool-snapshot.json), seed/reference baseline
and provisional Humming/DeepGEMM additions. Marlin proposes no new identity. Reviewed
index is empty; no provisional source binding is independently admitted.

Stopping condition: reconcile full entry/build/dispatch/device inventory; account
for each mechanism group and its variants; bind existing rules or propose paired
Template/ProofOutline records with discriminating checks; one correction pass and
short review. Admission, GPU measurements, WP gates and commits are separate work.

Status: partially studied — paused by user direction. Do not resume in this catalog pass unless asked. Twenty-nine Blackwell/GeForce kernel entry files have source-backed
mechanism dispositions: four SM100/103 blockscaled, two GeForce blockscaled, seven
dense, two blockscaled grouped, three blockwise, four convolution, one dense
grouped, three mixed-input, two MoE and one RMSNorm. Shared helper files are
accounted separately rather than counted as kernel entries. Twenty-three candidate
Template/ProofOutline versions across twenty-one names are proposed; forty-one
source/schema/scalar/lint commands pass. [Review](review.md) is the reading entry
point; [reuse audit](reuse-audit.md) supplies structural cues and exact old fields
or gaps. [Inventory](inventory.md) and [DSL family index](dsl-family-index.md)
retain outstanding scope. Full CUTLASS extraction is not closed.

Current closure: both GeForce convolution files are source-extracted across the
[frontend](trace-geforce-conv-frontend.md), [inputs](trace-geforce-conv-inputs.md),
[outputs](trace-geforce-conv-outputs.md) and [layouts/host](trace-geforce-conv-layouts.md)
traces. Scale-coordinate checks, separate stream phases, output source retirement
and13-field storage-probe agreement complete the source correction pass. No new
candidate is needed. Native layout/encoding, fixed-width address ranges, dense
capacity, numerical precision and failed-CLC-query/progress remain explicit
implementation obligations. Source extraction does not admit a native capability.

Earlier family traces and their named obligations remain linked in the review:
CLC query/participant bounds, empty reductions, descriptor retirement, masked
terminal metadata, physical lane/group maps, output observation and numerical
rounding are not erased by later source closure. The GeForce standalone GEMM
three-file family is complete across dispatch, schedules and layouts/host traces;
its source-specific order-padding, alignment and output-tail boundaries remain.

The [GeForce dense trace](trace-geforce-dense.md) is now source-extracted, with
static ownership, output type selection, host fixtures and resource boundaries
accounted. Fully extracted entry files now total29; candidate versions stay21.
Current: Blackwell attention's12 files are pinned in
[its inventory](inputs/attention-variants.json), across FMHA, mixed-input FMHA,
MLA and Mamba2 SSD. Next: FMHA forward/backward, then the remaining attention
groups and outstanding DSL/C++ families.
End-to-end agent replay is deferred. No CUDA/DSL compilation, GPU execution or
timing occurred in this source study.

Pool note: the original starting manifest remains preserved. The
[reconciled current view](inputs/pool-reconciled-snapshot.json) retains one externally
changed Humming candidate body used only for comparison, without a reused field
binding; all other original bodies are unchanged. The reuse audit and
inputs/reuse-bindings.json identify the concrete versions used here.

Current FMHA work spans [entry decisions](trace-fmha-contracts.md),
[reference/gain frames](trace-fmha-frames.md) and
[certified pruning](trace-fmha-pruning.md). Candidate versions now23 across21
names; entry coverage remains29. The frame revision preserves version1; the new
pruning pair requires conservative mass/value/error certificates absent from the
source heuristic. Sink gain and frozen-device/host-mask discrepancies are retained.
Next: full mask/empty-row policy, in-place/TMEM lifetime and forward scheduling,
then backward and other attention groups. No FMHA entry is source-complete yet.

The [FMHA mask/channel slice](trace-fmha-masks.md) checks selected mask domains,
host empty-row counterexamples and offset read-retirement channels. It adds no
candidate. Three-segment mask fields, full MMA/TMEM cross-work lifetimes and
forward scheduling/resources remain next, followed by backward and other groups.
Candidate/entry counts remain23/29.

## User-directed checkpoint

CUTLASS work stops here; the earlier three-hour allowance is superseded by the
user's instruction to move on immediately. Preserve all retained sources and
existing records. This is partial coverage, not full library completion.

Covered designs: Blackwell/GeForce dense, blockscaled, grouped, blockwise and
mixed-input GEMM; convolution; MoE and row reduction/RMSNorm. Attention work stops
after FMHA reference/gain, bounded pruning and mask/channel studies. The two-edge
masked_tile_pruning@2 pair/check is saved as a draft outside the23-pair validation
registration; backward has only an initial survey. No further checks are required
before leaving by this instruction.

Unread/unreconciled: remaining FMHA forward/backward and other attention groups,
distributed designs, other architecture DSL families/CuTe extensions, C++ device
families and generated build/dispatch routes. Native obligations in prior traces
remain named. Continue with tile-ai/tilelang under representative-design and
difference-only reading, not an entry-by-entry source audit.
