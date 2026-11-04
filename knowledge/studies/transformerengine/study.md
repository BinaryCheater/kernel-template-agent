# TransformerEngine NVIDIA design study

Assignment: full NVIDIA catalog study by distinct computation, representation,
pipeline and scheduling design. CUTLASS remains partially studied and user-paused;
do not resume it in this pass.

Source/license: [manifest](inputs/sources.json), full retained top-level archive
and tree, Apache 2.0. External submodules and vendor device implementations were
not materialized. The [starting pool](inputs/pool-snapshot.json) pins earlier
proposals plus six TileLang pairs; this study does not perform Store admission.

Status: source-design pass complete. The [inventory](inventory.md) reconciles
public operator families, common source registrations and device-source locations.
The [review](review.md) links eight proposed Template/ProofOutline pairs, concrete
source bindings and two focused logical checks. Most related variants reuse
existing rules with explicit numeric/layout/ownership limits.

Closure follows distinct-design coverage with named gaps, not a per-entry audit.
Native provider schedules, exact converter/layout behavior, concurrent visibility
and occupancy/progress remain binding obligations. No GPU compilation or benchmark
was performed; end-to-end replay remains deferred. Next library: NVlabs/tiny-cuda-nn.
