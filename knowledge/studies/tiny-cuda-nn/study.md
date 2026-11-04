# tiny-cuda-nn NVIDIA design study

Assignment: account for distinct NVIDIA computation, layout, pipeline and training
mechanisms across the full library. Read one representative per design and only
new differences in relatives. CUTLASS remains user-paused; its external dependency
is an implementation boundary, not a request to reopen that study.

Inputs: [retained source manifest](inputs/sources.json), full top-level archive at
the pinned revision; BSD-3-Clause terms in `LICENSE.txt`. Submodule device bodies
are not materialized. [Starting pool](inputs/pool-snapshot.json) pins seeds and
prior proposals through eight TransformerEngine pairs; proposals are provisional.

Status: source-design pass complete. The [inventory](inventory.md) reconciles all
registered groups, eleven CUDA units and device/header locations with the traces.
The [review](review.md) links five proposed Template/ProofOutline pairs. All five
pass schema, expression and proof-pair checks; the retained fragment serialization
check passed ten rectangular/layout cases. Explicit extraction and implementation
gaps remain named in the inventory. Source substitutions are separate from native
execution and numerical/performance evidence. No Store admission, GPU run or
end-to-end replay was performed. Next library: Dao-AILab/flash-attention.
