# Choose the completion boundary of partial-grid reductions

When a kernel emits independent FP32 partials, either join and reduce inside a
cooperative grid or launch an ordinary producer followed by a completion-ordered
reducer. Keep the grid geometry in the numerical contract: changing the CTA
budget changes token partitions and therefore the partial sums, even when both
forms are deterministic.

[partial_grid_join_placement@1](delta/partial_grid_join_placement@1/template.yaml)
and its [outline](delta/partial_grid_join_placement@1/proof.yaml) expose join_site
and reserved_sms. Concrete [retained source](inputs/sources.json) is the public
`cake_causal_conv1d_det_train_backend/` deterministic KCP path on SM100/103/107,
with BF16 activations/dx and FP32 four-tap rank-local dweight. The halo and
activation-stage boundary is in [document stencil halos](trace-document-stencil-halo-pullback.md).

## Geometry and live partials

Let B be actual CTA count and T=n_ch_tiles. CTA b owns tile t=b%T and row
rank=b//T. Tile t has G_t=ceil((B-t)/T) existing producers. The allocation has
ceil(B/T) rows, but some high rows for later tiles are holes. Native main
`5288d132a5cc927ae729_kcp_kernel.cu:219-233` sets streams=G_t*n_sub,
L=ceil(S/streams), and substream start=(rank*n_sub+sub)*L. The token loop masks
the final interval. It initializes acc to zero, so an existing CTA with empty
token work still publishes a neutral partial; a nonexistent CTA publishes
nothing and must never be read.

| Concrete decision | Rule field / binding | Evidence |
|---|---|---|
| One cooperative main, inline final merge | join_site=cooperative_grid,reserved_sms=0 | Registry kcp_bwd fallback5288d132a5cc927ae729 uses128 threads,50176 shared bytes and3 CTAs/SM on SM100/103; SM107 sibling uses2. Binding sets cudaLaunchAttributeCooperative |
| Ordinary main followed by reduce | join_site=following_kernel,reserved_sms=1 | Registry kcp_bwd_split fallbackfacdd7ca702cfb7d821a removes the whole-grid join and records reducer86b5ecfbd90aca4b52e5; kcp.py:163-190 subtracts one SM-equivalent CTA budget |
| Grid budget and channel coverage | pre_geometry | B=ctas_per_sm*(SM_count-reserved_sms); host rejects nonpositive budget or n_ch_tiles>B. It is a CTA-count reduction, not a request to reserve particular physical SMs |
| Workspace allocation versus live rows | pre_partition / pre_slots | cake_jit.py:3415-3439 allocates ceil(B/T)*C*4 FP32 values and documents tile-specific G_t. kcp.py:1209-1245 uses the same actual grid for allocation and main_ctas |
| Local CTA substream sum precedes its partial store | pre_numeric | Split mainfacdd7ca702cfb7d821a:1626-1653 stores per-thread acc in shared memory, sums sub=0..n_sub-1 in order, writes float4 tap partials and triggers dependents |
| All producers join before inline reads | pre_cooperative | Cooperative main5288d132a5cc927ae729:1560-1636 writes exclusive partials, calls this_grid().sync(), then merges channel/tap rows |
| Following reducer waits before any payload read | pre_separate | Reducer86b5ecfbd90aca4b52e5:86-132 derives G_t from main_ctas, executes griddepcontrol.wait, then loads only r=lane+32*k<G_t. Its binding enables programmatic stream serialization |
| Fixed lane-strided row fold and halving tree | pre_numeric | Cooperative inline reducer uses XOR16/8/4/2/1; following reducer stages lane sums in shared memory and explicitly adds those same half-pairings. Equal algebra is not a native bitwise proof across different grids |
| Large-channel st8 is a different resource menu, not another join protocol | pre_backend | Thresholds C>=512 on SM100/103 and C>=1024 on SM107 select st8;128 threads,82944 shared bytes,2 CTAs/SM, same cooperative/split pairing |

The ordinary main has no grid-wide barrier. Its final scheduling trigger
follows partial stores at line1653; it does not advertise a per-tile readiness
protocol or an early trigger at entry. Reducer prologue index arithmetic is
independent, but its wait still joins predecessor completion before reads.
Observed source instructions and launch attributes support this dependency
binding; actual runtime PDL and cooperative residency/progress were not tested.
Source concurrency/speed comments are not measurements from this study.

Reducing B changes G_t and L, hence which tokens enter each partial and the
order of the final FP32 sum. The completion choice may be local to each rank
because it changes no halo collective; dweight remains rank-local. A caller
must arrange any desired global weight-gradient reduction separately.

## Existing-rule composition and focused recovery

Pinned [reduction_partition@2](inputs/reuse-additions/reduction_partition@2/template.yaml)
provides exact-once partition and empty identities. For a fixed recovered
shape, its uniform policy binds num_groups=G_t*n_sub,group_size=L and
assignment=contiguous_parallel, with p.empty_groups_emit_identity=True.
For S=0 the host returns zero dweight after the halo protocol; no positive
source partition is instantiated. A dynamically recomputed L is part of the
native shape/geometry binding, not a claim that that rule's compile-bound
group_size is a free runtime field. Version1 c_no_empty excludes some valid
native short-token participants, so it is not used for them.

Pinned [early_dependent_grid_launch@1](inputs/reuse-additions/early_dependent_grid_launch@1/template.yaml)
supplies prewait independence, completion-before-read and buffer lifetime.
Here the primary has griddepcontrol.wait at entry and launch_dependents after
partial stores; the following binding opts into programmatic stream
serialization, and following reads occur after its own wait. There is no
inferred overlap benefit beyond those observed scheduling facts. Native
launch-chain support remains that rule's explicit target premise.

[ordered_workspace_combine@1](delta/ordered_workspace_combine@1/template.yaml)
uses ascending linear slot addition and floor-split ranges, so cannot supply
this lane-strided/halving reducer or the cooperative/following join choice.
The new rule owns actual-grid identity, tile-specific live rows and completion
placement; the pinned partition/dependency rules supply its lower-level
coverage and publication arguments.

The [focused check](checks/check_partial_grid_join_placement.json) covers292
grid/tile layouts,19080 exact token partitions and486 unread capacity holes.
It compares the cooperative XOR lane-tree mapping with the explicit shared
halving mapping at equal partial values. A cancellation witness gives1 from
two token partials and2 from four partials with the same final tree, proving
why changing grid size needs a separate rounding contract. This is an FP32
logical witness, not native kernel execution. Resource residency, PDL data
visibility and the producer's FP32 arithmetic remain native obligations.

This completes the KCP join/partial representative alongside the halo trace.
Zero-history training dispatch/pipeline/precision variants and inference
state paths still need their own representative reconciliation before the
convolution family can close.
