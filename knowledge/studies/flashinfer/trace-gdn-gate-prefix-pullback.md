# Recover activated gate gradients with a chunk-local suffix walk

When forward gates are represented by within-chunk cumulative sums, backward
must pull their cotangents back through that prefix transform. For local
G_t=sum(i<=t)g_i and endpoint contribution E to G_last, the result is

dg_i=E+sum(t>=i)dG_t. Reset at every chunk; E is injected once into the initial
suffix accumulator, not added once per step.

The [retained](inputs/sources.json) deterministic KCP GDN facade accepts FP32
natural-log decay g and post-sigmoid beta. It does not own a raw softplus or
sigmoid-parameter backward in this interface. That external chain rule is not
an unfinished raw-gate kernel in this package.

Native producer `cake_gdn_det_train_501d82df9928cfc02551_kernel.cu:2850-2892`
stores FP32 dg_part for valid chunk rows (zero for padded rows), dbeta for real
tokens, and gamma_dot as a separately reduced endpoint contribution. Its FP32
subgroup sums and FMA recipe remain part of that producer; gate_epilogue does
not reconstruct them from narrow tapes.

Concrete native consumer
`csrc/cake_gdn_det_train_b902b0b0bef82d9f7c53_kernel.cu:79-182` assigns one
thread to item=chunk*num_heads+head, with128 threads per CTA. It stages64
values, reads only rows s<chunk_len, initializes acc=gamma_dot and loops
s=63..0, updating/writing only valid s. It writes token-major output at
chunk_bos+s, whereas dg_part uses padded chunk*64+s. Empty chunks write
nothing. There is no cross-thread or cross-chunk sum in this epilogue.

Concrete existing bindings:

| Choice | Existing field / binding | Obligation |
|---|---|---|
| Independent whole (chunk,head) suffix problems | Seed output_tiling@2: tile_m=1,tile_n=1,executor=thread,index_map=custom over item=chunk*HV+head | Refined map outputs are entire per-item vectors, not independent time samples within a suffix. item owner covers each valid problem once |
| Ordered scalar affine recurrence on reversed valid row domain | [affine_piece_prefix_state@2](delta/affine_piece_prefix_state@2/template.yaml): piece_chunks=64,prefix_supply=piece_replay, one piece per chunk/head, scalar L=1,c=dG_reverse,seed=E | Its pre_affine/order premise applies after reversing the domain. There are no preceding pieces, so no summary consumer or extra prefix communication; an unused identity-linear summary can be eliminated |
| Padded physical capacity versus real rows | Existing affine rule short-piece/identity-padding clause; valid rows are clen and origin is chunk_bos | Do not add padded history or consume another chunk's prefix; recurrence addition order remains native reverse serial order |

The source64-slot private array is the concrete thread Realisation, not an
additional parallel scan policy. A reassociated warp/block scan would require
its own finite-precision contract; no such native alternative is claimed here.
The cumulative-gate derivative is a fixed workload dependency, not a free
optimizer variable, so this pass adds an instance rather than a new Template.

The [exact prefix-Jacobian check](checks/check_gate_prefix_pullback.json)
covers195 length/seed cases (including0 and64) and6240 independent loss
responses to original gate coordinates. It validates endpoint and suffix
algebra. It does not establish native dg_part/gamma_dot arithmetic or FP32
stage equality. This finishes the activated-gate epilogue representative;
other training backends, selective state and convolution organizations still
remain in the recurrent inventory.
