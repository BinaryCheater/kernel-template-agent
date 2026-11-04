# Restore a partition queue after its last claimer retires

When many independent token chunks share one channel tile, let its resident
CTAs take complete chunks from a local ticket counter. The final claimer to
retire can restore the counter to zero, removing a separate per-call clear
for later completion-ordered launches. Retirement protects counter reuse;
it is not a publication barrier for output values.

[self_resetting_partition_queue@1](delta/self_resetting_partition_queue@1/template.yaml)
and its [outline](delta/self_resetting_partition_queue@1/proof.yaml) capture
the protocol. The concrete [retained source](inputs/sources.json) is
`cake_causal_conv1d_det_train_backend/`, native zero-history forward
`cake_causal_conv1d_det_train_f5497330bfd286211643_kernel.cu`. Target is
SM100/103/107, width4 depthwise causal convolution plus SiLU. Each token
chunk rereads its original-input left history and document boundaries, so its
result does not depend on the CTA that owns its ticket.

| Concrete decision | Rule field / binding | Evidence |
|---|---|---|
| One queue per channel tile | pre_geometry / pre_members | Native `:211-240`: ct=bid%T, P_t=ceil((B-ct)/T), chunk=n_sub*8, N=ceil(S/chunk). Partitions have different worker counts when T does not divide B |
| Two initial tickets per CTA leader | pre_counter / task_coverage | `:243-257`: tid0 fetch-adds work[ct] twice, shares tickets in cells and stages only valid tasks |
| Consume current ticket and claim another lookahead | pre_retire / native_pipeline | `:301-320`: invalid current q breaks; valid q claims a new pend ticket, then computes its complete chunk and advances private ticket/stage queues |
| Invalid tickets never address payload | pre_items | Initial and future stages test q<N; valid task masks token/channel tails and document-local history. A later ticket of a worker cannot be valid after its earlier invalid ticket because atomic ticket values increase |
| Per-tile exact participant retirement | pre_members / pre_reset | `:587-599`: one leader executes atom.acq_rel.gpu.global.inc.u32 on work[T+ct] with limit P_t-1. Previous value P_t-1 identifies the final claimer and wraps this bank to0 |
| Only final claimer resets next-ticket bank | pre_reset | That leader stores work[ct]=0 after all other claimers have retired. No CTA leader may fetch a ticket after its retirement |
| Counter retirement is distinct from output completion | pre_epoch | No all-CTA output join is provided by the retirement word. Individual nonleader stores and kernel cleanup still need whole-kernel completion before external consumption or next invocation |
| Zeroed workspace reused per stream, capture-local ownership | pre_counter / pre_epoch | Backend `__init__.py:515-563` allocates zero int32 counters, validates caller workspace capacity/type, caches by (device,current stream), and creates a fresh captured zeroed buffer inside graph capture |
| Two concurrent launches require separate epochs | pre_epoch | Host facade requires separate buffers for overlapping calls. Same stream or explicit whole-kernel event order permits reuse; zero retirement count alone does not |

## Invariants and bounds

Successful fetch-add tickets totally order all claims within a partition.
The in-range values0..N-1 each occur once. Each worker's private lookahead
sequence is increasing, so encountering its first invalid current ticket cannot
strand a later valid private ticket. It owns and drains its valid prefetched
items before retiring. Every valid consumed task issues one extra claim, in
addition to two initial claims per worker: total N+2P. That bound and all
addresses must fit the source signed integer ticket convention; wrapping into
negative signed tickets would invalidate q<N as a payload guard.

Retirement happens after the worker's final ticket access. The acquire-release
wrapping increment joins exactly P claimers, so the final reset cannot race
with an earlier epoch ticket claim. It does not prove all payload writes have
completed; source CTA cleanup follows, and the next invocation is ordered
after the entire kernel. The two banks then start the next invocation at0.
A changed channel geometry may reuse sufficiently large zeroed storage only
after this completion boundary; the retirement-bank offset changes with T.

The registry records128 threads,46208 shared-memory bytes and4 CTAs/SM for this
forward. The two-stage native transfer pipeline and actual register/resource
usage are implementation bindings, not measured occupancy or speed claims.
The host returns for S=0; zero-task logical protocol cases check its identity
edge without claiming the host launches that kernel for an empty input.

## Existing-rule boundary and recovery

[two_family_cta_pool@1](delta/two_family_cta_pool@1/template.yaml) supplies
unique atomic work claims and complete output ownership, but has two-family
preference/fallback and an externally initialized launch epoch. It has no
partition-specific retirement-wrap count or last-claimer reset. This design
adds a queue lifecycle rather than another family-selection policy.
[partial_grid_join_placement@1](delta/partial_grid_join_placement@1/template.yaml)
provides tile-specific producer counts and distinguishes valid participants
from capacity holes. Its grid completion joins partial payloads; this queue's
retirement count joins only counter claimers. Substituting one for the other
would expose incomplete outputs or reuse a live epoch.

The [focused check](checks/check_self_resetting_partition_queue.json) explores
29448 abstract states for16 small worker/task configurations, with separate
initial claims, lookahead claims, consume and retire transitions. Every terminal
state has each valid task consumed once, N+2P claims and both banks restored to
zero. It includes workers with no valid tasks. Counterexamples distinguish an
incorrect participant limit (early reset duplicates a ticket) and zero counters
with pending output stores. These are protocol identities, not a simulated
CUDA memory model; native atomic scope, CTA transfer pipeline and stream/graph
lifetime remain explicit obligations.

The zero-history backward's preactivation-precision, FP64 final merge and
mixed bias/parameter routes remain to trace, alongside inference state paths.
This queue representative does not close the convolution family.
