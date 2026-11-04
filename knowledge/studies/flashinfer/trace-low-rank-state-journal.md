# Accepted low-rank history beside a recurrent checkpoint

When recent state changes are rank one, avoid materializing the full current
state at every verify. Keep a checkpoint plus accepted key/correction/gate
records; project this virtual state directly, then fold accepted history when
the window fills. [low_rank_state_journal@1](delta/low_rank_state_journal@1/template.yaml)
and [outline](delta/low_rank_state_journal@1/proof.yaml) capture the representation,
flush threshold and speculative cursor contract.

Paths below refer to [retained source](inputs/sources.json). The representative
is `gdn_kernels/gdn_decode_bf16_wy_ucache_flush.py`, HMMA on SM90+, K=V128,
T4/8. Its earlier verify-only sibling uses a16-slot ring and external flush;
the fused-fold body actually uses32 physical slots with a16-entry live window.
Copied comments showing16 physical entries are not this binding's layout.

Let H0 be V-major checkpoint, and accepted record j store normalized k_j,
correction u_j and cumulative log gate G_j. For P accepted entries,
H=exp(G_P)*H0+sum_(j<P) exp(G_P-G_j)*u_j*k_j^T, where G_P is the final
live entry's gate; P0 uses anchor0. Thus H*x can be evaluated as a checkpoint
projection plus weighted history dot products. u_j already contains the
state-dependent beta*(v-prediction) correction; it is not raw V and must retain
the same preceding-state generation.

| Source decision | Rule field / binding | Evidence |
|---|---|---|
| `:1412-1458`: immutable base/P snapshot, pool/head strides from descriptors and64-bit pool offsets | `pre_ring`, `pre_ownership` | Logical old rows map(base+j)&31. Block-strided pool pages are not assumed contiguous in dim0; negative padded request exits before state/ring work |
| `:1788-1846`: mask j<P before exp(G_P-G_j), bdec=exp(G_P), new G either old anchor+local or flush-local | `pre_frames`, `pre_flush` | Stale G must not be exponentiated and masked afterward. Finite/zero inactive K/U is also necessary: zero times NaN/Inf is not neutral |
| `:1892-1940`: append normalized keys before bdec rescales their temporary tile; key writer iff pid_hv%(HV/H)==0 | One writer per shared key group | Every sibling value-head CTA may still read the old key window. Appending at(base+P+s)&31, for both paths, is outside all old-window readers |
| `:2421-2445,2465-2595`: checkpoint projection against packed K/Q, then weighted history score/u contraction | `pre_state`, virtual-state projection identity | H need not be assembled just to compute its key/query projections |
| `:2785-2861`: R=V-exp(local_G)*hw_k; U=Tmat*R after output's folded QT*R | `pre_state`, `pre_numeric`; chunked_delta_feedback@2 composition | beta is inside Tmat. Stored U is derived feedback, not input tape. Output and new U have separate multiplication/staging/rounding paths |
| `:2358-2394`: old key tile register snapshot before shared buffer becomes U; `:2985-3060`: weighted-U contraction plus bdec*H0 | `pre_flush` and scratch lifetime | Fold old P only, preserving a resident half-state tile and reloading its other half. New speculative T entries are never folded in that call |
| `:3298-3348`: threshold/accepted cursor/backend contract | draft_tokens4/8, flush_min1..17-T | Lazy threshold17-T flushes when a full draft would exceed live capacity16; smaller thresholds fold earlier. Physical32 retains live16 plus all current drafts without overwrite |
| `:3855-3865`: standalone convenience slides base and resets old P only; shared-layer serving leaves cursor mutation to caller | `pre_cursor` | After all layers, caller adds actual accepted a: nonflush P+a; flush base+P and a. Committing shared base/P after layer1 makes later layers read the wrong window |

The accepted prefix makes new journal entries causal: u_s depends on prior
accepted history and earlier drafts, never later draft suffix. Publishing a
entries therefore selects the correct recurrence prefix. Rejected entries
remain stale outside the live prefix; they need no physical clearing if finite
payload and future read/write contracts hold. When acceptance is zero on a
flush call, old accepted history still folds into H0, but no current draft
enters the logical state.

The [exact check](checks/check_low_rank_state_journal.json) compares476
virtual-state/fold/accepted-prefix cases with a sequential affine recurrence.
It covers11968 ring/threshold combinations and79424 accepted cursor updates
across every physical base, live length0..16, T4/8 and legal threshold. It
checks28 elected key writers and shows1024 cases where restarting append at
the old base would overlap old readers. Counterexamples also distinguish
absolute versus flush-local draft gates and an early multilayer cursor slide.
These are algebra/address checks, not native mixed-precision or publication tests.

This is distinct from
[deferred_prefix_state_commit@1](delta/deferred_prefix_state_commit@1/template.yaml):
that rule freezes one checkpoint/window and retains full snapshots or raw
transition inputs, then recomputes accepted state transitions once. Here the
journal stores already-derived corrections across verify calls and periodically
changes the physical checkpoint while preserving the declared virtual state.
It trades dense state traffic for small repeated projections and low-rank
retention; raw input replay is not the same numeric recipe.

The UMMA sibling supplies additional menus: T1..8, FP16 state with BF16 IO/rings,
FP16 U next to BF16 K, and optional Philox/LCG stochastic checkpoint rounding.
The wrapper's target/layout/rounding gates are concrete implementation boundaries.
The first journal proposal records common representation/cursor requirements;
read the UMMA precision/RNG/fragment implementation separately before binding
those extensions. No backend bitwise identity is inferred from the common math.
