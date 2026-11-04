# Frozen verification and accepted-prefix state commit

When recurrent state is large and speculative acceptance is not known yet,
freeze its authoritative checkpoint and retain either complete post-token states
or the smaller inputs required to replay updates. The new
[deferred_prefix_state_commit@1](delta/deferred_prefix_state_commit@1/template.yaml)
and [outline](delta/deferred_prefix_state_commit@1/proof.yaml) express this
retention choice, prefix identity, tracking convention and one-use epoch.

Paths below are relative to the retained [tree](inputs/sources.json). The
representative is FlashInfer GDN MTP ReplaySSM, not the separately assigned
standalone Mamba library.

| Source decision / concrete binding | Rule field or boundary | Evidence |
|---|---|---|
| `flashinfer/gdn_decode.py:1434-1447,1655-1674`: cache_replayssm requires disable_state_update=True; full intermediate snapshots/scatter cannot coexist | `retention=transition_tape`, `pre_freeze`; state_snapshots is the alternative interface | Verification leaves the FP32 checkpoint unchanged. A later restore from a speculative all-token state is not this design |
| Same entry stores raw BF16 K/V and FP32 log-decay/beta windows by pool slot | `pre_retention`, `pre_parameters` | State update needs K,V,decay,beta, but no query Q. Cached K is before L2 normalization; the commit must use the same normalization/epsilon and gate stage |
| `gdn_kernels/gdn_replayssm_spec_fold.py:80-150`: ignore invalid/zero-accepted requests, load checkpoint, normalize K, form decay from cached log gate | `pre_prefix`, zero-length no-op, `pre_slots` | Accepted count is a consecutive prefix in[0,T]. Invalid slots and zero acceptance do not touch checkpoint/track output |
| `:157-199`: decay FP32 state rows, predict with K, form beta*(V-prediction), add rank-one correction | Declared ordered transition F_t | For V-major H: Hbar=alpha*H; Hnew=Hbar+beta*(v-Hbar*k)*k^T. The decay occurs before prediction |
| `:202-219`: optional track write after token update, then final checkpoint store | `pre_tracking`: zero-based b saves S_(b+1) only when b<a | Track slots are mutually distinct and disjoint from all live checkpoint slots. Step outside the accepted prefix leaves destination unchanged |
| `gdn_decode.py:1817-1868`: commit once across layers, then overwrite windows on next verify; SIMT versus tcgen05 target/shape/tracking menu | `pre_once`, `pre_lifetime`, `pre_backend` | Replay windows are consumed once without shift/clear. Next verify waits for all commit/track readers. Parallel layer/state stores are not atomic publication of a whole checkpoint generation |
| Source SIMT uses FP32 contractions; selected tcgen05 fold uses TF32-style contractions with narrower supported T/normalization/tracking menu | `pre_backend`, numerical recipe | Snapshot, SIMT replay and tensor-core fold do not inherit bitwise equality merely because their real recurrence matches |

The [exact check](checks/check_deferred_prefix_state.json) compares independent
matrix-affine delta updates with row-wise decay/predict/correct replay. It
covers80 accepted-prefix cases and440 tracking cases, including zero acceptance
and out-of-prefix tracking. It also gives counterexamples for double commit,
using diagnostic acceptance count instead of emitted-prefix count, stale tape,
changed key normalization, colliding state/track slots and reordered updates.
The small dimensions/windows are logical recurrence cases, not native shape claims.

The byte tradeoff is explicit. For one layer/slot, key heads H2, value heads HV4,
K=V128 and T8, full FP32 snapshots use2097152 bytes. Raw BF16 K/V plus FP32 g/beta
use12544 bytes: per token256*H+264*HV. These are retained payload sizes before
alignment/metadata and unchanged-checkpoint storage, not a runtime memory or
performance measurement.

The accepted count must come from actual consecutive draft emission, not the
independently rechecked diagnostic counter in
[draft_target_residual_correction@1](delta/draft_target_residual_correction@1/template.yaml).
Even after an earlier rejection, that diagnostic can count later passes.
Replay additionally needs the original checkpoint/window epoch; applying the
same accepted record prefix again to its already-mutated in-place destination
repeats transitions and changes future outputs.

The FlashInfer Mamba endpoint `mamba/replayssm_materialize.py:234-323` shares the
retain-inputs/replay idea but has different semantics: pointer-selected source
and destination slots, ring_start mapping, accepted-history bounds, optional
quantized-state scales/stochastic rounding, and exact source-state/scale copy
when replay length is zero. Those are not direct bindings of this first linear,
in-place, zero-no-op candidate. Its pointer-backed dependency anchors also belong
to the compiler's read/write contract. Retain these adapter/codec differences in
the [recurrent inventory](recurrent-inventory.md); do not claim the whole Mamba
endpoint through the GDN rule.

A source-reading correction prevents a spurious inverse candidate:
`gdn_kernels/gdn_decode_bf16_wy_output_only.py:1256` still labels Phase2 a
log-depth Neumann inverse, but`:1298-1405` actually implements two8x8 diagonal
forward substitutions and dependent off-diagonal solve with diagonal-beta RHS.
Its4/8-token specializations omit zero padded block work. Existing
[blocked_unit_lower_inverse@1](../tilelang/delta/blocked_unit_lower_inverse@1/template.yaml)
is specifically four16-row blocks producing a full inverse; it cannot bind this
two8-row weighted solve unchanged. Trace the actual solve/feedback producer next.
