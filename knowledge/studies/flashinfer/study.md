# FlashInfer NVIDIA design study

Assignment: full-library NVIDIA source-design pass in the continuing
[catalog](../../../docs/runbooks/learndemo-1/operator-source-catalog.md). Extract
distinct reusable computation, layout, pipeline, quantization and scheduling
choices. Closure requires a source-backed disposition for every design family,
with remaining gaps named; the classic attention subset does not close the study.

Status: partial; paused by user direction at the Worker A convergence checkpoint.
Do not start FlashMLA or ThunderKittens or expand this study without a new
instruction. Preserve the retained source, all proposals and all prior checks.
There are56 indexed provisional Template/Proof pairs,51 logical check scripts,
and one additional preserved draft pair. The most recent existing
[schema report](inputs/schema-check.json) covers all57 pairs; the draft has no
focused numerical check or completed source trace. No new checks were run for
this convergence checkpoint. These are file proposals, not Store admission,
native execution evidence or full-library closure.

[Source manifest](inputs/sources.json) pins the retained full archive and external
extracted tree. Root Apache-2.0, additional KDA license, NOTICE and per-file notices
remain there. CUTLASS, spdlog, CCCL and nixl submodule bodies are absent. Do not
fetch CUTLASS or resume its user-paused study. External/precompiled provider
kernels are implementation boundaries. Source AGENTS.md/CLAUDE.md files are
study material, not construction instructions.

[Starting pool](inputs/pool-snapshot.json) retains the preceding pinned pool and
fifteen FlashAttention proposals. Preserve historical versions and inspect actual
fields before reusing a rule. The [delta index](delta/index.md) gives the current
proposals; trace reuse must retain numerical, coordinate and protocol premises.

Use the [full-library inventory](inventory.md) and retained
[implementation-file list](inputs/source-file-inventory.json) for coverage.
Focused inventories carry the larger families:

- [GEMM](gemm-inventory.md): grouped/routed ownership, scales, low-rank branches,
  deterministic partials and native-provider boundaries.
- [Quantization/norm](quantization-norm-inventory.md): distinct scale recipes,
  layout/packing, complete statistic domains and rounded checkpoints.
- [Rotary/cache](rotary-cache-inventory.md): original partner supply, positions,
  normalization scope, cache generation and destination ownership.
- [Recurrent](recurrent-inventory.md): frozen checkpoints, selective commit, feedback solves,
  state layouts, scans and provider boundaries.
- [Sampling](sampling-inventory.md): conditional rejection, speculative correction,
  weighted radix, fixed-point prefixes, tie support and typed processing chains.

Latest indexed extraction: [self-resetting partition queue](trace-self-resetting-partition-queue.md)
adds zero-history forward task claims and last-claimer counter reset. Existing
checks explore29448 protocol states and distinguish retirement from output
completion. The grid-join and halo traces retain the separate KCP backward
completion and distributed-edge contracts.

## Exact resume point

Resume the zero-history convolution backward precision study, not a new library.
The draft [rematerialized_activation_gradient_recipe@1](delta/rematerialized_activation_gradient_recipe@1/template.yaml)
and [proof](delta/rematerialized_activation_gradient_recipe@1/proof.yaml) are
preserved and schema-validated, but are not yet indexed as finished extraction.
Next complete their source trace and consequential stage check, then reconcile
the draft against the existing rule pool before registration. No focused check
for this draft has been written or run.

Already-read source is the retained public
`flashinfer/cake_causal_conv1d_det_train_backend/`:

- `cake_jit.py:2638-2697` and `backward_route` define rounded/unrounded,
  bias and dtype routes. Load/store flags derive from actual operand and
  gradient destination types, independently.
- `csrc/cake_causal_conv1d_det_train_9c8310a1d3fa9b8ccffa_kernel.cu:450-470`
  uses the rounded preactivation recipe; the unrounded representative
  `cake_causal_conv1d_det_train_c8027a64ba3cc38333bf_kernel.cu:435-455`
  also changes the tap arithmetic to its FMA recipe. Do not reduce this
  distinction to one dtype flag.
- Reducer `cake_causal_conv1d_det_train_ce49146b9fef24adb47e_kernel.cu:93-176`
  widens stored FP32 partials for FP64 final merging, converts the total to
  FP32, then optionally to BF16. It does not compute local partials in FP64
  or directly round a FP64 total to BF16.
- Bias and mixed parameter/output format representatives, token-grid trimming,
  and inference state/ring/accepted-prefix paths still need their distinct
  source bindings. The existing KCP and queue traces do not close these paths.

## Materially unfinished families

- Recurrent designs: remaining GDN/KDA target pipelines, prepacked state,
  persistent/small-head/packed/output-only KDA, other training backends,
  Mamba selective-state update/checkpoint/materialization codecs and mutable
  dependency publication, and SSD portfolio/scheduler variants. See the
  [recurrent inventory](recurrent-inventory.md).
- Causal convolution: the preserved backward precision draft above, remaining
  bias/dtype and grid-trim variants, inference ring/accepted-prefix state,
  and distinct experimental training bodies.
- MoE routing/local experts: permutation/capacity, expert scheduling,
  quantized contraction/finalization and gradient organizations.
- Hyperconnection and gated/composite model blocks, including Sinkhorn,
  projection/quantization dependencies and saved training state.
- Communication/distributed fusion: rank maps, readiness, collective protocols,
  compression and shared/remote buffer lifetimes.
- Vocabulary loss and LM head designs.
- Remaining attention/task-native/training/model variants, B12x native maps,
  CLC/progress, ExpCast/orthogonal preconditioning and endpoint/layout gaps;
  remaining GEMM native/cluster split-K, quantization/norm/activation,
  rotary/cache and sampling variants listed in their focused inventories.
- Provider/precompiled boundaries and final full-file reconciliation.

The [full inventory](inventory.md) remains authoritative for family-level scope;
all focused traces retain their specific gaps. Listing these families does not
close their coverage. FlashMLA and ThunderKittens are not started. CUTLASS
remains separately user-paused and must not be fetched or reopened.

Keep all retained sources and study artifacts. Source extraction does not require
installation, GPU benchmarking, end-to-end replay, WP gate, commit, push or
promotion. Do not touch concurrent performance-model edits in the shared checkout.
