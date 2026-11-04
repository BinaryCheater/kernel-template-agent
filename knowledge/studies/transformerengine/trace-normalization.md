# Normalization: centered statistics and repeated row reductions

When a normalization row spans multiple warps or CTAs, separate the statistic
algebra from its communication protocol. Paths are under the retained source's
`transformer_engine/common/`.

| Source and concrete binding | Rule/choice | Boundary |
|---|---|---|
| `normalization/layernorm/ln_fwd_kernels.cuh:85–127`, `utils.cuh:844–863` | Local mean followed by centered squared deviations; new [centered_moment_merge@1](delta/centered_moment_merge@1/template.yaml) combines partial statistics | Local implementation is two-pass over retained register values, not necessarily per-sample online Welford. Tuned equal partitions infer counts; a tail cannot silently inherit a full partition count. |
| `utils.cuh:672–700,706–827` | Count-aware Chan merge over warp/CTA partials; equal-count mode transports only mean/M2 | Global M2 includes the between-partition mean correction. Merely summing or averaging local variances is wrong. Count and final population denominator are part of the contract. |
| `normalization/layernorm/ln_fwd_kernels.cuh:232–278` | General/tail path first allreduces mean, then centered deviation sum | Reuse partition, scalar allreduce and storage rules. This differs from computing raw E[x²]-E[x]² or merging per-CTA means/M2. Mean must be visible before computing deviations; padded values cannot enter the valid-row count. |
| `utils.cuh:400–435,445–488,606–648` | New [reusable_group_partial_allreduce@1](delta/reusable_group_partial_allreduce@1/template.yaml) | GPU-global release-add/acquire spin counters, not cluster remote transaction barriers. Two counters and two partial banks alternate across calls. Every participant follows the same generation order, and launch progress must be guaranteed. `replicated_partial_reduce@1` is one-shot and does not supply this reused-counter realization. |
| `normalization/rmsnorm/rmsnorm_fwd_kernels.cuh:96–119,161–248` | Tuned path reconstructs second moment as M2/N+mean²; general path directly reduces x²/N | Both target RMS, but arithmetic grouping differs. LN uses rsqrt(M2/N+epsilon); RMS omits centering of output. Zero-centered gamma adds one in the specified weight/compute dtype. |
| `normalization/layernorm/ln_bwd_kernels.cuh:99–166,209–300`; RMS backward relative | Two scalar row reductions for dX plus separate dGamma/dBeta partials and a finalizer | With y=(x-mu)*rs, dy=gamma*dZ, LN dX=rs*(dy-mean(dy)-y*mean(dy*y)). Parameter-gradient reduction is across rows; dX ownership is along columns. Reuse `reduction_partition@1` and `split_combine_kernel@1` rather than merging unrelated ownership domains. |
| `normalization/layernorm/ln_api.cpp:70–121`, `normalization/common.cpp` | Shape/type/alignment/backend plan registry and workspace query, then execution | cuDNN norm is an external implementation boundary. MXFP8 can select it; source also has a native RMS MXFP8 specialization. Training determines saved-statistic/column-scale requirements. A workspace-size query is not computation. Same-scale FP8 transpose may follow normalization as a separate launch. |
| `normalization/rmsnorm/rmsnorm_fwd_mxfp8.cu:7–19,67–80,439–452` | Statistics prepass followed by 32×128 normalize/quantize tiles | Whole-row rsigma is stored first; each output tile normalizes in FP32 and quantizes that FP32 value. This exposes more output CTAs than requiring a complete row in each quantizer. It does not silently insert a BF16 normalized tensor between the two steps. Caller-reserved SMs affect row launch chunking, not the row's numerical domain. |

## Two distinct invariants

For A=(1,3), B=(5,7), both local variances are 1. The merged variance is 5:
the missing term is the distance between means. `centered_moment_merge@1`
states the count/mean/M2 identity; native arithmetic and count inference are
separate obligations. Seed `online_rescaled_reduction@1` mentions Welford-like
statistics in scope but its fields/program carry max/sum/output, so it does not
instantiate this mean/M2 merge merely by name.

For repeated inter-CTA communication, each generation t publishes to bank t%2.
Counters follow `(0,0) → (P,0) → (P,P) → (0,P) → (0,0)`. Arrival at t+1 proves
that participant finished reading t; before anyone reuses t's counter/data at
t+2, the t+1 rendezvous proves all old readers/waiters retired. One shared bank
would allow a fast CTA's next write to overwrite a slow CTA's current read.
One alternating up/down counter would allow a fast CTA to remove the terminal
count before a slow CTA observes it.

The [finite abstract check](checks/check_reusable_group_barrier.py), run with
`.venv/bin/python knowledge/studies/transformerengine/checks/check_reusable_group_barrier.py`,
explores two participants over eight generations. Its
[result](checks/check_reusable_group_barrier.json) finds safe termination with
two counters/two banks, deadlock with one counter, and overwrite with one bank.
This resolves the logical reuse question. It assumes release/acquire publication
and fair progress; it does not validate native CTA visibility or occupancy.
In particular, all contributing threads' writes must be ordered before their
leader's release, and a spinning group must not exclude a required unscheduled
peer. These are premises, not consequences of the counter arithmetic.
