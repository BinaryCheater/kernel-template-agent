# Auxiliary NVIDIA operator designs

When attention is embedded in training blocks, the remaining kernel choices
concern saved intermediates, row statistics, affine-gradient ownership and
distributed class normalization. Paths below are relative to the
[retained tree](inputs/sources.json). CUDA extensions, current Triton operators
and historical retained operators are all included; provider device internals
and native performance remain separate evidence.

## Dense, activation and training checkpoints

| Source decision | Rule / field or explicit gap | Concrete binding and boundary |
|---|---|---|
| csrc/fused_dense_lib/setup.py:30-32 -> fused_dense.cpp:94-150 -> fused_dense_cuda.cu:128-226 | operator_fusion@2 elementwise/bias scale choices; vendor epilogue realization | FP16/BF16 GEMM uses FP32 compute, bias and GELU/ReLU epilogues. AUX is optional. Workspace is4MiB before SM90 and32MiB at major>=9. Up to five heuristic results are requested; availability/alignment and selected result remain provider obligations, not guaranteed by a heuristic index. No vendor device implementation is extracted. |
| csrc/fused_dense_lib/fused_dense.cpp:122-125,164-174; fused_dense_cuda.cu:462-482 | Activation-specific AUX encoding gap | GELU retains full preactivation in the input dtype; ReLU retains one bit per element in uint8. Derivative epilogues consume the matching representation. The buffer named pre_act is not universally an affine value. ReLU AUX bit packing/provider layout is an external realization boundary. |
| flash_attn/ops/fused_dense.py:249-344,347-395; ops/triton/mlp.py:15-92 | [affine_activation_rematerialization@1](delta/affine_activation_rematerialization@1/template.yaml) | GELU and squared-ReLU expose retain-both, retain-affine, or retain-inputs/recompute-affine+activation. Keep forward weights/bias and cast stage immutable. Main ReLU level1 instead keeps output plus tiny derivative AUX: the AUX cannot regenerate magnitudes. That representation is not admitted as a full affine checkpoint. |
| flash_attn/ops/triton/linear.py:175-245,258-346 | output_tiling@2, block_rasterization@1, operator_fusion@2 checkpoint=materialize | GROUP_M8 groups rows for L2 reuse. K-tile dot accumulates FP32, then bias, optional preactivation save and activation. Wrapped load coordinates fill padded output lanes; only original output bounds may be stored. The preactivation save path uses wrapped coordinates without the final output predicate, so duplicate tail writes need a native ownership binding; do not infer safe arbitrary tails. All active configs use SPLIT_K1; commented split-K choices are not connected capability. |
| flash_attn/ops/triton/linear.py:428-529; ops/triton/k_activations.py:55-162; ops/activations.py | Existing tiled contraction; derivative-epilogue/numeric binding | dgrad contracts incoming gradient with weights then multiplies the selected activation derivative at saved inputs. Exact-erf GELU, tanh GELU approximation, squared-ReLU and ReLU are distinct functions, with matching derivatives; their names are not interchangeable numeric policies. Packed activation launch/fuser calls remain provider/compiler boundaries. |
| flash_attn/ops/fused_dense.py:291-310,369-395,548-550,667-687; utils/distributed.py:18-113 | Collective handoff concepts; whole-tensor wrapper binding | Sequence-parallel MLP all-gathers input before GEMM and reduce-scatters output; async input gather overlaps dtype conversion or other independent work and is waited before first read. This is not TE's per-ready-chunk pipeline unless chunk/event boundaries are added. AllGather backward is ReduceScatter and vice versa; the all-reduce wrapper's backward identity assumes its declared tensor-parallel ownership convention. |

For A=affine(X,W,b), B=activation(A), saving only A can regenerate B;
saving X,W,b can regenerate both. Retaining only sign(A) cannot reconstruct
ReLU(A): affine values1 and2 share the same derivative bit but give outputs1
and2. The [logical check](checks/check_auxiliary_choices.py) and
[observations](checks/check_auxiliary_choices.json) compare the three full-value
retention modes and reject changed-weight replay. Norm reconstruction below
uses a different saved-state boundary and is not silently added to this rule.

## Normalization, residuals and dropout

| Source decision | Rule / field or explicit gap | Concrete binding and boundary |
|---|---|---|
| csrc/layer_norm/setup.py:115 onward -> ln_api.cpp -> ln_fwd_kernels.cuh:95-178,ln_utils.cuh:674-787 | centered_moment_merge@1 and reduction_partition@1 | CUDA forward retains the combined row, computes mean then squared deviations per warp and merges valid-count statistics across warps. Padded lanes are excluded from counts. Selected forward asserts CTAS_PER_ROW1; generic inter-CTA utility code does not establish a multi-CTA forward path. RMS rsqrt uses M2/N+mean^2+epsilon, while LN uses M2/N+epsilon. |
| csrc/layer_norm/ln_fwd_kernels.cuh:75-138; ln_bwd_kernels.cuh:95-182 | Materialized dropout mask, distinct from random replay | curand Philox state is thread-owned and advanced over the row loop; forward stores the mask and backward reads it. The source does not replay FA2's byte law or Triton's float law. Preserve row/column scales, dropout, residual addition and accumulation dtype in their original order. |
| csrc/layer_norm/ln_fwd_kernels.cuh:103-108,172 onward; ln_bwd_kernels.cuh:97-117 | indexed_gather@1 only for input reads; subset output/inverse-gradient mapping gap | Subset indices are one-based;0 means absent. A missing x0 can still leave a residual-defined row. z_subset independently decides output position. Forward output stores and backward input stores are not collision atomics; unique map ownership or an explicit combine is needed, not indexed_additive_scatter by name. |
| csrc/layer_norm/ln_parallel_residual_{fwd,bwd}_kernels.cuh:11 onward | Shared row-statistic reuse; branch fusion boundary | Two independently dropped inputs join one residual row, then tied or separate affine normalization outputs use the same statistics. Backward sums both affine branches into the common row derivative before distributing to each input/dropout mask. A branch-local derivative is insufficient. Standard/subset/parallel registrations are related variants, not new statistic algebra. |
| flash_attn/ops/triton/layer_norm.py:210-289 | operator_fusion@2 norm_scale, reduction_partition@1 | One program owns a full row. Compute scaled/dropped x, optional independently seeded x1 and residual, save combined pre-norm row when required, then LN mean/centered square or RMS direct square. FP32 row math precedes output casts; zero-centered gamma adds1 explicitly. Whole feature row must fit the64KiB bound. |
| ops/triton/layer_norm.py:227-250,609-624 | counter_indexed_mask_replay@1 with a distinct generator binding | Triton tl.rand(seed[row],column,n_rounds7)>dropout_p; the second input uses seed[M+row]. Replay keeps this seed/component/float-comparison identity. Seven rounds alone does not make it equivalent to FA2's byte-threshold generator. Optional returned masks do not replace replay requirements. |
| ops/triton/layer_norm.py:555-642,750-843 | reduction_partition@1 then split_combine_kernel@1 add for affine gradients | LN dX=rs*(wdy-mean(wdy)-xhat*mean(xhat*wdy)); RMS omits mean(wdy). Sum a second affine branch before these reductions. Each program visits rows_per_program rows and writes private FP32 dGamma/dBeta partials; final sum over those program rows, then cast. Extra empty row programs write identity partials; count/empty handling must accompany the generic partition. |
| ops/triton/layer_norm.py:1129-1218 | Normalized-output reconstruction binding gap | Norm+linear saves pre-norm row, gamma/beta, mean/rsigma and linear weight, but omits normalized y. Backward reconstructs y while computing norm gradients, then uses it for the linear weight gradient. This needs matching norm state/casts and all consumers; it is not affine_activation_rematerialization@1's affine-plus-activation graph. |

The common invariant is two different reduction domains: dX reduces over
features of one row, while dGamma/dBeta reduce over rows for one feature.
Combining them into one generic reduction owner would lose output ownership.
For parallel affine outputs, wdy=gamma0*dY0+gamma1*dY1 enters the shared
normalization Jacobian; individual parameter gradients still use each branch's
own dY. CUDA and Triton agree on this structure but differ in statistics, mask
representation, precision and schedule. The historical CUDA README's migration
notice does not remove its retained designs from coverage.

## Vocabulary-partitioned cross entropy

flash_attn/losses/cross_entropy.py calls ops/triton/cross_entropy.py. Each row
program streams at most16384 columns per tile into a local online LSE and,
when smoothing is enabled, a local logit sum. It reads the target logit only
on its owning shard. Source :215-248 all-gathers local LSE scalars and sums
additive loss terms; logsumexp of local LSE yields the global normalizer.
Add that global LSE and its squared penalty once. The source marks the
separate z-loss return non-differentiable; its value is already included in
the primary loss, so it is a diagnostic rather than a second gradient path.

[vocabulary_partitioned_loss@1](delta/vocabulary_partitioned_loss@1/template.yaml)
records this graph, smoothing/global-class-count contract and local gradient
ownership. Source backward :104-146 reconstructs probabilities from the final
global LSE, adds the derivative of squared-LSE regularization, subtracts the
global smoothing/target terms and applies incoming gradient and logit scale.
Ignore-index rows zero both loss and gradient. Optional in-place dlogits is
legal only after forward logits' last required external read.

PRECOMPUTED_LSE is enabled only without smoothing and at scale1. In distributed
mode it must be a local-shard statistic, because the host still merges it with
other local LSE values. Giving each of two ranks an already global L yields
L+log2, not L. The logical check compares target owners on both shards, smoothing
on/off, squared-LSE penalty on/off and finite-difference local gradients. Native
collective row identity, provider communication and accuracy remain bindings.

## Rotary and composition wrappers

ops/triton/rotary.py:25-99,128-193 rotates interleaved pairs or half-split pairs
in FP32, preserves the feature suffix, and optionally writes in place. Every
pair's two inputs load before its outputs overwrite them. Varlen token offsets
come from cumulative lengths; cos/sin positions add either a scalar or a
per-sequence offset. Positions beyond the table use identity cos1/sin0 in the
kernel, while the scalar host path asserts a valid range. Conjugate mode
negates sin for backward; it is the transpose linear map even if finite stored
cos/sin do not form an exact orthogonal inverse. layers/rotary.py packs Q/K
heads when contiguous and preserves their distinct offsets; cache rotation
reuses the earlier append conventions. Exact pair/sequence maps are bindings.

modules/{mha,mlp,block,embedding}.py compose the already listed attention,
rotary, dense, normalization, activation, dropout and distributed wrappers.
Embedding vocabulary partitions zero unowned contributions and reduce them;
the framework embedding implementation is an external device boundary.
Generation utilities manage cache metadata but introduce no additional local
NVIDIA kernel family. No new standalone kernel design is inferred from a
Python module/class count. The remaining native gaps are provider epilogue/AUX
layout, subset write ownership, stochastic laws, normalization reconstruction,
collective completion and compiler tail behavior, each distinct from the two
new reusable choices extracted above.
