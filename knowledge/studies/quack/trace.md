# Representative bindings and decision ledger

Source locations are relative to the pinned retained tree. Backend is NVIDIA CuTe DSL. No lowering or timing was run.

| Observed choice / concrete binding | Rule field / gap | Evidence |
|---|---|---|
| L=8, S=4, E=1, stride=1; fold XOR4 then swap/shuffle at 2,1 | distributed_slice_reduce@1 lanes/slices/stride, unique_store | reduce.py:371–458. Lane 5 owns slice1 but cannot store; lane1 stores it. Logical values use distinguishable lane/slice tags in replay. |
| SM80 softmax cluster=1; larger SM90 rows use threshold-selected cluster count | reduction_partition@1 assignment; physical_storage@1; dispatch only | softmax.py:37–64, 143–174; reduce.py:271–368 rescales each partial sum to final max before combine. |
| serial / parallel / separate split-K; each commits raw FP32 | reduction_partition@1 combine_complete; no nonlinear epilogue before final sum | gemm_config.py:22–33; determinism differs by combine schedule. |
| Tile integer anchor k=roundeven(max(z)*log2(e)); E=bf16(exp2(z*log2(e)-k)) | shifted_weight_checkpoint@1 provisional resemblance; anchor convention is an explicit local extension gap | epilogue/scaled_exp.py:1–22,35–68,78–95. Sum stored E, retain k. Full tile required: OOB exp2(-k) must not enter sum. |
| L=k_r*ln2+log(sum_t se_t*2^(k_t-k_r)); v=exp(k_r*ln2-L) | online_rescaled_reduction@2; shifted checkpoint compose | linear_cross_entropy.py:271–324; target correction uses exact FP32 target logit before downstream contractions. No claim of exact gradients after BF16 E rounding. |
| Shard-major rotated consumption + flag gate before first TMA | chunk_ready_collective_pipeline@1 (provisional), publication/progress premises | distributed/all_gather_gemm.py:7–38,65–74; tile_scheduler.py:124. Data/flag stream order and epoch lifetime belong to composition. Source comments' speeds are not measurements by this study. |
| QTIP lane has 32 weights, a 128-bit tail-biting stream, 16-bit windows advancing4 bits | Gap: correlated code stream rather than independent 4-bit affine codes | operand_transform/formats/qtip.py:2–29; QTIP2 changes state and fragment grouping. Reusing affine decode would be invalid. |

GEMM tile dimensions, stage count and cluster choices are optimizer choices inside architecture/config menus. Two-CTA M extent derives from selected MMA and cluster shape (gemm_config.py:57–68); no independent CTA-M knob is inferred. SM120 cannot inherit SM100 TMEM/cluster assumptions. Full instruction coordinate replay remains open.
