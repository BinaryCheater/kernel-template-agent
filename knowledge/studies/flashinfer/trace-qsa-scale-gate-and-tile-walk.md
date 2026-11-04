# QSA consumers: scales, gated output and contiguous tile walking

Sources refer to the retained [revision](inputs/sources.json). The explicit
`default` route is `flashinfer/experimental/qsa_nvfp4/backend.py` -> `jit.py`
-> `csrc/qsa_nvfp4_attention.cu/.cuh`. The Cake route comes from
`experimental/cake_qsa_nvfp4/cake_backend.py`, `cake_jit.py` and its retained
generated CUDA. They share a public semantic contract, not identical arithmetic.

## Direct quantized sparse consumer

| Source choice | Rule binding or gap | Concrete applicability |
|---|---|---|
| Default `qsa_nvfp4_attention.cuh:35-55,130-139`: key-owned QK, feature-owned PV, paired Q/K feature permutation | output_tiling@2 and explicit paired operand/output map | Four warps, head256. NVFP4 QK dimension is64*quad_lane+4*k_step; Q staging uses that same map. Shared selection/page offsets serve K and V. Do not apply a physical permutation to only one operand. Native fragment map remains a realization obligation. |
| :254-296,426-445: E2M1 values and E4M3 group16 scales converted with reciprocal exponent biases, then multiplied | decoded_tile_materialization@1 only after fixed decoder binding; register decoder realization | Source uses 2**120 and2**-120 factors to avoid a separate conversion bias multiply. Supported finite values and native multiplication/subnormal behavior must be verified; the comment's exactness is not inferred for arbitrary special encodings. No whole shared decoded tile is selected here. |
| `.cu` Params bind score_scale and output_scale; `.cuh:489,624` applies global K scale to logits and V scale at output | Scale-constancy binding; online_rescaled_reduction@2 gain1 | Group scales are per16 features and remain in local K/V decoding. Global K/V scales are separate. A feature-varying group scale cannot be moved outside the QK feature sum. |
| :489 onward uses finite masked sentinel and shared max/mass; :624-650 finalizes direct output | Existing row-state algebra with numeric/tail binding | Has-values/empty logic and finite sentinel must match the admitted range. Group normalization is not independently recomputed for each feature owner. |
| :668-755 merges normalized split outputs and base2 LSE, then BF16 cast, FP32 sigmoid gate, BF16 store | split_combine_kernel@1 plus explicit intermediate rounding | The gate applies once after all splits. Direct output :637-650 preserves the same first BF16 rounding; moving it through that cast changes output values. |

The [logical witness](checks/check_contiguous_tile_walk.json) includes
attention=1.001953125 and sigmoid(1) as the gate weight. BF16 attention first,
then gate gives0.73046875; gate before BF16 gives0.734375. This checks the
ordering contract, not the source's native sigmoid approximation or a full
attention error bound. A fused implementation must retain the first conversion.

## Source dispatch and alternate execution forms

Cake `README.md:66-114` and `cake_backend.py:255-337` derive selection tiles
of64 tokens and a split count from rows, KV heads and actual resident CTA
budget, with architecture/SM-count overrides. Actual split count remains
bounded by available tiles and merge capacity. Source lookup performance
comments are dispatch provenance, not measurements from this study.

The menu has direct single-split output, decode+separate merge, cooperative
same-launch merge and persistent tcgen05 walk. Cooperative forms require a
co-resident complete grid and ordered state publication; a source form being
compiled is not evidence of a reachable public branch. The recorded eager
host allowlist for SM100/103 is empty, while SM107 selects split form, so those
public calls do not exercise the cooperative build. No need to test or invent
performance for that dormant path.

`mma.sync` and tcgen05 are distinct realization groups: source supports
8.x/9.0/12.0 with the former and10.0/10.3/10.7 with the latter. Native TMEM,
CTA count, register budget and pipeline protocol differ. Capability12.0 does
not imply the SM100 instruction set; nearby unlisted versions do not inherit
this package's capability-exact automatic route.

## Contiguous walk changes per-group split geometry

[contiguous_group_tile_walk@1](delta/contiguous_group_tile_walk@1/template.yaml)
and its [proof](delta/contiguous_group_tile_walk@1/proof.yaml) capture the new
work-assignment object. Source `cake_jit.py:1134-1168` and generated persistent
kernel `csrc/cake_qsa_nvfp4_dfd8600d03040c9f808d_kernel.cu:310-347` agree:

- Flatten group g=(row,KV head) and tile t into s=g*T+t, S=groups*T.
- Worker c of G owns [floor(c*S/G),floor((c+1)*S/G)). The source chooses
  G=min(SM count,S), then walks all group intersections in that interval.
- The exact inverse owner of slot s is floor(((s+1)*G-1)/S).
- A group's first/last slot owners delimit its consecutive segment slots;
  this worker writes segment c-owner(g*T). Maximum segment count across
  groups determines the partial-state storage/merge capacity.

The rectangles are allocation domains; runtime valid lengths and page validity
still predicate actual contributions. When changing group, reset row/softmax
state but preserve input-ring generations carried across the walk. The source
keeps a cumulative raw-ring tile count and double-buffers Q across groups;
resetting every pipeline phase to zero at each row would require another
protocol. Partial state carries normalized output plus matching LSE, and
unused capacity must have defined identity or be excluded by merge bounds.

The logical check enumerates1620 small(group,tile,worker) cases, checks every
slot and partial ID once and reconstructs each group from uniquely valued tile
contributions. For S10,G3 the ranges are[0,3),[3,6),[6,10); slot3 belongs to
worker1, whereas floor(s*G/S) incorrectly gives0. This inverse error would
misnumber group partial slots even if aggregate tile coverage appeared right.

persistent_tile_scheduler@1 has static-stride/atomic fetch and lacks these
contiguous intersections and variable per-group partial counts.
cost_partitioned_worklists@1 emits greedy explicit task lists rather than
this division arithmetic. reduction_partition@1 supplies the partial-combine
algebra but not the worker interval/slot decoder. These existing rules remain
unchanged; the new candidate introduces exactly that missing choice.

Native group accumulator reset, carried mbarrier generations, TMEM lifetime,
empty partial representation and combine publication remain evidence obligations.
This QSA subgroup now has source-backed decoder/scale, output checkpoint,
execution-form and persistent assignment dispositions. Continue remaining
model/target attention families and then GEMM/recurrent/MoE and auxiliaries.
