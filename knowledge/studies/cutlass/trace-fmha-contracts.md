# Blackwell FMHA — entry and softmax decision slice

When attention streams score tiles into a normalized weighted sum, retrieve
`online_rescaled_reduction@1` for the running max/sum/output recurrence and
`approximate_math_rewrite@1` for exp2 with folded scale. Preserve the old running
maximum invariant and numerical premises. A source policy that freezes the
reference or discards a nonzero tile needs its own explicit decision and proof;
neither a generic fast-math label nor exact mask pruning supplies it.

The full Blackwell attention assignment contains12 files in four groups, retained
in [its inventory](inputs/attention-variants.json). Current scope is the FMHA
forward/backward pair under
`examples/python/CuTeDSL/cute/blackwell/kernel/attention/fmha/` in the
[pinned tree](inputs/sources.json). This slice extracts forward host predicates and
two softmax decisions. Device schedules, full recurrence/masks/precision, backward
and the other attention groups remain active; no new entry is marked complete.

## Decisions and pool comparison

| Source and binding | Observed choice | Old field or gap |
|---|---|---|
| Forward constructor:175–205 | CTA handles two M tiles; QK MMA=(M,N,D), PV MMA=(M,Dv,N). Scalar D shares Dv; tuple head dimension is restricted to(192,128). Cluster1, optional persistent work and TMA output. | Source geometry/menu. Workload D/Dv and software tiling are separate. Scheduler, operand layout and target/resource bindings remain to trace. |
| Constructor:207–239 | Two four-warp softmax groups, one four-warp correction group, one MMA warp, load warp, output warp and idle warp. | Existing warp-role names do not supply this full TMEM producer/consumer graph or its resource cost. Record the actual roles; no old four-warp MMA-consumer model is silently reused. |
| `check_supported_dtypes`:3280–3298 | QK input, PV input and output each in E4M3/f16/bf16; QK/PV accumulators both f32. | Actual frontend predicate, not native capability closure. Mixed precision's P conversion and selected MMA lowering still need numerical/layout binding. |
| `check_invalid_shape`:3300–3327 | Matching Q/K batch and D; D32/64/128/192; Q heads divisible by KV heads; tuple sequence lists have length batch. D192 requires QK E4M3. | No positivity checks on sequence values or head counts. KV heads0 raises division by zero rather than returning unsupported. `can_implement` catches only its unsupported exception. Host wrapper may impose additional checks; these are predicate-only boundaries. |
| `_setup_attributes`:334–357 | Q stage2, KV min(4 for8-bit else3), P/acc/softmax-correction stage1, MMA-correction/output stages2. Correction threshold8 log2 units when enabled, otherwise0. FP8 P pre-scale offset=max(0,floor(log2(448)−threshold)); LSE offset correction=offset*ln2. | Derived source menu and numeric range strategy. The D192/16-bit KV-stage2 branch exists but the public predicate rejects that QK dtype. Native capacities and consistent scale/LSE cancellation remain to trace. |
| `softmax_step`:2454–2466 | Guard −inf reference as0; with correction skipping enabled, keep old reference if its scaled increase is≤8. Send old/new references to correction. | Old `online_rescaled_reduction@1.program` requires m_new=max(old,tile max), and `invariant` names the exact prefix maximum. A frozen finite reference is algebraically usable but is not that old state invariant. Needs a reference-policy variant/revision or explicit gap; no old field supplies this choice. |
| `calculate_skip_softmax_flag`:1957–1987 | Per live row, request skip if `(tile_max−old_ref)*scale_log2 < log2(tau)`; out-of-range query rows also request skip. Vote all lanes, exchange four warp flags, require all four. Keep old reference when skipped. | Value-dependent nonidentity pruning, distinct from mask structure. `masked_tile_pruning@1.pre_neutral` requires every skipped value to be the reduction identity; it cannot supply this choice. A threshold on the largest individual weight is not a global discarded-mass/error certificate. |
| `softmax_step`:2514–2560 and `mma_pv`:1871–1912 | A skipped block bypasses probability computation/sum update and PV MMA, while still releasing/committing pipeline handles. | Changes both numerator and denominator. `approximate_math_rewrite@1` rewrites a map and has no tile-removal decision/error account. Native skip-flag publication and forward progress remain untraced here. |
| `correction_rescale`:3037–3049 | Read old/new reference pair, ballot whether any differ, rescale O with exp2(scale*(old−new)) only when needed. | Old recurrence's correction factor is a conditional match when its reference policy is respected. Skip-rescale control depends on the reference actually used for P and sum, not merely on whether the mathematical max changed. |

The exact existing bodies are retained in the reconciled pool:
[online recurrence](../../seed/references/flash_attention_2_fwd/templates/online_rescaled_reduction.yaml),
[math rewrite](../../seed/references/flash_attention_2_fwd/templates/approximate_math_rewrite.yaml)
and [mask pruning](../../seed/references/flash_attention_2_fwd/templates/masked_tile_pruning.yaml).
Historical gap files next to those candidates do not mean the current pinned
bodies are absent. No version is changed in this slice.

## Discriminating witnesses

[Check](checks/fmha_contracts.py) executes retained predicates at16807 dtype tuples
(27 accepted) and1440 shape cases; it also checks predicate acceptance of zero
sequences/negative tuple lengths and the KV-head0 exception. Those results apply
to this predicate rather than claiming the complete public `run` accepts every
such geometry. [Result](checks/fmha-contracts-result.json) records the domains.

The same check executes216 row-skip decision cases,16 four-warp vote combinations
and24 finite reference-freeze cases. The vote model assumes each stored flag is
one byte0/1; native exchange width, packing and barrier visibility remain separate
obligations. Reference freezing with old0/new max4 at threshold8 retains0, which
violates the old exact-max invariant. Yet weights(1,16) at reference0 and(1/16,1)
at reference4 normalize to the same16/17. This separates a potentially exact
change of representation from nonidentity work removal. Finite-range/overflow,
P rounding, LSE correction and the packed sum recurrence are still unverified.

For pruning, take a retained weight1 with value0 and a new block whose every
base2 score is−10. Tau0.01 accepts skipping because2^−10<0.01. With128 elements,
the block's total mass is0.125, and values1000 make the full normalized output
111.111…, while the pruned output is0. Six cardinality/value cases expose the
dependence on both tile size and value range. This is a scalar decision witness,
not a native GPU error measurement. A useful pruning rule must bind a numerical
contract and account for total dropped mass; source threshold alone does not
establish an arbitrary requested tolerance.

Next: trace the complete forward recurrence, P pre-scale/LSE compensation and
skip flag protocol before choosing paired rule deltas for reference selection and
bounded pruning. Then finish masks/schedules/layouts and backward. The remaining
attention groups and the full CUTLASS catalog scope are unchanged.

Continued in the [frame trace](trace-fmha-frames.md): version2 now makes reference
and common-gain choices explicit, with version1 preserved and sink compensation
counterexamples retained. Nonidentity pruning remains excluded and pending.
