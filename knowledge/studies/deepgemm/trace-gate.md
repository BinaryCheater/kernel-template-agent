# MegaGate: score production, selection and routing

All paths below are relative to [the pinned DeepGEMM tree](inputs/sources.json).
This slice follows the single public BF16 MegaGate API through its configuration,
SM100 mainloop, score workspace, barrier, epilogue and routing controls. The retained
source tests supply caller examples; their CUDA checks and benchmarks were not run.

## Contract and representative instance

`csrc/apis/mega_gate.hpp:29–183` accepts contiguous BF16 x `[T,H]` and weights `[E,H]`,
with positive H divisible by 256, E divisible by 4 and at most 512, and requested
`0 < k <= E`. The total output slots, k plus optional shared experts, must fit 32.
Nonempty input selects architecture major 10; empty T returns allocated empty outputs
before that gate. Up to two shared experts are allowed only under the API's divisibility
conditions. `out` contains int64 indices and FP32 weights. Optional unmapped indices
have contiguous slots but may have a wider row stride. Bias, image bias, masks,
fixed routing, random routing and physical mappings are compile-time presence branches;
their per-token values remain runtime data.

Take T16, H4096, E128, k8, no bias, deterministic mode, num_sms132. The host menu has
one expert group. Its single-CTA candidate is pruned by task doubling; deterministic
mode forbids K splitting, so the surviving pair has two logical CTAs, one token block
of width16, one worker group actually launched and two gate warpgroups (256 gate
threads plus 128 other threads). These are source-formula substitutions, not a timed
configuration result. Each CTA owns 64 experts; the pair divides the token paths into
eight-token halves. There are 64 K64 blocks per slice. Workspace is
`1 * 16 * 1 * 128 * 4 = 8192` bytes. Tails use an effective UMMA N rounded to 16 while
stores of final routing outputs guard the true token count.

Weights are UMMA operand A, tokens operand B, so a tensor-memory row/lane holds an
expert and a column a token. Expert-major accumulator orientation is deliberate:
the gate consumes expert vectors for a single token. The epilogue has a separate
UMMA_M128/two-CTA mapping: warp pairs see the same experts for different token halves.
Instruction fragment lowering is still an implementation obligation.

## Decisions and bindings

| Source choice | Rule / field or gap | Evidence and boundary |
|---|---|---|
| `heuristics/mega_gate.hpp:58–68,107–156`: align E to128; groups with128 or256 experts; one or two MMA CTAs; K splits1/2/4/8, equal whole-K64 ranges | `reduction_partition@1`, `num_groups=S`, `group_size=H/S`, contiguous_parallel; legal menu | H must divide S*64, pair+S8 is excluded, logical CTAs must fit SM count and be below64. Deterministic mode forces S1 for batch invariance. |
| Same file `:74–104,158–178`: task doubling prune then lexicographic preference | Dispatch policy, not a measured optimum | Prefer longer transpose, more launched SMs, pairs, fewer splits, fewer waves. Token width is smallest16-aligned width preserving maximal-width256 wave count. Gate WGs balance transpose/top-k units up to7; multiple waves use7. Stage count follows remaining SMEM, capped32, requires at least2. Full menu winner/resource replay remains open beyond the representative substitution. |
| `layout/mega_gate.cuh:46–77`: 4 KiB metadata cache and runtime-stride x/weight rings | `physical_storage@1`, `multi_buffer_rotation@2` partial matches | Cache bias then image bias, count only if it fits; padded aligned count controls device allocation. For E512 with both biases, count stays global. Source ring can exceed seed S8 domain, so no widened binding is implied. |
| `impls/sm100_bf16_mega_gate.cuh:70–77,124–128`: static worker-group striding and slice decoder | `persistent_tile_scheduler@1.fetch=static_stride`, grouped CTA decoder external | slice maps to expert group then K split; worker group strides token blocks. No new task-descriptor array. |
| Device `:133–244`: TMA producer, leader-only UMMA, four K16 MMA per K64 block, two accumulator slots | `warp_specialization@1`, `per_slot_handshake@1`, `compute_instruction_selection@2` partial bindings | Input stage/phase persists across token blocks. With eight K blocks and five input stages, successive starts are0/3/1/4, not all0. The accumulator ring has a separate phase. Pair tail waits for the final remote empty arrival before barrier teardown. Exact ISA/address/resource bindings remain open. |
| Device `:270–303`, `scheduler/mega_gate.cuh:12–43`: per-token-block score publication | Related to `epoch_tagged_completion@1`, exact binding gap | Gate uses `%gridid`, tag spacing64 and fewer than64 producers; MHC rule fixes spacing128 and shared epoch retirement. Do not substitute either protocol. Grid-id freshness during graph replay, tag wrap, async release visibility, co-residency and progress require device evidence. Single logical CTA omits global join. |
| Device `:283–320,377–398`: score activation depends on presence of bias and K splitting | Pointwise map / materialization placement; nonlinear split-combine boundary | With bias,S1 store sqrt-softplus once. With bias,S>1 store raw partials and sum in fixed split order before activation. Without bias rank raw reduced logits and activate only selected scores. Activating each partial before combining changes the function. |
| Device `:379–398`: image bias overrides ordinary bias only for image tokens; invalid experts become negative infinity | `boundary_predication@1`, ranking policy | E divisible4 makes validity uniform for each lane vector. Bias changes selection but never the weight numerator. NaNs/nonfinite ranking are outside the proposed exact-selection rule. |
| `epilogue/sm100_mega_gate.cuh:76–152`: five-comparator local sort, head merge, packed cursors | New `sorted_run_topk@1`, run_length4, k=requested | 32 lanes, up to4 waves. Each wave holds four consecutive experts per lane. Original local offsets use two bits; cursor uses three bits so exhausted value4 fits. Per-lane equal heads favor earlier waves; warp ties choose minimum global identity. |
| Device `:412–427`, epilogue `:158–181`: get selected unbiased scores | Selected-value gather | S1 rereads global score; S>1 shuffles the saved local reduced scores. These are different storage/recompute choices. They do not allow using biased rank values as weights. |
| Epilogue `:188–216`: append shared experts and normalize selected scores | Output numerical/routing contract | Routed weight is score/(selected sum+1e-20)*factor, shared weight1. Sum width8/16/32 follows k. Score-based physical mapping uses count and uint32(rank+token*23333) modulo count. Preserve unsigned wrap. |
| Device `:323–357`, epilogue `:172–184,219–253`: mask, random, fixed, ranked controls | Workload dispatch; separate selection contracts | Precedence mask > random > fixed > ranked. Mask emits -1/0; fixed reads logical slots, permits duplicates, reweights current scores. Random emits distinct physical IDs, independent positive random weights and logical -1. Fixed duplicate slots are outside sorted_run_topk's unique-ID domain. |

## Ranking and numerical boundaries

The stored or reduced logit l is transformed by `sqrt(log1p(exp(l)))`, with the source
choosing l directly as softplus for l>20. Exp/log/sqrt accuracy and overflow behavior
are not supplied by the scalar replay. Bias can change membership: unbiased scores
`[1,2,4,8]` with bias `[10,0,0,-20]` select ids0/2, yet their routed weights at factor1.5
are0.3/1.2, not the normalization of biased values. Image-token bias replaces ordinary
bias rather than adding both.

The no-bias branch uses real monotonicity to avoid activating all experts. Finite
precision can collapse distinct logits into equal scores. The scalar FP32 exp-rounding
witness l=-200/-199 produces two zero activations: raw ranking chooses id1 whereas
activated-score ranking with smallest-id tie breaking chooses id0. This is a numerical
policy boundary, not a GPU failure claim. The proposed selection rule operates on the
actual supplied ranking values; it does not assert that raw and rounded-activated ties
have identical identities. A future transform-deferral rule needs that extra premise.

Random routing samples each lane in range n-i, repeatedly removes the smallest current
candidate and shifts affected later lanes. The replay enumerates1904 small draw domains
and verifies bounds/uniqueness, not Philox output or uniformity (modulo reduction itself
has a distribution question). With a physical map the branch sets n to the sum of
logical counts and emits dense IDs0..n-1 directly. It does not read mapping entries.
Non-dense/arbitrary physical mappings therefore need an additional caller premise or
an API correction; current host validation checks only shapes/dtypes. Score-based
mapping does use the table. Fixed indices and logical duplicate counts have device
assertions, but host validation does not inspect their data values.

## Checks, reuse and remaining obligations

[mega_gate.py](checks/mega_gate.py) and [retained output](checks/mega-gate-result.json)
check81 local sorting-network inputs,168 full top-k domains against an independent
sort,540 token ownership domains,48 wrapped duplicate selectors and the random draw
cases. Equal scores, negative scores, exhausted runs, padded E260/264 and all four
expert waves distinguish tie/identity/cursor errors. Input stage carry and control
precedence are checked separately. These are source-equation checks, not GPU results.

The starting pool has reductions and indexed gathers but no candidate-run/cursor object
or exact ordered top-k consumption invariant. [sorted_run_topk@1](delta/sorted_run_topk@1/template.yaml)
and its outline isolate that object without coupling selection to activation, physical
expert mapping or weight normalization. The [partial chain](delta/gate-chain.yaml)
composes K partition and raw-logit selection for the representative instance; publication,
selected activation and output normalization remain explicit external edges. All records
remain declared proposals. Full heuristic replay, tensor-memory/TMA mapping, release
visibility, grid-id lifetime, arithmetic tolerances and real repeated execution remain
open before implementation admission. Continue the library inventory with MegaMoE.
