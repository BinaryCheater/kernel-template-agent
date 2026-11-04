# CuTe Blackwell forward: first representative

SM100 separates MMA, softmax, output correction and loading into distinct warp
roles. Their readiness cycle is more specific than generic producer/consumer
warp specialization. Pinned [inputs](inputs/sources.json); all paths below start
at `flash_attn/cute/`. Full family comparison remains active.

Representative source binding: oneCTA, FP16, head/value dimension128, tile_m128,
tile_n128, q_stage2, two Q tiles/CTA, no S ping-pong, dense nonpaged, no split,
no custom score/mask callback. This fixes a source menu member for tracing; it is
not a claim that a particular runtime workload was compiled or selected by timing.

| Choice / source | Existing field or delta | Binding / outstanding premise |
|---|---|---|
| `flash_fwd_sm100.py:203–218,322–375`: QK MMA tile128×128 per stage, two Q stages,512threads/CTA | `output_tiling@2` + physical_storage@1 stage occupancies; **multi-role specialization extension gap** | Softmax warps0..3 and4..7, correction8..11,MMA12,epilogue13,load14,unused15. q_stage1 and non-TMA variants reassign idle groups. Old warp_specialization@1 only parameterizes producer plus homogeneous consumer WGs and cannot express this role graph fully. |
| `:1764–1860`: tcgen05 QK SS into TMEM S; PV reads TMEM P and SMEM V, accumulates TMEM O | `compute_instruction_selection@1`: tcgen05_mma, operand/source realization gap; physical_storage@1 level tmem | Pointer/descriptor helpers show actual instruction construction but not emitted machine code. One/two-CTA atom ownership differs, not merely the same tile size twice. |
| `:1032–1058`: cyclic S-ready → P-prefix-ready **and** O-corrected → PV, plus separate suffix and final-O events | [prefix_ready_contraction@1](delta/prefix_ready_contraction@1/template.yaml) with prefix_k96; ordinary slot ownership remains separate | The pipeline's nominal consumer release is a fan-in of softmax and correction participants. First correction signals are seeded because no old O needs scaling. Neither P-prefix publication nor O correction alone admits PV. |
| `:2680–2755`: post-mask maximum, scale publication, exponent/convert, two-phase TMEM P write | Prefix candidate + online_rescaled_reduction@2 | Prefix is floor(floor(K/4)*3/32)*32; K128→96,K192→128. Producer fences TMEM stores before publication; tail wait is inside generated MMA helper. Mapping must align with actual MMA K atoms and TMEM copy fragments. |
| `blackwell_helpers.py:540–611`: prefix atoms, parity wait, suffix atoms in one inline assembly sequence | Prefix candidate implementation boundary | Native `tcgen05`/barrier/proxy ordering at that inline wait is an open binding obligation, including required lowering fences. Do not infer that a source barrier call alone supplies every native ordering edge. |
| `flash_fwd_sm100.py:2360–2382`, `softmax.py:341–418`: finite frame policy | online_rescaled_reduction@2: 16-bit lag8,gain0; FP8 running_max,lag0,gain8 | FP8 gain256 applies to P and denominator, LSE subtracts8 in base2. Actual exponent and narrow-format ranges constrain gain+lag. Current schema integer domain includes these source choices, without admitting every arbitrary combination. |
| `:2868–2930,3122–3170`: correction group receives scale and optionally loads/rescales/stores O in16-column chunks | online_rescaled_reduction@2 rescale order + physical_storage@1; **role-publication mapping gap** | For ordinary two-Q-stage schedule, next S readiness implies previous PV completion by issue order; the source avoids an extra O wait. In S ping-pong that implication fails and correction explicitly waits O-ready. These are different ordering premises. |
| `:2964–2977,3080–3087`: finite sink gain and empty split state | empty_support_normalization@1; exact-one extra-term ownership | Only split0 owns the sink. Empty ordinary state plus finite sink sets L=2^max_offset. Final LSE subtracts max_offset. Padded/negative-infinite sink heads are outside the new finite-sink rule's scope. |
| `:254–265`: SM103 fused TMEM-load row-max reduction only without callbacks and non-D32 | **Fused load/reduction realization gap** | Masked iterations use post-mask software max (:2687–2691); hardware pre-mask max cannot replace the semantic reduction without additional correction. A helper's hardware maximum is not automatically the maximum of the masked logical row. |

Stage witness: with q_stage2, Q stage0 and1 have distinct S/P state and O
accumulators. For each stage P's prefix signal joins correction's O-ready arrival
before any PV prefix is issued. Tail publication can lag those prefix MMAs; the
consumer blocks before crossing k96, then consumes96..127 and waits for total
completion before reuse. The source's split-P event is an operand-ready frontier,
not permission to free a whole slot early.

The [PTX ISA 9.2 fence definition](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.2.pdf)
specifies specialized tcgen05 ordering composed with execution synchronization.
The document is [retained locally with provenance](inputs/references.json).
That contract supplies an obligation for the inline tail wait, not evidence that
this source instance's native lowering satisfies it. No native test or bug chase
was performed. Remaining SM100 inventory includes S ping-pong and twoCTA details,
uneven KV storage, exp2 emulation, scheduler/descriptor variants and low-precision
transport. These will be compared by their new choices before closing the group.
