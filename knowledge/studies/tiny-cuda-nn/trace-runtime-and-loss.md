# Loss, reduction and runtime fusion

Sources are pinned in the [manifest](inputs/sources.json). Workload formulas,
GPU scheduling and host launch reuse are separate decisions here.

| Source design | Rule or explicit gap | Concrete binding and scope |
|---|---|---|
| `src/loss.cu:54–65`, `losses/*.h`: nine registered pointwise losses, padded predictions but compact targets | `output_tiling@2`: flat sample×padded-feature index; explicit pointwise arithmetic contract | L2,L1,RelativeL1,RelativeL2,RelativeL2Luminance,Mape,Smape,CrossEntropy,Variance. Prediction padding emits zero loss and zero derivative; target index is sample*logical_dims+feature. Data PDF divides values/derivatives and requires the selected formula's valid finite domain. |
| `losses/l2.h:40–82`: compute loss and derivative from shared difference/PDF/normalizer | `operator_fusion@2` structural idea; **loss-value-plus-gradient fused_ops realization gap** | Two logical outputs share arithmetic in one pass. With sample_count2,dims3,padded_width16, normalizer6 counts logical values, not32. Gradient has loss_scale; reported loss does not. |
| `losses/{relative_l1,relative_l2,relative_l2_luminance,mape,smape}.h` | Explicit gradient convention, not an interchangeable mathematical derivative | RelativeL2 denominator prediction²+.01 is reused as a frozen weight in the supplied derivative; SMAPE similarly omits differentiating its normalization. RGB luminance variant uses three channels. Do not derive supplied gradients by blindly differentiating reported scalar values. |
| `losses/{cross_entropy,variance_is,constant}.h`: log/reciprocal or externally supplied gradient | Same ownership/cast binding; workload-domain gate | Positive/nonzero prediction/PDF requirements follow the formulas. ConstantGradientLoss exists as a header API but is not registered by name; Trainer also accepts external dL/dy and sets its reported loss accordingly. These add no new pipeline. |
| `reduce_sum.h:41–143`: vector load16 bytes, scalar transform per component, warp XOR sums, warp partials in smem, first warp reduces and atomically adds one block result | `reduction_partition@1` twice, vectorized_access@1 for16-byte grouping subject to its realization; global atomic-combine realization is explicit | 1024threads/CTA; FP16 loads8,FP32 loads4,FP64 loads2. Byte count and base pointer must be16-byte aligned, no scalar residue path. output[sum] must be initialized before accumulation. Tail vector indices contribute0 but the element-count alignment remains mandatory. |
| `reduce_sum.h:145–224`, `src/reduce_sum.cu:39–80`: scalar-return wrapper and legacy multi-pass alternative | Current wrapper initializes accumulator and synchronizes return; legacy **implementation gap** | Existing split_combine_kernel@1 describes multi-pass sum conceptually. The old loop/tree path is not promoted: source lane32 contribution and in-place later-pass ownership require independent replay before native use. The main path does not invoke it. |
| `trainer.h:96–160`: offline model forward, optional logistic output perturbation, pointwise loss or external gradient, backward, optimizer | Existing map/reduction/storage bindings with ordered graph edges | Perturbation changes the training objective; RNG state must advance consistently. Optimizer waits for full parameter gradients and consumes linear parameter layout. Inference params may come from optimizer custom_weights (`:497`). |
| `trainer.h:163–251,254–357`: generated forward→loss→backward in one CTA, optimizer remains separate | `physical_storage@1`: forward context in smem or gmem; **whole training-sequence fusion kind remains a gap** | Source tries8 then4warps with shared forward context; when fewer would be needed switches to8warps with global context then halves to1. Shared bytes = backward-reduction scratch + per-sample forward-context bytes*threads. This is additive because lifetimes can overlap. Full batch granularity/warp participation is required. |
| `trainer.h:205–234`: forward context is warp-grouped, then immediate local loss and backward | operator_fusion@2 can describe supported local edges only; explicit producer/consumer chain | Forward context at shared offset after reduction scratch, or per-CTA global range. Each warp adds previous_multiple(threadIdx.x,32)*context_bytes; the comment's “bytes per warp” omits the32-sample factor used by the generated model. Loss/output gradients may remain local; gradient atomics still target initialized global buffers. |
| `cuda_graph.h:61–156`: capture a non-default stream sequence, update existing graph execution or reinstantiate, launch once capture closes | **Host launch-graph realization gap**, independent of device fusion | Nested captures defer to outer owner. Topology/pointer changes affect update success. Captured tensor/context/parameter allocations must survive launch. Reusing a graph launch does not mean its kernels share on-chip state. |
| `multi_stream.h:42–244`: entry event from parent, private child streams, exit child events back to parent | Explicit fan-out/fan-in dependency binding | Side outputs/scratch must be disjoint. This is not chunk_ready_collective_pipeline@1 (no collective chunks), nor a reused CTA barrier. A stream split is legal only when side consumers' values are ready and parent waits before reuse. |
| `src/rtc_kernel.cu:89–105,170–310`: gate JIT on FP16,target>=SM75,CUDA runtime>=11.8; NVRTC compile and retained source/header cache | Build/target binding; no measured kernel instance | Cache includes generated code and embedded/extra headers; selected options/target and parameter layout belong to the instance. Compile failure disables JIT/falls back; absence of support throws or returns capability false. Native emitted PTX/SASS for this study was not generated. |
| `src/{cpp_api,common_host,object}.cu`, matrix/memory/random headers | Adapters and common primitives bound through their callers | API wrappers construct model/loss/optimizer, expose precision and alignment, and route forward/backward. Memory arena/graph interactions require lifetime ordering; helper existence alone does not add an optimization family. Random generation, activation functions and matrix slicing are accounted by consumers above. |

The important reduction witness is FP16 length8192:16-byte grouping gives1024
vector loads, one1024-thread block,32 warp partials, then one atomic contribution
to an initialized scalar. Length8200 gives1025vectors and twoCTAs, with all unused
vector positions zero. It does not justify arbitrary scalar lengths, misaligned
segments, or reusing a nonzero output accidentally. Atomic association and
low-precision input conversion require numerical admission.

No runtime tests or measurements were performed. Registered loss variants reuse
the same organization; their gradient/domain conventions remain explicit, and
unrelated source defects are recorded without chasing them. Full training-sequence
fusion, launch-graph reuse and legacy reduction are named gaps rather than
silently covered by broader-sounding Template names.

The common RNG helper (`random.h:40–79`) gives each thread four strided output
indices and advances its private generator by4*thread_index; the host advances
its retained generator by the requested element count. Uniform/logistic transforms
share that organization. Exact generator epoch, rounded launch dimensions and
range/tail behavior are instance obligations; no distribution/independence claim
or general RNG Template is admitted from this helper. Common parallel-for helpers
choose flat/AoS/SoA index ownership and bounds predicates already used by the
pointwise/padding consumers, with no separate new pipeline.
