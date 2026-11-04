# Blackwell row reduction and RMSNorm

When each CTA owns only part of a row but needs the row's complete scalar before
writing its slice, consider replicating sender-indexed scalar partials and
combining them locally at every reader warp. This differs from assigning final
output shards: every CTA receives the full reduction result. The new
[replicated_partial_reduce@1](delta/replicated_partial_reduce@1/template.yaml)
records that ownership, byte-completion and allocation-lifetime contract.

Scope is both `reduce/reduce.py` and `rmsnorm/rmsnorm.py` under
`examples/python/CuTeDSL/cute/blackwell/kernel/`, plus their selected CuTe primitives.
[Inventory](inputs/row-reduction-variants.json) retains every function/method span.
Paths below refer to the pinned CUTLASS tree in [the source manifest](inputs/sources.json).
`reduce.py` declares helpers, not a standalone kernel; RMSNorm imports its row_reduce
and selects its actual warp/block/cluster routes.

## Concrete witness and decision ownership

The [source-derived witness](inputs/row-reduction-binding.json) uses FP16, M2,
N32768, weight present, eps1e-6 and symbolic SM100 configuration. It selects two
CTAs per row,128 threads per CTA,128 threads/four warps per row, one row per block,
N16384 per tile and128-bit/eight-element vectors. Declared shared storage is32808
bytes:32768 input bytes,32 partial-table bytes and one8-byte barrier. These are
configuration values, not measured allocation or timings.

| Source decision | Concrete choice and invariant | Rule or explicit boundary |
|---|---|---|
| RMS config:158–239; host:306–352 | Cluster count depends on dtype width, N and symbolic architecture. FP16/BF16 thresholds16K/32K/64K/128K choose1/2/4/8/16; FP32 thresholds double. Below SM90 choose1. Thread menu8/16/32/64/128/256; total128 or256. | Source dispatch heuristic, not a measured optimum. Actual support/residency, cluster16 launch policy and positive N/M remain required. get_sm_version returns80 if CUDA is absent; scalar checks pass explicit architecture values and make no hardware claim. |
| TV layout:242–257 | Thread(t,row), value(e,b) maps to row+R*(t*V+e+b*V*T). Each tile coordinate has one owner; whole-warp row ownership holds when T>=32. | vectorized_access@1 vector_bytes16/thread_arrangement=strided_loop/threads128 matches the R1 witness. Its contiguous-extent and base-alignment guards still apply. General multirow TV mapping is retained source glue, not a free mapping chosen by that enum. |
| Bounds:128–140,460–478 | N predicate is built from the vector's first coordinate; row predicate guards loads/stores, while collective reduction remains converged. | Coarse vector predicates need whole-vector boundary legality. N4097 gets only4096 scheduled columns; N32769 gets32768. At N32767, selected vector32760..32767 extends past the logical end. Predication cannot recover unscheduled columns; native within-vector tail safety is not established. |
| Input staging:394–399,480–496 | cp.async global→shared, commit/wait group0; weight loads overlap that wait. The same shared input is reread after the reduction. | async_transfer@1 supports cp_async readiness intent; physical_storage@1 level=smem/dedicated preserves the chunk through both reads. No claimed connection to a TMA controller or destination-stage rotation. Exact copy ownership and predicate-false zero fill remain implementation obligations. |
| `reduce.py:503–532`; wrapper `cute/arch/nvvm_wrappers.py:792–824` | Local FP32 register reduction, then xor butterfly with width min(T,32). T8/16 divides a warp into independent row subgroups; T32 uses a full warp. | Existing cross_lane_reduction@1 lanes8/16/32,result_to=all_lanes,combine_op=add. Complete/converged groups and admitted FP32 regrouping are required. Helper also selects max with fmax and -inf identity under its own NaN policy. |
| Local block helper:259–317 | Lane0 of each contributing warp writes [row,warp]; CTA barrier; every row warp rereads that row's partials into lanes and does a full warp reduction. | New replicated_partial_reduce@1 peers1,warps_per_row>1. smem_partial_combine@1's serial/tree/CUB implementation and its cost do not supply this repeated per-reader-warp table fold. Subwarp rows never select this path in the automatic config. |
| Cluster helper:326–431 | Sender warp lanes0..C-1 each write the same scalar to a different destination. Table slot is local_warp+W*sender_cta. Every destination registers4*R*W*C bytes, waits phase0, then every row warp folds its full table. | New rule peers2/W4 for witness,32 bytes per destination. Old scatter_reduce_output_shards@1 partitions final output fragments and cannot supply a full replicated scalar. Uniform whole-warp input, injective slots and exact destination/generation incidence are required. |
| Remote primitives:151–232; RMS init:416–428 and post-reduce:513–520 | Both data pointer and completion barrier are mapped to the destination CTA; fresh barrier init count1, collective cluster initialization, then a collective barrier after reduction before local output/exit. | The source uses one phase0 generation; repeated helper calls require a new generation/retirement protocol. byte_arrival_completion_join@1 requires>=1 independent copy arrival, so its fields do not represent this bytes-only store completion. No fake arrival is bound. Native proxy semantics/progress remain open. |
| RMS math:499–539 | FP32 square sum divided by original N, eps added, fast rsqrt, reread shared input, multiply rstd and optional FP32 weight, then cast to input dtype. | Workload formula and precision are fixed source choices. approximate_math_rewrite@1 rewrite=rsqrt_approx/fused_fma=false requires admitted error; operator_fusion@1 can keep the weighted consumer on chip. Neither rule proves a new numeric tolerance. |
| Optional weight and caller:306–352,561–637 | Weight may be absent, or read along N with row broadcast. Cache key is dtype/N/has_weight; M and eps remain runtime arguments. Inputs/outputs use contiguous(M,N) layouts; arbitrary input strides are not accepted by this pointer interface. | No-weight is a source variant of the same reduction. Weight-presence/pointer agreement is required; direct has_weight=true with w_ptr=None skips it. Host reference promotes to FP32 and casts back; tolerances are explicit test options, not evidence that every shape is correct. |

The source carries padded vector slots in sX and reads them into the square sum
without an explicit zeroing step. For N20000, configuration schedules20480 columns.
Correctness therefore needs neutral predicate-false copy behavior or another
verified fill implementation. The CopyG2SOp wrapper selects an IR async-copy atom;
it does not expose enough lowering here to claim zero fill. Invalid M-tail rows
also skip the input copy but still read shared storage and join collectives; their
values must stay unobserved and confined to their row. The selected M2/N32768
witness avoids both kinds of padding. Source shape acceptance alone does not
establish those general tails.

## Communication contract and composition

Each source CTA has R*W contributing warps. Replicating their scalars to C
destinations generates R*W*C stores per sender and R*W*C incoming scalars at each
destination. Four bytes per scalar gives the registered count. Sender slots are
unique, and lane-strided reads partition0..W*C-1 before the final butterfly.
The [NVIDIA PTX st.async specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async)
associates completed stores with byte completion at the supplied peer barrier;
its cluster lifetime requirements make collective allocation retirement necessary.
[Reference provenance](inputs/row-reduction-isa-reference.json) records this narrow
ISA use. Instruction availability does not prove this source's native execution.

The [partial chain](delta/row-reduction-chain.yaml) starts from defined local
square sums, reduces full-warp partials, combines the replicated table, computes
the reciprocal root, then applies weighted normalization locally. Input staging,
TV mapping, vector coverage, numeric admission and final store implementation are
explicit external edges. The source retains the input chunk in shared memory;
it does not use the older vLLM reference's global-memory reread implementation.

## Retained evidence and disposition

[Scalar checks](checks/row-reduction-result.json) execute the actual config methods
for108 dtype/architecture/width cases. Eight TV ownership domains cover69632
identifiable coordinates.38 replicated-table domains,15342 scalar deliveries and
738 reader results check add/max, local/cluster modes and lane-strided final
folds; three subgroup domains verify aligned8/16/32-lane separation. Tail witnesses
retain missing coverage, vector overhang and the unproved neutral-padding boundary.
[Old bindings](inputs/row-reduction-reuse.json) check the exact vector-alignment,
warp-combine and fast-math applicability fields; the vector rule rejects N32767.
New rule schemas and premise mutations pass independently of those native gaps.

Both files are source-extracted with these named boundaries. No GPU execution,
RMS output comparison, native instruction timing, Realisation, ProofSummary or
Store admission occurred. Continue the remaining CUTLASS inventory.
