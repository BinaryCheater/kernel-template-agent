# Optimizer state and matrix preconditioning

Optimizer kernels are state transitions, so fusion or skipped work must preserve
which state advances at which event. The [pinned source](inputs/sources.json)
registers ten optimizer names in `src/optimizer.cu:52–91`, including the conditional
Shampoo path. Most arithmetic formulas below are workload semantics; their
implementation choices are ownership, retained precision, reduction and update
cadence, not freely interchangeable optimizer algorithms.

| Source design | Rule or explicit gap | Binding and constraints |
|---|---|---|
| `optimizers/sgd.h:44–98`, `adam.h:47–133`: one parameter/thread reads low-precision gradient, unscales it, updates FP32 master and casts execution copy | `output_tiling@2`: thread owner; `operator_fusion@2` only for admitted scale/cast edges; `physical_storage@1` for persistent moments/master/copy | FP32 master+FP16 copy is not carry_compensated_parameter_split@1's bit codec. Do not bind that unrelated rule. L2 regularization, clipping, moment update, debiasing and decay order are fixed by the requested optimizer. |
| `adam.h:82–95,111–132`: non-matrix parameters may skip zero incoming gradients before regularization; local counters advance only for active owners | [selective_parameter_state_update@1](delta/selective_parameter_state_update@1/template.yaml) | Host step and per-parameter step differ. Inactivity preserves moments as well as weight. Matrix/non-matrix enable flags and non-matrix learning-rate multiplier are explicit groups. Source also exposes AdaBound effective-rate clamp, weight and gradient clips; these remain arithmetic-contract choices. |
| `novograd.h:46–89,124–166`: per-layer sum of squared gradients, scalar second-moment update, then per-parameter first moment/update | `reduction_partition@1`, `physical_storage@1`; exact reduction implementation bound in runtime trace | Loss scale squared applies to norm statistic. First step uses beta0 to initialize moments from exact observed statistic. Every layer's scalar must be complete before its parameter update. Shared layer norm differs from elementwise Adam second moments. |
| `ema.h:45–80,107` onward: nested update then debiased parameter EMA, FP32 accumulator optional | Explicit filter-state transition; `physical_storage@1` precision choice | Nested custom weights can be the input. Keeping only T-valued EMA introduces a cast each epoch; full-precision mode retains FP32 state. Global nested step is the filter clock, not a sparse parameter's active count. |
| `average.h:45–105`: finite ring of historical weights, running average adds(new-old)/sample_count | `physical_storage@1` ring occupancy; explicit sliding-window state formula | Ring slot = nested_step modulo sample_count, initialized zero. In warmup this is a fixed-denominator zero-filled window, not an average over samples seen so far. It is not multi_buffer_rotation@2's async transfer protocol. |
| `lookahead.h:45–101`, `exponential_decay.h:46–76`: wrapper schedule around nested transition | Explicit epoch/order binding; no new synchronization rule | Lookahead initializes slow state on first call and blends before nested step when current nested step is divisible by interval. Decay modifies nested learning rate before its transition. Moving wrappers or substituting a different clock changes the update program. |
| `batched.h:46–96`: accumulate gradients in FP32 divided by batch multiplier, cast and call nested optimizer only on completed group | `reduction_partition@1` partial group; `physical_storage@1` persistent pool; explicit update-epoch gap | Multiplier4 resets pool on calls0,4,..., updates nested on completed calls4,8,... . Loss scale must match the completed group; changing weights before the group completes would change the batch being represented. Skipping three nested transitions is part of the requested effective-batch semantics, not equivalence to four Adam updates. |
| `composite.h:45–111`: disjoint parameter intervals route to nested optimizers, optional custom inference vector assembled by copies | `output_tiling@2` custom ownership; `physical_storage@1` | Segment boundaries must match layer metadata and cover intended parameters. Different nested clocks and custom weights are preserved, not flattened into one global optimizer state. |
| `shampoo.h:167–284,370–424,645–850`: batch equal matrix shapes, momentum and left/right Gram statistics, use lagged roots for update | Existing contract/output-tiling and physical storage bindings; **two-sided preconditioner algorithm and root-refresh schedule remain explicit gaps** | Matrix weights use L_root * momentum * R_root; trailing parameters use Adam-style momentum update. Three streams/batch handle update,L,R with entry readiness and final joins. First step initializes statistics/roots; matrix update starts later. Old roots are read before a scheduled refresh. |
| `shampoo.h:435–632`: coupled inverse fourth-root iteration in FP32/FP64 using batched GEMMs | **Iterative inverse-root Template/convergence gap**, distinct from blocked_unit_lower_inverse@1 | Scale from squared Frobenius norm of A^4, initialize M and X, iterate T=(5I-M)/4, M←T^4 M, X←X T. Temporary copies avoid unsupported in-place GEMMs. Stability/SPD/ridge and stop-error semantics are not supplied by generic GEMM Templates. |
| `shampoo.h:830–862`: stagger root refreshes and symmetrize+ridge before inverse root | Explicit schedule/state binding | Source refresh interval10 before step100,200 thereafter, divided across matrix batches; first epoch refreshes all. Positive interval and valid batch counts are required. This is a source heuristic, not a measured preferred cadence. Vendor GEMM kernels and convergence of the chosen thresholds are unverified. |

Reference state substitution: scalar SGD master1, gradient2, loss_scale2,
learning_rate.1, no L2 yields master.9 and its cast execution copy. Sparse Adam
with beta1=beta2=.5 and first gradient2 produces moments(1,2), clock1; an inactive
zero-gradient call preserves all three, whereas dense Adam would give(.5,1),2.
For a completed batch of four equal-scale gradients(1,3,5,7), Batched passes4 to
the nested update, not four separate updates.

For Shampoo, the scalar positive model makes its algebra visible: at fixed scalar
A=a, M=a*X^4 and T=(5-M)/4, the recurrence preserves M'=a*(X*T)^4. The fixed point
M=1 implies X=a^(-1/4), but this identity alone does not prove convergence,
finite-precision behavior or the source stopping test. Thus the inverse-root
choice is recorded as an extraction gap rather than represented by an unrelated
triangular inverse rule. No new benchmark or broad optimizer test is needed for
these source dispositions; native execution and training-quality claims are absent.
