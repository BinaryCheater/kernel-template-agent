# Sparse support classification and no-work protocol

Sources below start at flash_attn/cute/ in the [pinned tree](inputs/sources.json).
Separating tile support, per-element validity and protocol participation prevents
a sparse optimization from either dropping nonzero mass or unbalancing a barrier.

| Source choice | Rule/field or gap | Binding / premise |
|---|---|---|
| compute_block_sparsity.py:121–333: classify each logical tile as empty/full/partial | General classified-list traversal remains an extraction gap; dense masked_tile_pruning@1 only supplies its narrower boundary case | Full evaluator OR-reduces both has-valid and has-invalid over the logical in-bounds tile. Partial ids enter mask list, full ids enter full list, empty ids are omitted. Counts bound valid list prefixes. |
| compute_block_sparsity.py:203–250; block_sparsity.py:698–701: optional four corners plus center | Explicit mask-family certificate gap | fast_sampling only marks the callback as safe; it does not prove arbitrary masks can be classified from five points. A tile with one isolated allowed interior point away from all five samples would be falsely empty. This optimization requires a callback-family property or a full evaluation. |
| block_sparse_utils.py:25–95,150–236,620–833: fixed/varlen lists, full versus partial streams and split ranges | Concrete sparse support/index binding; indexed_gather@1 supplies operand address indirection only | Lists must be in range, disjoint and complete for intended support; full tiles may omit callback masking only if certified full. Producer and all consumers use the same physical block order/subtile expansion. |
| block_sparse_utils.py:833–953,1097–1146: empty SM100 forward tile retains control handshake, omits P/MMA channels | [empty_work_pipeline_projection@1](delta/empty_work_pipeline_projection@1/template.yaml), empty_protocol control_only | Softmax emits synthetic no-work stats event, correction acknowledges, required epilogue runs. P-prefix/tail readiness is not signaled because no P exists. Prior acknowledgment drain is retained across empty/nonempty transitions. |
| block_sparse_utils.py:868–932: empty correction seeds placeholder stats and zero output scale | Explicit tagged-empty state encoding | Helper uses row_sum1,row_max-inf and scale0, not an ordinary positive-mass state or raw L0 state. The empty_support_normalization@1 semantic convention needs this encoding mapped explicitly; do not pass the placeholder directly to its L==0 detection. Finite sink handling follows the separate owned helper path. |
| block_sparse_utils.py:136–147 and backward load helpers | empty_work_pipeline_projection@1, empty_protocol skip_data_epoch | Empty KV work advances neither data producer nor consumer state; required dK/dV zero stores can still have an owner. This differs from forward's per-work control-only handshake. |
| topk_gather_kv.py:137–212: caller-provided token-index stream and validity bits | indexed_gather@1 row/index_prefetch registers; per-slot handoff for bitmask | Tokens0<=index<sequence_limit are valid; invalid payload transfers are suppressed/zeroed by the selected backend binding and scores must also be masked. The indexed stream's semantics must specify duplicate entries; deduplicating would change multiplicity unless the workload admits it. |
| topk_gather_kv.py:165–195: lane bit plus warp reduction builds one32-bit validity mask | Explicit lane→score bit mapping; cross_lane_reduction@1 for OR/add on disjoint bits | A bit is owned by one lane, so OR equals add without carries. Register interleaving used for wide copies is not the same order as the non-interleaved bitmask. Softmax must use the corresponding logical-key order. |
| topk_gather_kv.py disable_bitmask branch and MLA callers | Caller-supplied bounds premise | Turning off validity work is legal only when all represented indices are admitted by the intended support contract. A bounds-valid address does not by itself establish causal validity or head padding correctness. |

The pipeline witness is two empty work items followed by one nonempty item. A
per-work control channel may need three no-work/work publications and three
acknowledgments, while a P-only data channel participates only for the third.
Removing the first two control publications while keeping its consumer waits
deadlocks; inventing two P-ready events lets MMA consume unwritten payload.
This argument concerns event ownership, not native barrier phase correctness.

For sparse classification, a logical16×16 tile whose only allowed point is(3,5)
has no allowed corner or center sample. The full evaluator classifies it partial;
an unqualified five-point shortcut would omit nonzero support. No general
completeness claim or extra mask audit follows from the decorator.

No new script is needed for these separating witnesses. Native phase/tail,
bit-order, asynchronous zero-fill and runtime support contracts remain evidence
obligations; unrelated implementation issues are not chased.
