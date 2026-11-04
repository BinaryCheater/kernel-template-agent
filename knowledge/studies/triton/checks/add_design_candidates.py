"""Write extraction candidates, without Store admission."""

from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1] / "delta"


def add(name, obj, variables, pre, program, backward, obligations, scope, summary, note, refines):
    p = root / f"{name}@1"
    p.mkdir(exist_ok=True)
    t = dict(
        name=name,
        version=1,
        level_from=2,
        level_to=3,
        change="vertical",
        refines=refines,
        produces={
            "sync": ["sync", "work_assign"],
            "access": ["access", "store"],
            "transfer": ["transfer", "store", "sync"],
            "work_assign": ["work_assign", "map", "store", "sync"],
            "store": ["store", "transfer", "sync"],
            "contract": ["contract", "transfer"],
            "reduce": ["reduce", "store", "sync"],
        }.get(refines, ["map", "store"]),
        object=dict(type=name, description=obj),
        variables=variables,
        constraints=[],
        preconditions=[
            dict(id=f"pre{i}", expr=e, meaning=m, cls="semantic") for i, (e, m) in enumerate(pre)
        ],
        program=dict(description=program),
        backward_map=backward,
        obligations=[dict(name=n, statement=s, method=m) for n, s, m in obligations],
        effects=dict(transactions=[], storage=[], overlap=[]),
        scope=dict(predicate=" and ".join("(" + e + ")" for e, m in pre), text=scope),
        requires=[],
        probed_on=[],
        relations=[],
        signature=dict(
            algorithm=obj,
            storage="Binding-specific counters, partials or output ranges.",
            boundary="Tile prefixes and logical membership.",
            resources=[],
        ),
        numeric_risk=(
            "Integer counts must not overflow; ordering/rounding policy belong"
            "s to the bound operation."
        ),
        summary=summary,
    )
    (p / "template.yaml").write_text(yaml.safe_dump(t, sort_keys=False))
    (p / "proof.yaml").write_text(
        yaml.safe_dump(
            dict(
                template=f"{name}@1",
                version=1,
                relation=backward,
                rank="Unprocessed source items or unmerged partials.",
                assumes=[
                    dict(
                        name="contract",
                        statement=(
                            "The stated partition, comparator/count and publication premises hold."
                        ),
                    )
                ],
                guarantees=[],
                models=[],
                note=note,
            ),
            sort_keys=False,
        )
    )


def choice(n, domain, meaning):
    return dict(
        name=n, kind="structure", binding="compile", role="software", domain=domain, meaning=meaning
    )


def num(n, meaning):
    return dict(
        name=n,
        kind="numeric",
        binding="compile",
        role="software",
        domain=dict(lo=1, hi=65536),
        meaning=meaning,
    )


add(
    "striped_locked_partial_accumulation",
    (
        "Independent row contributors are hashed to locked vector partial "
        "accumulators, then combined in a separate pass."
    ),
    [num("groups", "Number of lock-protected vector partial slots.")],
    [
        (
            "p.contributors_partition_complete and p.row_to_slot_is_mod_groups",
            "Every row contribution enters one designated partial slot.",
        ),
        (
            ("p.locks_and_initialized_flags_zeroed and p.lock_release_publishes_whole_vector"),
            "A zero lock protects the entire vector and its first-writer flag.",
        ),
        (
            (
                "p.final_pass_waits_all_contributors and p.read_only_actual_nonemp"
                "ty_slots and p.rounding_order_admitted"
            ),
            ("Combiner sees complete nonempty slots and admits lock-arrival accumulation order."),
        ),
    ],
    (
        "Contributor row r uses slot r%groups. Compute its partial vector."
        " Acquire that slot mutex with compare-and-swap. If initialized fl"
        "ag is zero, set it and store the partial without reading old scra"
        "tch; otherwise load the complete prior vector and add the partial"
        " under the lock. Publish all vector stores before releasing the m"
        "utex. After the contributor kernel completes, reduce only slots w"
        "ith actual contributors into final output. groups controls lock c"
        "ontention versus scratch and final-combine traffic; it does not s"
        "plit a row numerical statistic."
    ),
    (
        "Expand each locked slot into the original row partial vectors in "
        "actual acquisition order, then combine all slots once."
    ),
    [
        (
            "first_writer",
            (
                "No slot is read before a completed initializing contribution, and"
                " updates cannot lose concurrent vector elements."
            ),
            "proof",
        ),
        (
            "slot_combine",
            (
                "Each original contribution enters one slot and every nonempty com"
                "plete slot enters the final combine once."
            ),
            "test",
        ),
        (
            "progress",
            (
                "Lock holders and waiting contributors can make progress under the"
                " launch/runtime contract."
            ),
            "test",
        ),
    ],
    (
        "Vector partial accumulation from independent rows with whole-vect"
        "or mutexes and fresh flags per invocation. Floating-point order d"
        "epends on lock acquisition. Unused slots must be excluded or init"
        "ialized by a separate policy."
    ),
    (
        "When many rows contribute the same parameter-gradient vector, cho"
        "ose a bounded number of striped partial slots instead of one part"
        "ial per row. Hash row modulo groups, initialize on first lock acq"
        "uisition and update the entire vector while locked. Reject a fina"
        "l reduction that reads never-initialized slots or overlaps active"
        " writers."
    ),
    (
        "Source python/tutorials/05-layer-norm.py:140-220. Modulo assignme"
        "nt partitions rows; mutex serialization yields an ordered fold in"
        " each slot, and the post-kernel combine joins those folds. Unlike"
        " first_writer_atomic_combine@1, there is no fixed partial0 initia"
        "lizer and the numeric vector updates are ordinary stores under a "
        "lock."
    ),
    "reduce",
)
add(
    "early_dependent_grid_launch",
    (
        "A predecessor grid releases dependent launch scheduling before it"
        "s completion while dependents still join completion before reads."
    ),
    [],
    [
        (
            "p.pdl_supported_and_enabled_for_both_launches and p.same_dependency_chain",
            ("Both grids use supported programmatic dependent launch in the actual stream chain."),
        ),
        (
            (
                "p.every_predecessor_program_triggers_or_finishes and p.consumer_w"
                "aits_before_dependent_reads"
            ),
            "Scheduling release is not mistaken for data visibility.",
        ),
        (
            "p.prewait_work_independent_of_predecessor and p.buffer_lifetime_valid",
            "Only independent prologue work may run before the completion join.",
        ),
    ],
    (
        "Predecessor programs issue a launch-dependents trigger after thei"
        "r necessary prologue, potentially before output epilogue stores. "
        "Once all programs have triggered or finished, the runtime may sch"
        "edule the dependent grid if resources permit. A dependent may com"
        "pute only independent indices/resources before grid-dependency wa"
        "it; it must execute the completion wait before reading any predec"
        "essor-produced value. Triggering alone provides no memory publica"
        "tion. Keep ordinary lifetime/stream completion before final exter"
        "nal observation."
    ),
    (
        "Move independent dependent-grid prologue work after predecessor c"
        "ompletion and erase the early scheduling trigger; all dependent d"
        "ata reads still see the completed predecessor result."
    ),
    [
        (
            "dependency_visibility",
            "Every predecessor-dependent read follows the whole-grid completion join.",
            "proof",
        ),
        (
            "release_progress",
            (
                "All required predecessor programs release scheduling or finish wi"
                "thout circular dependency."
            ),
            "test",
        ),
    ],
    (
        "NVIDIA supported PDL launch chains, CUDA SM90+ source gate. This "
        "is early scheduling with whole-grid completion, not per-tile read"
        "y publication or a transfer completion barrier."
    ),
    (
        "When launch overhead or independent dependent-grid prologue can o"
        "verlap a predecessor epilogue, release scheduling early and retai"
        "n a completion wait before dependent loads. In vector add, comput"
        "ing offsets can run early; reading x/y written by the previous gr"
        "id cannot. A launch trigger is never permission to consume unfini"
        "shed output."
    ),
    (
        "Source python/tutorials/11-programmatic-dependent-launch.py:28-67"
        ". Any execution with early independent work can be serialized acr"
        "oss the completion wait because that work touches no predecessor-"
        "produced state; the wait restores the original data dependency. N"
        "ative GDC launch/wait and resource progress are explicit obligati"
        "ons."
    ),
    "sync",
)
add(
    "blocked_thread_register_layout",
    (
        "Hierarchical register/lane/warp digits map a rectangular logical "
        "tile to physical thread registers."
    ),
    [
        num("S0", "Rows per thread register subtile."),
        num("S1", "Columns per thread register subtile."),
        num("T0", "Lane rows per warp."),
        num("T1", "Lane columns per warp."),
        num("W0", "Warp rows per CTA."),
        num("W1", "Warp columns per CTA."),
        choice(
            "order",
            ["row_major", "column_major"],
            "Fastest-varying dimension when decoding each hierarchy digit.",
        ),
    ],
    [
        (
            "T0 * T1 == 32 and p.factor_sizes_are_powers_of_two",
            "NVIDIA warp has 32 lanes with compatible block factors.",
        ),
        (
            "p.tensor_shape_multiple_of_block_shape and p.logical_elements_assigned_once",
            (
                "This version admits full block shapes or whole repetitions withou"
                "t physical broadcast replicas."
            ),
        ),
        (
            "p.register_capacity_and_launch_legal",
            "Register subtiles and warps fit the selected launch.",
        ),
    ],
    (
        "Define block extents B0=S0*T0*W0,B1=S1*T1*W1. Decode lane id into"
        " (l0,l1) with extents (T0,T1), warp id into (w0,w1) with extents("
        "W0,W1), and within-thread register id into (r0,r1) with extents(S"
        "0,S1), using the selected fastest dimension. Coordinate i=(w0*T0+"
        "l0)*S0+r0+q0*B0, j=(w1*T1+l1)*S1+r1+q1*B1 for outer repetition di"
        "gits(q0,q1). Higher shape repetitions add registers. Source tenso"
        "rs smaller than block shape physically broadcast replicas; that r"
        "egime requires an explicit replica ownership policy and is outsid"
        "e this first version."
    ),
    (
        "Erase physical thread/register hierarchy to recover the same logi"
        "cal tensor coordinates and values."
    ),
    [
        (
            "hierarchy_bijection",
            ("Every valid logical coordinate has one register/lane/warp/repetition decomposition."),
            "proof",
        ),
        (
            "consumer_layout",
            (
                "Loads, stores, collective and MMA consumers interpret the chosen "
                "mapping consistently."
            ),
            "test",
        ),
    ],
    (
        "Rank-two whole-block distributed tensors on 32-lane NVIDIA warps,"
        " power-of-two hierarchy factors and whole repetitions. Smaller-sh"
        "ape broadcast replicas are explicitly excluded; consumer-specific"
        " MMA layouts require a realization binding."
    ),
    (
        "When choosing how a tensor tile spreads over registers, lanes and"
        " warps, vary the hierarchy factors and their digit order separate"
        "ly. S=[2,4],T=[16,2],W=[2,2] gives a 64x16 block and eight regist"
        "ers per thread. A 128x128 tensor repeats that block 2x8 times, gi"
        "ving 128 registers per thread before liveness effects. Do not tre"
        "at smaller replicated tensors as unique-writer maps."
    ),
    (
        "Source python/tutorials/gluon/02-layouts.py:30-120,:160-175. Mixe"
        "d-radix division uniquely inverts the three hierarchy levels and "
        "outer repetitions. Native duplicate-replica suppression and speci"
        "al MMA layout rules remain outside this version; unique-coordinat"
        "e replay is recorded in checks/design_witnesses.py."
    ),
    "access",
)
add(
    "load_compute_store_role_pipeline",
    ("Three warp roles communicate through independent input and output generation rings."),
    [
        num("load_slots", "Input ring slots shared by load and compute roles."),
        num("store_slots", "Output ring slots shared by compute and store roles."),
        num("compute_warps", "Default compute-role warp count; source binds four or eight."),
    ],
    [
        (
            "p.role_warps_and_register_budgets_fit and p.all_roles_same_trip_count",
            (
                "Role resources fit rounded NVIDIA warpgroup allocation and every "
                "role processes the same generations."
            ),
        ),
        (
            (
                "p.barriers_initialized_and_input_output_phase_maps_match and p.pr"
                "oxy_fences_between_shared_and_async_accesses"
            ),
            "Ready/empty barriers and asynchronous proxy visibility match both rings.",
        ),
        (
            ("p.store_source_reads_retire_before_empty_publication and p.final_stores_drained"),
            "Output storage is not released before pending store reads finish.",
        ),
    ],
    (
        "Allocate input ring with full/empty barriers per load_slots and o"
        "utput ring with separate full/empty barriers per store_slots. Loa"
        "d role waits empty at slot i%load_slots and phase opposite floor("
        "i/load_slots)%2, registers expected input bytes and issues TMA lo"
        "ads. Compute waits input full at that phase, captures input in re"
        "gisters, fences shared/async proxy access and signals input empty"
        ". Compute forms output, waits output empty at its independently d"
        "erived phase, stores/fences it and signals output full. Store rol"
        "e waits output full, issues TMA stores, permits store_slots-1 gro"
        "ups in flight and signals the oldest output slot empty only after"
        " its store source reads complete. Drain final stores. Source uses"
        " one load warp and one store warp with 24-register worker budgets"
        ", plus default compute warps; actual allocation rounds the worker"
        " roles into warpgroups."
    ),
    (
        "Serialize each generation load, computation and output store whil"
        "e erasing the two physical rings and three actor roles."
    ),
    [
        (
            "generation_identity",
            (
                "Every role reads the same generation and each ring is released on"
                "ly after its last reader."
            ),
            "proof",
        ),
        (
            "proxy_and_store_completion",
            (
                "Native ready/empty publication includes proxy fences and complete"
                "d store-source reads."
            ),
            "test",
        ),
        (
            "role_progress",
            "Role allocation and waits have no circular missing-generation dependency.",
            "test",
        ),
    ],
    (
        "Finite equal-trip three-role CTA pipelines with independent input"
        "/output ring lengths and native TMA/mbarrier support. This expose"
        "s the separate store-role ring/resources absent from warp_special"
        "ization@1 fields."
    ),
    (
        "When compute should overlap both input movement and output stores"
        ", use separate load, compute and store roles with independently s"
        "ized rings. A two-slot input ring and three-slot output ring need"
        " different modulo/phase counters. A ready barrier does not replac"
        "e the async proxy fence, and output empty may be published only a"
        "fter TMA finishes reading that slot."
    ),
    (
        "Source python/tutorials/gluon/08-warp-specialization.py:96-123,:1"
        "24-232,:234-307. Per ring, ready(i) precedes read(i), empty(i) fo"
        "llows last read(i), and write(i+slots) waits for that empty phase"
        ". Input capture lets computation outlive input slot lifetime; out"
        "put TMA source retirement is independent of input retirement."
    ),
    "work_assign",
)
add(
    "indexed_descriptor_row_transfer",
    (
        "One descriptor transfer gathers or scatters a list of independent"
        "ly indexed rows with one shared contiguous column interval."
    ),
    [
        choice(
            "direction",
            ["gather", "scatter"],
            (
                "Read indexed global rows into a dense shared tile or write a dens"
                "e shared tile to indexed global rows."
            ),
        ),
        num("rows", "Number of row offsets, at least eight in the source."),
        num("cols", "Transferred contiguous columns per row."),
    ],
    [
        (
            (
                "rows >= 8 and p.descriptor_rank_two_and_block_rows_one and p.row_"
                "offset_layout_valid"
            ),
            (
                "Blackwell source descriptor is [1,cols], with replicated lane bas"
                "es and required register bases for offsets."
            ),
        ),
        (
            "p.column_offset_bytes % 16 == 0 and cols >= (32 / p.element_bits) * 8",
            ("Column origin is 16-byte aligned and source dtype-dependent extent is legal."),
        ),
        (
            ("direction == 'gather' or p.scatter_offsets_nonnegative_and_destinations_disjoint"),
            ("Scatter excludes negative offsets and requires conflict-free destination rows."),
        ),
        (
            ("p.descriptor_and_shared_layout_match and p.transfer_completion_before_read_or_reuse"),
            ("Descriptor layout and asynchronous completion protect the dense shared tile."),
        ),
    ],
    (
        "Form row index vector offsets[r] and a common column origin y. Ga"
        "ther maps shared[r,c] to global[offsets[r],y+c], zero filling out"
        "-of-range coordinates. Scatter maps dense shared[r,c] to that glo"
        "bal address, skipping permitted positive out-of-range coordinates"
        "; negative scatter coordinates are illegal. Convert row offsets t"
        "o the source-required replicated lane layout, allocate matching s"
        "hared layout and for gather register rows*descriptor_block_bytes "
        "on the mbarrier before waiting for completion. Scatter fences gen"
        "eric shared writes before TMA and waits source-read completion be"
        "fore reuse. Duplicated row offsets are allowed for gather but req"
        "uire an explicit separate conflict policy for scatter."
    ),
    (
        "Resolve every row index and column offset to the original logical"
        " global coordinate; dense result rows retain their occurrence ide"
        "ntity."
    ),
    [
        (
            "index_identity",
            (
                "Each dense row occurrence uses exactly its requested global row a"
                "nd common column interval."
            ),
            "proof",
        ),
        (
            "layout_and_bounds",
            (
                "Native offset register/lane layout, alignment and asymmetric gath"
                "er/scatter bounds are obeyed."
            ),
            "test",
        ),
        (
            "completion",
            (
                "Payload visibility and source/destination lifetimes match the asy"
                "nchronous direction."
            ),
            "test",
        ),
    ],
    (
        "Blackwell NVIDIA TMA gather/scatter, two-dimensional descriptor w"
        "ith block [1,cols], rows>=8, cols >= (32/element_bits)*8 and 16-b"
        "yte aligned column origin. Gather accepts negative offsets as zer"
        "o-fill; scatter does not. Not an arbitrary paged tensor transfer."
    ),
    (
        "When rows are irregular but each row needs the same contiguous co"
        "lumn window, use an indexed descriptor transfer instead of scalar"
        " gather loads. Choose direction, row count and column width subje"
        "ct to hardware layout/alignment. For FP16, cols must be at least1"
        "6 and column origin a multiple of8 elements. Preserve repeated ga"
        "ther occurrences; reject overlapping scatter destinations without"
        " a separate contract."
    ),
    (
        "Source python/tutorials/gluon/09-tma-gather-scatter.py:190-238,:2"
        "45-277,:355 onward. The coordinate equation is direct; the source"
        " explicitly distinguishes zero-filled gather from negative-coordi"
        "nate illegal scatter. Row-layout and native completion remain dev"
        "ice obligations."
    ),
    "transfer",
)
add(
    "spare_operand_slots_epilogue_alias",
    (
        "Output staging borrows one or two free slots in an operand ring a"
        "fter the previous MMA retires."
    ),
    [
        num("S", "Input prefetch ring slots; source uses at least three."),
        choice(
            "output_slices",
            ["one", "two"],
            "One B-sized output tile or two N-half slices sharing the spare slots.",
        ),
    ],
    [
        (
            "S >= 3 and p.B_ring_slots == S + 1 and p.next_tile_live_B_slots == S - 1",
            "B ring has exactly two spare slots after the next prologue.",
        ),
        (
            (
                "p.previous_mma_reads_retired and p.next_tile_prefetch_occupies_pr"
                "eceding_sequence_slots"
            ),
            ("Old operand reads are complete and next inputs occupy the preceding S-1 slots."),
        ),
        (
            (
                "((output_slices == 'one' and p.BM == p.BK) or (output_slices == '"
                "two' and p.BM == 2 * p.BK and p.BN % 2 == 0)) and p.output_slice_"
                "bytes_fit_B_slot and p.output_slice_layout_legal and p.next_load_"
                "waits_output_source_reads"
            ),
            (
                "Each slice fits its borrowed slot and no following load overwrite"
                "s a pending store source."
            ),
        ),
    ],
    (
        "After draining old MMA reads and issuing S-1 next-tile input gene"
        "rations, let producer be the next input sequence. B ring has S+1 "
        "slots; live next-tile operands occupy producer-(S-1) through prod"
        "ucer-1 modulo S+1. Borrow producer modulo S+1 and optionally prod"
        "ucer+1 modulo S+1 for finalized output. One-slice mode uses BM=BK"
        "; two-slice mode uses BM=2*BK and splits output along N into two "
        "byte-equivalent B-slot shapes. Fence shared output writes and iss"
        "ue asynchronous stores. The next compute may consume a ready old-"
        "prefetched input, but must wait output source reads before the ne"
        "xt destructive TMA load. Producer and consumer sequence counters "
        "continue across tile boundaries."
    ),
    (
        "Give output its own temporary buffer and retain the same input ge"
        "nerations; disjoint live slots and completion order preserve all "
        "input/output values."
    ),
    [
        (
            "live_slot_disjointness",
            (
                "The two future sequence slots do not overlap the S-1 live precedi"
                "ng B slots modulo S+1."
            ),
            "proof",
        ),
        (
            "output_coordinates",
            (
                "Split slices preserve each original output coordinate and fit bor"
                "rowed allocation/layout."
            ),
            "test",
        ),
        ("retirement", "Previous MMA and subsequent TMA/store events obey alias lifetime.", "test"),
    ],
    (
        "The source cross-tile Gluon matmul pipeline with B ring S+1 and n"
        "ext prologue S-1, retired old MMA reads, output BM=BK or BM=2*BK."
        " Not an arbitrary ring alias independent of schedule."
    ),
    (
        "When an operand ring has two spare slots after next-tile prefetch"
        ", borrow them for output staging instead of a separate C allocati"
        "on. For S=4, three next B tiles occupy a five-slot ring and the n"
        "ext two slots are free; a 128x256 output can split into two 128x1"
        "28 slices fitting 64x256 B slots. Reject a different prefetch cou"
        "nt or refill before output stores finish reading."
    ),
    (
        "Source python/tutorials/gluon/07-persistence.py:627-724. In a cyc"
        "le of S+1 slots, the S-1 preceding residues and two next residues"
        " partition all slots. MMA wait retires old reads; store_wait befo"
        "re the next load retires output source reads. Byte equality for B"
        "M=2*BK follows (BM*BN/2)=BK*BN. This resolves the earlier named a"
        "lias gap without claiming arbitrary async scheduling."
    ),
    "store",
)
add(
    "ordered_async_engine_chain",
    (
        "Operations on one ordered asynchronous engine reuse intermediate "
        "storage without an intervening SM-side completion wait."
    ),
    [],
    [
        (
            ("p.same_ordered_engine_and_scope and p.engine_operations_preserve_issue_order"),
            (
                "All chained operations have the same engine/group scope and requi"
                "red engine execution order."
            ),
        ),
        (
            ("p.shared_inputs_proxy_published and p.external_accesses_ordered_at_chain_boundaries"),
            ("SM generic writes/reads and other engines synchronize at the chain boundaries."),
        ),
        (
            (
                "p.intermediate_storage_live_through_chain and p.commit_wait_befor"
                "e_external_result_use"
            ),
            ("Storage remains allocated and external consumers wait actual committed completion."),
        ),
    ],
    (
        "Issue shared-to-tensor copy, tensor-core MMA consuming that tenso"
        "r data, and optionally a later tensor copy overwriting the same i"
        "ntermediate. The common engine executes them in issue order, so t"
        "he later overwrite cannot precede the earlier MMA read. Track com"
        "pletion for the whole sequence with one engine commit to a barrie"
        "r and wait only before SM-side consumption or incompatible extern"
        "al access. Generic shared writes before the engine copy require p"
        "roxy fencing; an mbarrier wait after completion may supply the so"
        "urce retirement boundary before generic reuse."
    ),
    (
        "Insert a completion wait after each engine operation; ordered eng"
        "ine semantics and final wait make the same values visible at all "
        "external observations."
    ),
    [
        (
            "engine_dependency",
            (
                "Every chained consumer executes after its engine producer and bef"
                "ore a later destructive engine write."
            ),
            "proof",
        ),
        (
            "external_visibility",
            (
                "Non-engine accesses observe only completed/proxy-published values"
                " at chain boundaries."
            ),
            "test",
        ),
    ],
    (
        "Blackwell tcgen05 copy/MMA/commit sequence with matching engine e"
        "xecution scope and legal shared/tensor layouts. Does not order ar"
        "bitrary TMA, SM arithmetic or other CTAs merely because the opera"
        "tions are asynchronous."
    ),
    (
        "When a tensor-core engine can consume a prior engine copy in issu"
        "e order, chain copy→MMA→next copy and wait once at the external o"
        "bservation boundary. This can reuse tensor operand storage withou"
        "t an SM wait between operations. Preserve proxy fences and final "
        "commit/wait; a TMA transfer from another engine needs its own rea"
        "diness join."
    ),
    (
        "Source python/tutorials/gluon/10-tcgen05-copy.py:5-74,:117-142. T"
        "he source explicitly states shared tcgen05 execution order for co"
        "py/MMA/commit. Serial engine order gives producer-before-read-bef"
        "ore-overwrite; final completion covers the chain. Native scope, c"
        "ompiler issue-order preservation and layouts remain obligations."
    ),
    "sync",
)
add(
    "block_scaled_instruction_operands",
    (
        "A scaled MMA consumes quantized operands plus independently index"
        "ed scale tensors instead of software-dequantized operands."
    ),
    [
        choice(
            "format",
            ["nvfp4", "mxfp4", "mxfp8", "mixed_e4m3_e2m1"],
            "Requested operand/code format pair from the source instruction menu.",
        ),
        num(
            "scale_group_k",
            "Logical consecutive K values sharing each operand scale, source16 or32.",
        ),
    ],
    [
        (
            (
                "p.code_and_scale_formats_match_instruction and p.group_membership"
                "_and_scale_coordinates_exact"
            ),
            ("Code format, scale format and K grouping match both operands and the instruction."),
        ),
        (
            "p.packed_byte_offsets_match_logical_K and p.scale_storage_layout_legal",
            ("Sub-byte operand descriptors and TMEM scale layouts preserve logical coordinates."),
        ),
        (
            (
                "p.input_and_scale_transfers_complete_before_mma and p.accumulatio"
                "n_rounding_admitted"
            ),
            (
                "Operands/scales are ready and hardware accumulation meets the req"
                "uested numerical contract."
            ),
        ),
    ],
    (
        "Keep encoded A/B values and scale tensors separate through transf"
        "er. For logical product (m,n,k), select A scale at (m,k//scale_gr"
        "oup_k) and B scale at (n,k//scale_group_k), independent of each o"
        "perand packed-byte coordinate. Feed those scale fragments to the "
        "scaled MMA instruction with the encoded operands and the admitted"
        " accumulator. NVFP4 uses E2M1 data/E4M3 scale and groups16; MX fo"
        "rmats use E8M0 scales/groups32. Convert packed element K to descr"
        "iptor bytes using each operand own elements_per_byte; scales use "
        "logical K/group, never byte K/group. Scale TMEM has its own Tenso"
        "rMemoryScalesLayout, distinct from accumulator TMEM."
    ),
    (
        "Replace the intrinsic by its specified contraction over decoded s"
        "caled operand values at each logical (m,n,k), retaining the admit"
        "ted native rounding contract."
    ),
    [
        (
            "scale_identity",
            ("Every encoded operand product receives the corresponding logical scale pair."),
            "proof",
        ),
        (
            "packing_and_layout",
            ("Packed bytes and physical scale fragments decode to the same logical K coordinates."),
            "test",
        ),
        (
            "native_numeric",
            (
                "Scaled instruction, accumulator and output casts meet the declare"
                "d numerical recipe."
            ),
            "test",
        ),
    ],
    (
        "NVIDIA Blackwell supported scaled-MMA format menu with explicit16"
        "/32 K scale groups and matching scale layout. Does not assert tha"
        "t hardware internally forms raw partial sums and promotes them ex"
        "actly as block_scaled_partial_accumulation@1."
    ),
    (
        "When hardware accepts block-scaled operands directly, preserve co"
        "de and scale as separate instruction operands. Bind format/group "
        "size and logical scale coordinates before packing. E2M1 byte offs"
        "et k//2 and scale offset k//16 are different maps; using byte off"
        "sets for scales silently doubles scale coverage. Native scaled MM"
        "A is an alternative realization to software dequantization or exp"
        "licit raw-partial promotion."
    ),
    (
        "Source python/tutorials/10-block-scaled-matmul.py:178-242 and glu"
        "on/11-tcgen05-mma-scaled.py:124-174. Direct intrinsic specifies s"
        "caled products; source independently indexes code bytes and scale"
        " groups. The proof of hardware fragment/numerical behavior remain"
        "s native obligation. This corrects the earlier conditional raw-pa"
        "rtial reuse description."
    ),
    "contract",
)
add(
    "striped_snake_rasterization",
    (
        "A rectangular tile grid is traversed in minor-axis strips while a"
        "lternating major-axis direction."
    ),
    [
        choice("minor_axis", ["m", "n"], "Axis grouped into strips."),
        num("strip_width", "Minor-axis tile width, with a short final strip."),
    ],
    [
        (
            "p.independent_tiles and p.tile_rows >= 1 and p.tile_cols >= 1",
            "Traversal permutes independent output tiles.",
        ),
        (
            "p.linear_work_ids_cover_exact_grid",
            "Work scheduler supplies every tile ID exactly once.",
        ),
    ],
    (
        "Let minor extent be tile_rows for minor_axis=m, otherwise tile_co"
        "ls; major is the other extent. For strip b with start=b*strip_wid"
        "th, actual width w=min(strip_width,minor-start), its IDs start af"
        "ter all prior full strips. Decode local ID q as minor=start+q%w, "
        "major=q//w for even b and major_extent-1-q//w for odd b. Swap axe"
        "s back if needed. The final short strip uses its actual width in "
        "both division and modulo."
    ),
    (
        "Erase strip order and alternating direction; Euclidean strip/loca"
        "l decoding yields every original tile exactly once."
    ),
    [
        (
            "bijection",
            (
                "Full and short strips partition the minor domain and alternating "
                "major traversal is a permutation."
            ),
            "proof",
        )
    ],
    (
        "Independent rectangular tiles, positive strip width and actual-wi"
        "dth decoding at the final strip. This adds alternating direction "
        "beyond block_rasterization@1 grouped_m."
    ),
    (
        "When persistent/CLC tile traversal should remain near the previou"
        "s strip endpoint, alternate the major-axis direction between narr"
        "ow strips. Choose strip width and minor axis; preserve actual fin"
        "al width. For five minor tiles,width2 the final one-tile strip us"
        "es width1, not2."
    ),
    (
        "Source python/tutorials/gluon/14-multicta.py:668-706. Strip inter"
        "vals are disjoint; q%w and q//w cover each strip rectangle, and r"
        "eversing major coordinates is bijective. See unique-coordinate ch"
        "ecks in checks/design_witnesses.py."
    ),
    "work_assign",
)
add(
    "descriptor_union_completion_count",
    (
        "Expected MMA completion arrivals are derived from the union of de"
        "scriptor-driven issuing CTA sets."
    ),
    [
        choice(
            "instruction_group",
            ["one_cta", "two_ctas"],
            "Two-CTA atoms elect only even CTA ranks in the source.",
        )
    ],
    [
        (
            ("p.descriptors_share_cluster_bit_count and p.cga_bases_match_payload_identity"),
            (
                "Descriptor CGA bases describe the same physical cluster and actua"
                "l operand replication."
            ),
        ),
        (
            (
                "p.backend_commit_issuers_match_source_union and p.barrier_count_u"
                "ses_derived_union_size"
            ),
            "Barrier registration counts unique issuers, not per-operand multiplicity.",
        ),
    ],
    (
        "Enumerate candidate CTA ranks c in the cluster; in two_ctas mode "
        "skip odd c. A descriptor admits c when every nonzero CGA basis bi"
        "t position i has (c & (1<<i))==0; zero bases impose no restrictio"
        "n. Count c once if any descriptor admits it. With no descriptors "
        "or multicast disabled use one arrival. Initialize stage-release a"
        "nd accumulator-ready barriers using this derived unique issuer co"
        "unt; do not multiply arrivals by the number of descriptors or ass"
        "ume count equals physical CTA count."
    ),
    (
        "Replace the barrier arrival count by the number of actual unique "
        "issuing CTA atoms whose engine completion the receiver joins."
    ),
    [
        (
            "union_count",
            (
                "Each actual source issuer is counted once even when several descr"
                "iptors elect the same CTA."
            ),
            "proof",
        ),
        (
            "backend_incidence",
            ("Descriptor predicate matches actual native issuer/recipient completion routing."),
            "test",
        ),
    ],
    (
        "The Gluon tcgen05 multicast helper with up to four shared descrip"
        "tors and matching cluster bit count. Derived arrival count is not"
        " an optimizer knob; cluster layout and instruction group are inpu"
        "ts."
    ),
    (
        "When multicast MMA has several operand replication patterns, deri"
        "ve completion arrivals from the union of elected CTA issuers. A C"
        "TA elected by both A and B still sends one commit arrival; two-CT"
        "A mode counts only its even-rank atom owner. Registering per-desc"
        "riptor totals can deadlock, while counting too few allows prematu"
        "re reuse."
    ),
    (
        "Source python/triton/experimental/gluon/language/nvidia/blackwell"
        "/__init__.py:665-700; selected caller gluon/14-multicta.py:972-10"
        "28. The helper enumerates a finite set and increments once after "
        "the first admitting descriptor, proving set-union cardinality. Na"
        "tive commit routing is the separate incidence obligation."
    ),
    "sync",
)
add(
    "opposite_order_topk_merge",
    (
        "Two bounded top-k lists in opposite order merge by pairwise maxim"
        "um into a bitonic candidate list."
    ),
    [num("capacity", "Power-of-two retained candidate count, at least the requested k.")],
    [
        (
            (
                "p.capacity_power_of_two and p.common_total_key_order and p.input_"
                "blocks_partition_candidates"
            ),
            (
                "Local top-k and retained state use the same total order over dist"
                "inct original occurrences."
            ),
        ),
        (
            ("p.left_list_ascending_and_right_list_descending and p.padding_below_all_valid_keys"),
            "Merge inputs are oppositely ordered and padding cannot win.",
        ),
        (
            "p.requested_k <= capacity and p.enough_valid_candidates",
            "Retained count covers requested outputs.",
        ),
    ],
    (
        "Scan disjoint input blocks. Retain each block top capacity keys u"
        "nder one total order. Sort the running retained list ascending, t"
        "ake the next block top list descending, and replace running state"
        " elementwise by max(left[i],right[i]). This bitonic sequence cont"
        "ains the largest capacity keys of their union; sort/bitonic-merge"
        " it before the next iteration. Finally sort retained keys descend"
        "ing and keep requested k. Source total key packs monotone floatin"
        "g value in high bits and reverse original index in low16 bits, so"
        " smaller expert index wins equal supplied floating-bit values."
    ),
    (
        "Expand the stream into the union of all original candidates; reta"
        "ined state is its top-capacity multiset and final prefix is the r"
        "equested exact top-k."
    ),
    [
        (
            "pairwise_union",
            (
                "Opposite sorted lists pairwise maxima retain the largest capacity"
                " elements of their union."
            ),
            "proof",
        ),
        (
            "key_identity",
            (
                "Packed keys preserve value order, original identity and admitted "
                "tie/exception semantics."
            ),
            "test",
        ),
    ],
    (
        "Power-of-two bounded exact top-k, disjoint input blocks and commo"
        "n total keys. Floating signed zero and NaN must follow explicit k"
        "ey semantics; mathematical equality of differently encoded values"
        " is not inferred. Source expert count<32768 bounds low16 index en"
        "coding."
    ),
    (
        "When streaming exact top-k should keep only k-sized state, merge "
        "opposite-sorted retained and block top lists by pairwise max, the"
        "n bitonic-sort the result. This needs neither a multi-run cursor "
        "nor an append/threshold buffer. Keep score/index key order identi"
        "cal at every block, and exclude padded zeros from valid keys."
    ),
    (
        "Source python/triton_kernels/triton_kernels/topk_details/_topk_fo"
        "rward.py:33-104 and topk.py:27-45. If A ascends and B descends, d"
        "iscarded min(A_i,B_i) cannot outrank the kth retained maximum; cr"
        "ossing monotone order gives the standard bitonic top-half partiti"
        "on. An induction over blocks retains global top-capacity. This di"
        "ffers from sorted_run_topk@1 cursor consumption and streaming_top"
        "k_filter@1 threshold compaction."
    ),
    "reduce",
)
add(
    "expected_observed_scale_pair",
    (
        "Quantization consumes one immutable expected scale while separate"
        "ly reducing an observed output scale."
    ),
    [],
    [
        (
            ("p.expected_scale_positive_finite_and_immutable and p.expected_scale_matches_decode"),
            "Current encoded values and decoder use the same retained expected scale.",
        ),
        (
            (
                "p.observed_scale_initialized_and_domain_complete and p.observed_c"
                "onsumed_after_all_producers"
            ),
            (
                "Observed maximum has fresh state and is not consumed before all p"
                "roducing CTAs complete."
            ),
        ),
        (
            "p.exception_and_saturation_policy_explicit",
            "NaN/Inf, epsilon, cast and optional saturation follow the requested recipe.",
        ),
    ],
    (
        "Load expected scale once for the invocation and quantize each com"
        "puted value as value/expected_scale with the declared optional fi"
        "nite saturation/cast. Independently track absmax of the unscaled "
        "computed values, map NaN to the source exceptional statistic (+in"
        "finity), normalize that observed maximum by the format finite max"
        "imum and source epsilon, and atomically maximize the observed-sca"
        "le destination. Never reinterpret a partial observed scale as the"
        " current encoding scale. Future expected scale choice is an exter"
        "nal policy after producer completion; no history roll or same-inv"
        "ocation feedback is invented."
    ),
    (
        "Decode current values with the immutable expected scale and expos"
        "e a separate complete statistic about their unscaled source value"
        "s."
    ),
    [
        (
            "epoch_separation",
            (
                "All values in this invocation use the same expected scale and obs"
                "erved updates cannot change their encoding."
            ),
            "proof",
        ),
        (
            "statistic_complete",
            (
                "Observed scale includes every valid source value once under the r"
                "equested exception/max policy before consumption."
            ),
            "test",
        ),
    ],
    (
        "Per-invocation expected scale plus independent observed max-stati"
        "stic channel. Source flexpoint supports optional scale pointers; "
        "finite input subgroup suffices for ordinary numeric binding, exce"
        "ptional policy is explicitly source-defined. This is not quantiza"
        "tion_scale_state_transition@1 history roll/reset."
    ),
    (
        "When scaling predictions and measured dynamic range are separate,"
        " retain the expected scale for current encoding and reduce an obs"
        "erved scale for later policy. Producer CTAs may atomically update"
        " observations, but current values must not divide by those partia"
        "l updates. A future controller may choose another expected scale "
        "only after the whole observation domain completes."
    ),
    (
        "Source numerics_details/flexpoint.py:119-176 and swiglu_details/_"
        "swiglu.py:67 onward in the shipped triton_kernels package. Immuta"
        "ble expected dataflow proves current encoding independence; max r"
        "eduction over all producer observations supplies the final statis"
        "tic. Caller initialization/completion and future policy remain ex"
        "plicit."
    ),
    "store",
)

add(
    "count_bucketed_sparse_row_reduction",
    (
        "Rows are stably bucketed by active contributor count, then reduce"
        "d by ascending bitmap member indices."
    ),
    [
        num("chain_factor", "Adjacent feature tiles processed by one row executor."),
        num("heavy_subtiles", "Independent row subtiles for the heavy bucket (source binds2)."),
    ],
    [
        (
            (
                "p.mask_feature_broadcast and p.reduction_extent <= 32 and p.nonze"
                "ro_mask_means_active"
            ),
            "One contributor bitmap describes every feature of a row.",
        ),
        (
            (
                "p.row_permutation_exact and p.original_row_identity_retained and "
                "p.bucket_limits_cover_counts"
            ),
            (
                "Reordering changes execution order only; dispatch limits cover ev"
                "ery active contributor."
            ),
        ),
        (
            "p.admitted_add_order_and_decode_scale_policy and p.output_tiles_disjoint",
            (
                "Ascending contributor accumulation, scale decoding and output own"
                "ership match the requested numeric contract."
            ),
        ),
    ],
    (
        "Count nonzero contributor-mask entries per row. Within each plann"
        "er CTA interleaved row enumeration, stable-partition into counts>"
        "=4, counts3, counts2, counts<=1, keeping the original row IDs and"
        " scattering back to that same enumeration of destination position"
        "s. This is not a global stable sort. For a tile of reordered rows"
        " dispatch from its maximum count: at most1, exactly2, exactly3 us"
        "e those static loop bounds; mixed/heavy tiles use the full reduct"
        "ion extent and may split independent row subtiles. Build each row"
        " uint32 bitmap from nonzero mask entries. At every iteration obta"
        "in ffs(bitmap)-1, mask an empty bitmap, then clear its least sign"
        "ificant set bit. Load only that contributor, apply its micro/glob"
        "al/per-element scales in declared order, and add. An empty row pr"
        "oduces zero. Chain adjacent feature tiles under the same row meta"
        "data, then store at original row IDs. This is count-class orderin"
        "g, not least-loaded heap assignment."
    ),
    (
        "Invert the row permutation and expand the bitmap loop into the or"
        "iginal masked contributor sum in ascending contributor index orde"
        "r."
    ),
    [
        (
            "bucket_cover",
            (
                "Every original row appears once, and every dispatched loop limit "
                "covers all of its set bits."
            ),
            "test",
        ),
        (
            "bitmap_order",
            (
                "ffs/clear-lowest enumerates each active contributor once in ascen"
                "ding original order, including empty rows."
            ),
            "proof",
        ),
        (
            "tile_ownership",
            (
                "Heavy row subtiles and chained feature ranges partition outputs w"
                "ithout duplicating row contributions."
            ),
            "test",
        ),
    ],
    (
        "Feature-broadcast sparse masks with at most32 contributors; gener"
        "al per-feature masks use the separate ordinary masked loop. Reord"
        "ering is safe for independent rows. Source thresholds1/2/3 are a "
        "compiled menu, not a measured universal optimum."
    ),
    (
        "When rows have a small bounded contributor axis and varying spars"
        "ity, bucket by active count and enumerate only bitmap members. Ke"
        "ep original row IDs through the permutation so dense output order"
        " is preserved. Static short loops and heavy row subtiles are inde"
        "pendent scheduling choices; feature-dependent masks cannot share "
        "one bitmap."
    ),
    (
        "Source reduce.py:237-296,350-510,570-667. Existing cost_partition"
        "ed_worklists@1 assigns tasks through a least-loaded heap and cann"
        "ot express this stable categorical permutation; size_class_cooper"
        "ative_assignment@1 changes executor width, not bitmap contributor"
        " traversal. Native casts, postprocess functions and scale arithme"
        "tic remain separate numeric obligations."
    ),
    "reduce",
)

add(
    "mixed_radix_materialization",
    (
        "Split logical coordinates into bounded digits, permute the digits"
        ", and materialize a consumer-oriented tensor layout."
    ),
    [
        choice(
            "profile",
            [
                "square_tile",
                "hopper_mma_v2",
                "hopper_mma_v3",
                "hopper_scale",
                "blackwell_weight_tile",
                "fp4_axis_repack",
            ],
            (
                "Pinned factorization/permutation family; all profiles are actual "
                "NVIDIA storage paths."
            ),
        ),
        num("tile_rows", "Square tile width, or Blackwell packed-K tile extent."),
        num("tile_columns", "Blackwell N tile extent."),
        num("warps", "Hopper scale warp count; power of two."),
    ],
    [
        (
            (
                "p.logical_extents_fit_factorization_or_neutral_padding and p.sele"
                "cted_profile_matches_consumer"
            ),
            (
                "Every valid coordinate has a unique complete mixed-radix expansio"
                "n compatible with the consumer."
            ),
        ),
        (
            (
                "p.destination_owned_once and p.padding_excluded_or_neutral and p."
                "element_codes_unchanged"
            ),
            (
                "Materialization copies exact element/code identities without race"
                "s; padding cannot become logical data."
            ),
        ),
        (
            "p.packing_axis_and_leading_strides_explicit",
            (
                "Packed bytes and logical scalar axes are distinguished and batch "
                "addresses preserve actual strides."
            ),
        ),
    ],
    (
        "Decode coordinates into the selected profile digits, permute thos"
        "e digits, then flatten using destination strides. square_tile spl"
        "its [R/t,t,C/t,t] and permutes [0,2,1,3]. blackwell_weight_tile t"
        "akes K-packed bytes and maps (e,k,n) to (e,k/tile_rows,n/tile_col"
        "umns,n%tile_columns,k%tile_rows); zero-pad tiles. hopper_scale sp"
        "lits [M/(32*warps),2,warps,2,8,K/2,2] and permutes [0,2,5,1,4,6,3"
        "], producing [M/32,K*32]. hopper_mma profiles split each of two a"
        "xes into [rest,k_tile,warp_tile,threads,scott,contig]. Bind k_til"
        "e=[1,4/u], warp_tile=[2,2], threads=[4,4], scott=[2,1], contig=[1"
        ",u], with u=4 for MMA v2 and u=1 for v3; permute the resulting12 "
        "digits by [0,3,6,10,4,9,7,1,8,2,5,11]. The Hopper bit recoding is"
        " a separate reversible step. fp4_axis_repack transposes each2x2 l"
        "ogical nibble tile: from adjacent bytes a,b storing one packed ax"
        "is produce lo=(a&15)|((b&15)<<4), hi=(a>>4)|(b&240), placing lo/h"
        "i at adjacent coordinates of the old packed axis; this map is its"
        " own inverse. Inverse materialization applies the inverse permuta"
        "tion and removes only padding."
    ),
    (
        "Apply the inverse digit permutation and flatten original logical "
        "coordinates, preserving each element or packed-code identity."
    ),
    [
        (
            "digit_bijection",
            (
                "The selected split product equals the padded extents and its perm"
                "utation is bijective on all valid coordinates."
            ),
            "test",
        ),
        (
            "padding_and_stride",
            ("Leading strides, packing axes and valid tails remain consistent in both directions."),
            "test",
        ),
        (
            "consumer_binding",
            ("Selected physical layout matches actual instruction/TMA/decode coordinate use."),
            "test",
        ),
    ],
    (
        "Materialized complete mixed-radix profiles from shipped NVIDIA te"
        "nsor layouts. This moves data; tensor_axis_permutation@1 is only "
        "a rank-three view and does not supply these digit factorizations."
        " Warp count and tile extents must satisfy consumer alignment and "
        "source guards."
    ),
    (
        "When a consumer needs contiguous tiles or instruction-owned fragm"
        "ents that ordinary strides cannot express, materialize a mixed-ra"
        "dix digit permutation. Bind the split factors and exact order, pr"
        "eserving code identities and excluding padding. Hopper fragment/s"
        "cale profiles and Blackwell pre-transposed weight tiles differ in"
        " their digit maps; they are not interchangeable swizzles."
    ),
    (
        "Source layout_details/tiled.py:27-55, hopper_value.py:145-230, ho"
        "pper_scale.py:110-143, blackwell_value_shuffled.py:_swizzle_data_"
        "torch and _convert_shuffled_mxfp4. Digit permutation is invertibl"
        "e under full factorization. Native conversion kernels and selecte"
        "d consumers still need address/layout assurance. Blackwell scale1"
        "28x4 uses existing blocked_scale_digit_layout@1 rather than this "
        "new profile."
    ),
    "store",
)

add(
    "fp4_bf16_bit_interleave",
    (
        "Eight FP4 payloads in four bytes are rearranged into one word opt"
        "imized for packed BF16 expansion."
    ),
    [],
    [
        (
            "p.input_four_bytes_each_two_fp4 and p.complete_word_ownership",
            "Each complete32-bit word owns exactly eight E2M1 codes.",
        ),
        (
            ("p.little_endian_byte_view_and_unsigned_shifts and p.decoder_matches_bit_positions"),
            "Encoder and decoder agree on byte order and masked unsigned shifts.",
        ),
    ],
    (
        "For each half h=0,1 take nibble q_j=(byte_j>>(4*h))&15 for j=0..3"
        ". Let C(q)=((q&8)<<12)|((q&7)<<6), D(q)=((q&8)<<11)|((q&6)<<9)|(("
        "q&1)<<13). Place C(q0), C(q1)>>3, C(q2)>>6, D(q3) in disjoint16-b"
        "it positions by OR, then place half h at word bits16*h. Decode th"
        "e first three nibbles from those disjoint fields and the fourth f"
        "rom its separate positions, retaining signed zero and all payload"
        " bits. Hopper consumer expands the prepared fields to BF16 repres"
        "entations; the storage transform itself changes no FP4 codes."
    ),
    (
        "Extract each source nibble using the inverse bit incidence map, r"
        "estoring its original byte and low/high position."
    ),
    [
        (
            "bit_incidence",
            (
                "Every one of32 input bits maps to one distinct output bit, with i"
                "nverse recovering all256 possible source byte patterns in each po"
                "sition."
            ),
            "test",
        ),
        (
            "consumer_decode",
            ("Native BF16 expansion interprets those positions as the original E2M1 values."),
            "test",
        ),
    ],
    (
        "Pinned Hopper MXFP4 four-byte bit layout before its matching BF16"
        " decoder. This is not contiguous bitstream packing or a per-byte "
        "payload shift. Complete words and source little-endian representa"
        "tion are required."
    ),
    (
        "When software FP4 decoding dominates, pre-interleave payload bits"
        " to match packed BF16 expansion. Preserve every code bit through "
        "an explicit bit permutation and bind the matching decoder. This c"
        "hanges stored representation, not the quantizer or the value/scal"
        "e grouping."
    ),
    (
        "Source hopper_value.py:_compress_fp4, _compress_fourth, _pack_bit"
        "s and _unpack_bits. Existing bitstream_packing@2 concatenates bit"
        "s and operand_bitfield_placement@1 shifts one byte; neither suppl"
        "ies this32-bit permutation. The finite bit-incidence witness is s"
        "eparate from native decoder assurance."
    ),
    "store",
)

add(
    "encoded_group_scale_recipe",
    (
        "A low-float group selects a scale-format and rounding recipe and "
        "quantizes using that exact encoded scale."
    ),
    [
        choice("code_format", ["e2m1", "e4m3", "e5m2"], "E2M1 codes pack two per byte."),
        choice(
            "scale_format",
            ["ue8m0", "e4m3"],
            "Power-of-two unsigned exponent or direct floating scale.",
        ),
        choice("rounding", ["up", "down", "nearest"], "Scale rounding; not value-code rounding."),
        num("group_size", "32 for OCP MX,16 for NVFP4."),
    ],
    [
        (
            (
                "(scale_format == 'e4m3' and code_format == 'e2m1' and group_size "
                "== 16 and rounding in ['up','nearest']) or (scale_format == 'ue8m"
                "0' and group_size == 32 and rounding in ['up','down'])"
            ),
            ("Pinned code/scale/group/rounding pairs are coupled, not a Cartesian product."),
        ),
        (
            ("p.requested_recipe and p.finite_valid_values and p.groups_have_valid_elements"),
            ("The caller requests this lossy quantizer and invalid lanes are excluded from amax."),
        ),
        (
            "p.nonzero_scales_normal_and_representable and p.no_scale_encoding_overflow",
            (
                "Nonzero group scales fit the declared normal metadata domain; ext"
                "reme underflow/overflow is outside this binding."
            ),
        ),
        (
            ("p.format_rounding_pair_supported and p.cast_reciprocal_and_saturation_admitted"),
            (
                "UE8M0 supports up/down, direct E4M3 supports up/nearest. Code rou"
                "nding/saturation is an explicit numeric contract."
            ),
        ),
    ],
    (
        "Reduce max(abs(valid values)) in each logical group, excluding in"
        "valid positions. UE8M0 up divides by the code finite maximum (6,4"
        "48,57344) and rounds scale up to a power of two by exponent/manti"
        "ssa bits. UE8M0 down divides by the maximum code power of two (4,"
        "256,32768) and rounds scale down to a power of two. Direct E4M3 d"
        "ivides by the finite maximum and casts nearest-even; up additiona"
        "lly increments a positive encoding if it undershoots, saturating "
        "scale encoding at448. Use the rounded FP32 scale represented by t"
        "hat encoding for reciprocal and value conversion. Zero groups use"
        " zero reciprocal and emit zero codes; UE8M0 metadata byte0 decode"
        "s to2**-127. Invalid codes become zero. FP4 values use satfinite "
        "nearest-even code conversion and even-index nibble low, odd high."
        " Decode with the same metadata; do not claim lossless recovery of"
        " original values."
    ),
    (
        "The emitted code/metadata pair is the requested rounded quantizer"
        " result; inverse decoding recovers represented values, not arbitr"
        "ary original inputs."
    ),
    [
        (
            "scale_recipe",
            (
                "Selected divisor, scale rounding and zero-group policy match the "
                "requested format recipe."
            ),
            "test",
        ),
        (
            "code_scale_identity",
            (
                "Each code was computed with the rounded stored scale belonging to"
                " its own group and logical coordinate."
            ),
            "proof",
        ),
        (
            "native_conversion",
            (
                "Hardware/software FP4 and FP8 conversion, reciprocal and saturati"
                "on implement the admitted numeric policy."
            ),
            "test",
        ),
    ],
    (
        "Finite groups within supported normal scale ranges. E4M3 scale gr"
        "oups16 bind NVFP4; UE8M0 groups32 bind MXFP4/MXFP8. This adds MXF"
        "P4/E5M2 and explicit down/up scale variants absent from groupwise"
        "_low_float_quantization@1; exceptional groups and subnormal scale"
        " construction remain outside the candidate domain."
    ),
    (
        "When a block-scaled output needs a different format or scale-roun"
        "ding recipe, select both explicitly and use the decoded stored sc"
        "ale to form codes. Upward and downward exponent recipes use diffe"
        "rent divisors; substituting one for the other changes the quantiz"
        "er. Keep group identity, packed-code order and zero-group behavio"
        "r visible."
    ),
    (
        "Source numerics_details/mxfp_details/_downcast_to_mxfp.py:9-150 a"
        "nd _upcast_from_mxfp.py:9-123. This candidate preserves a finite "
        "normal-scale subset; it does not certify extreme bit-rounding beh"
        "avior. The decoder special-cases UE8M0 zero as2**-127 and255 asNa"
        "N, and saturates decoded MXFP4 to destination finite range. NVFP4"
        " global row/fiber scales are applied once outside per-K tile deco"
        "ding."
    ),
    "map",
)

add(
    "owner_routed_occurrence_copy",
    (
        "Selected logical occurrences are copied to uniquely owned peer ro"
        "ws and published as a completed generation."
    ),
    [
        choice(
            "direction",
            ["token_to_expert", "expert_to_occurrence"],
            (
                "Forward fanout or inverse occurrence permutation; neither directi"
                "on adds expert outputs."
            ),
        )
    ],
    [
        (
            "p.expert_owner_partition_exact and p.peer_buffers_addressable",
            (
                "Each expert has exactly one owner rank; rank0 is a real owner, no"
                "t missing metadata."
            ),
        ),
        (
            (
                "p.destination_occurrence_map_injective and p.all_indices_valid an"
                "d p.rows_cover_requested_occurrences"
            ),
            (
                "Every peer destination row has one writer and all requested occur"
                "rences are represented."
            ),
        ),
        (
            ("p.old_readers_retired and p.peer_store_completion_barrier_before_consumption"),
            (
                "Buffers are fresh/reusable and the communication runtime publishe"
                "s all peer writes before consumers read."
            ),
        ),
    ],
    (
        "For token_to_expert, each source rank enumerates its local token "
        "t, restores global token g=t+rank*local_tokens, and loads selecte"
        "d expert e(g,j) plus destination row gate(g,j). Compute owner(e) "
        "from the unique membership bit across ranks; the source uses sum("
        "rank*membership). Copy the token vector to peer[owner(e)][gate(g,"
        "j)] for every selected occurrence j, retaining duplicates only if"
        " their destination rows are distinct. For expert_to_occurrence, s"
        "ource row s carries a global destination occurrence d and its exp"
        "ert e(d). Only rank owner(e(d)) writes s to peer[d/local_occurren"
        "ces][d%local_occurrences]. Launch completion plus the symmetric-m"
        "emory barrier publishes the whole destination generation. A later"
        " reduction across expert occurrences is separate; ordinary stores"
        " here must not be presented as additive scatter."
    ),
    (
        "Erase placement by following each output occurrence back to its o"
        "riginal token/expert occurrence; fanout preserves one value per s"
        "elected occurrence, inverse copy preserves its own expert result."
    ),
    [
        (
            "owner_and_address",
            (
                "Unique expert ownership and injective occurrence indices yield ex"
                "actly one writer per peer destination coordinate."
            ),
            "test",
        ),
        (
            "generation_publication",
            (
                "Consumers see complete stores only after runtime barrier completi"
                "on and no prior generation is reused early."
            ),
            "test",
        ),
    ],
    (
        "Direct peer-addressed copies with explicit occurrence identity an"
        "d unique expert owners. Replicated expert ownership, duplicate de"
        "stination rows or reduction-to-token require another protocol. Th"
        "e inverse global row is an occurrence index in the flattened sele"
        "cted-expert domain, not necessarily a token index."
    ),
    (
        "When routed expert work needs to move token/expert occurrences ac"
        "ross shards, retain a unique occurrence row and its expert owner."
        " Forward fanout copies a token to every selected expert row; the "
        "inverse route copies each expert result to its own destination oc"
        "currence. Publish peer stores before consumers and reduce occurre"
        "nces only in an explicit later stage."
    ),
    (
        "Source distributed.py:10-100,136-290 and distributed_details/mesh"
        ".py:119-172. The actual EP-to-DP host shape already includes its "
        "occurrence domain; decoding d by local extent must not silently d"
        "ivide it by top-k. Existing indexed_additive_scatter@1 atomically"
        " adds collisions, which is absent here. Native symmetric-memory o"
        "rdering/progress is a runtime obligation."
    ),
    "transfer",
)

add(
    "carry_isolated_bit_counts",
    ("Packed boolean columns are summed in progressively wider carry-isolated fields."),
    [
        num(
            "rows_per_tile",
            ("Supported power-of-two reduction length with source split factors dividing it."),
        )
    ],
    [
        (
            "p.uint32_boolean_words and p.rows_split_into_exact_stages",
            (
                "Each input word holds32 independent boolean columns and source st"
                "age factors partition all rows."
            ),
        ),
        (
            "p.stage_counts_fit_fields and p.final_integer_counts_do_not_overflow",
            (
                "First counts<=8 fit4 bits, second counts<=128 fit8 bits, final co"
                "unts fit their integer destination."
            ),
        ),
    ],
    (
        "Let a1=min(8,N), a2=min(16,N/a1), a3=N/(a1*a2), requiring integra"
        "l source stages. For each phase s=0..3 extract (word>>s)&0x111111"
        "11 and sum groups of a1 words: each4-bit field holds one column c"
        "ount without carry. Split those counts into phases t=0,1 by (pack"
        "ed>>(4*t))&0x0f0f0f0f; sum a2 groups into8-bit fields. Split by b"
        "yte phase u=0..3 using (packed>>(8*u))&255 and sum remaining a3 g"
        "roups into ordinary32-bit counters. Restore original column order"
        " by flattening [u,t,s], so column index is8*u+4*t+s. Store tile p"
        "artial counts and optionally atomically add disjoint tile totals "
        "into a fresh zero-initialized final array."
    ),
    (
        "Expand each packed field count into the sum of its original input"
        " bits, one independent reduction per column."
    ),
    [
        (
            "no_field_carry",
            (
                "Every packed intermediate count stays strictly below the next fie"
                "ld radix and cannot affect an adjacent column."
            ),
            "proof",
        ),
        (
            "column_identity",
            "Flattened phase order restores all32 original bit positions exactly.",
            "test",
        ),
        (
            "partial_combine",
            (
                "Final integer totals include all valid tile partials and no overf"
                "low or premature consumption."
            ),
            "test",
        ),
    ],
    (
        "Unsigned32-bit membership matrices and source legal stage lengths"
        ". Integer exactness requires field/headroom bounds; this is verti"
        "cal per-column population count, not scalar popcount of each word"
        " or bit-transpose alone."
    ),
    (
        "When summing many packed membership rows, count several columns a"
        "t once inside spaced fields, then widen before carries could mix "
        "columns. Bind each stage fan-in below its field capacity and rest"
        "ore column phase order explicitly. Tile counts can serve both pre"
        "fix metadata and final atomic totals."
    ),
    (
        "Source tensor_details/bitmatrix_details/sum_bitmatrix_rows.py:9-8"
        "0. N=128 gives fan-ins8,16,1 and counts8<16 then128<256; source m"
        "asks rows beyond the dynamic shape to zero. Native int atomic lif"
        "etime and bounds are separate from the packed-field identity."
    ),
    "reduce",
)

add(
    "code_domain_group_requantization",
    (
        "Requantize adjacent scaled low-float groups using code-domain max"
        "ima before expanding values."
    ),
    [],
    [
        (
            "p.e2m1_codes_and_direct_e4m3_group16_scales and p.outer_scale_immutable",
            (
                "Input scale metadata describes the actual input codes and optiona"
                "l outer row scale is fixed."
            ),
        ),
        (
            "p.output_mxfp8_group32_requested and p.groups_pair_without_crossing_rows",
            "Two adjacent16-value groups form each requested32-value group.",
        ),
        (
            (
                "p.finite_scales_and_values and p.new_scale_normal_representable a"
                "nd p.power_two_fold_rounding_admitted"
            ),
            (
                "The finite-range contract admits FP32 scale arithmetic and foldin"
                "g an inverse power-of-two scale; underflow/overflow is excluded."
            ),
        ),
    ],
    (
        "Clear each E2M1 sign bit and reduce its ordered3-bit magnitude co"
        "des within each16-value input group, using packed max.u16x2 field"
        "s or an equivalent exact max. Decode only that maximum code, mult"
        "iply by abs(input_group_scale*outer_row_scale), and max the adjac"
        "ent two groups. Divide by448 and round upward to an encoded E8M0 "
        "power-of-two output scale. Multiply each original signed input sc"
        "ale by the inverse output power of two once per group, decode FP4"
        " exactly through FP16 then promote to FP32, multiply by the folde"
        "d scale and cast to E4M3. Emit one output scale per32 values. No "
        "separate NVFP4 or outer scale remains in the result. Zero groups "
        "produce zero codes under the stated zero-reciprocal policy."
    ),
    (
        "Interpret output codes with their new E8M0 metadata as the reques"
        "ted MXFP8 quantization of the original decoded NVFP4 values inclu"
        "ding the outer row scale."
    ),
    [
        (
            "code_maximum",
            (
                "Ordered magnitude-code max equals max(abs(decoded E2M1)) in the s"
                "ame logical16-value group."
            ),
            "test",
        ),
        (
            "pair_grouping",
            (
                "Adjacent maxima and per-value source scales are paired into exact"
                "ly the same32-value output group."
            ),
            "test",
        ),
        (
            "scale_fold",
            (
                "Folding the inverse power-of-two scale preserves the admitted FP3"
                "2 rounding domain and native conversions."
            ),
            "test",
        ),
    ],
    (
        "Pinned finite normal-scale NVFP4-to-MXFP8 conversion. Does not re"
        "use a generic raw-product promotion rule or infer exactness for s"
        "ubnormal/extreme scales. Packed maxima require explicit logical g"
        "roups, not reliance on inline-assembly pack order."
    ),
    (
        "When converting block-scaled FP4 into MXFP8, find maxima in the o"
        "rdered magnitude-code domain before expanding every value. Pair t"
        "he two16-value source groups, absorb the outer scale, and fold th"
        "e new inverse power-of-two scale into each old group scale. This "
        "removes duplicated full decoding for scale selection while preser"
        "ving code/group identity."
    ),
    (
        "Source numerics_details/mxfp_details/_upcast_from_mxfp.py:130-190"
        ". Packed fields contain independent magnitude maxima, so the code"
        "-domain step is exact. Scale folding is conditional on the admitt"
        "ed finite normal arithmetic domain; native decode/cast and roundi"
        "ng assurance remain open."
    ),
    "map",
)

add(
    "cyclic_tile_reduction_partition",
    ("Parallel reduction participants own cyclic tiles rather than contiguous reduction ranges."),
    [
        num("participants", "Independent partial-result participants."),
        num("tile_size", "Consecutive logical reduction indices per owned tile."),
    ],
    [
        (
            "p.associative_commutative_exact_combine and p.changed_float_order_admitted",
            (
                "The requested combine admits this exact-arithmetic partition and "
                "its floating-point order."
            ),
        ),
        (
            (
                "p.tail_mask_identity and p.empty_participants_emit_identity and p"
                ".partial_generation_complete"
            ),
            (
                "Tails/empty owners contribute the identity and final combine wait"
                "s for all partials."
            ),
        ),
        (
            "p.additive_epilogue_once_and_nonlinear_epilogue_after_combine",
            (
                "Bias/additive checkpoints occur once; nonlinear activation/quanti"
                "zation waits for the combined value."
            ),
        ),
    ],
    (
        "Participant s owns logical tiles s+j*participants and within each"
        " owns k=(s+j*participants)*tile_size+u for u in [0,tile_size), ma"
        "sked by k<length. Accumulate those products in increasing j, stor"
        "e a separate partial indexed by s, then combine every participant"
        " exactly once. The matmul instance restores packed operand offset"
        "s independently: W tile increment is packed_tile_K*participants w"
        "hile logical X increment is tile_size*participants. Apply bias on"
        "ly to s=0 or after final combine; defer nonlinear activation and "
        "output quantization to the final reduction pass. Empty participan"
        "ts publish neutral partials instead of reading invalid data."
    ),
    (
        "For each logical k recover tile=k//tile_size, owner=tile%particip"
        "ants, iteration=tile//participants and within-tile=k%tile_size; c"
        "ombine all owned products once."
    ),
    [
        (
            "cyclic_cover",
            (
                "Every logical reduction index has exactly one owner and one itera"
                "tion, including the final partial tile."
            ),
            "test",
        ),
        (
            "partial_and_epilogue",
            (
                "All partials complete before combination; additive and nonlinear "
                "epilogues have the required exact-once placement."
            ),
            "test",
        ),
    ],
    (
        "Tile-cyclic split-K and equivalent reductions with separate parti"
        "cipant-indexed scratch. Existing reduction_partition@1 and @2 nam"
        "e strided threads/warps and contiguous ranges, but do not expose "
        "the cyclic tile width/participant map as a compiled assignment."
    ),
    (
        "When split-K should interleave consecutive K tiles across indepen"
        "dent partial owners, select participant count and tile width sepa"
        "rately. Advance each owner by tile_width*participants, retain its"
        " scratch identity and combine after all owners finish. Packed ope"
        "rand byte strides follow their own tile sizes; nonlinear epilogue"
        "s belong after the complete sum."
    ),
    (
        "Source matmul_details/_matmul.py K-loop and _p_matmul.py:388-413;"
        " matmul.py:185-193,791-824 and epilogue pid_k==0 bias. Division/r"
        "emainder gives a constructive inverse. Generated default split co"
        "unt considers SM/grid coverage and at least four tiles per owner,"
        " but is a dispatch heuristic, not semantic evidence or a universa"
        "l performance law."
    ),
    "reduce",
)

add(
    "contraction_numeric_recipe",
    (
        "A contraction explicitly selects operand preparation and intrinsi"
        "c accumulation policy before layout selection."
    ),
    [
        choice(
            "operand_preparation",
            ["identity", "widen_e4m3_to_peer16", "fp32_to_tf32_rn", "fp32_to_tf32_rna"],
            ("Numeric preparation of each affected operand, with source/target legality guards."),
        ),
        choice(
            "accumulator",
            ["f32", "f64"],
            "FP64 only when both source operands and selected backend support it.",
        ),
        choice(
            "imprecise_accumulation",
            ["backend_default", "bounded"],
            (
                "Pass through the requested intrinsic policy; this is not a promis"
                "e about a native instruction count."
            ),
        ),
        num("imprecise_limit", "Requested maximum imprecise accumulation interval when bounded."),
    ],
    [
        (
            "p.requested_numeric_recipe and p.error_and_exception_policy_admitted",
            (
                "The coarse operation requests the selected preparation/accumulati"
                "on recipe; no equivalence to a higher-precision contraction is in"
                "ferred."
            ),
        ),
        (
            "p.operand_formats_and_backend_supported and p.rounding_mode_matches_target",
            ("Widening and TF32 rounding match the selected operand types and architecture."),
        ),
        (
            (
                "p.preparation_before_layout_selection_and_exactly_once and p.prod"
                "uct_coordinates_unchanged"
            ),
            (
                "Preparation is exposed before backend chooses fragments, preservi"
                "ng logical contraction coordinates."
            ),
        ),
    ],
    (
        "For each affected input tile either preserve its representation; "
        "widen E4M3 to the other operand FP16/BF16 type before dot lowerin"
        "g; or explicitly round FP32 values to TF32 with nearest-even on H"
        "opper+ and nearest-away on the source Ampere path. Preserve unmod"
        "ified operands. Initialize accumulator as FP64 only for the both-"
        "FP64 route, otherwise FP32. Pass the declared allow_tf32 and max_"
        "num_imprecise_acc policy to dot; do not infer the emitted instruc"
        "tion or its internal reduction schedule from those flags alone. A"
        "pply identical policy in regular and persistent routes so a sched"
        "uling choice does not silently change the requested numeric prepa"
        "ration."
    ),
    (
        "Erase layout and execution assignment while retaining the selecte"
        "d prepared-input and accumulator recipe; this recovers the reques"
        "ted contraction, not an unrounded real-valued sum."
    ),
    [
        (
            "preparation_identity",
            (
                "Every input coordinate receives the requested cast/rounding once "
                "before fragment lowering."
            ),
            "test",
        ),
        (
            "native_policy",
            (
                "Selected intrinsic/backend implements the requested numeric and a"
                "ccumulation policy over the supported range and exceptions."
            ),
            "test",
        ),
    ],
    (
        "Explicit numeric recipe for mixed E4M3/16-bit and FP32-to-TF32 NV"
        "IDIA contraction paths. Source mode is architecture-gated, not un"
        "iversally interchangeable; native instruction details and error b"
        "ounds require separate lowering/assurance."
    ),
    (
        "When changing persistence or operand layouts could hide a numeric"
        " conversion, expose the operand preparation and accumulator polic"
        "y first. Widen mixed E4M3 operands before layout choice, or selec"
        "t the target-correct TF32 rounding explicitly. Bind the intrinsic"
        " accumulation limit independently from tile scheduling and verify"
        " the requested numeric contract."
    ),
    (
        "Source matmul_details/_common.py:16-48, _matmul.py:388-525 and _p"
        "_matmul.py:388-580. compute_instruction_selection@2 selects instr"
        "uction and operand bit compatibility but has no preparation-round"
        "ing or imprecise-accumulation variable. The source policy is reco"
        "vered here without claiming which machine instruction the compile"
        "r emits."
    ),
    "contract",
)
