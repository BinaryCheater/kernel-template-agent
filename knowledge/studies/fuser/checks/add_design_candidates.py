"Materialize source-study candidate records; does not admit them to the Store."

from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1] / "delta"


def write(
    name,
    obj,
    variables,
    constraints,
    pre,
    program,
    backward,
    obligations,
    scope,
    summary,
    note,
    refines="access",
):
    p = root / f"{name}@1"
    p.mkdir(exist_ok=True)
    t = dict(
        name=name,
        version=1,
        level_from=3,
        level_to=4,
        change="vertical",
        refines=refines,
        produces=(
            ["access", "store", "transfer", "sync"]
            if name == "thread_ownership_exchange"
            else {
                "reduce": ["reduce", "store", "sync"],
                "work_assign": ["work_assign"],
                "transfer": ["transfer", "sync"],
                "map": ["map", "access"],
            }.get(refines, ["access"])
        ),
        object=dict(type=name, description=obj),
        variables=variables,
        constraints=[
            dict(id=f"c{i}", expr=e, meaning=m, cls="semantic")
            for i, (e, m) in enumerate(constraints)
        ],
        preconditions=[
            dict(id=f"pre{i}", expr=e, meaning=m, cls="semantic") for i, (e, m) in enumerate(pre)
        ],
        program=dict(description=program),
        backward_map=backward,
        obligations=[dict(name=n, statement=s, method=m) for n, s, m in obligations],
        effects=dict(transactions=[], storage=[], overlap=[]),
        scope=dict(predicate=" and ".join(e for e, m in constraints + pre), text=scope),
        requires=[],
        probed_on=[],
        relations=[],
        signature=dict(
            algorithm=obj,
            storage=(
                "Source-bound physical buffer; exact byte count and capacity supplied by binding."
            ),
            boundary="Coordinate identity and storage lifetime.",
            resources=["smem", "reg"],
        ),
        numeric_risk=(
            "Address/storage transformation preserves values; arithmetic chang"
            "es require separate obligations."
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
                rank="Unmapped coordinates or unretired readers.",
                assumes=[
                    dict(
                        name="binding",
                        statement="Declared source layout and lifetime premises hold.",
                    )
                ],
                guarantees=[],
                models=[],
                note=note,
            ),
            sort_keys=False,
        )
    )


def numeric(n, meaning):
    return dict(
        name=n,
        kind="numeric",
        binding="compile",
        role="software",
        domain=dict(lo=1, hi=65536),
        meaning=meaning,
    )


write(
    "blocked_scale_digit_layout",
    (
        "A padded two-dimensional scale table split into row/column digits"
        " and physically interleaved."
    ),
    [
        numeric("R_outer", "Fast row digit: row modulo R_outer."),
        numeric("R_inner", "Other within-tile row digit: row//R_outer modulo R_inner."),
        numeric("C", "Contiguous scale columns per column tile."),
    ],
    [
        (
            "p.padded_rows % (R_outer * R_inner) == 0 and p.padded_cols % C == 0",
            "Allocation contains whole digit tiles.",
        ),
        (
            "p.padded_rows >= p.rows and p.padded_cols >= p.cols",
            "Every logical scale is inside the padded table.",
        ),
    ],
    [
        (
            "p.scale_coordinates_paired_with_codes and p.consumer_uses_same_digit_layout",
            "Producer and decoder agree on the logical scale coordinate.",
        ),
        (
            "p.allocation_lifetime_valid and p.integer_offsets_fit",
            "Storage and offset arithmetic are valid.",
        ),
    ],
    (
        "For logical (r,c), let rt=r//(R_outer*R_inner), ro=r%R_outer, ri="
        "(r//R_outer)%R_inner, ct=c//C, ci=c%C. Store in contiguous digit "
        "order [rt,ct,ro,ri,ci]. Offset=rt*p.padded_cols*R_outer*R_inner+c"
        "t*R_outer*R_inner*C+ro*R_inner*C+ri*C+ci. Padding storage is sepa"
        "rate from logical scale ownership and must not emit scales for no"
        "nexistent code groups."
    ),
    (
        "Invert the radix digits to r=rt*R_outer*R_inner+ri*R_outer+ro and"
        " c=ct*C+ci, recovering the original scale table."
    ),
    [
        (
            "digit_bijection",
            "Every valid scale has exactly one physical address and inverse.",
            "proof",
        ),
        (
            "consumer_pairing",
            "Code and decoded scale use the same logical group despite physical layout.",
            "test",
        ),
    ],
    (
        "Positive row/column digit sizes, padded allocation and exact inte"
        "ger addressing. Fuser NVFP4 binding R_outer=32,R_inner=4,C=4. Doe"
        "s not change scale values or choose quantization groups."
    ),
    (
        "When a scale consumer expects small column groups interleaved wit"
        "h row subtiles, split coordinates into digits rather than copying"
        " a plain transpose. Bind row digit sizes and column group; r=33,c"
        "=5 at (32,4,4) maps to [0,1,1,1,1]. Reject consumers using an ord"
        "inary row-major or differently ordered digit table."
    ),
    (
        "Euclidean division uniquely yields bounded digits; mixed-radix fl"
        "attening is injective and inverse reconstructs r,c. Source runtim"
        "e/block_quantization_kernels.cu:16-58 and :290-315. Unique-coordi"
        "nate witness in checks/design_witnesses.py."
    ),
)
write(
    "thread_ownership_exchange",
    (
        "A materialized tile at a selected producer/consumer boundary tran"
        "sfers ownership between two thread mappings."
    ),
    [
        dict(
            name="exchange_boundary",
            kind="structure",
            binding="compile",
            role="software",
            domain=["cached_input", "cached_output"],
            meaning="Observed placement alternatives where the fusion switches thread ownership.",
        )
    ],
    [
        (
            "p.writer_map_bijective and p.reader_map_covers_required_tile",
            (
                "Each logical tile point has one writer and every consumer can fin"
                "d its required points."
            ),
        ),
        ("p.tile_bytes <= p.shared_capacity_bytes", "Exchange tile fits shared storage."),
    ],
    [
        (
            (
                "p.all_writes_visible_before_reads and p.no_overwrite_before_last_"
                "reader and p.all_participants_converge"
            ),
            "CTA publication, convergence and storage lifetime protect exchange.",
        ),
        (
            "p.boundary_preserves_value_and_coordinate_semantics",
            ("The selected cache boundary contains values matching the consumers at that point."),
        ),
    ],
    (
        "At exchange_boundary allocate shared tile S. Threads of writer ma"
        "pping W store value V(x) at the shared address for logical coordi"
        "nate x. Synchronize the participating CTA. Threads with consumer "
        "mapping R read S at the address for their required x, then releas"
        "e the tile only after all readers retire. Input and output vector"
        " widths may differ provided vector_width*unroll equals each threa"
        "d mapping assigned element count."
    ),
    (
        "Erase shared storage and the ownership maps: consumer of coordina"
        "te x reads the same logical V(x) written at the chosen boundary."
    ),
    [
        (
            "coordinate_identity",
            "Writer and reader shared addresses agree for each logical coordinate.",
            "proof",
        ),
        (
            "lifetime",
            ("Publication precedes every read and retirement precedes every destructive reuse."),
            "test",
        ),
    ],
    (
        "CTA-local materialized tensor tiles with two explicit bijective o"
        "wnership maps and legal vector accesses. This is not the metadata"
        "-only tensor_axis_permutation@1 and not the GEMM-specific shared_"
        "memory_tiling@1."
    ),
    (
        "When fused inputs and outputs require different contiguous lane d"
        "irections, choose an explicit cache boundary to exchange a tile t"
        "hrough shared memory. Bind cached_input or cached_output and both"
        " thread maps; a 32x32 tile on 256 threads may use four-element ve"
        "ctors on one side and two-element vectors with two unrolled opera"
        "tions on the other. Synchronize before readers switch ownership."
    ),
    (
        "For each logical x, writer stores V(x) to S[address(x)] once; vis"
        "ibility and lifetime imply the corresponding reader sees V(x). Pl"
        "acement changes only the value representation at the selected gra"
        "ph boundary. Source csrc/scheduler/transpose_non_tma.cpp:248-271,"
        ":457-494,:627-633."
    ),
)
write(
    "butterfly_reduce_scatter_chunks",
    (
        "A warp subgroup reduces vector-valued partials while scattering d"
        "isjoint result chunks among its lanes."
    ),
    [
        numeric("lanes", "Power-of-two subgroup lanes spanning the reduction axis."),
        numeric("values", "Vector components contributed by each subgroup lane."),
    ],
    [
        (
            "values % lanes == 0 and values >= lanes and p.lanes_power_of_two",
            "Each recursive half has an integral chunk.",
        ),
        (
            "p.subgroup_mapping_exact and p.lanes_participate",
            (
                "Peer shuffles stay within the logical subgroup and every required"
                " lane participates."
            ),
        ),
    ],
    [
        (
            "p.combine_associative_commutative_under_contract and p.valid_partial_states",
            "The combine and state contract admit the butterfly order.",
        )
    ],
    (
        "Start each of lanes logical lanes with values partial states. At "
        "each descending XOR peer bit halve the live component vector. Low"
        "er-half owners keep the lower half and send the upper half; upper"
        "-half owners keep the upper half and send the lower half. Combine"
        " received peer states componentwise with the retained states. Rep"
        "eat until each lane owns values/lanes complete states, the contig"
        "uous component range [lane*values/lanes,(lane+1)*values/lanes). P"
        "ublish only these states for subsequent cross-warp combining."
    ),
    (
        "Gather the disjoint per-lane output chunks to reconstruct the vec"
        "tor reduction of every original lane partial."
    ),
    [
        (
            "component_membership",
            (
                "Each final component contains each subgroup input exactly once wi"
                "th no component mixing."
            ),
            "proof",
        ),
        (
            "shuffle_and_state",
            (
                "Physical XOR peers, active mask and any coupled statistical state"
                " implement the logical exchange."
            ),
            "test",
        ),
    ],
    (
        "Power-of-two warp subgroups with divisible vector length; combine"
        " may be a coupled state such as count/mean/M2 provided its merge "
        "contract is satisfied. Source BDIMX divides 32, lanes=32/BDIMX an"
        "d NumVals divisible by lanes."
    ),
    (
        "When each lane holds several outer-reduction statistics, reduce a"
        "nd scatter vector chunks before writing shared partials. Choose s"
        "ubgroup width and vector length; BDIMX=8,NumVals=8 gives four red"
        "uction lanes and two final components per lane, reducing shared p"
        "ayload fourfold. Reject component counts not divisible by subgrou"
        "p width or shuffles crossing another output column."
    ),
    (
        "After one level, each owned half combines the same component from"
        " two disjoint lane sets. Induction doubles the reduced lane set w"
        "hile halving owned components. At log2(lanes) all input lanes are"
        " covered; component intervals partition the vector. Source runtim"
        "e/block_welford_outer.cu:11-58,:105-195. Numerical Welford merge "
        "binds centered_moment_merge@1 explicitly, separate from address p"
        "roof."
    ),
    refines="reduce",
)
write(
    "cluster_partial_destination_table",
    (
        "Sender-indexed scalar partial tables placed at either one cluster"
        " root or every cluster CTA."
    ),
    [
        dict(
            name="destinations",
            kind="structure",
            binding="compile",
            role="software",
            domain=["last_cta", "all_ctas"],
            meaning="Deliver partials only to the last cluster rank or replicate to every rank.",
        ),
        numeric(
            "peers", "Resident cluster CTA count, at most 32 in the lane-selected realization."
        ),
        numeric("warps", "Contributing full warps per CTA."),
    ],
    [
        (
            "peers <= 32 and p.scalar_bytes >= 4 and p.scalar_bytes <= 8",
            ("The lane-selected sender and float/double remote stores support the bound type."),
        ),
        (
            "p.sender_slot_count == peers * warps and p.partial_partition_exact",
            "Sender table has one injective slot per warp partial.",
        ),
    ],
    [
        (
            (
                "p.fresh_phase_zero_barrier and p.cluster_initialization_published"
                " and p.remote_store_supported"
            ),
            "Destination barriers initialized once and published before remote stores.",
        ),
        (
            (
                "p.transaction_completion_orders_reads and p.allocations_live_unti"
                "l_remote_access_retired"
            ),
            "Waits publish payload and allocations survive every sender/reader.",
        ),
        (
            "p.combine_order_admitted and p.participants_match_launch",
            "Numerical order and cluster participants match the requested operation.",
        ),
    ],
    (
        "Each warp reduces its local values to a lane-uniform scalar. Send"
        "er rank r,warp w owns slot r*warps+w. In all_ctas mode lane d sen"
        "ds that scalar to destination d for d<peers; in last_cta mode one"
        " elected lane sends only to rank peers-1. Each consuming CTA regi"
        "sters peers*warps*p.scalar_bytes expected bytes against its fresh"
        " phase-zero barrier, waits for transaction completion, then folds"
        " all sender slots with lane-strided reads and a warp reduction. O"
        "nly designated destinations expose final results."
    ),
    (
        "Project each destination table fold to the complete reduction of "
        "the disjoint sender warp partials; redundant copies expose the sa"
        "me admitted result."
    ),
    [
        (
            "delivery",
            (
                "Each consuming destination receives every sender partial once in "
                "its injective slot."
            ),
            "proof",
        ),
        (
            "byte_completion",
            "Each destination expects exactly the bytes actually sent before reading.",
            "test",
        ),
        (
            "retirement",
            ("No peer exits or reuses storage while remote accesses or consumer reads remain."),
            "test",
        ),
    ],
    (
        "One-shot SM90+ resident cluster scalar float/double reductions. T"
        "ables have peers*warps slots per destination. Repeated phase-zero"
        " use requires another generation protocol. Source helper permits "
        "lane-strided folds of tables larger than one warp."
    ),
    (
        "When a cluster reduction needs its result at one CTA or all CTAs,"
        " choose destination tables before selecting the final writer. Fou"
        "r warps in each of two FP32 CTAs deliver eight scalars: 32 expect"
        "ed bytes per destination. Root mode sends eight stores total; all"
        " mode sends sixteen. Reject mismatched byte counts, reused phase-"
        "zero barriers and peers leaving before remote completion."
    ),
    (
        "Source runtime/cluster.cu:140-319 and tests/cpp/cluster_runtime_t"
        "est/cluster_test_kernels.cu:82-112. Slot r*warps+w is injective; "
        "the destination selection changes replication but not contributor"
        " membership. Each final lane visits indices lane+32*j, which part"
        "ition the table. Compare CUTLASS replicated_partial_reduce@1: thi"
        "s rule adds last-CTA destinations and 8-byte scalar realization e"
        "xplicitly, not a silent binding."
    ),
    refines="reduce",
)
write(
    "rectangular_section_rasterization",
    (
        "A two-dimensional independent tile grid traversed by rectangular "
        "sections and then intra-section coordinates."
    ),
    [
        numeric("section_m", "Rows per traversal section."),
        numeric("section_n", "Columns per traversal section."),
    ],
    [("p.tile_rows >= 1 and p.tile_cols >= 1", "Independent output grid is nonempty.")],
    [
        (
            "p.independent_tiles and p.padded_coordinates_guarded",
            (
                "Tile permutation is legal and padded section coordinates are not "
                "executed as valid outputs."
            ),
        )
    ],
    (
        "Create a padded grid of ceil_div(tile_rows,section_m)*ceil_div(ti"
        "le_cols,section_n) sections. Decode section index in row-major or"
        "der, then intra-section coordinates in row-major order. Map m=sec"
        "tion_row*section_m+local_row,n=section_col*section_n+local_col. S"
        "kip coordinates m>=tile_rows or n>=tile_cols. Fuser split/reorder"
        " domains realize this traversal; output masks remain the underlyi"
        "ng tiled operator obligation."
    ),
    (
        "Forget traversal order and padding; every valid output tile has o"
        "ne section and one intra-section coordinate."
    ),
    [
        (
            "bijection",
            ("Euclidean division maps every valid tile to exactly one section/local coordinate."),
            "proof",
        )
    ],
    (
        "Independent rectangular output tiles, positive section sizes, gua"
        "rded padded sections. This adds independent M and N section facto"
        "rs beyond block_rasterization@1 grouped_m; it makes no cache-traf"
        "fic savings guarantee."
    ),
    (
        "When locality requires grouping both tile axes, choose separate M"
        "/N section sizes instead of one grouped-axis width. A 2x4 section"
        " groups eight nearby output tiles; a short final section must ski"
        "p padded coordinates. The optimizer varies section sizes while ti"
        "le extents come from upstream output tiling."
    ),
    (
        "For a valid (m,n), section=(m//section_m,n//section_n), local=(m%"
        "section_m,n%section_n) uniquely invert the construction. Source c"
        "src/scheduler/matmul_hopper+.cpp:220-290. Native scheduling follo"
        "ws split/reorder/merge with independent tail predication."
    ),
    refines="work_assign",
)
write(
    "multimem_peer_address_operation",
    (
        "A multicast virtual address applies one vector operation to corre"
        "sponding bytes in each registered peer allocation."
    ),
    [
        dict(
            name="operation",
            kind="structure",
            binding="compile",
            role="boundary",
            domain=["replicate_store", "sum_load"],
            meaning=(
                "Copy one vector to all registered peers, or sum corresponding FP3"
                "2 vectors from those peers."
            ),
        )
    ],
    [
        (
            "p.size_bytes % 16 == 0 and p.address_alignment_bytes >= 16",
            "Source realization uses full 16-byte vectors only.",
        )
    ],
    [
        (
            (
                "p.multicast_membership_matches_collective and p.peer_allocations_"
                "live and p.sm90_multimem_supported"
            ),
            "Virtual address membership, peer lifetime and hardware gates are explicit.",
        ),
        (
            ("p.all_sources_ready_before_operation and p.all_readers_retired_before_source_reuse"),
            "Upstream stream publication and downstream retirement protect peer payload.",
        ),
        (
            "p.sum_rounding_admitted or operation == 'replicate_store'",
            "SUM mode requires admitted peer reduction arithmetic.",
        ),
    ],
    (
        "Bind corresponding peer offsets through a multicast allocation. F"
        "or each independent 16-byte vector, replicate_store loads its bit"
        "s from one ordinary source and uses multimem.st.global.v4.f32 at "
        "the multicast destination, writing identical bits to every member"
        ". sum_load executes multimem.ld_reduce.global.add.v4.f32 at the m"
        "ulticast source and stores the resulting four FP32 sums to an ord"
        "inary destination. Peer ready and completion stream-memory operat"
        "ions bracket the source realization; a result at one rank versus "
        "all ranks is a caller consumer-set choice."
    ),
    (
        "Expand the multicast address to its member allocations: replicate"
        "_store copies identical source bits to every peer and sum_load ad"
        "ds each peer FP32 value once per coordinate."
    ),
    [
        (
            "membership",
            (
                "Multicast registration denotes exactly the intended peer addresse"
                "s for every vector."
            ),
            "test",
        ),
        (
            "payload_lifetime",
            ("Ready signals follow input writes and retirement signals follow all relevant reads."),
            "test",
        ),
        (
            "operation_values",
            (
                "Vector store preserves bits or vector load performs the requested"
                " peer SUM under the numerical policy."
            ),
            "test",
        ),
    ],
    (
        "NVIDIA SM90+ NVLink SHARP multicast allocations with explicit reg"
        "istered membership and full 16-byte vectors. SUM load is FP32 onl"
        "y; replicate store transports bit patterns. Distinct from CTA-clu"
        "ster shared multicast and contraction instruction selection."
    ),
    (
        "When peer buffers have corresponding offsets and a registered mul"
        "ticast address, replace software fanout stores or peer-load sums "
        "with the corresponding multimem address operation. Choose operati"
        "on from the collective contract, keep stream ready/retired handsh"
        "akes, and reject unregistered peers or scalar tails. Two peers co"
        "ntaining [1,2,3,4] and [10,20,30,40] produce [11,22,33,44] in SUM"
        " mode."
    ),
    (
        "Source runtime/multicast.cu:12-35, runtime/multicast_reduce.cu:14"
        "-42; selected caller csrc/multidevice/cuda_p2p.cpp:830-960. Corre"
        "ctness reduces to address membership, hardware operation semantic"
        "s and caller publication/retirement. No independent speed or nati"
        "ve rounding evidence is claimed."
    ),
    refines="transfer",
)
write(
    "cyclic_warp_row_skew",
    (
        "A warp-indexed shared tile cyclically shifts each warp row digit "
        "to vary its bank placement."
    ),
    [
        numeric("row_width", "Contiguous elements per row digit; divides the NVIDIA 32-lane warp."),
        dict(
            name="warp_skew",
            kind="numeric",
            binding="compile",
            role="software",
            domain=dict(lo=0, hi=31),
            meaning="Coefficient of warp index in cyclic row shift; source binds one.",
        ),
    ],
    [
        (
            "32 % row_width == 0 and warp_skew < 32 / row_width",
            "Within-warp row digits fit the lane decomposition.",
        )
    ],
    [
        (
            "p.writer_reader_use_same_skew and p.shared_lifetime_valid",
            "Writes and reads agree on the same address permutation and lifetime.",
        )
    ],
    (
        "Let R=32/row_width. A logical element indexed by chunk,warp,row,c"
        "olumn uses address chunk*p.block_threads+warp*32+((row+warp_skew*"
        "warp)%R)*row_width+column, with 0<=row<R and 0<=column<row_width."
        " Invert row=(stored_row-warp_skew*warp)%R. Choose warp_skew to ch"
        "ange peer-warp bank incidence without changing values; validate t"
        "he actual reader request footprint before claiming fewer bank pas"
        "ses."
    ),
    ("Invert the cyclic shift for each warp and recover the original logical shared tile."),
    [
        (
            "address_bijection",
            "Cyclic row addition is a permutation independently within each warp chunk.",
            "proof",
        ),
        (
            "bank_footprint",
            ("Chosen skew achieves its claimed bank behavior for the actual vector/read grouping."),
            "test",
        ),
    ],
    (
        "NVIDIA warp-indexed shared tiles with row_width dividing32 and ma"
        "tching read/write mapping. Source row_width8 gives R4,warp_skew1."
        " No universal conflict-free guarantee is encoded."
    ),
    (
        "When a cross-warp reduction reads same-bank rows from several war"
        "p partials, shift each warp row digit cyclically before storing. "
        "Choose skew zero for identity or one for the observed outer-stati"
        "stic layout; preserve the inverse reader map. This exposes additi"
        "on modulo row-count rather than mislabeling it an XOR swizzle."
    ),
    (
        "Source runtime/block_welford_outer.cu:157-195. For fixed chunk/wa"
        "rp, modular addition has a unique inverse, proving coordinate ide"
        "ntity. Bank benefit depends on the consumer transaction and remai"
        "ns an explicit check, not part of the permutation proof."
    ),
    refines="access",
)
write(
    "pointwise_slice_demand_pushdown",
    (
        "A slice after a pure pointwise producer narrows producer evaluati"
        "on to the requested input-coordinate domain."
    ),
    [
        dict(
            name="start",
            kind="numeric",
            binding="runtime",
            role="boundary",
            domain=dict(lo=0, hi=2147483647),
            meaning="First retained logical input coordinate; supplied by the slice contract.",
        )
    ],
    [
        (
            "start + p.output_length <= p.input_length and p.output_length >= 0",
            "Demand interval is within the producer output domain.",
        )
    ],
    [
        (
            ("p.producer_pure_pointwise_and_coordinate_local and p.input_mapping_preserved"),
            ("Each requested output depends only on corresponding producer input coordinates."),
        ),
        (
            (
                "p.other_consumers_keep_original_producer_domain and p.replay_nume"
                "ric_semantics_preserved"
            ),
            ("Shared producers are cloned as needed and arithmetic/casts retain their contract."),
        ),
    ],
    (
        "Replace slice(F(x),start,length) with evaluating the same pointwi"
        "se F only at original coordinates start+j for j in [0,length). Pr"
        "opagate that demanded coordinate interval backward through each p"
        "ure pointwise producer without changing its scalar operations. If"
        " another consumer needs the unsliced producer domain, keep its or"
        "iginal producer and clone the producer slice for this consumer. P"
        "adding is not pushed through F by this version; preserve the orig"
        "inal padding value with a separate guarded output path."
    ),
    (
        "Substitute the original coordinate start+j into the unchanged sca"
        "lar producer DAG to recover every sliced result, leaving other co"
        "nsumers unchanged."
    ),
    [
        (
            "coordinate_identity",
            ("Each retained output evaluates the same original-coordinate producer inputs."),
            "proof",
        ),
        (
            "consumer_isolation",
            ("Cloning/narrowing does not change another consumer required domain or values."),
            "test",
        ),
    ],
    (
        "Pure coordinate-local pointwise producer DAG followed by an in-bo"
        "unds contiguous slice. No reduction, RNG, mutation or arbitrary p"
        "ad-through-nonlinearity transformation. Native resize scheduler h"
        "as additional shape restrictions; this candidate records the obse"
        "rved slice mechanism."
    ),
    (
        "When a pointwise producer computes values later discarded by a sl"
        "ice, propagate the demanded coordinates backward instead of mater"
        "ializing the full producer output. For y[j]=F(x[j+L]), compute on"
        "ly that input interval. Clone a shared producer for the sliced co"
        "nsumer so an unsliced consumer still gets its full domain; do not"
        " move a padding constant through F."
    ),
    (
        "Source csrc/scheduler/resize.cpp:70-112,:397-411,:438-447. Coordi"
        "nate-locality proves each retained scalar DAG unchanged. Shared-u"
        "se cloning preserves the old domain for other consumers. operator"
        "_fusion@2 has no slice-demand field, so this is an explicit separ"
        "ate refinement."
    ),
    refines="map",
)
