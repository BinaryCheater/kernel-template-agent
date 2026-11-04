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
    "privatized_bin_accumulation",
    ("A per-worker bin table absorbs sample updates before combining nonzero bin totals globally."),
    [
        choice(
            "table_location", ["shared", "global_private"], "Where each CTA-private table lives."
        ),
        choice(
            "update_grouping",
            ["individual", "local_equal_runs"],
            ("Count each decoded bin or coalesce adjacent equal decoded bins within one thread."),
        ),
    ],
    [
        (
            ("p.sample_partition_exact and p.bin_decoder_consistent and p.invalid_bins_ignored"),
            "Each valid sample has one decoded bin and every private/final map agrees.",
        ),
        (
            (
                "p.private_tables_zeroed and p.final_table_zeroed and p.counts_fit"
                " and p.private_storage_fits"
            ),
            "Initial counter state and resource capacity are explicit.",
        ),
        (
            (
                "p.private_updates_atomic and p.private_updates_complete_before_fl"
                "ush and p.final_updates_atomic"
            ),
            "Concurrent updates and table-flush ordering are safe.",
        ),
    ],
    (
        "Partition samples among workers. Initialize one private counter t"
        "able per worker/channel. Decode each valid sample to a bin. Indiv"
        "idual mode atomically adds one to its private bin. local_equal_ru"
        "ns mode forms maximal equal-bin runs in each thread item sequence"
        " and atomically adds each run length once, ignoring invalid bins."
        " After all worker updates retire, flush nonzero private counters "
        "through the output-bin decoder with atomic additions to the initi"
        "alized global histogram. Shared/global-private tables realize the"
        " same count decomposition; source blended mode assigns location b"
        "y CTA parity."
    ),
    (
        "Sum each private count by logical output bin to recover exactly t"
        "he original sample histogram."
    ),
    [
        (
            "count_conservation",
            (
                "Every valid sample contributes exactly one, even when equal-bin r"
                "uns are compressed."
            ),
            "proof",
        ),
        (
            "flush_visibility",
            (
                "Private table publication precedes flush and final atomic sums re"
                "tain every worker contribution."
            ),
            "test",
        ),
    ],
    (
        "Integer sample histograms with consistent channel/bin mapping and"
        " bounded counters. RLE grouping is thread-local decoded-bin equal"
        "ity, not sorting and not global run encoding."
    ),
    (
        "When many samples contend on the same output bins, accumulate per"
        "-worker tables and flush counts. Choose shared/global-private sto"
        "rage and optionally coalesce adjacent equal bins inside each thre"
        "ad. Bins [2,2,5,5,5] become updates (2,+2),(5,+3); invalid bins c"
        "ontribute nothing. Extra private capacity and initialization trad"
        "e against fewer global atomics."
    ),
    (
        "Source agent_histogram.cuh:295-399,:646-675. Private sample sets "
        "partition inputs; replacing equal-bin +1 updates by one run-lengt"
        "h update preserves integer sum. Decoding private bins to final bi"
        "ns must commute with aggregation; native atomics/counter range re"
        "main obligations."
    ),
    "reduce",
)
add(
    "merge_path_partition",
    "Output-diagonal partitions of a stable merge of two sorted sequences.",
    [num("tile_items", "Merged outputs per parallel tile.")],
    [
        (
            "p.inputs_sorted_same_strict_weak_order and p.comparator_consistent",
            "Inputs are sorted using the same comparator.",
        ),
        (
            "p.diagonals_cover_output and p.partition_and_consumer_tile_sizes_equal",
            ("Partition launch and consumer agree on diagonal spacing including final tail."),
        ),
        (
            "p.first_input_wins_equivalent_ties and p.output_ranges_disjoint",
            "Tie rule is consistent across binary partition and serial merge.",
        ),
    ],
    (
        "For output diagonal d search i in [max(0,d-B_length),min(d,A_leng"
        "th)] with j=d-i. Binary search advances i when B[j-1] is not stri"
        "ctly less than A[i], giving first-input priority for equivalent k"
        "eys. Neighboring diagonals d,d+tile_items determine disjoint A/B "
        "slices. Each tile merges only those slices into its matching outp"
        "ut interval; optionally partition each tile again by per-thread d"
        "iagonals and serial merge short ranges. Clamp last diagonals to t"
        "otal length and move associated values with their key source indi"
        "ces."
    ),
    (
        "Concatenate output intervals in diagonal order to obtain the stab"
        "le global merge of the original two sequences."
    ),
    [
        (
            "stable_partition",
            (
                "Each diagonal identifies the unique stable-merge prefix and adjac"
                "ent intervals contain every source once."
            ),
            "proof",
        ),
        (
            "tie_and_payload",
            (
                "Equivalent-key priority and associated value identity agree betwe"
                "en search and local merge."
            ),
            "test",
        ),
    ],
    (
        "Two sorted random-access sequences, strict weak comparator, first"
        "-input-before-second equivalent ties, exact output-length arithme"
        "tic. Partition tile size must match any selected fallback consume"
        "r agent."
    ),
    (
        "When merging long sorted runs, partition by output diagonals so e"
        "ach executor receives a bounded amount of merged output even if o"
        "ne run dominates. Choose tile_items; search both input coordinate"
        "s at each diagonal and preserve the same tie priority during loca"
        "l merge. Reject a partition computed for a different fallback til"
        "e size."
    ),
    (
        "Source block_merge_sort.cuh:41-61; device/dispatch/dispatch_merge"
        ".cuh:106-134; agent_merge.cuh:180-270. The stable merge prefix is"
        " monotone on each diagonal. Its unique A/B counts invert prefix l"
        "ength; adjacent prefix differences partition both sources. Test e"
        "qual keys to distinguish tie bias."
    ),
    "map",
)
add(
    "stable_radix_digit_scatter",
    (
        "One stable digit partition using global bin bases, per-tile count"
        " prefixes and local stable ranks."
    ),
    [
        num("digit_bits", "Bits in one digit pass; legal source menu is backend-specific."),
        choice(
            "count_schedule",
            ["separate_upsweep", "onesweep_lookback"],
            "Precompute per-tile digit prefixes or publish them with online lookback.",
        ),
    ],
    [
        (
            (
                "p.key_twiddle_order_preserving and p.local_digit_ranks_stable and"
                " p.payload_follows_key"
            ),
            "Ordered key transform and stable local ranks preserve key/payload identity.",
        ),
        (
            "p.bin_totals_exact and p.tile_digit_prefix_exact and p.counts_fit",
            "Global bin bases and earlier-tile bin counts are exact.",
        ),
        (
            (
                "p.output_capacity_sufficient and p.tail_items_excluded and p.coun"
                "t_publication_visible"
            ),
            "All destinations exist and invalid tile items cannot appear as real keys.",
        ),
    ],
    (
        "For each digit bin b compute global base(b)=sum counts of smaller"
        " bins. Each tile computes its stable rank within bin and bin coun"
        "t. Destination of item x in tile t is base(b)+sum_{u<t} count(u,b"
        ")+rank_within_bin(x). Source may sort into shared memory by full "
        "local rank then subtract the tile-local digit base from the globa"
        "l offset. Separate upsweep materializes count prefixes; onesweep "
        "publishes PARTIAL digit counts, looks back to a COMPLETE prefix a"
        "nd then scatters within the same pass. The onesweep full-tile all"
        "-one-bin shortcut skips local ranking/scatter rearrangement but s"
        "till completes count-prefix publication. Repeating stable passes "
        "from least to most significant digit sorts the complete transform"
        "ed key."
    ),
    (
        "Erase count tables and scatter order: outputs are a stable partit"
        "ion by this digit, and repeated LSD passes give full transformed-"
        "key order."
    ),
    [
        (
            "position_bijection",
            (
                "Per-bin output intervals and stable within-bin ranks give unique "
                "complete destinations."
            ),
            "proof",
        ),
        (
            "prefix_visibility",
            "Online prefixes read completed matching counts and keep tile order.",
            "test",
        ),
        (
            "payload_stability",
            (
                "Equal digits preserve original logical order and values follow th"
                "e same key permutation."
            ),
            "test",
        ),
    ],
    (
        "Fixed-width ordered keys, stable digit ranks, bounded counters an"
        "d initialized output. Onesweep lookback progress/publication is a"
        " separate backend obligation; availability is not a speed claim."
    ),
    (
        "When integer or transformed scalar keys can be sorted by digits, "
        "scatter through bin base plus earlier-tile count plus local rank."
        " Choose separate count/scatter launches or online onesweep public"
        "ation. For digits [1,0,1] in tile0 and [0,1] in tile1, digit0 own"
        "s output positions0,1 and digit1 owns2,3,4 in original within-bin"
        " order. A homogeneous full tile can copy directly only after pres"
        "erving its count-prefix effects."
    ),
    (
        "Source agent_radix_sort_onesweep.cuh:321-356,:399-459,:514-570,:6"
        "90-713. Within each bin, tile prefixes partition the bin interval"
        " and stable local ranks partition each tile interval. Across bins"
        " the global bases partition output; LSD induction preserves lower"
        "-digit ordering inside equal higher-digit bins."
    ),
    "map",
)
add(
    "head_reset_scan_state",
    (
        "An ordered scan state carries head count and aggregate of the suf"
        "fix after the most recent head."
    ),
    [],
    [
        (
            "p.associative_value_operator and p.head_flags_match_segments",
            (
                "Value combine admits reassociation and head flags identify exact "
                "logical segment starts."
            ),
        ),
        (
            "p.scan_order_preserved and p.cross_tile_neighbors_match and p.counts_fit",
            "Ordered scan and boundary context do not invent/split segments.",
        ),
    ],
    (
        "Lift value operator op to pair (h,v), where h counts segment star"
        "ts and v is the suffix aggregate since the last start. Combine A "
        "then B as h=A.h+B.h and v=B.v if B.h>0, otherwise op(A.v,B.v). Sc"
        "an in original item order, carrying the same state across tile bo"
        "undaries. A nontrivial-run encoder derives head/tail flags from n"
        "eighboring values, feeds h=head&&!tail and v=!head||!tail, then u"
        "ses head-prefix counts to compact offsets and the previous suffix"
        " aggregate as completed length. Explicitly flush the final suffix"
        " and run count at end of input."
    ),
    (
        "Project each prefix state to its number of segment starts and the"
        " value aggregate of its final open segment."
    ),
    [
        (
            "lift_associativity",
            "The lifted operator is associative but generally noncommutative.",
            "proof",
        ),
        (
            "boundaries_and_flush",
            (
                "Tile neighbors and final suffix handling preserve each original s"
                "egment exactly once."
            ),
            "test",
        ),
    ],
    (
        "Ordered segmented scans with an associative value operator and ex"
        "act head markers. Nontrivial RLE binding ignores singleton runs v"
        "ia paired head/tail predicates; ordinary scan-by-key uses all seg"
        "ment heads instead."
    ),
    (
        "When a scan must reset accumulated values at segment starts, lift"
        " the value to head-count plus suffix aggregate. The right state r"
        "eplaces the left suffix exactly when it contains a new head; othe"
        "rwise extend it. This permits parallel ordered scans and cross-ti"
        "le carry without treating resets as commutative reduction. A run "
        "crossing a tile boundary must remain one run, and the final suffi"
        "x needs an explicit flush."
    ),
    (
        "Source thread/thread_operators.cuh:312-380; agent/agent_rle.cuh:3"
        "53-434,:448-515,:1026-1078. Three-state associativity: if C has a"
        " head both parenthesizations choose C.v; else if B has a head bot"
        "h choose op(B.v,C.v); else both choose op(op(A.v,B.v),C.v), equal"
        " by value associativity. Head counts add independently. Reverse o"
        "rder changes the suffix."
    ),
    "scan",
)
add(
    "bounded_candidate_materialization",
    (
        "A filter pass uses compact candidate storage when it fits, otherw"
        "ise replays its predicate over immutable original inputs."
    ),
    [num("capacity", "Maximum candidate identities held by the temporary buffer.")],
    [
        (
            (
                "p.original_inputs_immutable and p.active_predicate_replayable and"
                " p.original_identity_preserved"
            ),
            (
                "Original values and identities can be filtered again with exactly"
                " the current prefix predicate."
            ),
        ),
        (
            ("p.buffer_exact_when_selected and p.capacity_test_uses_actual_candidate_count"),
            "Buffer fallback never truncates or mistakes capacity for candidate count.",
        ),
    ],
    (
        "At a refinement pass maintain an active membership predicate over"
        " original item identity. If the previous active count fits capaci"
        "ty and its compact buffer is complete, scan buffered keys plus or"
        "iginal indices. Otherwise scan original input and reevaluate the "
        "same active predicate. Apply the next filter/histogram only to ac"
        "tive items. Preserve original indices when fetching values. The f"
        "irst pass uses original inputs; later passes may switch represent"
        "ation without changing membership."
    ),
    (
        "Erase the selected physical input representation; every pass sees"
        " exactly the active original occurrences."
    ),
    [
        (
            "membership_identity",
            (
                "Buffered and replayed paths enumerate the same original active oc"
                "currences exactly once."
            ),
            "proof",
        ),
        (
            "capacity_and_lifetime",
            (
                "Only complete in-capacity buffers are read and original data rema"
                "in live through every replay."
            ),
            "test",
        ),
    ],
    (
        "Iterative selection/refinement whose active set is defined by a c"
        "ompact replayable predicate and immutable original input. Candida"
        "te buffer capacity is a resource decision, not a semantic bound o"
        "n input length."
    ),
    (
        "When boundary candidates may exceed scratch capacity, keep origin"
        "al identities and fall back to replaying the current prefix predi"
        "cate over original input. Choose capacity without discarding over"
        "flow candidates. Radix top-k may scan the original array again wh"
        "en previous_len exceeds buffer_length; it must not process only t"
        "he truncated buffer."
    ),
    (
        "Source agent_topk.cuh:595-646 and following filter/histogram path"
        "s. The complete buffer is a bijective materialization of active o"
        "riginal occurrences. Replacing it by predicate replay preserves m"
        "embership, so subsequent histogram/rank operations are unchanged."
        " Performance differs through repeated input reads."
    ),
    "work_assign",
)
add(
    "resident_overflow_repeated_scan",
    (
        "A repeated predicate scan keeps a fixed input window resident and"
        " re-streams the remaining immutable chunks."
    ),
    [
        num("resident_chunks", "Input chunks retained across repeated passes."),
        num("stream_slots", "Reusable temporary chunk slots for overflow scans."),
    ],
    [
        (
            "p.resident_and_overflow_partition_input and p.original_inputs_immutable",
            "Resident and streamed chunks cover the original input once per pass.",
        ),
        (
            ("p.current_predicate_applied_to_both_regions and p.original_coordinates_preserved"),
            "All regions apply the same active predicate and retain original identities.",
        ),
        (
            "p.stream_slot_completion_before_read_and_reuse and p.resident_storage_fits",
            (
                "Asynchronous loads complete before consumption and every slot ret"
                "ires before refill."
            ),
        ),
    ],
    (
        "Partition each worker input chunks into resident window and overf"
        "low. Load resident data once. For each iterative radix/predicate "
        "pass, consume resident values and stream all overflow through bou"
        "nded reusable slots, applying the same current predicate. Residen"
        "t histogram work may overlap the first overflow load. Source alte"
        "rnates overflow traversal direction between passes so turnaround "
        "chunks already in slots need not be reloaded. If final ties requi"
        "re ascending original indices, retain the low-index first-visited"
        " window; descending ties retain the high-index window. Final filt"
        "ering observes that required original-coordinate order rather tha"
        "n slot arrival order."
    ),
    (
        "Forget resident/stream slot placement and traverse each original "
        "occurrence once under the pass predicate; the same histogram or s"
        "elected set is obtained."
    ),
    [
        (
            "pass_cover",
            (
                "Each pass consumes all resident and overflow chunks once with mat"
                "ching predicates and no stale slot generation."
            ),
            "proof",
        ),
        (
            "tie_order",
            (
                "Deterministic final traversal preserves requested original-index "
                "preference independent of streaming order."
            ),
            "test",
        ),
        (
            "slot_lifetime",
            (
                "Load completion, consumption and next-generation refill of every "
                "stream slot are ordered."
            ),
            "test",
        ),
    ],
    (
        "Repeated scans over immutable chunks with a fixed retained window"
        " and bounded asynchronous overflow slots. Histogram passes are or"
        "der-independent; deterministic final filtering requires explicit "
        "original-index ordering. No arbitrary reversal of an ordered scan"
        " is allowed."
    ),
    (
        "When repeated radix passes outgrow resident shared memory, keep a"
        " prefix or suffix window and re-stream the rest through bounded s"
        "lots. Choose resident capacity and stream slot count; derive fina"
        "l visit order from the tie policy. Reversing histogram sweeps can"
        " reuse turnaround chunks, while deterministic filtering must stil"
        "l select original indices in its requested order."
    ),
    (
        "Source agent_batched_topk_cluster.cuh:253-290,:879-922,:2895-2907"
        ",:2919-2984,:3137-3217. Resident and overflow chunk sets partitio"
        "n original coordinates. Predicate equality makes their union hist"
        "ogram identical to a full scan; stream generations and final inde"
        "x order are separate obligations."
    ),
    "work_assign",
)
add(
    "duplicate_balanced_set_partition",
    (
        "A merge-diagonal partition adjusted inside equal-key runs so mult"
        "iset pairs stay consistently assigned."
    ),
    [num("tile_inputs", "Nominal total input diagonal spacing per parallel tile.")],
    [
        (
            ("p.inputs_sorted_same_strict_weak_order and p.equivalent_key_classes_consistent"),
            "Both sequences share comparator and equality classes.",
        ),
        (
            (
                "p.balanced_boundary_algorithm_and_consumers_match and p.output_mu"
                "ltiplicity_contract_explicit"
            ),
            ("All boundaries and local set operations use the same duplicate pairing convention."),
        ),
        (
            "p.output_prefixes_exact and p.payload_source_matches_operation",
            (
                "Output compaction and key/value source identity match union/inter"
                "section/difference."
            ),
        ),
    ],
    (
        "At diagonal d find ordinary MergePath (i,j). If j<B_length, let x"
        "=B[j] and find lower bounds a0,b0 of its equal run in the visited"
        " A/B prefixes. Let total=(i-a0)+(j-b0), advanceB=max(total//2,tot"
        "al-(i-a0)), bounded by the available B equal-run end found by upp"
        "er_bound in the source search range. Set advanceA=total-advanceB."
        " If advanceA=advanceB+1 and another equal B remains, set star=1; "
        "otherwise star=0. Return A_boundary=a0+advanceA, B_boundary=d-A_b"
        "oundary+star. Neighboring corrected boundaries define disjoint so"
        "urce subranges. Apply the requested serial multiset operation loc"
        "ally and compact surviving outputs through exact output-count pre"
        "fixes. Intersection/difference output payloads come from A; union"
        "/symmetric difference preserve their declared source multipliciti"
        "es."
    ),
    (
        "Concatenate local outputs to recover the requested sorted multise"
        "t operation, with equal-key counts paired consistently across par"
        "tition boundaries."
    ),
    [
        (
            "duplicate_pairing",
            (
                "Corrected boundaries do not separate an equal-key matched pair in"
                " a way that changes the requested multiplicity."
            ),
            "proof",
        ),
        (
            "count_and_payload",
            (
                "Output prefixes compact survivors once and payload source follows"
                " the operation contract."
            ),
            "test",
        ),
    ],
    (
        "Two sorted sequences and the explicit balanced-path correction us"
        "ed by CCCL; integer coordinate packing must fit selected backend "
        "fields. This differs from a stable merge partition, which emits e"
        "very duplicate rather than pairing equal occurrences."
    ),
    (
        "When parallel set operations encounter duplicate runs crossing ti"
        "le diagonals, rebalance the cuts through equal-key prefixes befor"
        "e serial intersection/difference/union. Choose input diagonal spa"
        "cing, not independent A/B cuts. A run with three A copies and two"
        " B copies contributes two intersection outputs and one A-minus-B "
        "output even when a partition crosses its middle."
    ),
    (
        "Source agent_set_operations.cuh:107-150,:156-207,:578-637. Interp"
        "ret equal occurrences as paired by ordinal across A and B. The st"
        "air-step correction assigns matched pairs consistently; the star "
        "advances the B side of an otherwise split pair. Native biased-sea"
        "rch and packed-coordinate realization need the explicit witness i"
        "n checks/design_witnesses.py; general proof remains reviewer-visi"
        "ble."
    ),
    "map",
)
add(
    "inplace_neighbor_boundary_checkpoint",
    ("Snapshot cross-tile input neighbors before any in-place stencil output can overwrite them."),
    [choice("neighbor", ["left", "right"], "Which adjacent input outside a tile is required.")],
    [
        (
            "p.tile_partition_exact and p.input_output_alias_only_same_elements",
            "Tiles cover the input with the admitted in-place aliasing.",
        ),
        (
            (
                "p.checkpoint_launch_completes_before_output_writes and p.checkpoi"
                "nt_live_through_last_read"
            ),
            "A preceding checkpoint pass preserves boundary input values.",
        ),
        (
            "p.tile_inputs_loaded_before_local_overwrite and p.endpoint_policy_explicit",
            (
                "Within-tile inputs are captured before stores and global endpoint"
                "s follow the caller rule."
            ),
        ),
    ],
    (
        "For every tile boundary snapshot the preceding input element for "
        "left-neighbor differences, or the following tile first element fo"
        "r right-neighbor differences. Complete the checkpoint pass before"
        " any tile writes in-place outputs. Each CTA loads its own tile in"
        "to local values; internal neighbors come from those captured inpu"
        "ts, cross-tile neighbor comes from checkpoint storage, and the gl"
        "obal edge uses the declared endpoint policy. Then store outputs o"
        "ver the original input allocation."
    ),
    (
        "Replace checkpoint and captured tile reads by their original pre-"
        "write input values, recovering an out-of-place adjacent-neighbor "
        "operation."
    ),
    [
        (
            "snapshot_identity",
            ("Each cross-tile neighbor equals the original input before any output overwrite."),
            "proof",
        ),
        (
            "local_capture",
            ("All local inputs used by a tile are captured before its destructive output stores."),
            "test",
        ),
    ],
    (
        "One-dimensional adjacent-neighbor operations on disjoint tiles wi"
        "th exact in-place input/output aliasing. Arbitrary overlapping sh"
        "ifted allocations or larger stencil halos need a different checkp"
        "oint mapping."
    ),
    (
        "When an in-place adjacent difference reads across CTA boundaries,"
        " save just those cross-tile neighbors before launching the output"
        " pass. Choose left or right direction; avoid retaining the full i"
        "nput. A boundary after index3 must read original x3 for output4 e"
        "ven if the previous CTA already replaced x3."
    ),
    (
        "Source agent_adjacent_difference.cuh:130-245 and AgentDifferenceI"
        "nit below it. Global checkpoint-before-write order preserves all "
        "hazardous remote reads; local tile capture preserves the remainin"
        "g reads. This is input preservation, not adjacent_boundary_reuse@"
        "1 endpoint function optimization."
    ),
    "map",
)
add(
    "monotone_first_match_pruning",
    ("An atomic best index bounds increasing tile searches for the first matching input."),
    [],
    [
        (
            (
                "p.request_is_first_matching_index and p.tile_indices_increase_per"
                "_worker and p.predicate_deterministic"
            ),
            (
                "Search requests a minimum index and every worker future range has"
                " a known monotone lower bound."
            ),
        ),
        (
            (
                "p.best_initialized_to_length and p.best_updates_atomic_min and p."
                "tile_search_finds_local_first"
            ),
            "Published best is always a valid found index or sentinel.",
        ),
    ],
    (
        "Initialize global best=N. Workers visit disjoint increasing index"
        " tiles. Before a tile with lower bound L read global best; if bes"
        "t<L, exit because no future index can improve it. Otherwise evalu"
        "ate the tile predicate, reduce to its earliest match, atomically "
        "minimize global best and exit that worker on a found tile. A stal"
        "e relaxed best may postpone pruning but cannot suppress an improv"
        "ing match. Sentinel N means no match after all workers complete."
    ),
    (
        "Erase skipped ranges whose indices exceed a witnessed matching in"
        "dex; remaining atomic minima equal the earliest matching original"
        " input."
    ),
    [
        (
            "safe_pruning",
            ("Every skipped index is strictly greater than an already found valid matching index."),
            "proof",
        ),
        (
            "coverage",
            (
                "Every potentially improving index is searched unless a smaller va"
                "lid match was already published."
            ),
            "test",
        ),
    ],
    (
        "First-index queries over immutable inputs with deterministic pred"
        "icate, disjoint increasing worker tiles and a globally atomic min"
        "imum. Does not justify pruning arbitrary find-any or max-value se"
        "arches."
    ),
    (
        "When looking for the first matching index, use the best published"
        " index to stop workers whose future tiles start later. A relaxed "
        "stale bound costs extra work and preserves correctness because be"
        "st only decreases. Preserve monotone worker order; an arbitrary l"
        "ater-index match cannot justify skipping an earlier unseen tile."
    ),
    (
        "Source agent_find.cuh:150-220. Best monotonically decreases throu"
        "gh actual predicate matches. If best<L all future worker indices>"
        "=L are worse; otherwise search local tile and contribute its mini"
        "mum. Atomic linearization plus complete worker coverage yields th"
        "e global minimum."
    ),
    "map",
)
add(
    "binned_floating_accumulator",
    (
        "A floating sum uses exponent-indexed primary/carry collectors ins"
        "tead of one floating scalar."
    ),
    [
        num(
            "fold",
            (
                "Number of primary collectors, paired with the same number of carr"
                "y collectors; device dispatch currently binds three."
            ),
        )
    ],
    [
        (
            (
                "p.input_is_finite_float_or_double and p.requested_sum_contract_ad"
                "mits_binned_rounding"
            ),
            "Input and result use the specified finite summation recipe.",
        ),
        (
            ("p.bin_width_matches_type and p.native_rounding_and_bit_operations_match_recipe"),
            ("Float uses 13-bit bins, double 40-bit bins with the source bit-deposition recipe."),
        ),
        (
            ("p.collector_range_and_renormalization_bound_valid and p.merge_aligns_bin_indices"),
            ("Collector shifts, carries and merged states respect bin/endurance invariants."),
        ),
    ],
    (
        "Represent a partial sum using fold primary and fold carry floatin"
        "g values with an exponent-indexed bin anchor. Before deposit rebi"
        "n to accommodate the maximum magnitude: shift retained collectors"
        " toward lower-priority bins and initialize newly exposed bins. De"
        "posit an input by forced-low-bit floating additions into each pri"
        "mary collector, propagating the residual to the next. The highest"
        "-range bin uses the source compression/expansion path. Renormaliz"
        "e primary carry bits into the carry vector before the endurance b"
        "ound is exceeded. Merge two partial accumulators by aligning bin "
        "indices, combining corresponding primary/carry components and ren"
        "ormalizing. Convert to output scalar only after the complete part"
        "ial merge. Source device dispatch binds fold=3; manually batched "
        "unsafe deposits additionally require a declared magnitude bound a"
        "nd bounded count before renorm."
    ),
    (
        "Interpret each binned state through its primary/carry decoding re"
        "cipe; completed state represents the admitted reproducible sum of"
        " its deposited inputs."
    ),
    [
        (
            "encoded_sum",
            (
                "Rebin, deposit, merge and renorm preserve the specified encoded-s"
                "um interpretation within its numerical error contract."
            ),
            "proof",
        ),
        (
            "order_independence",
            (
                "For the admitted range/endurance, changing partition or merge ord"
                "er yields the declared reproducibility guarantee."
            ),
            "test",
        ),
        (
            "conversion",
            "Final scalar conversion follows the specified bin/carry rounding sequence.",
            "test",
        ),
    ],
    (
        "Finite float/double sum and the exact CCCL binned recipe, 13/40-b"
        "it bins, default fold3 in device dispatch. This is not an arbitra"
        "ry exact superaccumulator, not scalar Kahan summation, and not a "
        "claim of correctly rounded real summation. Full numerical proof/n"
        "ative checks remain open obligations."
    ),
    (
        "When reproducibility must survive changes in reduction partition "
        "rather than merely fixing a tree, select an exponent-binned parti"
        "al-state representation. The source device path uses three primar"
        "y and three carry collectors, paying more state and arithmetic th"
        "an a scalar sum. Preserve bin alignment and renormalization endur"
        "ance; replacing its low-bit deposit by ordinary addition changes "
        "the algorithm."
    ),
    (
        "Source detail/rfa.cuh:45-103,:249-402,:409-463,:645-688; selected"
        " dispatch/dispatch_reduce_deterministic.cuh:52-95 and kernels/ker"
        "nel_reduce_deterministic.cuh:130-198. Collector decoding and nume"
        "rical reproducibility require independent analysis/native witness"
        "es; the proposal records the actual design and its unresolved num"
        "erical obligations, rather than assuming scalar associativity."
    ),
    "reduce",
)
add(
    "size_class_cooperative_assignment",
    (
        "Independent tasks are partitioned by size and assigned thread-, s"
        "ubgroup- or CTA-cooperative executors."
    ),
    [
        num("thread_threshold", "Maximum task size in the thread-cooperative class."),
        num("subgroup_threshold", "Maximum size in the subgroup-cooperative class."),
        num("subgroup_width", "Threads collaborating on one middle-class task."),
    ],
    [
        (
            "thread_threshold <= subgroup_threshold and p.size_classification_consistent",
            "Size classes are ordered, exhaustive and use one size unit.",
        ),
        (
            (
                "p.task_ids_preserved and p.class_worklists_exact_once and p.coope"
                "rative_lane_maps_partition_task"
            ),
            ("Partitioning and cooperative item maps preserve every original task exactly once."),
        ),
        (
            "p.scratch_and_launch_legal and p.task_outputs_independent",
            ("Cooperative resources/participants are legal and task outputs do not conflict."),
        ),
    ],
    (
        "Classify each task by its actual or safe upper-bound size: thread"
        " class up to thread_threshold, subgroup class up to subgroup_thre"
        "shold, otherwise CTA class. Preserve original task IDs in class w"
        "orklists. A subgroup lane l handles offsets l+j*subgroup_width; C"
        "TA thread l uses l+j*block_threads; a single thread iterates its "
        "whole tiny task. Locally service smaller classes and enqueue larg"
        "er tasks when the source uses a separate kernel; allocate queue s"
        "lots by exact count prefixes. An alternative uniform launch may c"
        "hoose one class for all tasks using a declared maximum task size."
        " Operator-specific reduction/scan/copy is unchanged inside each t"
        "ask."
    ),
    (
        "Erase class queues and executor grouping to recover the original "
        "independent per-task computations over all assigned item offsets."
    ),
    [
        (
            "task_cover",
            (
                "Every task enters one size class and each of its items is assigne"
                "d to one cooperating lane."
            ),
            "proof",
        ),
        (
            "queue_and_resources",
            "Class queues, lane masks and per-task scratch match the selected launch.",
            "test",
        ),
    ],
    (
        "Independent variable-size tasks with legal cooperative bodies and"
        " explicit size units. Classification threshold is a scheduling po"
        "licy, not a numerical change. Subgroup collectives require their "
        "own full-participation contract."
    ),
    (
        "When a batch mixes tiny and large tasks, choose size thresholds a"
        "nd cooperative width rather than giving every task a full CTA. Pr"
        "eserve descriptor identities through partitioning. Memcpy may ser"
        "vice tiny buffers locally and queue large ones; segmented reducti"
        "on can choose a subgroup launch from a safe maximum segment size."
        " Reject queues that lose task identity or lanes that execute anot"
        "her task collective."
    ),
    (
        "Source agent_batch_memcpy.cuh:595-650,:1030-1123; kernels/kernel_"
        "segmented_reduce.cuh:130-227. Class intervals partition sizes; mo"
        "dular lane assignment partitions task offsets. Exact prefix queue"
        " allocation gives distinct slots. Each body retains its own value"
        " and synchronization obligations."
    ),
    "work_assign",
)
add(
    "sparse_lane_rank_reduction",
    (
        "A masked warp reduction communicates by logical member rank rathe"
        "r than physical XOR lane distance."
    ),
    [
        choice(
            "result_to",
            ["first_member", "all_members"],
            "Expose only the first logical member or broadcast the completed value.",
        )
    ],
    [
        (
            (
                "p.member_mask_nonempty_and_same_for_all_members and p.logical_ran"
                "k_ascending_physical_lanes"
            ),
            "Participating lanes share one mask and rank by set-bit order.",
        ),
        (
            "p.operator_associative_under_contract and p.every_member_converges",
            "Ordered grouping is legal and all mask members execute each shuffle.",
        ),
    ],
    (
        "For physical lane l in mask M define rank=popcount(M & lanes_belo"
        "w(l)), n=popcount(M). First reduce each lane local items in order"
        ". At strides 1,2,4,...<next_power_of_two(n), fetch the current va"
        "lue of member rank+stride through the masked rank-to-lane shuffle"
        "; combine only if that member exists. First member obtains the co"
        "mplete ordered reduction. In all_members mode broadcast from the "
        "least physical set bit of M. Do not use physical lane+stride or a"
        "n XOR mask for a sparse group."
    ),
    (
        "Erase masked shuffles; the first member or every member receives "
        "the ordered reduction of all original members in increasing logic"
        "al rank."
    ),
    [
        (
            "rank_interval",
            (
                "After stride s each rank summarizes its ordered suffix interval o"
                "f at most 2*s members; the first rank eventually covers all membe"
                "rs."
            ),
            "proof",
        ),
        (
            "masked_shuffle",
            "Rank-to-physical-lane mapping and convergence preserve group membership.",
            "test",
        ),
    ],
    (
        "Nonempty subsets of one NVIDIA warp, including noncontiguous lane"
        "s and non-power-of-two member counts. Associativity is required, "
        "commutativity is not; this is not cross_lane_reduction@1 fixed-wi"
        "dth XOR butterfly."
    ),
    (
        "When predicate partitioning leaves sparse warp members, reduce ov"
        "er set-bit rank instead of physical neighbors. For lanes {1,4,9},"
        " logical ranks0,1,2 must combine A,B,C in that order. A three-mem"
        "ber group needs no padding lane to contribute. Bind first-member "
        "or all-member result policy explicitly."
    ),
    (
        "Source cudax coop/__group/mapping/binary_partition.cuh:ballot/pop"
        "count map; coop/__algorithm/reduce.cuh:316-345 and shuffle_down.c"
        "uh masked-group route. Stride induction concatenates adjacent log"
        "ical-rank intervals; invalid right intervals are absent. Native m"
        "asked shuffle mapping remains an executable obligation."
    ),
    "reduce",
)
