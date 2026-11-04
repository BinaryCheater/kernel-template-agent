# Parallel chunk factors before a serial state chain

When a small sequence/head grid leaves little parallel state work, move the
state-independent chunk preparation off that chain. The new
[state_independent_chunk_factors@1](delta/state_independent_chunk_factors@1/template.yaml)
and [outline](delta/state_independent_chunk_factors@1/proof.yaml) choose on-chain
work or a complete prepared workspace. They compose with
[chunked_delta_feedback@2](delta/chunked_delta_feedback@2/template.yaml), rather
than pretending the recurrent state itself becomes independent across chunks.

Source paths refer to [retained FlashInfer](inputs/sources.json), particularly
`flashinfer/kda_kernels/kda_chunked_bt16.py`. Its SM100-family BT16 design uses
K=V128 and two64-row value shards per state chain. From the feedback rule, let
R=T*(V-EK*H_operand), H_operand be the declared state conversion, and KR be the
outgoing insertion key matrix. Prepare W=KR^T*T and P=C*T, plus EK,EQ,E_end.
Then a chain uses Z=V-EK*H_operand,
Y=(EQ*H_operand+P*Z)*scale, H_next=diag(E_end)*H+W*Z.
W/P do not depend on incoming state or V. Original V remains a chain input;
there is no need to store value-dependent U/O factors for this organization.

| Concrete source decision | Rule field / reuse | Evidence |
|---|---|---|
| `:7609-7705`: weighted inverse transpose times KR stages W halves; `:7960-8158`: KK/L/inverse, W halves, causal P, EK/EQ and diagonal records | `factor_supply=prepared_workspace`, `pre_independence` | Prep accepts Q/K/gates/beta and metadata; serial H and V are absent. Two W producers own disjoint key-column halves and retire their asynchronous stores before scratch reuse |
| `:6516-6529`: P=C*T stored directly in pairwise SW32 physical order | `pre_layout`; seed compute_instruction_selection@2 `fragment_layout` remains an additional lowering obligation | A linear512-byte global-to-shared TMA copy must yield the native pairwise tile, not ordinary row-major coefficients |
| `:7490-7585` uses `tcgen05_decay_b_key_storage_dim_runtime`; helper`:547-565` returns key^8 | `pre_layout`, distinct reader agreement | The header near6358 still advertises an extra token-dependent XOR. The actual current helper removed it because the state MMA reader displaced key channels. Do not reconstruct the binding from that stale header |
| `:9580-9595`: generated packed metadata is a separate prior launch; `:9611-9665`: host_prep then host_chain_dv2 on one stream | `pre_publish` | Complete preparation is the cross-kernel publication boundary. The removed chain-first/two-stream flag ring is not retained as a supported alternative |
| `:8642-8705`: one CTA per sequence/head/value-half, common factor reads, disjoint V/output/state rows | `pre_chain` | Factor work is shared by two value shards; each shard still traverses its own ordered chunk prefix. It cannot process arbitrary state chunks out of order |
| `:9200-9375`: prior projection/update completion before packing/rescaling next state; checkpoint readers retire before replacement | `pre_numeric`, `pre_chain` | State operand conversion and carried FP32 update have different stages. Factoring W/P changes low-precision rounding; ideal associativity is not native bitwise equivalence |
| `:9600-9609`: decomp iff n_seq*heads*2<=SM_count unless compile-time forced | `pre_backend`, source tuning policy | It estimates whether doubled state grid fits a wave. It is not a proof that prepared mode is faster or valid for every shape/target |

The [focused check](checks/check_prepared_chunk_factors.json) covers32 prepared
one-/two-chunk comparisons with an independent sequential channelwise recurrence.
It varies state and V while reusing the same Q/K/gate factors, making the moved
dependency boundary explicit. Exact rational products do not check native factor
rounding or asynchronous retirement.

Labeled coordinate checks cover2048 decay entries,2048 W entries and256 pairwise
entries. Decay workspace key^8, W token^8 and SW32 pairwise maps are bijections.
The obsolete token-dependent decay map disagrees with the current state-reader
binding on1024 entries. This checks the declared workspace coordinate composition;
actual TMA descriptors, ldmatrix registers and tcgen05 lowering remain native
obligations. A locally self-consistent writer/auxiliary-reader pair alone would
miss the second-reader displacement.

Logical retained payload per chunk/head is13312 bytes for two-byte operands:
three16x128 records(EK,EQ,W), one16x16 P and128 FP32 diagonal entries. This omits
workspace capacity slack, alignment, metadata, original V and state. The actual
useful packed chunk prefix must fit the allocated capacity; its counts and
sequence origins must match in prep and chain. This is the same useful-versus-
capacity distinction as
[runtime_partition_capacity@1](delta/runtime_partition_capacity@1/template.yaml),
but its prepared attention partitions are not directly this recurrent work map.

Existing materialization rules do not supply this choice unchanged:
[decoded_tile_materialization@1](delta/decoded_tile_materialization@1/template.yaml)
fixes code/metadata decoder identity and shared/register decode sites;
[affine_activation_rematerialization@1](../flash-attention/delta/affine_activation_rematerialization@1/template.yaml)
fixes forward/backward checkpoint boundaries. Neither expresses Q/K-only factor
hoisting, folded W/P with original V, or a serial state chain after parallel
chunk preparation. Their lifetime concerns recur, while the new rule owns this
dependency split and its concrete payload/publication contract.

This disposes the independent-prep/serial-chain design. Persistent dispatch,
small-head execution, frozen output/checkpoint layouts, context-parallel state
propagation and reverse/training designs remain separate representatives in the
[recurrent inventory](recurrent-inventory.md).
