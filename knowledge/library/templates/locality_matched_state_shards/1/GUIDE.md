# locality_matched_state_shards@1: independent state rows with matched memory and execution domains

## When to consider

When recurrent value rows are independent and state traffic dominates, split every state by value row, place halves in matching memory domains and bind execution to those same domains. Keep the full key axis in each shard and join both completions. A current-stream fallback preserves results but carries no domain-local traffic promise.

## Computation before and after

Independent value-row state shards with matching physical memory and execution locality plus complete join.

Allocate domain d state half[slots,HV,64,128] in memory locality domain d for d0/1, retaining padded domain stride and owner. Expose logical pool[slots,HV,2,64,128] where original rowv=64*d+r. Domain launch d selects pool[:,:,d],V[...,64*d:64*(d+1)],output same slice, with identical request slot/head/key/query/gate metadata and complete key dimension. matched_domain_streams records one ready/fork event on main; both locality-restricted streams wait, run their row kernels, record completion, and main waits both. ordered_current_stream runs the same two slices serially on the current stream. Each kernel performs full row prediction/correction/readout and writes only its owned state/output rows. No state combine is needed because rows are independent; consumers still wait for both halves. Declare locality only for matched memory/execution domains under the target contract, not for all replicated inputs or for ordered fallback. Preserve mappings and lifetimes across graph capture and allocator release.

Recover the original computation: Erase domain/stream labels and place each shard row at64*d+r. The two disjoint row sets cover the full original state/output, each using every key term and original inputs. Fork/join makes their completion equivalent to one completed full-row update under the admitted per-row numerical recipe.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| execution_mode | compile / software | ["matched_domain_streams", "ordered_current_stream"] | Use per-domain restricted streams plus fork/join for locality, or execute the two functional shards on the current stream when host launch overhead dominates. Only matched mode supplies the locality placement/execution claim. |

## Coupled constraints and scope

Rubin source ugpu_state_pool and _run_rubin_t1_ugpu in FlashInfer gdn_decode.py allocate BF16 two-domain value shards and run the SM107 T1 complete-row kernel on each64-row slice. Matched green-context streams are used under CUDA graph capture and above the source eager threshold; smaller eager calls run both slices on current stream. Threshold8192 batch*value_heads is cost policy, not semantic legality. Source hardware traffic/performance claims are not measurements of this study. KV/key-axis partition, arbitrary nonlinear coupling, atomic whole-pool commit and generalized NUMA topology are outside this first version.

Scope predicate: `(p.domains == 2 and p.value_features == 128 and p.key_features == 128 and p.tokens_per_request == 1) and (p.domain_allocation_bytes >= p.slot_capacity * p.value_heads * 64 * 128 * 2 and p.slot_capacity >= 1 and p.value_heads >= 1) and (p.value_row_state_updates_independent_and_each_shard_keeps_complete_key_axis) and (p.state_shards_physically_allocated_in_corresponding_locality_domains) and (p.pool_slot_head_value_and_output_slice_maps_agree) and (p.shared_key_query_gate_inputs_identical_and_live_through_both_shards) and (execution_mode == 'ordered_current_stream' or p.green_context_execution_domain_matches_memory_and_backfill_contract) and (p.main_input_event_precedes_all_shards_and_all_shard_completions_precede_consumers) and (p.mutable_state_slots_unique_and_no_cross_request_seed_write_overlap) and (p.state_inputs_and_localized_allocation_owner_live_through_all_stream_and_graph_users) and (p.per_row_reduction_fma_conversion_and_store_recipe_admitted) and (p.two_domain_locality_allocation_context_and_shard_kernel_available)`.

- Precondition `pre_independence` (implementation): `p.value_row_state_updates_independent_and_each_shard_keeps_complete_key_axis` — A scalar-gated delta decode updates each value/state row independently once shared key/query/gate inputs are fixed. Each domain owns half the value rows and all128 key columns; no inter-domain key partial reduction is omitted.
- Precondition `pre_allocation` (implementation): `p.state_shards_physically_allocated_in_corresponding_locality_domains` — Two physical locality-domain allocations hold state rows0..63 and64..127 for every slot/head. Domain stride includes allocator-granularity padding. Permuted/select views retain actual strides and allocation owner; reshaping as a dense ordinary pool is not equivalent.
- Precondition `pre_mapping` (implementation): `p.pool_slot_head_value_and_output_slice_maps_agree` — Domain d local row r maps to original value row64*d+r for state,V and output; request slot indices and every head mapping remain identical across the two launches. Local domain0 must not accidentally consume the first half for both domains.
- Precondition `pre_shared_inputs` (implementation): `p.shared_key_query_gate_inputs_identical_and_live_through_both_shards` — Shared inputs are read-only or coherently supplied until both domains finish. Each launch independently normalizes/loads the complete key/query data under the same declared recipe; replicated setup is not a cross-domain state dependency.
- Precondition `pre_execution` (implementation): `execution_mode == 'ordered_current_stream' or p.green_context_execution_domain_matches_memory_and_backfill_contract` — Matched mode uses supported green-context SM groups with the same locality-domain IDs as allocation; source backfill policy is part of the admitted target resource contract. Current-stream fallback preserves functional ownership but does not imply domain-local state traffic.
- Precondition `pre_order` (implementation): `p.main_input_event_precedes_all_shards_and_all_shard_completions_precede_consumers` — In matched mode main stream records a fork event, each domain stream waits, each records done, and main waits every done before later consumers. Current-stream mode orders both launches directly. No half-complete checkpoint/output may publish as a complete state.
- Precondition `pre_slots` (implementation): `p.mutable_state_slots_unique_and_no_cross_request_seed_write_overlap` — Live mutable state slices have unique request ownership. A write must not overwrite another concurrent request seed; same-request in-place update is allowed only after its own row seed readers finish.
- Precondition `pre_lifetime` (implementation): `p.state_inputs_and_localized_allocation_owner_live_through_all_stream_and_graph_users` — State contents are initialized; allocation/green contexts stay alive through stream and CUDA-graph users. Releasing/mapping reuse waits for readers, including deferred release around capture. Required context/kernel warmup precedes capture.
- Precondition `pre_numeric` (implementation): `p.per_row_reduction_fma_conversion_and_store_recipe_admitted` — Value-row partition does not alter each row complete key reduction, scalar decay/prediction/correction order or declared state/output rounding. Bitwise equality additionally requires the exact per-row input and arithmetic tree, not just real row independence.
- Precondition `pre_backend` (implementation): `p.two_domain_locality_allocation_context_and_shard_kernel_available` — Target/driver exposes two compatible memory/execution locality domains and the actual V64,K128 T1 BF16 kernel path. Generic memory placement APIs, other architectures, more domains and arbitrary recurrent feature dependencies are not inferred.
- Constraint `c_shape` (implementation): `p.domains == 2 and p.value_features == 128 and p.key_features == 128 and p.tokens_per_request == 1` — Concrete source split is two64-value-row halves, each with complete128 key columns, for T1 decode.
- Constraint `c_allocation` (implementation): `p.domain_allocation_bytes >= p.slot_capacity * p.value_heads * 64 * 128 * 2 and p.slot_capacity >= 1 and p.value_heads >= 1` — Each BF16 physical allocation holds its half-state payload; extra allocation-granularity padding remains in actual strides.

## Local example and misuse

For a local instance of `store`, identify the object `independent_state_rows_with_matched_memory_and_execution_domains` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/locality_matched_state_shards@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/locality_matched_state_shards@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two localized padded allocations exposed through a logical slot/head/domain/row/key view.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: localized_state_pool
  level: dram
  scope: kernel
  bytes: 2 * p.domain_allocation_bytes
  note: Physical state payload plus per-domain allocation-granularity padding. Ordinary
    shared inputs/output storage and stream metadata are separate.
overlap: []
```

## Obligations and numerical checks

- `row_and_address_partition` (test): All slot/head/value/key coordinates map exactly once to the intended physical-domain shard with preserved padded strides and matching V/output slices.
- `complete_row_computation` (proof): Each shard keeps all key columns and preserves every row reduction/update/store recipe; no missing cross-domain reduction exists.
- `fork_join_publication` (test): Both shards see initialized inputs and every consumer observes both completions before a complete state/output generation is published.
- `physical_locality_and_lifetime` (test): Actual driver allocation and SM groups match domain IDs/backfill policy; allocations/contexts remain live through all stream/graph consumers.

Wrong domain/view stride or V slice mixes state rows; dropping a join exposes half-old state. Bitwise row equivalence requires the actual reduction/FMA tree; locality alone has no numeric guarantee.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
