# paged_recurrent_boundary_state@1: page boundary recurrent state with shared immutable prefix

## When to consider

When recurrent prefixes are reused, store page-end states and share only immutable complete pages. Continue private partial tails in place, checkpoint every crossed absolute page boundary and derive slot records once per immutable step if layers reuse them.

## Computation before and after

Map current paged sequence lengths to recurrent prefix/checkpoint/final-state rows.

For prefill_append sequence i let first/end be its CSR entries, T=(end-first-1)*P+last_len and p=T-L. If p0 initialize zero; otherwise read page[first+floor((p-1)/P)], which must already hold the exact prefix state. Process only new tokens. For every absolute boundary q*P with p<q*P<=T, write the inclusive state to page[first+q-1]. First checkpoint entry is first+floor(p/P). If L>0 and T%P!=0, also write final state to last page[end-1]; otherwise suppress this separate final write because zero-new work changes nothing or the boundary checkpoint already owns it. Prefix may end inside a page; its private state is loaded before in-place continuation/checkpoint writes, with any virtual compute padding neutral. For decode_one always write last page; when last_len1 read the preceding page, otherwise read/write the last page. Require nonempty old prefix so that preceding page exists when needed. Derive these row records per layer or once per step, publish them before all users, and retain matching metadata/state generations until all layer/stream readers retire. Complete shared prefix pages are never written; allocator copy-on-write of partial shared pages is a separate prerequisite, not performed here.

Recover the original computation: Follow each sequence original token recurrence from its cached state after p tokens. Every checkpoint/final destination stores exactly the after-token state for its absolute page position. Split read/write rows preserve complete shared prefix states, and suppressing the duplicate full-page final store retains the same boundary state.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| page_tokens | runtime / software | {"lo": 64, "hi": 2147483584, "step": 64} | Positive64-aligned logical tokens per page. Controls checkpoint granularity and prefix-sharing/state storage; not the64-token internal compute block itself. |
| operation | compile / boundary | ["prefill_append", "decode_one"] | Bound input operation: multiple new tokens with page-boundary checkpoints, or one token following a nonempty prior prefix. |
| slot_preparation | compile / software | ["per_layer", "per_step"] | Derive small pool-row records within each layer call, or prepare once and reuse across all layers of the same immutable step. |

## Coupled constraints and scope

FlashInfer gdn_paged.py and Triton row preparation feed NVIDIA GDN split pool/checkpoint kernels with page_size a positive multiple64. Prefill accepts zero new length and unaligned prefix; decode_one excludes an empty old prefix. Pool row stride may be padded but each[H,V,K] row is dense. Page allocator/COW, prefix identity lookup, all-layer readiness and source GPU metadata value validation are caller/system boundaries. This rule stores recurrent states at page positions, not per-token KV payloads.

Scope predicate: `(p.sequences >= 1 and page_tokens % 64 == 0) and ((operation == 'prefill_append' and p.slot_record_bytes == 16) or (operation == 'decode_one' and p.slot_record_bytes == 8)) and (p.page_table_complete_ids_valid_and_after_step_lengths_coherent) and (p.cached_prefix_state_matches_sequence_layer_parameters_and_recipe) and (p.every_written_page_private_and_no_other_sequence_reads_it_this_step) and (p.checkpoints_are_after_absolute_page_boundary_tokens_and_virtual_padding_neutral) and (p.slot_records_derived_from_same_immutable_step_metadata_and_published_before_layers) and (operation == 'prefill_append' or p.single_token_decode_has_nonempty_prior_prefix) and (p.all_seed_checkpoint_and_layer_readers_complete_before_page_reuse_or_mutation) and (p.selected_paged_mode_state_dtype_stride_and_checkpoint_kernel_supported)`.

- Precondition `pre_table` (implementation): `p.page_table_complete_ids_valid_and_after_step_lengths_coherent` — Each sequence has at least one valid page ID in CSR order, final last-page length1..P, total T=(page_count-1)*P+last_len, and nonnegative new length L<=T. Tables, host page size and token-prefix metadata describe the same after-step sequence.
- Precondition `pre_generation` (implementation): `p.cached_prefix_state_matches_sequence_layer_parameters_and_recipe` — The read page contains exactly the state after the prior prefix p=T-L for this layer/parameter/numeric generation. Shared complete-prefix pages have the same valid state meaning for every sharing sequence; shape equality or page ID reuse alone does not certify it.
- Precondition `pre_ownership` (implementation): `p.every_written_page_private_and_no_other_sequence_reads_it_this_step` — Complete prefix pages may be shared read-only. A partial prefix/tail page and all newly written/checkpoint pages are private. Written page destinations are unambiguous; no other sequence in this call reads or writes them. Own read/write alias is allowed only after its original seed readers retire.
- Precondition `pre_boundaries` (implementation): `p.checkpoints_are_after_absolute_page_boundary_tokens_and_virtual_padding_neutral` — Checkpoint at absolute token count q*P is the inclusive state through that token, not a page-start state. Unaligned p may be virtually left-padded to the64-token compute boundary using identity transitions and masked outputs, without replaying old prefix tokens.
- Precondition `pre_slots` (implementation): `p.slot_records_derived_from_same_immutable_step_metadata_and_published_before_layers` — Prepared read/write/checkpoint-prefix records match the current page table and new-token counts. They may be reused across layers of this step, never implicitly across changed decode/page metadata. Caller-supplied records are not source-validated against table values.
- Precondition `pre_mode` (implementation): `operation == 'prefill_append' or p.single_token_decode_has_nonempty_prior_prefix` — Decode mode requires at least one token before the new token; if it opens a page, at least two page entries exist. The first token is handled by the prefill zero-seed path, not by reading an absent previous page.
- Precondition `pre_lifetime` (implementation): `p.all_seed_checkpoint_and_layer_readers_complete_before_page_reuse_or_mutation` — Page/slot generations remain live through every layer and stream user. Page allocator or subsequent steps cannot recycle/remap a page early, and published checkpoint states complete before prefix reuse.
- Precondition `pre_backend` (implementation): `p.selected_paged_mode_state_dtype_stride_and_checkpoint_kernel_supported` — Actual backend supports split read/write pool rows, absolute checkpoint destinations and unaligned-prefix identity masking at64-token compute granularity. This first binding uses NVIDIA GDN routes; it does not infer support for every provider/CP combination.
- Constraint `c_domain` (implementation): `p.sequences >= 1 and page_tokens % 64 == 0` — Nonempty sequence batch and native checkpoint-aligned page size; zero new-token sequences remain valid.
- Constraint `c_slot_bytes` (implementation): `(operation == 'prefill_append' and p.slot_record_bytes == 16) or (operation == 'decode_one' and p.slot_record_bytes == 8)` — Source uses four or two int32 row/prefix records per sequence.

## Local example and misuse

For a local instance of `store`, identify the object `page_boundary_recurrent_state_with_shared_immutable_prefix` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/paged_recurrent_boundary_state@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/paged_recurrent_boundary_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One state per page boundary or private partial tail plus optional per-step row records.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: prepared_step_slots
  level: dram
  scope: kernel
  bytes: p.sequences * p.slot_record_bytes
  when: slot_preparation == 'per_step'
  note: Small shared read/write/checkpoint-prefix records only; recurrent page pool
    and page table are separate.
overlap: []
```

## Obligations and numerical checks

- `absolute_state_positions` (test): Read page, crossed checkpoints and final tail correspond to the correct absolute token counts, including unaligned prefixes, full final pages and zero new tokens.
- `shared_prefix_ownership` (test): All complete shared prefix pages remain immutable; mutable pages cannot overlap another sequence read/write set, and own read aliases retire before overwrite.
- `slot_generation_reuse` (test): Prepared records remain identical across layers of the same step and are regenerated before any page/length generation change.
- `native_checkpoint_recipe` (test): Virtual left padding, numerical state storage and actual device checkpoint publication implement the declared absolute boundaries.

A page can have the correct shape and wrong prefix/parameter generation. Reusing stale prepared rows, treating first-token decode as previous-page read, or mutating a shared partial page changes sequence state.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
