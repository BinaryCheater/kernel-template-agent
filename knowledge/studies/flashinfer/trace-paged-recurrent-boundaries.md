# Paged recurrent checkpoints and private partial tails

When a sequence can reuse an existing recurrent prefix, retain state at complete
page boundaries and share those completed pages read-only. Continue a partial
tail in private storage; a page ID alone does not establish which absolute
prefix, layer or parameter generation its state represents.

[paged_recurrent_boundary_state@1](delta/paged_recurrent_boundary_state@1/template.yaml)
and [outline](delta/paged_recurrent_boundary_state@1/proof.yaml) express that
state-position/ownership contract and the choice to prepare small slot records
per layer or once per immutable step. They do not allocate pages or implement
copy-on-write; those are prerequisites supplied by the cache owner.

Paths below refer to [retained source](inputs/sources.json): `gdn_paged.py`,
`triton/gdn_paged.py`, and `gdn_kernels/delta_rule_dsl/delta_rule_sm90.py`.
The concrete endpoint feeds NVIDIA GDN's split pool-read/write and absolute
checkpoint path; its prefill wrapper forces use_cp=False.

| Source decision | Rule field / binding | Evidence |
|---|---|---|
| gdn_paged`:265-317`: after-step CSR page list, total T=(pages-1)*P+last_len, new-token length L and prefix p=T-L | pre_table, pre_generation | Read page must contain the state after exactly p old tokens. Nonempty prefix can end inside a page; zero new length leaves the pool unchanged |
| Triton prefill slots`:36-83`: read entry first+floor((p-1)/P) or -1 for p0; first checkpoint entry first+floor(p/P) | operation=prefill_append | Empty prefix uses zero state. First checkpoint is the next absolute page boundary, not a count restarted at the new-token origin |
| Same kernel suppresses separate final write when L0 or T%P0 | absolute_state_positions | A full-page ending is already written by its boundary checkpoint; tail ending writes the last private page. Duplicate final stores need not race the checkpoint writer |
| gdn_paged`:265-317`: completed prefix pages read-only, partial tail private; no written page read/written by another sequence of this call | pre_ownership | Sharing a complete prefix is safe only while its state stays immutable. Own partial-tail read/write alias requires old seed reads before overwrite |
| delta_rule_sm90`:2338-2360,650-698`: prefix is virtually left padded to64-token work origin; gate/beta loads mask padding to identities | pre_boundaries | New-token work may begin at any absolute phase. Padding does not replay the earlier prefix already included in the cached seed |
| Same file`:795-845`: checkpoint key=(first_entry-1)*P+origin%P; boundary test uses key+block_end and original work length | page_tokens positive multiple64 | A crossed checkpoint lands on its actual page-table entry. A later tail block does not create a checkpoint beyond the real sequence |
| Triton decode slots`:84-118`; gdn_paged`:365-425`: last_len1 reads previous page, otherwise last; writes last | operation=decode_one | Existing prefix must be nonempty. If one new token opens a page, at least two page entries exist; first-token decode does not read an absent predecessor |
| gdn_paged`:135-230`: GPU slot preparation, optionally shared by all layers;`:86-104,303-307`: structural checks only | slot_preparation=per_layer/per_step, pre_slots | Caller-supplied records are not compared with table values. Reuse is valid across layers of the same step, not across changed lengths/pages just because shapes match |

The [focused check](checks/check_paged_recurrent_boundaries.json) compares9652
prefill metadata cases and14080 absolute checkpoint events against independent
inclusive token-position labels. It verifies46 single-token decode boundary
cases, all prefix phases for selected64/128-page endings, zero-new work,
private partial continuation and virtual64 checkpoint keys. Counterexamples
separate shared mutable tails, cross-sequence read/write overlap and stale
prepared slots after a page-opening decode. State contents/generation, native
rounding and allocator/layer publication remain source contracts.

Prepared metadata is16 bytes per sequence for prefill(read,final-write,first
checkpoint entry,prefix length),8 for decode(read/write). Computing once per
step saves repeated row-derivation launches across layers but extends the
metadata lifetime and binds every layer to the same page/length generation.
This is a mechanism-level reuse choice, not a performance measurement.

[semantic_block_page_fragments@1](delta/semantic_block_page_fragments@1/template.yaml)
resolves attention support fragments to KV payload pages. It does not supply
recurrent after-page state positions, shared prefix state validity or duplicate
checkpoint/final-write suppression. [deferred_prefix_state_commit@1](delta/deferred_prefix_state_commit@1/template.yaml)
selects one speculative window prefix; it does not provide persistent CSR page
identity or private partial-tail policy. The new rule owns those missing
choices and constraints while leaving the recurrence equations to their bound
GDN implementation.

This disposes the paged read/checkpoint/write state mechanism. Locality-domain
placement in `gdn_decode.py:117-205` is distinct: it physically shards every
state's independent value rows across memory domains and restricts matching
launches to those domains. The allocator/green-context and actual row kernel
remain the next locality representative in the [recurrent inventory](recurrent-inventory.md).
