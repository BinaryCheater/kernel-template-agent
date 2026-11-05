# atom_owned_cluster_multicast@1: separate cluster payload full owner and empty release maps

## When to consider

When a physical cluster contains several cooperative matrix-instruction atoms, derive data destinations, full-barrier owners and release targets separately. Payload fanout may exceed the readiness-owner count, especially for scale operands. Match release counts to consuming atoms and adapt every map together on fallback.

## Computation before and after

Factor cluster multicast into payload delivery, atom-owned readiness and producer release relations.

Rank r=x+dim_x*y; instruction-atom owner(r)=r & ~1. For x_column, elected issuers lie at y=0: payload P={x+dim_x*yy:0<=yy<dim_y}; release targets R={2*floor(x/2),2*floor(x/2)+1}; expected arrivals E=dim_y. For x_parity_row, elected issuers lie in the first X atom of each row: P={(x%2)+2*a+dim_x*y:0<=a<dim_x/2}; R={dim_x*y,dim_x*y+1}; E=dim_x/2. For whole_x_row, issuer is x=0 in each row: P={xx+dim_x*y:0<=xx<dim_x}; R={dim_x*y}; E=dim_x/2. For every relation, full barrier owners O={owner(r):r in P}; bitmask form O=(P|(P>>1)) & 0x5555. Arm only the intended full-owner barriers with complete transaction accounting, issue payload to P, and let each consuming atom release R once after its readers retire. Producer waits for E atom completions before stage reuse. Recompute P/O/R/E for the actual fallback dimensions; do not replace O or R with P.

Recover the original computation: Each receiving CTA sees the same operand bytes as an independent local load, while projected owner barriers aggregate the instruction atom readiness and the separate release relation prevents reuse until every consuming atom completes.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| dim_x | runtime / boundary | {"lo": 2, "hi": 4, "step": 2} | Active physical cluster X dimension; adjacent pairs form independent2-CTA instruction atoms. |
| dim_y | runtime / boundary | {"lo": 1, "hi": 4} | Active physical Y dimension, fixed or selected from a prepared fallback. |
| payload_relation | compile / boundary | ["x_column", "x_parity_row", "whole_x_row"] | Source A/SFA share an X column, B shares issuer X parity across a row, SFB shares the whole row. Select only when the workload has that operand identity. |

## Coupled constraints and scope

Pinned PrimsTS batched GEMM wider/flexible CGAs on SM100-family block-scaled/tcgen05 paths, with adjacent2-CTA UMMA atoms and16-bit masks. Source examples2x1/4x1/2x2/4x2/4x4; other admitted active dimensions require native launch support. Relations are operand/workload constraints, not arbitrary permutations. Combined A+B release, routed loads and compact local scale-copy variants require their own binding.

Scope predicate: `dim_x * dim_y <= 16 and dim_x * dim_y <= hw.max_cluster_size and (p.active_dimensions_shared_by_task_election_payload_and_barrier_logic and p.two_adjacent_cta_instruction_atoms_supported) and (p.selected_payload_relation_preserves_same_operand_tile_and_stage_identity) and (p.elected_issuers_cover_each_needed_payload_once_with_declared_slice_ownership) and (p.full_owner_transaction_byte_accounting_matches_all_issued_payloads and p.full_barriers_armed_before_transfer) and (p.one_release_event_per_consuming_instruction_atom_after_all_operand_reads and p.release_targets_and_expected_arrivals_agree) and (p.stage_generation_and_cluster_entry_exit_handshakes_valid and p.selected_multicast_and_remote_barriers_supported) and (p.fallback_dimensions_within_prepared_bounds_and_all_masks_counts_recomputed)`.

- Precondition `pre_topology` (implementation): `p.active_dimensions_shared_by_task_election_payload_and_barrier_logic and p.two_adjacent_cta_instruction_atoms_supported` — Every adjacent physical X pair is one instruction atom, with an even-rank barrier owner; all participants use the same actual topology.
- Precondition `pre_data` (semantic): `p.selected_payload_relation_preserves_same_operand_tile_and_stage_identity` — Payload recipients need the identical logical operand tile in the same pipeline generation; operand roles determine the legal relation.
- Precondition `pre_issuers` (implementation): `p.elected_issuers_cover_each_needed_payload_once_with_declared_slice_ownership` — CTA task election supplies each intended payload/slice once; task selection and payload masks are separate decisions.
- Precondition `pre_full` (implementation): `p.full_owner_transaction_byte_accounting_matches_all_issued_payloads and p.full_barriers_armed_before_transfer` — Projected owner barriers count every required TMA transaction/byte, including slices from both parity issuers when present.
- Precondition `pre_release` (implementation): `p.one_release_event_per_consuming_instruction_atom_after_all_operand_reads and p.release_targets_and_expected_arrivals_agree` — One logical atom completion contributes exactly once to each intended producer empty barrier after every operand reader retires.
- Precondition `pre_lifetime` (implementation): `p.stage_generation_and_cluster_entry_exit_handshakes_valid and p.selected_multicast_and_remote_barriers_supported` — Data/full/empty state advances in matching generations; no peer exits or overwrites a stage while remotely addressed.
- Precondition `pre_fallback` (implementation): `p.fallback_dimensions_within_prepared_bounds_and_all_masks_counts_recomputed` — A flexible fallback changes all topology-dependent masks, issue elections and arrival counts together.
- Constraint `c_cluster` (resource): `dim_x * dim_y <= 16 and dim_x * dim_y <= hw.max_cluster_size` — Active ranks fit the native16-bit masks and hardware cluster limit.

## Local example and misuse

For a local instance of `transfer`, identify the object `separate_cluster_payload_full_owner_and_empty_release_maps` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/atom_owned_cluster_multicast@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/atom_owned_cluster_multicast@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Per-CTA payload stages and full/empty barrier state with independent rank maps.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `payload_owner_map` (test): Projection maps every payload rank to its adjacent atom owner without changing operand coordinates or omitting required transfer accounting.
- `release_census` (test): Every acquired producer empty barrier receives exactly E logical atom completions and no release occurs before all local/peer operand reads finish.
- `election_and_fallback` (test): Task election, payload/full/release masks and expected counts agree for both preferred and active fallback shapes.
- `native_visibility` (test): TMA completion, remote barrier ownership, transaction bytes and exit/retirement satisfy the native instruction protocol.

No numeric risk is declared by this proposal.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
