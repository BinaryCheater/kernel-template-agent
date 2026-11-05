# generation_tagged_payload_lines@1: generation tagged payload lines

## When to consider

When a small-message transport embeds readiness with payload, tag every physical line with its FIFO generation and accept data only after all corresponding tags match. prims_ll.h:49-165 clean-mask rollover; prims_ll128.h:190-280 warp vote+reload and target-specific completion fence.

## Computation before and after

When a small-message transport embeds readiness with payload, tag every physical line with its FIFO generation and accept data only after all corresponding tags match.

Map step g to circular slot g%S. Wait for retired credit before writing. Emit payload with expected generation g+1 in every designated tag field. Receiver repeatedly loads line and tests all expected tags; cooperative variants vote across flag lanes and reload payload after successful vote. Reduce/copy only matching payload. Retire slot after all readers finish. On finite tag rollover, clean every stale line tag including unused tail lines before allowing alias; otherwise require nonwrapping generation epoch.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| line_bytes | compile / software | {"lo": 16, "hi": 128} | Transport line size; source LL16 and LL128128 have distinct layouts. |
| tag_bits | compile / software | {"lo": 32, "hi": 64} | Generation tag width; finite reuse needs cleanup/epoch separation. |

## Coupled constraints and scope

Source-derived conditional transport protocol. Volatile vector operations alone are not a portable release/acquire proof. LL/LL128 are distinct hardware realizations. Resource effects: LL carries8 payload bytes in16; LL128 reserves flag words in128-byte warp lines.

Scope predicate: `p.line_payload_tag_mapping_complete and p.tagged_store_load_visibility_contract and p.sender_credit_prevents_live_slot_overwrite and p.generation_alias_prevented and p.collective_flag_poll_participation_and_abort_contract`.

- Precondition `pre_0` (semantic): `p.line_payload_tag_mapping_complete and p.tagged_store_load_visibility_contract` — Expected flags imply visibility of matching payload only under the selected hardware transport ordering contract.
- Precondition `pre_1` (semantic): `p.sender_credit_prevents_live_slot_overwrite and p.generation_alias_prevented` — Consumer retires before reuse; stale tags cannot match future live generation.
- Precondition `pre_2` (semantic): `p.collective_flag_poll_participation_and_abort_contract` — All required flag lanes participate; abort is failure, never successful data.

## Local example and misuse

For a local instance of `map`, identify the object `generation_tagged_payload_lines` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/generation_tagged_payload_lines@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/generation_tagged_payload_lines@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: LL carries8 payload bytes in16; LL128 reserves flag words in128-byte warp lines.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ready` (proof): No payload from an older generation may be accepted for g; the hardware visibility premise connects matching tags to data.
- `reuse` (proof): Credits prevent overwrite while any reader holds the old slot.
- `rollover` (proof): Unused lines cannot retain a tag that aliases a later live generation.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
