# cooperative_probe_atomic_claim@1: cooperative probe atomic claim

## When to consider

When a hash insertion probes several buckets together, elect one available lane and atomically claim a whole packed slot before advancing the probe. Two different keys colliding at one empty slot: one CAS wins, loser retries rather than skipping all other lanes in the round. Two equal keys: loser recognizes duplicate. Non-atomic check-then-store loses either entry.

## Computation before and after

When a hash insertion probes several buckets together, elect one available lane and atomically claim a whole packed slot before advancing the probe.

For one input key/value, each cooperative lane reads its current probe bucket and classifies slots as equal, available or unequal. If any lane observes an equal key, unique-key insertion returns duplicate. Otherwise elect the lowest available lane, attempt whole-slot compare-exchange from empty to desired key/value and broadcast status. On success return inserted; on same-key failure return duplicate; on other-key interference retry the same probe round before advancing. Advance only when the group has no available slot.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| group_size | compile / software | [1, 2, 4, 8, 16, 32] | Cooperative tile size, must divide CUDA warp and match selected probe implementation. |
| slot_bytes | compile / software | [4, 8] | Narrow scope: whole key/value representation fits native packed CAS. |

## Coupled constraints and scope

Unique-key packed static map/set insertions on NVIDIA CUDA. Larger nonpacked key/value protocols, erase races and multimap duplicates require separate premises and are excluded. Linear/double hashing must supply their own full-cycle argument. Resource effects: Group-wide bucket reads plus one CAS attempt per elected slot; smaller groups reduce per-key parallelism while larger groups read more candidates together. Table storage remains capacity*slot_bytes.

Scope predicate: `p.keys_exclude_sentinels and p.equal_keys_same_probe and p.capacity_sufficient and p.probe_covers_capacity and p.slot_aligned and p.no_concurrent_erase and p.group_participation`.

- Precondition `pre_0` (semantic): `p.keys_exclude_sentinels and p.equal_keys_same_probe` — Valid keys exclude empty/erased markers and equal keys follow the same probing sequence.
- Precondition `pre_1` (semantic): `p.capacity_sufficient and p.probe_covers_capacity` — Probe reaches free storage under valid capacity/load assumptions.
- Precondition `pre_2` (semantic): `p.slot_aligned and p.no_concurrent_erase and p.group_participation` — Packed atomic alignment is valid, no concurrent erase, all tile lanes participate.

## Local example and misuse

For a local instance of `map`, identify the object `cooperative_probe_atomic_claim` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/cooperative_probe_atomic_claim@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/cooperative_probe_atomic_claim@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Group-wide bucket reads plus one CAS attempt per elected slot; smaller groups reduce per-key parallelism while larger groups read more candidates together. Table storage remains capacity*slot_bytes.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `linearization` (proof): Whole-slot CAS is the successful insertion linearization point and no observer can see a claimed key without its paired payload in this packed scope.
- `unique_key` (proof): Equal-key insertions converge to one retained slot under the same probe and retry order.
- `coverage` (proof): Cooperative probe buckets cover capacity without skipping an available earlier round.
- `publication` (proof): A consuming phase starts after insertions complete; relaxed atomic claim alone does not publish unrelated memory.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
