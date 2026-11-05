# head_partitioned_hierarchical_collective@1: head partitioned hierarchical collective

## When to consider

When local peers share several network heads, stripe chunks across heads and compose local reduce/scatter, inter-group reduction, then local broadcast/gather. all_reduce.h:248-635; registered order differs across channel/head dimensions. CollNetChain is the head_count1 sequential local path, not the same topology performance.

## Computation before and after

When local peers share several network heads, stripe chunks across heads and compose local reduce/scatter, inter-group reduction, then local broadcast/gather.

For each loop stripe assign (channel,head) a disjoint chunk. Scatter source contribution to its head; head reduces all local rank inputs. Network offload or an inter-head tree combines one partial per local group. Returned completed value is finalized by the declared unique post-op owner then broadcast/gathered locally. Bind chunk address permutation consistently: ordinary ((channel*heads)+head)*chunk or registered ((head*channels)+channel)*chunk. Run scatter/reduce/return/gather in disjoint barrier groups; registered direct buffers may replace copies with notifications, preserving protocol steps and buffer lifetime.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| head_count | compile / software | {"lo": 1, "hi": 64} | Declared local head/rail count; actual topology and arity cap constrain it. |
| chunk_size | compile / software | {"lo": 1, "hi": 1073741824} | Elements per head/channel stripe. |

## Coupled constraints and scope

Explicit local and network collective contracts; no proof of correctness for an arbitrary offload provider. NVLS is an alternative local hardware realization with its own capability guard. Resource effects: Four worker roles pipeline stages; head count and stripe size control traffic distribution.

Scope predicate: `p.head_stripes_partition_output and p.local_groups_partition_ranks and p.stage_layout_maps_agree and p.network_or_head_tree_reduction_contract and p.local_transport_contract and p.post_operation_owner_explicit and p.role_groups_progress and p.registration_and_notification_contract`.

- Precondition `pre_0` (semantic): `p.head_stripes_partition_output and p.local_groups_partition_ranks and p.stage_layout_maps_agree` — Each output stripe and rank contribution has one path through the hierarchy.
- Precondition `pre_1` (semantic): `p.network_or_head_tree_reduction_contract and p.local_transport_contract and p.post_operation_owner_explicit` — External offload or tree reduces each local partial once and returns final result; post-op location is explicit.
- Precondition `pre_2` (semantic): `p.role_groups_progress and p.registration_and_notification_contract` — Bypassing payload copies does not bypass control credit/notification obligations.

## Local example and misuse

For a local instance of `map`, identify the object `head_partitioned_hierarchical_collective` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/head_partitioned_hierarchical_collective@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/head_partitioned_hierarchical_collective@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Four worker roles pipeline stages; head count and stripe size control traffic distribution.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `hierarchical_partition` (proof): Local subgroups cover contributors once, and head stripes cover each output once.
- `permutation` (proof): Both address layouts are bijections on valid channel/head pairs and every stage uses the same selected map.
- `completion` (proof): Final post-op occurs once after cross-group reduction; notification and receive completion precede use.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
