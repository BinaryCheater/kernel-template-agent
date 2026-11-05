# persistent_value_rematerialization@1: retained producer frontier

## When to consider

When a derived tensor must survive a reduction but its immutable producer inputs occupy fewer live bytes, choose whether to retain the tensor or retain its producer frontier and reconstruct it. FP16 x may replace a persistent FP32 affine output if the same cast/multiply/add sequence is replayed. Reject hidden RNG/state, missing dependencies and frontiers overwritten before the last replay.

## Computation before and after

Retain inputs of a pure producer slice and replay its output at later uses.

For retention=derived_value compute v=F(frontier) once and retain v through the last consumer. For producer_frontier retain all frontier values instead, discard v after its early uses and reevaluate the same producer DAG F before each later consumer. Keep each replay live through its local last use. Frontier may be input registers or a shared-memory cache; all frontier data survive until the last replay finishes.

Recover the original computation: Replace every replay by the same logical derived value F(frontier); immutable inputs and preserved operations make consumer values equal.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| retention | compile / software | ["derived_value", "producer_frontier"] | Keep derived output or reconstruct it from a smaller retained frontier. |

## Coupled constraints and scope

Pure deterministic producer DAGs crossing a phase boundary. No mutation, RNG, changed reductions or implicit extra precision. Saving storage is conditional on the actual frontier/replay live set, not guaranteed by selecting the transformation.

Scope predicate: `p.frontier_covers_every_dependency and p.producer_slice_pure and p.frontier_immutable_until_last_replay and p.replay_numerical_contract_preserved and p.no_rng_or_external_state`.

- Precondition `pre_replay` (semantic): `p.producer_slice_pure and p.frontier_immutable_until_last_replay and p.replay_numerical_contract_preserved and p.no_rng_or_external_state` — Repeated producer evaluation has the same inputs and numerical semantics with no hidden state.
- Constraint `c_frontier` (semantic): `p.frontier_covers_every_dependency and p.frontier_live_bytes <= p.storage_capacity_bytes` — The retained frontier reconstructs every required value and fits storage.

## Local example and misuse

For a local instance of `store`, identify the object `retained_producer_frontier` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/fuser/delta/persistent_value_rematerialization@1/template.yaml`. Original outline: `knowledge/studies/fuser/delta/persistent_value_rematerialization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Producer frontier versus derived persistent tensor; replay-local temporaries.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `replay_identity` (proof): Every replay receives identical frontier coordinates/values and preserves the admitted arithmetic semantics.
- `lifetime` (test): Frontier values survive all replays and no derived replay is overwritten before local consumption.
- `capacity` (test): Retained frontier plus each live replay fits the chosen resource capacity.

Different contraction, cast removal or fusion on replay needs its own numerical premise; algebraic equality alone is insufficient.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
