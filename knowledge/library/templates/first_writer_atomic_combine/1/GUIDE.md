# first_writer_atomic_combine@1: initialized atomic accumulator

## When to consider

Use when several blocks contribute to one output tile and the numerical contract allows unordered accumulation. Bind partials to the actual contributor count (2..10 here) and choose an update engine supported by the target. For four partials, publish the first store before letting the other three add; reuse the tile only after all four finish. The partial count comes from the upstream partition.

## Computation before and after

Initial store followed by atomic partial accumulation with completion countdown.

Partial 0 waits for lock<=0, stores its tile, completes its writes, then publishes lock=1-partials. Each other partial waits for lock<=-1, atomically adds each owned output element, completes the update, then atomically increments lock. The last completed update restores lock=0. No fixed ordering is imposed among nonzero partials.

Recover the original computation: Sum all partial values in their actual atomic linearization order; erase the lock and output staging.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| partials | runtime / software | {"lo": 2, "hi": 10} | Number of partials of this output tile; this version uses the source scheduler's ten-slice bound. |
| update | compile / boundary | ["packed_atomic_add", "bulk_reduce_add"] | Atomic element update engine; selected by the target and output path. |

## Coupled constraints and scope

Split reductions with one designated initializing partial and 2..10 partials per tile. Humming selects this path for TMA stores or native packed atomic add with more than three slices. That threshold is a dispatch policy, not a correctness requirement. Effects remain unpriced until each element update and completion event is modeled.

Scope predicate: `p.reassociation_allowed and p.atomic_supported`.

- Precondition `pre_contract` (semantic): `p.reassociation_allowed and p.atomic_supported` — Numeric contract allows nondeterministic update order and the target implements the update.
- Precondition `pre_generation` (semantic): `p.lock_initial == 0 and p.no_overlapping_generation` — Lock and output tile have one live launch generation.
- Constraint `c_count` (semantic): `partials == p.partial_count and partials >= 2` — Countdown matches the actual number of producers.

## Local example and misuse

For a local instance of `reduce`, identify the object `initialized_atomic_accumulator` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/humming/delta/first_writer_atomic_combine@1/template.yaml`. Original outline: `knowledge/studies/humming/delta/first_writer_atomic_combine@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Output tile and one global countdown per synchronization group.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `initialization_visible` (test): Every atomic update follows visible completion of the initializing store.
- `exact_updates` (exhaustive): Every partial contributes once per output element and each element update is atomic.
- `completion_before_reuse` (test): Countdown increments follow completion of all updates of that partial and zero occurs only after every partial finished.
- `progress` (test): The initializing partial and all awaited producers can run under the actual launch residency and scheduler.

Output precision rounding and nondeterministic atomic order; exact arithmetic witnesses do not settle floating-point error.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- alternative_of: ordered_global_combine (outside menu); Other partials are concurrent rather than lock-ordered by slice index.

Structured contract: [template.yaml](template.yaml).
