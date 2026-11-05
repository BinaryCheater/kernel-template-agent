# cache_prefetch@1: cache prefetch schedule

## When to consider

When future global operand tiles are addressable while a bounded shared-memory pipeline is already in use, consider independent L2 lookahead. Preserve the demand-copy indices and all real completion/release dependencies; tune distance with zero disabling hints, without allocating another shared slot. The pinned CuTe examples expose this distance for immutable A/B or A/B/SF streams; T5,D4,S2 is the index witness, while other backends and invalid source addresses require their own guarded implementation.

## Computation before and after

Prime a finite cache-hint stream then issue one bounded rolling hint after each demand-copy issue.

Preserve the original demand-copy stream for tiles0..T-1 and all its waits/releases. For distance D>0, issue L2 hints for tiles0..min(D,T)-1 before that stream; after demand-copy issue k, hint tile k+D iff k<T-D. D=0 issues no hints. Restart hint indices at zero for each independent output-tile stream, preserving any shared-buffer slot/phase state from the original program. A hint has no program-visible destination slot and grants no readiness or release permission.

Recover the original computation: Erase all hint events; the original demand-copy, synchronization and arithmetic trace remains in the same order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| distance | compile / software | {"lo": 0, "hi": 2147483647} | Zero disables hints; a positive value primes that many tiles then hints distance ahead after each demand-copy issue. |

## Coupled constraints and scope

Finite immutable global-memory tile streams with an L2-only hint mechanism. Distance may exceed the actual shared-memory stage count or T: prologue bounds still hold and no extra shared destination is allocated. Hint count is T per selected stream when D>0 and zero otherwise; physical cache transactions, latency, residency and cache pollution are unmeasured and are not priced by these empty effects. TMA descriptor prefetch and eviction priority are separate choices.

Scope predicate: `p.hints_value_neutral and p.hint_addresses_valid and p.immutable_source and p.tiles >= 0`.

- Precondition `pre_hint_semantics` (implementation): `p.hints_value_neutral and p.hint_addresses_valid` — Hints may affect cache state but cannot change application values or replace demand-copy completion.
- Precondition `pre_source` (semantic): `p.immutable_source and p.tiles >= 0` — Source storage and metadata remain valid and immutable throughout this finite tile stream.
- Constraint `c_hint_semantics` (implementation): `p.hints_value_neutral and p.hint_addresses_valid` — The selected backend supports non-faulting value-neutral hints for every selected source tile.

## Local example and misuse

For a local instance of `transfer`, identify the object `cache_prefetch_schedule` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/cache_prefetch@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/cache_prefetch@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Hint-selected source tiles may subsequently be consumed by one or several CTAs.. Storage: L2 cache state only; no program-visible destination buffer is introduced.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `hint_bounds` (exhaustive): Every hint refers to a tile in0..T-1 and a valid source descriptor/address for that tile.
- `demand_trace` (exhaustive): Erasing hints preserves the demand-copy indices and every completion/release dependency of the original stream.
- `value_neutral_backend` (unspecified): The selected hint backend has no application-visible value effect and all source/metadata lifetimes cover its use.

No arithmetic change under the stated backend and immutable-source premises; performance may improve or degrade.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

- enables: [async_transfer@1](../../async_transfer/1/GUIDE.md); May bring source data toward L2 before an actual async transfer; does not establish its readiness.

Structured contract: [template.yaml](template.yaml).
