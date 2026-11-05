# poison_sentinel_rotating_completion@1: poison sentinel rotating completion

## When to consider

When a transport guarantees whole-vector payload visibility and valid output excludes a sentinel word, detect arrival from a payload sentinel and rotate separate receive/clear buffers to retain returned output. contrib/custom_algos/alltoall/lsa_poison_alltoall_kernel.cuh and allreduce/twoshot_mc_poison_allreduce_kernel.cuh:80-180; current k%3, clear(k+2)%3. Warmup policy comments differ: bind actual host choice before use.

## Computation before and after

When a transport guarantees whole-vector payload visibility and valid output excludes a sentinel word, detect arrival from a payload sentinel and rotate separate receive/clear buffers to retain returned output.

Initialize all rotating receive buffers with sentinel in chosen word. At generation k receive into bank k%B while clearing a different bank protected by the proved peer-skew and output lifetime. Each unique producer writes full vectors with valid data; each receiver polls its local vector sentinel word until changed. Complete all required slots before returning. Clear only the assigned disjoint old-bank range back to sentinel. Warmup entry barriers establish initial clean state; skipping later entry barriers requires a proof of launch-order/skew and clear visibility, not merely buffer rotation arithmetic.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| buffer_count | compile / software | {"lo": 3, "hi": 16} | At least enough buffers for proved peer skew plus live output and clear generation; source3. |

## Coupled constraints and scope

Source custom NVLink alltoall and multicast AllReduce demos assume native128-bit arrival behavior and sentinel exclusion. Hardware visibility and steady-state barrier elision remain premises, not established by source comments. Resource effects: Avoids separate ready array/exit barrier but consumes sentinel domain and rotating full output buffers.

Scope predicate: `p.valid_payload_excludes_sentinel and p.vector_payload_sentinel_visibility_atomic and p.all_receive_slots_poisoned_before_generation and p.matched_launch_order_and_bounded_peer_skew and p.output_readers_retired_before_clear and p.clear_visible_before_reuse and p.input_publication_and_peer_progress`.

- Precondition `pre_0` (semantic): `p.valid_payload_excludes_sentinel and p.vector_payload_sentinel_visibility_atomic` — A non-sentinel observed word certifies the complete matching vector only under the native transport contract.
- Precondition `pre_1` (semantic): `p.all_receive_slots_poisoned_before_generation and p.matched_launch_order_and_bounded_peer_skew` — No stale non-sentinel data can be accepted; skew bound excludes concurrent write to clear buffer.
- Precondition `pre_2` (semantic): `p.output_readers_retired_before_clear and p.clear_visible_before_reuse` — Returned output stays readable until caller consumers retire; rotating clear cannot erase live output.
- Precondition `pre_3` (semantic): `p.input_publication_and_peer_progress` — Sentinel arrival does not establish source input readiness; caller publishes it separately.

## Local example and misuse

For a local instance of `map`, identify the object `poison_sentinel_rotating_completion` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/poison_sentinel_rotating_completion@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/poison_sentinel_rotating_completion@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Avoids separate ready array/exit barrier but consumes sentinel domain and rotating full output buffers.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `arrival` (proof): Every accepted non-sentinel denotes exactly one complete vector of current generation, not legal sentinel-valued data or stale prior data.
- `rotation` (proof): Clear bank is disjoint from all live producer/consumer banks under the declared skew and lifetime bound.
- `retention` (proof): Current output is not cleared before its downstream readers finish.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
