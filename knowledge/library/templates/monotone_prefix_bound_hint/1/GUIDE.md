# monotone_prefix_bound_hint@1: monotone prefix bound hint

## When to consider

When independent ordered batches can use conservative prefix attenuation to skip work, publish an atomic packed bound and ignore hints from a future batch. PrimingChainEncoding.cuh:14-190; ParallelBatchFwd.cu:520-564,675-696. Positive batch keys below65535; reserved sentinels excluded.

## Computation before and after

When independent ordered batches can use conservative prefix attenuation to skip work, publish an atomic packed bound and ignore hints from a future batch.

Encode one word ordered by (!saturated, upward-rounded positive-half bound, wrapped negative next-batch index). atomicMin makes established saturation sticky and prefers tighter bounds. Publish at batch b with K=b+1. Consumer at b accepts only stored K<=b; otherwise use bound 1 and no saturation. For accepted unsaturated upper bound U, local attenuation stopping threshold is global_threshold/U; U must be positive. An upstream saturated flag permits skipping. Treat packed state as a self-contained hint: relaxed atomic access does not publish unrelated payload. Init K=0,U=1; zero sentinel reserved for invalid ray. Equal-U initial K=0 may remain preferred, losing only optimization.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| key_bits | compile / software | {"lo": 1, "hi": 16} | Bounded positive next-batch index field; source uint16 with reserved states. |

## Coupled constraints and scope

Monotone attenuation and conservative hints only. Source priming may be stale and forward-only full-batch fallback remains approximate. Resource effects: One packed 32-bit atomic word per pixel, no payload publication implied.

Scope predicate: `p.prefix_attenuation_nonincreasing and p.every_published_bound_conservative and p.finite_unit_bounds and p.atomic_word_covers_all_fields and p.rounding_toward_upper_bound and p.key_range_legal and p.saturation_flag_is_sound and p.consumer_rejects_future_index`.

- Precondition `pre_0` (semantic): `p.prefix_attenuation_nonincreasing and p.every_published_bound_conservative and p.finite_unit_bounds` — Each preceding-batch value upper-bounds true attenuation at later batches.
- Precondition `pre_1` (semantic): `p.atomic_word_covers_all_fields and p.rounding_toward_upper_bound and p.key_range_legal` — No torn state; finite nonnegative float encoding preserves numeric order; quantization rounds upward.
- Precondition `pre_2` (semantic): `p.saturation_flag_is_sound and p.consumer_rejects_future_index` — Only an established preceding saturation can suppress work.

## Local example and misuse

For a local instance of `map`, identify the object `monotone_prefix_bound_hint` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/gsplat/delta/monotone_prefix_bound_hint@1/template.yaml`. Original outline: `knowledge/studies/gsplat/delta/monotone_prefix_bound_hint@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One packed 32-bit atomic word per pixel, no payload publication implied.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `conservatism` (proof): Admitted earlier prefix and upward quantization keep U>=true initial attenuation; rejection of K>b prevents downstream contamination.
- `atomic_priority` (proof): Unsigned ordering implements sticky saturation and tighter nonnegative bounds without field tearing.
- `hint_only` (proof): No output equivalence follows unless exact composition/replay independently recovers the accepted candidate prefix.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
