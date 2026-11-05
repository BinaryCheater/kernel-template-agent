# split_slot_claim_publication@1: split slot claim publication

## When to consider

When a key/value slot cannot be claimed in one atomic operation, separate key ownership from payload readiness and constrain which readers may observe the intermediate state. ThreadA wins keyK,threadB loses key but temporarily wins payload CAS: B must clear its speculative payload, then A publishes its own value. A find returning merely because keyK exists can expose the wrong value.

## Computation before and after

When a key/value slot cannot be claimed in one atomic operation, separate key ownership from payload readiness and constrain which readers may observe the intermediate state.

Stable path: atomically CAS empty key to desired key; only the successful claimant stores its payload. A reader finding the key must wait until payload differs from its empty sentinel before consuming it. In insertion-only speculative path, perform key CAS and payload CAS back-to-back: key winner retries payload CAS from sentinel until it succeeds; a key loser that speculatively acquired payload clears it back to sentinel, letting rightful winner complete. The speculative path cannot publish a complete key/value snapshot until all insertion participants finish. Whole-slot packed CAS is a separate realization with no split readiness wait.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA nonpacked wide map slots. Source gates ordinary insertion back-to-back CAS at sm70+, older and stable mixed-operator paths use dependent payload write; packable16-byte slots at sm90 can use atomic whole-slot CAS instead. No erase or arbitrary mixed-operation guarantee added. Resource effects: Avoids requiring a native wide CAS but adds payload polling or speculative retry/cleanup, with two atomic locations per slot.

Scope predicate: `p.payload_excludes_empty_sentinel and p.no_concurrent_erase and p.key_claim_atomic and p.writer_eventually_completes and p.reader_policy_matches_protocol`.

- Precondition `pre_0` (semantic): `p.payload_excludes_empty_sentinel and p.no_concurrent_erase` — A real payload is distinguishable from empty and erase/reuse cannot create ABA during insertion.
- Precondition `pre_1` (semantic): `p.key_claim_atomic and p.writer_eventually_completes` — Atomic key claim elects one writer and that writer can complete payload publication.
- Precondition `pre_2` (semantic): `p.reader_policy_matches_protocol` — Stable key-then-payload readers wait for nonempty payload; speculative back-to-back insertion has no concurrent readers until quiescence.

## Local example and misuse

For a local instance of `map`, identify the object `split_slot_claim_publication` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/split_slot_claim_publication@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/split_slot_claim_publication@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Avoids requiring a native wide CAS but adds payload polling or speculative retry/cleanup, with two atomic locations per slot.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ownership` (proof): Only the successful key claimant determines the final payload for that key.
- `reader_readiness` (proof): Observing key alone never suffices to expose an uninitialized or speculative payload.
- `cleanup` (proof): A losing speculative payload claim is retired without erasing a completed rightful payload under the admitted insertion-only state machine.
- `progress` (proof): Key winner and all speculative cleanup participants can run, and valid payload never equals the wait sentinel.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
