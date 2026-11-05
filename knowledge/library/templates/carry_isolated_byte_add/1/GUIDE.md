# carry_isolated_byte_add@1: carry isolated byte add

## When to consider

When reducing packed bytes modulo256, isolate alternating byte lanes before wide addition so carry cannot corrupt neighboring results. reduce_kernel.h:361-370. triton carry_isolated_bit_counts@1 instead widens exact boolean counts; cannot bind modulo wrap unchanged.

## Computation before and after

When reducing packed bytes modulo256, isolate alternating byte lanes before wide addition so carry cannot corrupt neighboring results.

Let mask=0x00ff00ff. Compute x=(a&mask)+(b&mask), y=(a&~mask)+(b&~mask) in unsigned32 arithmetic. Extract low8 bits of each9-bit lane and interleave them in original byte order (source __byte_perm selector0x7250). Equivalent to ((x&mask)|(y&~mask)), since masking removes carry bits. Repeat binary reduction only under modulo256 byte semantics.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Unsigned byte pack addition, source signed byte dispatch may reinterpret bits with its declared modular semantics. No inference to wide accumulation or averages before widening. Resource effects: Two masked word additions and permutation instead of four scalar byte additions.

Scope predicate: `p.four_unsigned_bytes_per_word and p.result_is_per_byte_modulo_256`.

- Precondition `pre_0` (semantic): `p.four_unsigned_bytes_per_word and p.result_is_per_byte_modulo_256` — Widened exact integer sum or saturating sum is a different workload.

## Local example and misuse

For a local instance of `map`, identify the object `carry_isolated_byte_add` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/carry_isolated_byte_add@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/carry_isolated_byte_add@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Two masked word additions and permutation instead of four scalar byte additions.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `carry` (proof): Even-byte additions occupy16-bit fields; odd-byte carries fall only into discarded even positions or wrap outside word.
- `lane_identity` (proof): Reinterleaving returns (a_i+b_i) mod256 independently for each of four original byte coordinates.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
