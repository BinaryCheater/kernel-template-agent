# hashed_max_rank_sketch@1: hashed max rank sketch

## When to consider

When approximate cardinality can use a fixed register sketch, hash each item to a register and retain the maximum leading-zero rank, keeping identical hash/precision across merges. Two hashes assigned register3 with ranks2 and5 leave5 regardless of order or duplicate repetition. Adding ranks to get7 would be a histogram, not this sketch. Different precisions cannot merge by raw register index.

## Computation before and after

When approximate cardinality can use a fixed register sketch, hash each item to a register and retain the maximum leading-zero rank, keeping identical hash/precision across merges.

For hash h of width W, choose register h>>(W-precision). Compute rank=countl_zero((h<<precision)|(1<<(precision-1)))+1, where padding bounds the all-zero suffix rank. Atomically maximize that register. Merge sketches registerwise max. Estimate from z=sum(2**(-register)) and v=number of zero registers; apply the declared HLL/HLL++ finalizer with linear-counting/bias correction branches. Duplicate items repeat idempotent register updates, not increments.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| precision | compile / software | {"lo": 4, "hi": 30} | Number of register-index hash bits; exact source constructor/resource bounds narrow the usable range. |

## Coupled constraints and scope

NVIDIA HLL register-state refinement. Precision changes the approximation contract/storage; constructor limits and hash width must be checked before choosing precision. Standard-error claims are outside this source extraction. Resource effects: Fixed2**precision integer registers regardless of item count; per-item one hash/rank/max update, final estimate scans registers rather than stored unique keys.

Scope predicate: `p.hash_uniformity_contract and p.precision_fits_hash and p.registers_initialized_zero_and_atomic_max and p.merge_hash_precision_identical`.

- Precondition `pre_0` (semantic): `p.hash_uniformity_contract and p.precision_fits_hash` — Hash width supports the register prefix and rank suffix; any cardinality accuracy assumption is explicit.
- Precondition `pre_1` (semantic): `p.registers_initialized_zero_and_atomic_max` — Registers start zero and concurrent updates retain maximum rank.
- Precondition `pre_2` (semantic): `p.merge_hash_precision_identical` — Merged sketches have the same hash identity, precision, register order and rank definition.

## Local example and misuse

For a local instance of `map`, identify the object `hashed_max_rank_sketch` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/hashed_max_rank_sketch@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/hashed_max_rank_sketch@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Fixed2**precision integer registers regardless of item count; per-item one hash/rank/max update, final estimate scans registers rather than stored unique keys.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `register_identity` (proof): Prefix bits choose the correct register, suffix rank excludes those prefix bits and remains representable.
- `merge_equivalence` (proof): Registerwise max equals processing the union of item streams with the same policy.
- `estimator_contract` (proof): Estimate finalizer and precision-specific constants match declared policy; register equivalence does not prove statistical accuracy.
- `concurrency` (proof): Every max update and merge is atomic at the required scope and reads follow completed producers.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
