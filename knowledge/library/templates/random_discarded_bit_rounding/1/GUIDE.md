# random_discarded_bit_rounding@1: random discarded bit rounding

## When to consider

When narrowing a binary format by discarding low significand bits, use a declared random-bit carry to select the adjacent representable value and keep exact per-element RNG identity across realizations.

## Computation before and after

When narrowing a binary format by discarding low significand bits, use a declared random-bit carry to select the adjacent representable value and keep exact per-element RNG identity across realizations.

For finite FP32 values within BF16 finite range, reinterpret sign-magnitude bits as u32, add independent unsigned r16 to the low16-bit discarded field through integer addition, then truncate low16 bits. The carry selects away-from-zero rather than toward-zero with probability residual/65536 when r16 is uniform. Source packed pair consumes distinct halves of one random word. Special values bypass perturbation and use the declared saturating conversion; do not share one validity predicate across a mixed-special/normal pair. Hardware and software realizations are bit-compatible only if the same random bits reach the same element, including lane/subtile/tile seed mapping.

Recover the original computation: BF16 stochastic rounding from FP32 with source16-bit random carry. FP16/FP8/FP4 grid rescaling uses different realization proofs and cannot inherit this bit-truncation formula directly.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| discarded_bits | compile / software | {"lo": 16, "hi": 16} | Source BF16 narrowing discards16 FP32 low bits. |

## Coupled constraints and scope

BF16 stochastic rounding from FP32 with source16-bit random carry. FP16/FP8/FP4 grid rescaling uses different realization proofs and cannot inherit this bit-truncation formula directly.

Scope predicate: `(p.finite_bf16_range_or_guarded_special and p.random_bits_match_element_map and p.random_word_domain_defined)`.

- Precondition `pre0` (semantic): `p.finite_bf16_range_or_guarded_special and p.random_bits_match_element_map and p.random_word_domain_defined` — Normal range or exact special guard, fixed element RNG map, declared randomness domain.

## Local example and misuse

For a local instance of `convert`, identify the object `random_discarded_bit_rounding` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/random_discarded_bit_rounding@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/random_discarded_bit_rounding@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `carry_threshold` (proof): For residual d the carry occurs for exactly d of2^16 random words.
- `special_guard` (proof): No integer perturbation changes NaN/Inf/out-of-range classification before special conversion.
- `rng_identity` (proof): Realization substitutions preserve every logical element random word if bitwise replay is required.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
