# integer_operand_widening@1: exactly widened integer tile

## When to consider

When an MMA consumes a wider floating type than a stored integer operand, consider widening that operand's tile immediately before use. Preserve the stored signedness and every logical coordinate; select typed conversion or a matching paired prepack/converter. The pinned optimized path binds signed Int4 to BF16 with eight-value[0,2,1,3,4,6,5,7] storage order and CUDA>=13.1; raw code0 remains0, so an offset-binary q-8 decoder is not interchangeable. Scaling, tails and wide-buffer placement need their own supported bindings.

## Computation before and after

Signed/unsigned integer decode and exact floating widening with preserved logical order.

Extract each raw b-bit value u at its logical coordinate. For s4/s8 decode q=u when u<2**(b-1), otherwise q=u-2**b; for u8 decode q=u. Convert q exactly to f16 or bf16. The typed path consumes natural fragment order. The paired Int4/BF16 path consumes eight-value blocks already stored in order[0,2,1,3,4,6,5,7] and uses the matching converter to restore natural logical order while widening. Scaling, zero-point subtraction, accumulator grouping and wide-slot placement are separate transformations; this rule does not move a scale through an MMA.

Recover the original computation: Interpret the same stored integer values directly at each logical coordinate; the widened floating values are numerically identical integers.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| input_format | compile / boundary | ["s4", "s8", "u8"] | Stored integer interpretation; s4/s8 use two's-complement and u8 is unsigned. |
| compute_format | compile / boundary | ["f16", "bf16"] | Floating compute representation; all source-format integers are exactly representable. |
| conversion_path | compile / software | ["typed", "paired_shuffle"] | Ordinary typed conversion or the paired eight-value Int4/BF16 prepack-and-convert path. |

## Coupled constraints and scope

Exact widening of signed4/signed8/unsigned8 integer operands to f16/bf16. The shuffled implementation is limited to complete eight-value Int4/BF16 blocks; the pinned helper enables it for K-major Int4 with K divisible by8 and scale K at least8 on CUDA target version>=13.1. Typed conversion has natural ordering. New integer widths, zero-point semantics, mixed tails, packing permutations or target conversion policies require an explicit variant/binding. Register values are produced; wide shared/TMEM stage lifetime and all scale arithmetic remain outside this rule. No global pre-converted tensor allocation is implied.

Scope predicate: `p.input_format_matches_storage and p.fragment_coordinate_map_exact and p.storage_order_matches_path and p.converter_path_available`.

- Precondition `pre_format` (semantic): `p.input_format_matches_storage and p.fragment_coordinate_map_exact` — Raw integer bits and fragment coordinates match the requested signed/unsigned operand.
- Precondition `pre_pair` (implementation): `p.storage_order_matches_path and p.converter_path_available` — Encoder and converter agree on byte/value order and the target/compiler supports the selected path.
- Constraint `c_pair` (implementation): `conversion_path != 'paired_shuffle' or (input_format == 's4' and compute_format == 'bf16' and p.full_eight_value_chunks)` — The pinned shuffled path requires signed Int4 to BF16 and complete eight-value chunks.

## Local example and misuse

For a local instance of `map`, identify the object `exactly_widened_integer_tile` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/integer_operand_widening@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/integer_operand_widening@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Narrow source representation plus register-valued widened fragment.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `integer_identity` (exhaustive): Every admitted raw code becomes exactly its signed or unsigned integer value in the chosen floating dtype.
- `order_identity` (exhaustive): Prepack order composed with the selected conversion path preserves every logical coordinate exactly once.
- `backend_conversion` (test): Selected conversion and fragment loading implement the stated decode/order mapping for full chunks without unsupported target paths.

Conversion is exact for the declared formats; subsequent scale multiplication and reduction order have independent rounding obligations.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
