# ragged_descriptor_rebase@1: ragged descriptor rebase

## When to consider

When a fixed-shape transfer descriptor must enforce a per-sequence upper bound, translate the sequence to the end of a large descriptor axis and cancel the address translation through auxiliary coordinates.

## Computation before and after

When a fixed-shape transfer descriptor must enforce a per-sequence upper bound, translate the sequence to the end of a large descriptor axis and cancel the address translation through auxiliary coordinates.

Choose B=2^30 and logical stride s. A valid sequence of length L at offset O starts at descriptor coordinate B-L; local q is valid iff B-L+q<B, equivalently q<L. One-extra-axis store realization shifts pointer by -B*s and uses auxiliary coordinate O+L with stride s, so address is base+(O+q)*s. Two-extra-axis realization keeps base mapped, adds axis coordinates B and O+L with strides2^34-s and s; base+(B-L+q)*s+B*(2^34-s)+(O+L)*s equals base+(O+q)*s modulo2^64. Lower bounds and all nonragged axes still need caller validity. Use one-axis only on native paths accepting the shifted pointer; source excludes large backward-shifted base for TMA loads.

Recover the original computation: NVIDIA TMA ragged sequence views, one extra axis for selected store paths or two extra axes preserving a mapped base. Source original rank<=4 or<=3 respectively; no arbitrary pointer-wrap legality is implied.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

NVIDIA TMA ragged sequence views, one extra axis for selected store paths or two extra axes preserving a mapped base. Source original rank<=4 or<=3 respectively; no arbitrary pointer-wrap legality is implied.

Scope predicate: `(p.nonnegative_length_and_offset_in_i32 and p.length_le_big_axis and p.descriptor_rank_supported and p.native_address_wrap_contract and p.actual_valid_addresses_allocated and p.tail_zero_fill_or_store_skip_defined)`.

- Precondition `pre0` (semantic): `p.nonnegative_length_and_offset_in_i32 and p.length_le_big_axis and p.descriptor_rank_supported and p.native_address_wrap_contract and p.actual_valid_addresses_allocated and p.tail_zero_fill_or_store_skip_defined` — Descriptor extents, native wrap semantics, valid physical allocation and tail transfer behavior.

## Local example and misuse

For a local instance of `transfer`, identify the object `ragged_descriptor_rebase` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/ragged_descriptor_rebase@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/ragged_descriptor_rebase@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `address_identity` (proof): Auxiliary coordinates cancel synthetic offset for every valid q.
- `bound_identity` (proof): Upper bound comparison accepts exactly0<=q<L under nonnegative local coordinates.
- `native_descriptor` (proof): Descriptor rank, byte stride conversion and base pointer mapping are accepted by selected backend.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
