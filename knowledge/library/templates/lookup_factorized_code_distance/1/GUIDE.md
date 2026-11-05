# lookup_factorized_code_distance@1: lookup factorized code distance

## When to consider

When approximate vectors are sums of codebook entries, precompute query/codeword metric terms and scan integer codes by table lookups rather than reconstructing each vector. Two subquantizers with code[0]=1 and code[1]=0 require L0[1]+L1[0], not L0[1]+L0[0]. Different coarse lists must supply their own residual terms. Dropping a nonzero constant changes returned distances even if ranking survives.

## Computation before and after

When approximate vectors are sums of codebook entries, precompute query/codeword metric terms and scan integer codes by table lookups rather than reconstructing each vector.

For each query and selected coarse list, factor the squared-distance or inner-product expression into a list/query constant and per-subquantizer codeword terms. Materialize or combine lookup rows for the current query/list. Load rows cooperatively into shared memory and synchronize. Each candidate scans its packed codes, looks up the term at subquantizer*codes_per_subquantizer+code and sums all required terms with the constant. Prefetch the next candidate code into a separate register buffer, then rotate after consumption. Preserve list offsets for final identity recovery without writing one identity per distance.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| lookup_bits | compile / software | [16, 32] | Storage precision of distance tables; this changes rounding and needs numerical permission. |

## Coupled constraints and scope

NVIDIA PQ distance scan; approximate distance means distance to the quantized representation, not exact original input vectors. Precomputed residual and no-precomputed tables are distinct factorization bindings, with float/half lookup rounding explicit. Resource effects: Table size M*K*lookup_bits/8 per query/list versus per-candidate reconstruction dimension work; scan uses M lookups per code plus reduction and code loads. Global table precomputation adds storage amortized over candidates.

Scope predicate: `p.codebook_partition_defined and p.codes_valid and p.factorization_exact_for_contract and p.lookup_rounding_admitted and p.table_ready_and_capacity_valid`.

- Precondition `pre_0` (semantic): `p.codebook_partition_defined and p.codes_valid` — Every code maps to the correct subquantizer and codeword.
- Precondition `pre_1` (semantic): `p.factorization_exact_for_contract and p.lookup_rounding_admitted` — The metric decomposition matches decoded vectors and table rounding is permitted.
- Precondition `pre_2` (semantic): `p.table_ready_and_capacity_valid` — Tables fit selected storage and are published before scan.

## Local example and misuse

For a local instance of `map`, identify the object `lookup_factorized_code_distance` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/faiss/delta/lookup_factorized_code_distance@1/template.yaml`. Original outline: `knowledge/studies/faiss/delta/lookup_factorized_code_distance@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Table size M*K*lookup_bits/8 per query/list versus per-candidate reconstruction dimension work; scan uses M lookups per code plus reduction and code loads. Global table precomputation adds storage amortized over candidates.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `factorization` (proof): Expanded lookup expression equals the specified decoded-vector metric before admitted rounding.
- `code_table_alignment` (proof): Each code reads its own subquantizer row and query/list binding, never another row.
- `publication` (proof): Shared rows finish loading before any candidate lookup; no overwrite during use.
- `identity_recovery` (proof): Prefix/list offsets recover the exact original candidate identity from each emitted distance slot.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
