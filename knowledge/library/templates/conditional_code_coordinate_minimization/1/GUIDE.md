# conditional_code_coordinate_minimization@1: conditional code coordinate minimization

## When to consider

When additive codebook encodings have coupled code choices, update one coordinate against fixed other codes using unary/binary tables, then retain the best complete reconstruction after perturbation. For M2, changing code0 reads the old code1; subsequent code1 update sees the new code0. Updating both coordinates from stale state in parallel is a different search algorithm. A worse perturbation must not overwrite retained best codes.

## Computation before and after

When additive codebook encodings have coupled code choices, update one coordinate against fixed other codes using unary/binary tables, then retain the best complete reconstruction after perturbation.

Build unary query/codebook terms and binary codeword interaction tables. For each vector and coordinate m, assign one thread to each candidate code k and compute unary[m,k]+sum over m2!=m of binary[m,m2,k,current_code[m2]]. Block-reduce to the minimizing code and update only coordinate m. Stream-order coordinates and sweeps. Perturb selected codes between attempts; evaluate full reconstruction error and replace best codes only on strictly smaller error, then restore the best state before the next attempt.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| sweeps | compile / software | {"lo": 1, "hi": 2147483647} | Number of coordinate sweeps per perturbation; heuristic budget, not convergence guarantee. |
| perturbations | compile / software | {"lo": 1, "hi": 2147483647} | Outer attempts; caller-selected search budget. |

## Coupled constraints and scope

NVIDIA additive quantizer ICM/iterated-local-search encoding. Not an exact global encoding optimizer; floating comparison and tie behavior follow block minimum and strict best selection. Resource effects: Precompute unary M*N*K and binary M*M*K*K terms; each coordinate evaluates K candidates with M-1 lookups rather than reconstructing D-dimensional vectors. Adds kernel ordering and code/error snapshots.

Scope predicate: `p.tables_match_codebooks and p.codes_valid and p.coordinate_updates_ordered and p.candidate_evaluation_complete and p.launch_capacity_valid and p.random_state_defined`.

- Precondition `pre_0` (semantic): `p.tables_match_codebooks and p.codes_valid` — Unary and pairwise terms are derived from the same codebooks and every code index is valid.
- Precondition `pre_1` (semantic): `p.coordinate_updates_ordered and p.candidate_evaluation_complete` — Each coordinate sees prior updates in the sweep, and full error evaluation precedes best-state selection.
- Precondition `pre_2` (semantic): `p.launch_capacity_valid and p.random_state_defined` — K candidate threads and dimension evaluation fit launch limits; perturbation RNG ownership is defined.

## Local example and misuse

For a local instance of `map`, identify the object `conditional_code_coordinate_minimization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/faiss/delta/conditional_code_coordinate_minimization@1/template.yaml`. Original outline: `knowledge/studies/faiss/delta/conditional_code_coordinate_minimization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Precompute unary M*N*K and binary M*M*K*K terms; each coordinate evaluates K candidates with M-1 lookups rather than reconstructing D-dimensional vectors. Adds kernel ordering and code/error snapshots.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `conditional_objective` (proof): For fixed other coordinates, the tabulated objective differs from full error only by a k-independent constant.
- `update_order` (proof): No coordinate consumes partially updated state of another coordinate kernel.
- `best_retention` (proof): Best error does not increase across attempts under the actual evaluated arithmetic and its matching code snapshot.
- `budget` (proof): Finite sweep/attempt counts terminate without claiming a global optimum.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
