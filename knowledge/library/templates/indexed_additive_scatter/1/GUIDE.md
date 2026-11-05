# indexed_additive_scatter@1: collision preserving indexed accumulation

## When to consider

When gathered rows generate gradients for their original storage, scatter-add through the same indices and retain every duplicate contribution. For indices [2,0,2,-1], source rows0 and2 both add to destination2; the sentinel does not access memory. Initialize once across all chunks, guard the current logical extent, and wait for all adds before consumption. A read-side gather or disjoint peer-shard reduction cannot supply this collision contract.

## Computation before and after

Scatter each logical contribution through its original index and atomically add at collisions.

For logical source row s, runtime index I[s], and feature d, implement Y[k,d]=Y_initial[k,d]+sum_{s: valid(s) and I[s]=k} X[s,d]. Load the immutable index with its source-row identity. Predicate invalid rows before forming or dereferencing a destination address. Each valid source/feature owner issues atomic_add_f32(Y[I[s],d],X[s,d]). A four-element implementation groups four legal adjacent features without assuming a vector-wide atomic transaction. Duplicate indices issue all their separate contributions, including across CTAs, tokens and chunks. Initialize the destination once, not per producer or chunk; preserve a caller-supplied accumulator when accumulation is in-place. Publish completion only after every contributor finishes. No deterministic reduction order or gradients through the discrete index choice are introduced.

Recover the original computation: Group the completed atomic additions by destination coordinate and recover the original indexed sum with exactly the same contribution multiplicities and admitted floating-point order.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| vector_width | compile / software | ["scalar", "four_elements"] | Scalar FP32 adds or a supported four-element vector add with per-element atomicity. |

## Coupled constraints and scope

FP32 additive scatter through a runtime row index, with scalar or supported four-element native adds. Source sparse MLA accumulates dK/dV into existing FP32 buffers. Invalid sentinels and shortened sequence views need explicit guards or a proven caller bound. Native atomic costs are unpriced. Assignment scatter, unique-only deduplication, deterministic reductions and vector-wide snapshot consistency are outside this version.

Scope predicate: `p.immutable_row_index_map and p.invalid_rows_predicated_before_destination_access and p.valid_indices_within_logical_and_allocation_bounds and p.each_logical_contribution_owned_once and p.duplicate_indices_retain_all_contributions and p.output_initial_state_defined_once and p.atomic_fp32_add_supported and p.atomic_addresses_and_vector_lanes_valid and p.no_nonatomic_competing_writers and p.all_contributors_complete_before_consumer_or_reset and p.unordered_fp32_addition_admitted and p.finite_contributions_and_accumulation_range_admitted`.

- Precondition `pre_map` (semantic): `p.immutable_row_index_map and p.invalid_rows_predicated_before_destination_access and p.valid_indices_within_logical_and_allocation_bounds` — Each contribution retains its original index; invalid slots are skipped before pointer access and every remaining index is valid for its sequence/view.
- Precondition `pre_contributors` (semantic): `p.each_logical_contribution_owned_once and p.duplicate_indices_retain_all_contributions and p.output_initial_state_defined_once` — Each logical contribution is issued once even when destinations collide; output starts at the specified existing value or zero, initialized once for the full accumulation.
- Precondition `pre_atomic` (implementation): `p.atomic_fp32_add_supported and p.atomic_addresses_and_vector_lanes_valid and p.no_nonatomic_competing_writers` — Selected atomic width and alignment are legal for every lane; overlapping destination updates all use compatible atomics.
- Precondition `pre_completion` (implementation): `p.all_contributors_complete_before_consumer_or_reset and p.unordered_fp32_addition_admitted and p.finite_contributions_and_accumulation_range_admitted` — Consumers/reset wait for all adds, and the numerical policy admits the actual finite FP32 additions and nondeterministic interleaving.

## Local example and misuse

For a local instance of `reduce`, identify the object `collision_preserving_indexed_accumulation` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flash-attention/delta/indexed_additive_scatter@1/template.yaml`. Original outline: `knowledge/studies/flash-attention/delta/indexed_additive_scatter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: All contributors share one indexed destination accumulator without storing separate full-size partial outputs.. Storage: Destination accumulator and retained immutable indices; contention depends on index distribution.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `mapping_and_multiplicity` (test): Every valid source/feature contributes once to the intended destination; duplicates all contribute and invalid indices cause no destination access.
- `atomic_lowering` (test): Selected scalar/vector atomic lowering, alignment, lane bounds and memory scope prevent lost updates.
- `initialization_and_completion` (test): One defined initial state precedes every add, and every final reader/reset follows all contributors across chunks and CTAs.
- `numeric_order` (test): The actual finite FP32 accumulation and unordered contributor order satisfy the caller's numerical policy.

Atomicity prevents lost updates but does not fix FP32 addition order; colliding contributions can round differently across executions.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
