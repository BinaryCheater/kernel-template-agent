# bounded_lattice_parameter_lookup@1: shared lattice parameter map

## When to consider

When a multiresolution lattice uses bounded learned tables, make its coordinate map and collision semantics explicit. Choose dense/modulo or compiled coherent hashing under the field's representation contract; use the identical map for gradient sums. At R=4,H=8, coordinates (1,0) and (0,1) both hash to entry1, so gradients 2 and 3 must contribute 5 to the same parameter. Hashing an arbitrary dense field with unequal values at those vertices is rejected by pre_field.

## Computation before and after

Bound lattice storage through a shared coordinate map and pull gradients back through its preimages.

For resolution R and coordinate c, form the source uint32 linear index L=sum_d c[d]*R**d and stride R**dimensions when the source safe-resolution branch permits it; otherwise set stride=UINT32_MAX and L=0. Dense and tiled address modes use L modulo table_entries. Hash uses the same L when table_entries>=stride, otherwise uses uint32 XOR_d(c[d]*a[d]) modulo table_entries, with coherent-prime factors [1,2654435761,805459861,3674653429] truncated to dimensions. Read feature f at level_offset+(h(c)*features+f). For training, map every logical vertex's gradient through exactly the same h and sum all contributions, including aliases within one stencil or between samples. Level offsets are measured in scalar parameters here; convert any source vector-entry offsets before binding. Use atomic or an equivalent complete reduction, with declared initialization and parameter-update order.

Recover the original computation: Expand the logical lattice field as T(c,f)=theta[level_offset+h(c)*features+f]. For each table parameter, sum logical gradients over its preimage; this is the chain rule for the admitted parameter-sharing map.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| address_mode | compile / software | ["dense", "tiled", "hash"] | Coordinate map admitted by the field parameterization; this choice may change parameter sharing. |
| table_entries | compile / software | {"lo": 1, "hi": 1073741824} | Allocated parameter vectors per level, derived from selected budget and level geometry. |
| dimensions | compile / boundary | [2, 3, 4] | Lattice coordinate dimensions; pinned compiled CoherentPrime grid menu. |
| features | compile / boundary | [1, 2, 4, 8] | Contiguous scalar parameters in each table entry. |

## Coupled constraints and scope

Per-level shared trainable tables, pinned to tiny-cuda-nn's compiled 2D/3D/4D CoherentPrime path and dense/tiled fallbacks. Hash collisions are intentional parameter tying, not lossless compression of arbitrary dense fields. No equivalence of different hash families or stochastic corner gradients is claimed. Coordinate generation and interpolation weights are separate inputs.

Scope predicate: `p.logical_field_factors_through_selected_map and p.resolution_positive and p.coordinate_and_u32_arithmetic_match and p.level_offset_valid and p.all_colliding_gradient_contributions_combined and p.gradient_epoch_initialized and p.no_parameter_update_during_reads and p.lookup_and_gradient_rounding_admitted`.

- Precondition `pre_field` (semantic): `p.logical_field_factors_through_selected_map` — The workload defines or admits T(c)=theta[h(c)]; arbitrary independent dense values cannot be silently hashed or tiled.
- Precondition `pre_map` (implementation): `p.resolution_positive and p.coordinate_and_u32_arithmetic_match and p.level_offset_valid` — Coordinate conversion, uint32 arithmetic, resolution and level segment agree across all consumers.
- Precondition `pre_gradient` (semantic): `p.all_colliding_gradient_contributions_combined and p.gradient_epoch_initialized and p.no_parameter_update_during_reads` — Training sums contributions for shared parameters in one owned gradient epoch before updating parameters.
- Precondition `pre_numeric` (implementation): `p.lookup_and_gradient_rounding_admitted` — Lookup casts and atomic accumulation order/precision satisfy the numerical policy.
- Constraint `c_capacity` (resource): `table_entries * features * p.parameter_bytes <= p.level_capacity_bytes` — Each level's table fits its allocation.

## Local example and misuse

For a local instance of `access`, identify the object `shared_lattice_parameter_map` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/tiny-cuda-nn/delta/bounded_lattice_parameter_lookup@1/template.yaml`. Original outline: `knowledge/studies/tiny-cuda-nn/delta/bounded_lattice_parameter_lookup@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: Multiple lattice vertices may intentionally share one trainable vector.. Storage: table_entries times features scalar parameters per level.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `address_agreement` (test): Every forward and backward lookup uses the same level, coordinate, branch and feature address within the allocation.
- `sharing_contract` (proof): All aliased lattice values are constrained to equal the same table parameter by the declared workload representation.
- `gradient_pullback` (proof): Each contributing logical gradient appears once in its physical parameter sum, including collisions, with an initialized accumulator and admitted rounding.

Atomic gradient order and FP16 versus FP32 accumulation change rounding; collisions change the represented field unless parameter sharing is already admitted.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
