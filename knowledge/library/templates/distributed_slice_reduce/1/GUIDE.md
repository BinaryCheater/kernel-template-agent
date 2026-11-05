# distributed_slice_reduce@1: distributed slice reduce

## When to consider

When each lane needs only a slice of a vector reduction, exchange ownership while reducing instead of replicating the whole reduced vector in every lane.

## Computation before and after

When each lane needs only a slice of a vector reduction, exchange ownership while reducing instead of replicating the whole reduced vector in every lane.

Let L lanes each hold S slices of E values. First fold lane partners at XOR offsets S,2S,... below L. For m=S/2,...,1: lanes with group_index bit m clear swap slices r and r+m; shuffle the lower slot from the partner at XOR m*stride; merge it with the local upper slot into the lower slot. Move all coupled planes together. At completion lane g owns original slice g mod S in slots [0,E); permit only g<S to store. Compared with a full butterfly over S*E entries at every level, shrinking the live slice set reduces communication and spreads final stores.

Recover the original computation: Within one aligned NVIDIA warp subgroup, original slice g mod S is completely reduced in the first E slots of lane g; forget lane ownership to recover the full vector reduction.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| lanes | compile / software | {"lo": 1, "hi": 32} | L: selected subgroup size, power of two. |
| slices | compile / software | {"lo": 1, "hi": 32} | S: fragment size / slice_elems, derived from fragment shape. |
| slice_elems | compile / software | {"lo": 1, "hi": 1024} | E: coupled contiguous elements per owned slice. |
| stride | compile / software | {"lo": 1, "hi": 32} | Power-of-two lane spacing fixed by layout. |

## Coupled constraints and scope

Within one aligned NVIDIA warp subgroup, original slice g mod S is completely reduced in the first E slots of lane g; forget lane ownership to recover the full vector reduction.

Scope predicate: `(is_pow2(lanes) and is_pow2(slices) and is_pow2(stride) and slices <= lanes and lanes * stride <= 32) and (p.full_subgroup_active and p.same_plane_layout and p.associative_commutative_merge and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.full_subgroup_active and p.same_plane_layout and p.associative_commutative_merge and p.rounding_admitted` — Every lane participates; coupled planes have the same slice mapping.
- Constraint `c0` (implementation): `is_pow2(lanes) and is_pow2(slices) and is_pow2(stride) and slices <= lanes and lanes * stride <= 32` — Aligned subgroup fits a warp.

## Local example and misuse

For a local instance of `reduce`, identify the object `distributed_slice_reduce` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/quack/delta/distributed_slice_reduce@1/template.yaml`. Original outline: `knowledge/studies/quack/delta/distributed_slice_reduce@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `ownership` (proof): Induction on m halves the owned data domain and doubles represented input lanes.
- `unique_store` (proof): Only g<S stores slice g, avoiding duplicates when L>S.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
