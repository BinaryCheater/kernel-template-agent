# blocked_thread_register_layout@1: blocked thread register layout

## When to consider

When choosing how a tensor tile spreads over registers, lanes and warps, vary the hierarchy factors and their digit order separately. S=[2,4],T=[16,2],W=[2,2] gives a 64x16 block and eight registers per thread. A 128x128 tensor repeats that block 2x8 times, giving 128 registers per thread before liveness effects. Do not treat smaller replicated tensors as unique-writer maps.

## Computation before and after

Hierarchical register/lane/warp digits map a rectangular logical tile to physical thread registers.

Define block extents B0=S0*T0*W0,B1=S1*T1*W1. Decode lane id into (l0,l1) with extents (T0,T1), warp id into (w0,w1) with extents(W0,W1), and within-thread register id into (r0,r1) with extents(S0,S1), using the selected fastest dimension. Coordinate i=(w0*T0+l0)*S0+r0+q0*B0, j=(w1*T1+l1)*S1+r1+q1*B1 for outer repetition digits(q0,q1). Higher shape repetitions add registers. Source tensors smaller than block shape physically broadcast replicas; that regime requires an explicit replica ownership policy and is outside this first version.

Recover the original computation: Erase physical thread/register hierarchy to recover the same logical tensor coordinates and values.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| S0 | compile / software | {"lo": 1, "hi": 65536} | Rows per thread register subtile. |
| S1 | compile / software | {"lo": 1, "hi": 65536} | Columns per thread register subtile. |
| T0 | compile / software | {"lo": 1, "hi": 65536} | Lane rows per warp. |
| T1 | compile / software | {"lo": 1, "hi": 65536} | Lane columns per warp. |
| W0 | compile / software | {"lo": 1, "hi": 65536} | Warp rows per CTA. |
| W1 | compile / software | {"lo": 1, "hi": 65536} | Warp columns per CTA. |
| order | compile / software | ["row_major", "column_major"] | Fastest-varying dimension when decoding each hierarchy digit. |

## Coupled constraints and scope

Rank-two whole-block distributed tensors on 32-lane NVIDIA warps, power-of-two hierarchy factors and whole repetitions. Smaller-shape broadcast replicas are explicitly excluded; consumer-specific MMA layouts require a realization binding.

Scope predicate: `(T0 * T1 == 32 and p.factor_sizes_are_powers_of_two) and (p.tensor_shape_multiple_of_block_shape and p.logical_elements_assigned_once) and (p.register_capacity_and_launch_legal)`.

- Precondition `pre0` (semantic): `T0 * T1 == 32 and p.factor_sizes_are_powers_of_two` — NVIDIA warp has 32 lanes with compatible block factors.
- Precondition `pre1` (semantic): `p.tensor_shape_multiple_of_block_shape and p.logical_elements_assigned_once` — This version admits full block shapes or whole repetitions without physical broadcast replicas.
- Precondition `pre2` (semantic): `p.register_capacity_and_launch_legal` — Register subtiles and warps fit the selected launch.

## Local example and misuse

For a local instance of `access`, identify the object `blocked_thread_register_layout` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/triton/delta/blocked_thread_register_layout@1/template.yaml`. Original outline: `knowledge/studies/triton/delta/blocked_thread_register_layout@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Binding-specific counters, partials or output ranges.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `hierarchy_bijection` (proof): Every valid logical coordinate has one register/lane/warp/repetition decomposition.
- `consumer_layout` (test): Loads, stores, collective and MMA consumers interpret the chosen mapping consistently.

Integer counts must not overflow; ordering/rounding policy belongs to the bound operation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
