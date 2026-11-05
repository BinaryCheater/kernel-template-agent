# causal_window_state@1: causal window state

## When to consider

When a depthwise causal stencil is evaluated across invocations, keep its last W-1 inputs as explicit state instead of rereading the complete prefix; a ring buffer avoids shifting retained history.

## Computation before and after

When a depthwise causal stencil is evaluated across invocations, keep its last W-1 inputs as explicit state instead of rereading the complete prefix; a ring buffer avoids shifting retained history.

For each owned (state_row,channel), recover W-1 prior samples in chronological order. A ring of length H stores logical position q at q mod H, with caller-supplied next position p. Read positions p-W+1 through p-1 before overwriting any slot. For each new x_i write it at (p+i) mod H, append it to the register window, emit bias+sum_j weight[j]*window[j] for j0..W-1, optionally SiLU and cast, then shift the register window by one. Expose next position p+advance to the caller; this source consumes cache_seqlens but does not increment them. Linear state is an alternative layout containing the trailing H samples in chronological order. Initialize missing history to declared zeros or supplied initial samples; never infer it from uninitialized state.

Recover the original computation: Depthwise NVIDIA causal convolution widths2–4 with linear or circular per-channel state. This is a boundary-state realization of the same full-sequence stencil, not a different convolution or a commutative reduction over time.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| width | compile / software | {"lo": 2, "hi": 4} | W: workload stencil width; pinned compiled menu2/3/4. |
| history_capacity | compile / software | {"lo": 1, "hi": 2147483647} | H: caller-selected state length, at least W-1. |

## Coupled constraints and scope

Depthwise NVIDIA causal convolution widths2–4 with linear or circular per-channel state. This is a boundary-state realization of the same full-sequence stencil, not a different convolution or a commutative reduction over time.

Scope predicate: `(history_capacity >= width - 1) and (p.history_matches_logical_prefix and p.unique_state_row_channel_writer and p.position_nonnegative and p.state_addresses_valid and p.rounding_admitted)`.

- Precondition `pre0` (semantic): `p.history_matches_logical_prefix and p.unique_state_row_channel_writer and p.position_nonnegative and p.state_addresses_valid and p.rounding_admitted` — History/position agree, independent writers, valid allocation and admitted arithmetic.
- Constraint `c0` (implementation): `history_capacity >= width - 1` — State can recover the entire left halo.

## Local example and misuse

For a local instance of `map`, identify the object `causal_window_state` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/causal-conv1d/delta/causal_window_state@1/template.yaml`. Original outline: `knowledge/studies/causal-conv1d/delta/causal_window_state@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: not specified. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `window` (proof): Before output i the register window contains exactly x[p+i-W+1..p+i].
- `state_commit` (proof): After the call the ring decodes the trailing H samples at next position p+advance, including wraparound.
- `ownership` (proof): State row indirection must not cause concurrent writers to the same channel.

Floating-point reassociation, casts and approximate functions require an admitted numerical contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
