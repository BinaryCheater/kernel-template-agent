# tensor_axis_permutation@1: strided permuted view

## When to consider

Use when a consumer can read a rank-three tensor through permuted strides; bind axis0/axis1/axis2 to the desired source-axis order. Permute extents and strides together and rewrite coordinates by the inverse permutation, for example order1,0,2 changes a BHR descriptor into HBR without copying. Bind scale-metadata views separately and validate the consumer's stride requirements before composing the contraction.

## Computation before and after

Permute shape and stride metadata with an inverse coordinate map.

Construct an output view with unchanged allocation and base offset. Its extent i is source_extent[axis_i] and its stride i is source_stride[axis_i]. For output coordinate j set source coordinate at axis_i equal to j_i. This step performs no element movement. Apply it independently to other operands or metadata only with their own descriptors; packed scale rows are not ordinary data coordinates.

Recover the original computation: Invert the axis permutation on coordinates and restore the original extents/strides; the physical element address is unchanged.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| axis0 | compile / software | {"lo": 0, "hi": 2} | Source axis represented by output axis0. |
| axis1 | compile / software | {"lo": 0, "hi": 2} | Source axis represented by output axis1. |
| axis2 | compile / software | {"lo": 0, "hi": 2} | Source axis represented by output axis2. |

## Coupled constraints and scope

Rank-three tensor access descriptors with a retained backing allocation. This changes the view rather than the stored values; writable aliasing remains the same as the source descriptor. Each contraction operand and each packed metadata descriptor must be checked separately against the downstream contract. Consumers requiring contiguous storage need a separate materialization, not a silent reshape.

Scope predicate: `p.rank == 3 and p.consumer_accepts_permuted_strides`.

- Precondition `pre_rank` (implementation): `p.rank == 3 and p.backing_allocation_retained` — The source descriptor has three extents/strides and its allocation remains live.
- Precondition `pre_consumer` (implementation): `p.consumer_accepts_permuted_strides` — The selected consumer supports the resulting strides and rewrites coordinate semantics consistently.
- Constraint `c_permutation` (semantic): `axis0 != axis1 and axis0 != axis2 and axis1 != axis2` — Every source axis has one destination axis.

## Local example and misuse

For a local instance of `access`, identify the object `strided_permuted_view` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/deepgemm/delta/tensor_axis_permutation@1/template.yaml`. Original outline: `knowledge/studies/deepgemm/delta/tensor_axis_permutation@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Shared backing allocation; no new data buffer.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `index_bijection` (exhaustive): Every valid output coordinate maps to exactly one valid source coordinate and vice versa.
- `address_identity` (exhaustive): The base plus stride dot coordinate address is identical in the two views.
- `consumer_contract` (test): The consuming operation interprets the renamed axes and strides consistently; a subsequent contiguous reshape is a separate transformation.

None for the view itself; a changed reduction order in a later contraction is a separate obligation.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
