# offset_indexed_sparse_incidence@1: offset indexed sparse incidence

## When to consider

When sparse convolution reuses one geometric neighbor relation across channels, materialize an offset-indexed incidence map once and consume only present edges. For 1D stride-one padded convolution, inputs at 0 and 2 both contribute to output 1 at different offsets; output 1 has two bits, while absent input 1 leaves the center pair -1. Reject duplicate coordinates or insufficient pair capacity.

## Computation before and after

When sparse convolution reuses one geometric neighbor relation across channels, materialize an offset-indexed incidence map once and consume only present edges.

For each input and kernel offset, evaluate the integer convolution coordinate transform and bounds/divisibility predicate. Deduplicate valid output coordinates into output identities. Store pair[offset,output]=input, otherwise -1; set bit offset%32 in word offset//32 for each present pair. Consumers sum feature[input,c]*weight[offset,c,o] only for present entries. Native format instead compacts input/output pairs per offset. Submanifold format restricts output identities to existing input sites.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| mask_word_bits | compile / software | [32] | Source bit word width; number of words is ceil(kernel_volume/32), not a free choice. |

## Coupled constraints and scope

NVIDIA generated CUDA sparse convolution with unique active coordinates. General, transposed and submanifold coordinate predicates are separate bindings. Mask split partitions offset bits without omitting or double counting a contribution. Resource effects: Pair storage O(kernel_volume*active_outputs); bit masks add ceil(kernel_volume/32) words per output and amortize geometry across channels.

Scope predicate: `p.unique_input_coordinates and p.geometry_valid and p.capacity_sufficient`.

- Precondition `pre_0` (semantic): `p.unique_input_coordinates` — Input coordinates uniquely identify active sites.
- Precondition `pre_1` (semantic): `p.geometry_valid and p.capacity_sufficient` — Coordinate transform is valid and output/pair storage covers all admitted sites.

## Local example and misuse

For a local instance of `map`, identify the object `offset_indexed_sparse_incidence` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/spconv/delta/offset_indexed_sparse_incidence@1/template.yaml`. Original outline: `knowledge/studies/spconv/delta/offset_indexed_sparse_incidence@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Pair storage O(kernel_volume*active_outputs); bit masks add ceil(kernel_volume/32) words per output and amortize geometry across channels.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `edge_exactness` (proof): Each valid geometric edge appears once and no missing neighbor contributes.
- `output_identity` (proof): Competing input contributors share the same deduplicated output identity.
- `mask_publication` (proof): Masks begin at zero, competing bits use atomic OR, and map/mask writes complete before consumption.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
