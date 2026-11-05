# truncated_range_midpoint_quantization@1: truncated range midpoint quantization

## When to consider

When the requested scalar format defines range-truncated codes with midpoint reconstruction, separate original encoding metadata from prepared decoder scale and offset. a0,r15,bits4: x1.9 encodes1 and decodes1.5, while nearest-even gives2. x15 encodes15 and decodes15.5. Do not clamp the reconstructed endpoint or substitute nearest-even.

## Computation before and after

When the requested scalar format defines range-truncated codes with midpoint reconstruction, separate original encoding metadata from prepared decoder scale and offset.

With original minimum a, range r>0 and Q=2**code_bits-1, encode q=truncate(clamp((x-a)/r,0,1)*Q). Prepare decoder scale s=r/Q and offset b=a+0.5*s, then decode b+q*s. Cache per-feature (b,s) in shared memory when ranges vary by dimension; uniform ranges use scalar metadata. Keep original (a,r) for encoding, since prepared (b,s) would implement a different quantizer. Pack codes separately.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| code_bits | compile / software | [4, 6, 8] | Numerical format choice, not a free performance knob for an unchanged decoded vector. |

## Coupled constraints and scope

NVIDIA scalar quantizer QT4/QT6/QT8 and uniform4/uniform8 finite positive-range paths. FP16 and direct8 conversion have different policies and are separate realizations. This implements a requested lossy format; no original-vector error guarantee is implied. Resource effects: One scale/offset pair per dimension or one uniform pair; decoding replaces a division and half-step expression by prepared affine multiply/add. Low-bit storage handled independently.

Scope predicate: `p.finite_inputs_and_metadata and p.range_positive and p.metadata_owner_matches_feature and p.numeric_policy_requested`.

- Precondition `pre_0` (semantic): `p.finite_inputs_and_metadata and p.range_positive` — Original minimum and range are finite, range is positive, and normalized arithmetic is admitted.
- Precondition `pre_1` (semantic): `p.metadata_owner_matches_feature and p.numeric_policy_requested` — Uniform or per-feature range metadata agrees between encoder and decoder, with this truncation/midpoint policy requested.

## Local example and misuse

For a local instance of `map`, identify the object `truncated_range_midpoint_quantization` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/faiss/delta/truncated_range_midpoint_quantization@1/template.yaml`. Original outline: `knowledge/studies/faiss/delta/truncated_range_midpoint_quantization@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: One scale/offset pair per dimension or one uniform pair; decoding replaces a division and half-step expression by prepared affine multiply/add. Low-bit storage handled independently.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `numeric_policy` (proof): Clamp and truncate at all bucket boundaries according to the actual floating arithmetic.
- `metadata_identity` (proof): Each feature uses its original encoder metadata and corresponding prepared decoder metadata.
- `decode_boundary` (proof): Maximum code decodes to a+r+0.5*r/Q, which is the declared policy rather than an assumed clamp to a+r.
- `publication` (proof): Per-feature shared decoder metadata is initialized and synchronized before use.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
