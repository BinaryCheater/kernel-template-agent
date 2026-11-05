# conditional_log_magnitude_transport@1: conditional log magnitude transport

## When to consider

When small-magnitude expert outputs tolerate quantization, conditionally pack each signed value into a logarithmic magnitude code and retain raw blocks when the shared range is unsuitable. contrib/nccl_ep/device/ll_ep.cuh:1092-1277,1435-1546,1835 onward. Zero and equal-amplitude blocks fall through condition or explicit zero decoding.

## Computation before and after

When small-magnitude expert outputs tolerate quantization, conditionally pack each signed value into a logarithmic magnitude code and retain raw blocks when the shared range is unsuitable.

For magnitudes per128-channel block compute amax, amin, Lmax=log2(amax), Lmin=max(log2(amin),Lmax-32); zero amin uses clipped floor. Code only when Lmax<0 and Lmin<Lmax consistently across participating lanes, otherwise copy raw BF16. Let step=(Lmax-Lmin)/510, r=2-log2((1+2^step)/2)/step. Encode q=floor(max(log2(abs(x))/step+r-Lmin/step,0)), preserve sign separately. Require q fits9 bits. Pack16 q values into144 bits followed by16 sign bits (five32-bit words). Decode q0 as zero, otherwise signed 2^((q-1)*step+Lmin); accumulate decoded value times supplied routing weight in FP32. Range metadata and mode prefix determine payload offsets; sender and receiver must derive identical mode.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| magnitude_bits | compile / software | {"lo": 9, "hi": 9} | Observed9-bit magnitude menu; one additional sign bit. Other formats need a new realization. |
| block_channels | compile / software | {"lo": 128, "hi": 128} | Observed shared min/max range group. |

## Coupled constraints and scope

Concrete LogFMT-10 source, not FP8 E4M3 or affine scale quantization. Conditional lossy transport; no error bound or equality to uncompressed arithmetic is inferred. Resource effects: 16 BF16 values32bytes become20bytes plus shared block range metadata; variable segment addresses require mode prefix.

Scope predicate: `p.bfloat16_source_and_quantization_admitted and p.block_range_metadata_shared and p.finite_values_and_zero_handling_bound and p.native_approx_log_exp_error_admitted and p.packed_offsets_follow_block_mode_prefix_and_sign_mapping`.

- Precondition `pre_0` (semantic): `p.bfloat16_source_and_quantization_admitted and p.block_range_metadata_shared` — Source and receiver agree on128-channel amax/amin and raw-versus-coded decisions.
- Precondition `pre_1` (semantic): `p.finite_values_and_zero_handling_bound and p.native_approx_log_exp_error_admitted` — Approximate log2/exp2 and zero underflow behavior are explicit, not an exact identity.
- Precondition `pre_2` (semantic): `p.packed_offsets_follow_block_mode_prefix_and_sign_mapping` — Variable coded160-bit versus raw256-bit segments retain exact coordinate identity.

## Local example and misuse

For a local instance of `map`, identify the object `conditional_log_magnitude_transport` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/nccl/delta/conditional_log_magnitude_transport@1/template.yaml`. Original outline: `knowledge/studies/nccl/delta/conditional_log_magnitude_transport@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: 16 BF16 values32bytes become20bytes plus shared block range metadata; variable segment addresses require mode prefix.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `code_identity` (proof): Packed bit positions and sign bits invert without coordinate mixing; raw blocks are byte-preserving.
- `quantization` (proof): Decoded values equal declared lossy codebook; rounding threshold and approximate function error satisfy admitted policy.
- `range_mode` (proof): All participants choose same range/mode and prefix lengths so raw/coded boundaries remain synchronized.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).
