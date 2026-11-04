# Grouped mixed-input GEMM: segment work, scale placement and paired waits

When one tensor contains unequal contiguous groups, round work counts **inside each
group** before scheduling. A work index then needs both a group identity and its
physical start, not just a global tiled coordinate. If two adjacent transforms
share one empty-slot wait, the later credit must certify retirement of every
older reader that either transform could overwrite. These are the retrieval cues
for the two new rules below.

Scope: both `grouped_mixed_input_gemm.py` and
`grouped_mixed_input_gemm_acc_scale.py`, plus their grouped host preparation. Paths
are relative to the retained pinned CUTLASS tree in [the manifest](inputs/sources.json).
The [four-file inventory](inputs/mixed-input-variants.json) and
[concrete binding](inputs/grouped-mixed-binding.json) retain the source instances.
Batched conversion is covered by [the widening trace](trace-mixed-input.md).

## Contract and concrete branch

Both source frontends accept M256/n16/K512/L4, Int4 A, BF16 B/C, FP32 accumulation,
A scale M1/K256, MMA256×16×128, cluster2×1 and two-CTA instructions. The host's n is
an average: B/C contain fused N64; A and its scales have four group batches. Group
boundaries are immutable prefix metadata. A discriminating admissible partition is
lengths[8,0,40,16], prefix[0,8,8,48,64]. The actual random generator may choose other
aligned lengths. This witness tests the partition, not a claim about its random seed.

| Decision and source | Actual binding | Rule, invariant or gap |
|---|---|---|
| `mixed_input_host_utils.py:234–320` | B(1,fusedN,K), C(1,M,fusedN), fusedN=n×groups; prefix length groups+1, alignmentN=16×8/Bbits | Representation is fixed by the workload. Empty groups are allowed. Uniform generation uses floor division, but its actual caller makes fusedN divisible by groups; no caller remainder-loss defect is asserted. |
| `utils/mixed_input_helpers.py:795–832,862–873,938–983` | Six-field scalar cursor; each group contributes ceil(length/(CTA_N×clusterN)); one-based internal index, public index minus1 | New [segmented_axis_work_decode@1](delta/segmented_axis_work_decode@1/template.yaml): physical origin and signed remaining distance, immutable complete prefixes, nonnegative monotone queries and bounded Int32 sums. |
| Regular scheduler:1020–1079; acc-scale scheduler:923–950 | Static persistent scheduling with M inner; N queries increase or repeat. Work descriptor includes M,N,group and distance_to_boundary | Equal queries preserve cursor state. Terminal sentinel is checked before a consumer indexes A/scales. A peer whose remaining distance is nonpositive keeps the collective identity and participates; its loads must remain safe. |
| Regular epilogue:1817–1857 onward; acc-scale:1751–1802 | Full group-local N tiles use TMA. Group tails use SIMT with row and group-end predicates | Explicit family variant `group_boundary_hybrid`: domains are disjoint and cover each valid output once. `staged_async_output_store@1` applies only to the full-tile TMA stream; its ring clock advances only for TMA subtiles. `boundary_predication@1` runtime_length/fill=none supplies tail masking, not the whole hybrid ownership mapping. |
| Both transform paths; helper conversion | Signed4/8 or unsigned8 exact widening; paired shuffle only under its original gates | Same [integer_operand_widening@1](delta/integer_operand_widening@1/template.yaml) mapping/premises, new instances. Per-group A/scale selection is supplied by the new decoder. Wide TMEM/SMEM placement still follows A major mode. |
| Acc-scale constructor:130–158; MMA:1462–1525; epilogue:1641 onward | ConvertScale only; scaleK is a multiple of tileK. Reset raw FP32 partial for every scale group, multiply after partial completion, accumulate into zero-initialized final registers, cast once | Conditional reuse of DeepGEMM [block_scaled_partial_accumulation@1](../deepgemm/delta/block_scaled_partial_accumulation@1/template.yaml), partialK256, granK_A256, granK_B512, granM1, granN64. B has logical unit scale over the full K512/N64 domain; no B-scale tensor exists physically. Numeric reassociation must be admitted. The old SM90 WGMMA Realisation supplies no SM100 implementation. |
| Acc-scale transform pair; state helper:1852–1860; stage allocator:2074–2079 | Clone current slot then advance; wait both narrow inputs, acquire only the second wide slot, publish both separately. Odd tail acquires its own slot. Narrow/wide stage counts at least2 | New [ordered_slot_wait_coalescing@1](delta/ordered_slot_wait_coalescing@1/template.yaml), batch_width2. One cooperative producer sequence, exact generation, prefix-ordered retirement and batch_width<=stages are required. Consumer readiness/publication and individual last-read obligations remain unchanged. |
| Host preparation:131 | Computed A scales are overwritten with random integers | Upstream test-data fixture, not evidence of a production quantizer or a general numeric policy. Negative scale extents and empty-K frontend gaps from the batched trace remain explicit; accepted predicates alone establish neither support nor an initialized empty reduction. |

The decoder differs from `online_group_work_decode@1`: that old rule fixes a
32-lane prefix-window mapping and leaves physical decode external. It does not
supply scalar boundary traversal, signed peer tails or the new physical origin.
For CTA_N16/clusterN1 the witness has five work units, whereas rounding fusedN64
once gives four. Query1 means group2 at physical N8, not global N16. Cooperative
N peers can lie wholly beyond a group end; dropping them from collective work is
not justified by an output mask.

## Composition and numerical boundary

For the accumulator-scale branch, decode the group-local origin first. It selects
A/scale batch and the B/C fused-N window. Widen integer A exactly, then externally
stage operands and produce one raw FP32 partial per K256 scale group. The old
partial-accumulation rule promotes those partials only under its numerical scope.
Finally the hybrid output variant routes full group tiles through the old TMA
source ring and tails through predicated SIMT stores. Paired empty waits refine
only the transformed-A slot protocol; they do not establish the decoder, numeric
policy, staging layouts or destination completion. These dependency edges are
recorded in the [binding](inputs/grouped-mixed-binding.json), with open native
edges rather than a claimed whole-kernel reconstruction.

Scale relocation is not bit-identical even for valid BF16 scales. Put qA=7,B=1 at
k0 with scale145/128, and qA=-7,B=145/128 at k256 with scale1; all other inputs zero.
Scaling A in BF16 before MMA gives 7.9375-7×145/128=1/128. Promoting FP32 partials
gives 7×145/128-7×145/128=0. Both results survive the final BF16 cast. Therefore
exact-real factoring does not discharge `numeric_contract_allows_partial_scaling`;
the rule rejects the same binding when that premise is false.

A later empty credit also cannot justify paired writes merely because release
calls were issued in order. If old slot1 retires while old slot0 remains live,
waiting on1 and overwriting both is unsafe. The rule requires actual completion
to form a retired prefix. Native UMMA credit ordering and progress must establish
that premise. Its cost counts ceil(items/batch_width) empty waits **per finite
work sequence**, replacing only that component of `per_slot_handshake@1`; all
other costs remain. It is an API-event count, not a measured instruction or timing
claim. Sum counts over work tiles separately when each tile starts a fresh pair tail.

## Retained evidence

[Grouped checks](checks/grouped-mixed-input-result.json) execute the retained scalar
cursor and shape/scale predicates after documented AST substitution of Int32 and
ceil-div operations. There are1800 cursor domains and22506 monotone/repeated
queries, nine output coverage domains (including50 inactive peer instances),
924 ordered-credit domains and nine exact-rational scale partition cases. The
five-versus-four work-count witness, duplicate writes without group clipping,
out-of-order credit counterexample and BF16 rounding counterexample distinguish
the proposed boundaries. [Guard cases](inputs/grouped-mixed-binding-cases.json)
check positive bindings, rejected premise mutations and batch2/stage1 rejection.

This is source extraction with scalar/order/numeric checks. Native group clipping,
lane maps, converter instructions, prefix retirement, descriptor visibility and
accepted numerical tolerance remain implementation obligations. No CUDA compile,
GPU run, Realisation or Store admission is claimed.
