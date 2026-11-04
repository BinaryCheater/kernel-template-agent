# Representative decision ledger

NVIDIA CUDA CUB and Triton paths; no emitted PTX or GPU run.

| Decision / concrete binding | Rule / field | Source |
|---|---|---|
| Real pair compose (a1*a0,a1*b0+b1); complex pair same chronological function order | ordered_affine_prefix@1 composition | selective_scan_common.h:138–172. This cannot reuse reduction_partition@1's commutative scope to reorder tokens. |
| Each tail token becomes (1,0), including complex representation | ordered_affine_prefix@1 tail_and_reset | selective_scan_fwd_kernel.cuh:219–237; a zero padding coefficient would destroy the saved final state. |
| Prior chunk aggregate seeds next CUB inclusive scan; thread0 stores new aggregate | ordered_affine_prefix@1 chunk_carry | fwd_kernel.cuh:240–258; B/C constant/variable paths change coefficient factoring and output multiplication at209–211,262–268. |
| Example maps(2,3),(5,7),(11,13), s0=17, chunk2 | ordered_affine_prefix@1 witness | Sequential states37,192,2125; aggregate(110,255), 110*17+255=2125. Reversing maps gives a different B, so witness detects order. |
| Chunk end state sums dt*x*B*exp(dA_end-dA_t) | chunked_decayed_outer_product_state@1 provisional realization | ssd_chunk_state.py:192–261; masks retain only sequence id of chunk last token. Clamp difference at0 requires nonpositive decay domain to be semantic identity. |
| Chunk state passing s=exp(dA_chunk)*s+local_state | ordered_affine_prefix@1 composition, sequential realization | ssd_state_passing.py:59–89; seq_idx mismatch sets carry scale0. IDs must name contiguous sequences without reuse across separated segments. |
| State update s=s*exp(A*dt)+B*dt*x, output sum(s*C)+D*x then SiLU(z) | Direct recurrence instance, output_tiling@2 | selective_state_update.py:95–132; selected tied-head branch changes broadcasts, not recurrence. In-place state rows need unique writers. |
| Blackwell architecture major10/12 restrict num_stages1 | Executable backend guard, not performance hypothesis | mamba3_siso_fwd.py:19–46; Mamba3 algorithm itself remains a named gap. |

Composition: input-dependent affine coefficients → ordered_affine_prefix@1 → state projection/reduction → optional residual and gate. SSD instead forms chunk-local affine maps then passes boundary state and adds within-chunk contributions. No cross-sequence carry is allowed.
