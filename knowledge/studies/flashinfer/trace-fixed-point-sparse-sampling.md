# Row-relative integer masses and sparse sampling

When reproducible probability prefixes span mixed FP32 exponents, align each
value to one row-maximum exponent and quantize it downward into a bounded integer
mass. The new [row_relative_fixed_point_mass@1](delta/row_relative_fixed_point_mass@1/template.yaml)
and [outline](delta/row_relative_fixed_point_mass@1/proof.yaml) separate that codec
from sorting, top-p comparison and finite-grid categorical sampling.

Route: `flashinfer/cake_sampling.py:2842` -> two-launch radix/slab chain or fused
tail -> `csrc/cake_sampling/cake_sampling_binding.cuh:655-735` and generated
kernel bundle. The retained [manifest](inputs/sources.json) locates the full
source. Representative `csrc/cake_sampling/generated/cake_sampling_kernels_part15.cuh:46` is
`kernel_cake_sparse_topp_sample_t32_i2`,32 threads/two items, up to64 live entries.
The full slab has1024 entries; manifest/native shape selection chooses a wider
scan when needed. Fused part1`:1742-1798,1933-1957` contains the same codec.

| Decision / concrete source binding | Rule field or boundary | Evidence |
|---|---|---|
| Part15`:77-92` packs sanitized FP32 bits and inverse ID, then a sorting network orders packets | Fixed value-descending/ID-ascending order, separate from codec | Smallest ID wins equal values; exact-k stage and sorted sparse top-p can split a boundary tie cohort, unlike AIR's whole-cohort threshold |
| `:336-352`: maximum effective exponent e0=max(E0,1); significand includes hidden bit only for normals; shift into a padded uint64 | `fraction_padding_bits=29`; other widths are inferred numerical choices | Shared real unit Delta=2**(e0-179); q_i=floor(w_i/Delta). The guarded shift>=64 path emits0; positive values below one unit can disappear |
| `:353-402`: exact uint64 thread/warp prefixes and warp totals | `pre_count`, `pre_coordinates`, `pre_lifetime` | Each q_i<2**53 and1024 values total<2**63. All regroupings preserve the prefix only when original order/identity is preserved |
| `:405-431`: cast total and exclusive/inclusive prefixes to double; target=float_p*double(total); keep positive q with target>double(exclusive) | Downstream comparison policy, not an integer-codec guarantee | Sorted prefix ends at the first crossing and can split ties by ID. FP64 conversion is not exact for every integer above2**53 |
| `:433-577`: Philox counter uses offset/4, row subsequence and offset%4 component; uint32 word becomes FP32 uniform | Full counter/component and endpoint recipe must be bound | Different vector/warp shapes need matching counter maps for same-seed equality. Rounded uniform can reach1; final-cutoff fallback is explicit |
| `:578-595`: compare double(u)*double(kept_int) against double prefix intervals; endpoint uses cutoff ID | Native finite-grid CDF, separate from ideal q_i/sum(q) | Integer prefixes alone do not prove exact categorical probabilities or identical endpoint behavior to another sampler |
| `:596-623`: optional renorm uses original v_item divided by retained integer mass converted back via row maximum | Explicit output-numerator policy | This is not automatically q_i/sum(q_i). Original numerator and encoded denominator can give a real sum slightly above1 before output rounding |
| `:77-92,336-352,405-413` and entry contract: sanitize invalid probabilities, infinity units, nonpositive p and zero rows | Separate special-row contract outside finite-positive codec | Infinite-row branch uses unit mass only for infinities and ignores top-p; all-zero row returns a fallback ID with zero renorm. Do not infer these policies from finite-row algebra |

For finite positive rows, write w_i=q_i*Delta+r_i with0<=r_i<Delta. Discarded
mass D=sum(r_i)<N*Delta. Normalized original mass is a mixture of the decoded
integer distribution and the residual distribution, giving total-variation
bound D/sum(w) on the unchanged row. That bound does not automatically cover a
changed top-p support at a crossing.

The [exact-unit check](checks/check_row_relative_mass.json) covers336 cases,
including padding0/1/8/29, normal/subnormal peaks, large exponent gaps and up to1024
entries. An independent rational floor verifies every encoded q and the mass/TV
bound; regrouped integer prefixes match the original order exactly.
At maximum1 with source padding29, Delta=2**-52 and value2**-53 becomes zero.
This unit depends on the maximum's exponent, not an exact max/2**53 division.

Two consumer witnesses keep the contracts separate. For values[1,3*2**-53],
source-style original numerators divided by decoded total sum to
9007199254740995/9007199254740994 rather than1, and differ from the integer
categorical vector. For an exactly normalized seven-value FP32 row in the
check, every codec residual is zero and total integer mass is2**54. Its last
positive mass1 has exclusive prefix2**54-1; converting to double rounds that
prefix to2**54. The literal source `top_p=1` comparison therefore marks the last
entry unkept. This is a logical arithmetic witness of a consumer boundary, not
a native CUDA execution claim. It constrains any reading of the entry comment
that every positive integer mass is necessarily retained at p1.

The candidate captures the downward mixed-exponent codec and exact integer scan.
[shared_exponent_integer_sum@1](delta/shared_exponent_integer_sum@1/template.yaml)
instead sums exact fractions/counts inside one exponent and rounds once.
[mass_rank_radix_boundary@1](delta/mass_rank_radix_boundary@1/template.yaml)
keeps whole value cohorts; this sorted sparse prefix may stop inside a cohort.
PDL ordering, sorting-network native maps, per-shape coverage, special rows and
finite RNG/FP64 boundary policies remain explicit realization/contract bindings.

Next reconcile GVR/GVR2 selection and the remaining generated/native roles in the
[sampling inventory](sampling-inventory.md), then continue recurrent, MoE and
distributed designs. Full-library closure is still open.
