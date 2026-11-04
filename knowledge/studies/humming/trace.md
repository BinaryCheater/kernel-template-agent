# Instance decomposition

All source locations refer to the pinned revision in [inputs/sources.json](inputs/sources.json).
The [inventory](inventory.md) defines G01–G13. This file records the last verified boundary;
[family traces](trace-families.md) account for all remaining groups. Group dispositions
retain open obligations rather than converting branch presence into proof credit.

## Caller and selection boundary

The vLLM linear adapter (`vllm_linear_humming.py:24–94` in the partial caller archive)
permits CUDA at capability 7.5+, maps configured weight width and group size into a
Humming schema, prepares scales/zero points and locks, and delegates application through
`apply_humming_linear`. The followed utility (`vllm_humming_linear_utils.py:90–125,128–221,234–265`)
normalizes checkpoint dimensions, rejects asymmetric equal-width A/B, pads N to 256 and K
to 128 through Humming transform, flattens leading input dimensions and reshapes the result.
`vllm_utils_humming.py:69–79` maps these lazy helpers into `humming.transform` and
`humming.forward`; `humming/forward.py:166–222` optionally processes unquantized input
then calls GEMM. Other checkpoint adapters and schema fallbacks remain to be traced.

The MoE caller (`vllm_fused_humming_moe.py:112–126,160–182`) defaults to indexed GEMM,
or grouped contiguous GEMM for expert parallelism, with environment override. These paths
consume routing/expert metadata unlike the linear caller. Input processing and top-k
combination must be traced separately from the GEMM arithmetic.

Humming `ops/gemm.py:45–100` passes layer, compute and tuning configuration, inputs,
weights/scales, optional zero point/bias, routing metadata and locks to the launcher.
`kernel/humming.py:694–756` prepares either one configuration or an interval menu;
this ends at generated configuration and registration, not measured selection quality. `csrc/launcher/launcher.cpp:85–95`
selects a registered kernel from the runtime M interval. Architecture and shape heuristics
are host policy; source bounds and candidate generation are separate from measured performance.

## G09: contiguous bitstream storage

Concrete input: contiguous int32 tensor with 32 unsigned 3-bit codes, NVIDIA backend.
`ops/weight.py:50–58` requires last dimension divisible by 32, and allocates three int32
words. `kernel/pack_weight.py:23–44` binds `pack_weight_kernel<3>`, grid=1, block=32.
Thread 0 owns the complete group. Other threads have no visible global reads or writes.

For logical code i=10, its low bit is at global bit 30. Bits 0 and 1 go to word 0,
offsets 30 and 31; bit 2 goes to word 1, offset 0. `common_unpack_weight<3>` takes
two low bits from the first word and one from the second. The shift boundaries distinguish
this mechanism from packing eight INT4 codes wholly inside each word.

| Decision / binding and source | Kind | Rule and field | Evidence / outcome |
|---|---|---|---|
| `num_bits=3`, `ops/weight.py:50–58`, `kernel/pack_weight.py:21–25` | Independent compile variable | `bitstream_packing@1.code_bits` | Full 1..8 width enumeration in `checks/bitstream-result.json`; library quantized-weight domain motivates the proposed bounded width range. Public pack helper itself has no explicit 1..8 guard. |
| Container uint32, `kernel/pack_weight.cuh:4–6` | Fixed implementation binding | `word_bits=32` | No domain generalization to other word widths. |
| 32 codes per thread, `kernel/pack_weight.cuh:13,53–62` | Fixed grouping and derived storage | `program`, `c_whole_groups`; words per group=code_bits | A group has exactly 32*code_bits bits, so it always occupies a whole number of words. |
| Position divmod(i*width,32), `kernel/pack_weight.cuh:14–24` | Derived map and carry constraint | `bit_identity` | All 1,152 one-hot bit witnesses match independent concatenation; reverse-order mutation fails the oracle. Explained at the source arithmetic boundary. |
| Input mask, `kernel/pack_weight.cuh:18` | Numeric premise | `pre_code_range` | In-range codes preserve value; out-of-range high bits are intentionally discarded. No lossless claim outside the premise. |
| Block=32 threads, grid=ceil(length/1024), builder `:34–37` | Compiled launch policy | `group_ownership` | Lengths 32,64,992,1024,1056,2016,2048,2080 have exact-once code ownership. |
| Last CTA guards, `kernel/pack_weight.cuh:65–73,91–100` | Legal boundary and open language obligation | `memory_safety` | All-or-none global accesses per group. Inactive local arrays are still consumed by the unrolled helper. Open: compile exact code and inspect with sanitizer on NVIDIA. |
| Inverse, `kernel/pack_weight.cuh:29–50` | Recoverable storage contract | `backward_map` | Source-style two-word extractor agrees with the independent bitstream oracle for every case. No GPU result asserted. |

Proposed status: G09 partial — storage identity and ownership explained; inactive-thread CUDA
initialization and compiled memory safety remain open. Next evidence is an exact compiled
pack/unpack specialization with initcheck/memcheck, not a timing experiment.

## G08: one matrix prepack boundary

Concrete binding: a 64x64 INT4 code matrix, A width 16, unpacked input, no preprocessing,
no mini-block transpose, interleave_mode=3, no packed-K and no native conversion.
`kernel/repack_weight.py:19–69,90–94` binds `weight_repack_nk<4,16,false,false,false,false,0,false,false>`
with grid=(1,1,1) and block=32. `kernel/process.cuh:143–145` requires N padded to 64
and K padded to a multiple of 32; this witness satisfies both.

| Decision / binding and source | Kind | Rule and field | Evidence / outcome |
|---|---|---|---|
| Shared 64x64 gather, `kernel/process.cuh:135–164,188` | Fixed staging + barrier | `physical_storage@1`, `staged_relayout@1` has a different transfer direction; not bound | Shared input staging is visible; do not silently bind the epilogue-specific staged_relayout program to this input transform. |
| N=64, K subtiles=16, `kernel/process.cuh:190–204,277–319` | Instance tile and global layout | `offline_operand_prepack@1` tile_rows=16, tile_cols=64 | Four output rows of 128 words represent all 4096 INT4 codes. This is 2048 bytes. |
| Mixed-radix tmp order, `kernel/process.cuh:231–265` | Coordinate permutation | `element_order=mma_fragment_interleaved` | All 4096 uniquely labelled coordinates have distinct destinations and match an independently derived inverse. |
| A width=16, B width=4, mode=3, `kernel/process.cuh:8–15,32–49` | Consumer-coupled interleave | Additional source binding; old rule lacks a numeric A-width variable | Nibble permutation is [0,2,4,6,1,3,5,7] for both low bits and sign bit. Changing A width changes this permutation; this dependency belongs to the source binding until other widths are checked. |
| Coordinate (17,43) | Identifiable witness | `permutation_inverse` premise only | Output word=277, nibble=5. Identity interleave is bijective but wrong and is rejected. See `checks/repack-result.json`. |
| Packer-to-loader-to-MMA agreement | Structural composition obligation | `offline_operand_prepack@1.permutation_inverse`, `scale_alignment` | Recovered for A16/B4 by `checks/composition.py`: S2R half-group load, nibble extraction and PTX B map agree for all 8192 coordinates of N128/K64. Other formats and the A/output maps remain open (GAP01/GAP09). |

G08 closes as partial. The A16/B4 consumer composition is checked; odd-width compression,
other A widths, preprocessing, alternate interleave modes, packed-K/native layouts, expert axes
and padded tails are retained in GAP09. Word277 refers to N64 storage; the same coordinate
is word 533 in N128 storage, as used by `checks/composition.py`.

## Remaining family traces and closure

See [trace-families.md](trace-families.md) for decisions D01–D53 across G01–G13,
including source policy substitution on SM90, ordinary/WGMMA/UMMA/MXMMA bindings,
expert scheduling, epilogues, quantization, input phases, MXFP4 rebase and the benchmark.
Every row closes as explained, partial or open. [delta/gaps.yaml](delta/gaps.yaml) gives
next evidence for the open boundary of every group. This closes the source-study pass;
it does not certify all source branches or all compiled specializations.
