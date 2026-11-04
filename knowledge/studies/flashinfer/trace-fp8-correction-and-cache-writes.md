# FP8 correction and complete quantized-cache writes

Sources refer to the retained [revision](inputs/sources.json). This trace adds
one record-ownership candidate and concretizes numeric/cache boundaries without
changing the user-paused CUTLASS study or its rule versions.

## Monolithic FP8: one serial online state across alternating groups

`flashinfer/cute_dsl/attention/monolithic/mla_decode_fp8.py:3525-3574` makes
two softmax groups inherit each other's previous **global** row max/mass in
serial tile order. They are not independent key-partial states to combine at
the end. Ping-pong readiness publishes the previous complete state before its
peer starts the next online update.

At :3560-3565, a small max increase rolls back the reference itself before
forming the correction. Thus old numerator and denominator keep one frame;
skipping just the O multiplication while adopting a new denominator reference
would be a different, incorrect recurrence. Constructor :217-226 bounds the
threshold by log2(448), since lagged-reference weights can reach 2**threshold
before E4M3 conversion. Public preparation in `monolithic/mla_decode.py:860`
uses threshold zero. online_rescaled_reduction@2 binds that default and the
nonnegative threshold subset through 8 with gain 1. The constructor's larger
interval (8,log2(448)] is outside its variable domain, so it remains an explicit
range extension gap rather than a silent binding or an edit to paused rules.

At :3571-3588, correction/reference metadata can publish before final row mass;
intermediate correction work does not consume that mass. :3654-3673 patches
updated row_sum, then :3675-3710 hands control to the next group. Final tiles
publish the completed mass through a separate branch and perform a parity-tail
rendezvous before persistent next-work reuse. These are **field-specific
publication and serial state relay** realization gaps: a single record-ready
flag does not prove every field is already valid, and the last-tile path cannot
reuse an intermediate placeholder. Native TMEM loads/stores and named-barrier
ordering still need the selected implementation binding.

The modular `cute_dsl/attention/mla_decode_fp8.py:37-40,469-621` delegates
loader, MMA, compute and correction roles to separate objects; `roles/mla_compute.py`
retains row-max/exponent/mass and P/correction publication. This is a separate
role/resource implementation of the same chain, not evidence that monolithic
ping-pong or partial-field publication carries over unchanged.

## Cache reads: payload and metadata share one valid identity

`include/flashinfer/attention/sparse_mla_sm120/common/kv_cache_io.cuh:26-46,81-145`
uses runtime flat row pitch for inline records and page/entry addressing for
footer records. Data-copy width and record advance are distinct. The bulk-ready
byte count includes every staged row, including invalid entries redirected to
an immutable zero row. Invalid slots must not gather cache slot zero: unrelated
NaN bytes could survive into 0*NaN in PV. The source `mask_idx_past_len` also
predicates stale positive padding on the math-side rope path; valid IO alone
does not protect a separate synchronous metadata/rope read.

Bind indexed_gather@1 only for valid mapped identities and add explicit invalid
zero-row/tail predication. decoded_tile_materialization@1 still requires the
matching scale group, dtype/cast sequence and generation. Footer scale loads
and inline payload copies follow the layout from the
[earlier trace](trace-mla-cute-cutile-cache.md); neither can borrow the other's
record stride.

## Cache writers and duplicate-slot ownership

| Source decision | Existing rule or new candidate | Concrete binding |
|---|---|---|
| `csrc/sparse_mla_sm120/cake_dsv41_fp8_cache_ops.cu:165-228`: two lanes per32-feature group reduce amax, form a power-of-two UE8M0 scale, encode E4M3 and write one scale byte | power_of_two_scale_encoding@1 metadata domain; quantizer-specific numeric binding | Finite scale trajectory is 2**ceil(log2(max(max(amax,1e-4)/448,1e-4))). Scale ownership is one group, not one lane half. Special-value branches are source-specific and not supplied by a generic finite quantizer. |
| `compute/nvfp4_quantization.cuh:76-115`: group16 scale is rounded E4M3(amax/6), decoded back before reciprocal/packed E2M1 conversion | Explicit E2M1/group16 decoder/quantizer binding | No global NVFP4 tensor scale is implied. A zero encoded scale gives zero reciprocal. This differs from the FP8 UE8M0/group32 record, even though both are quantized cache paths. |
| `dsv4_nvfp4_cache_ops.cu:54-84`: quantize latent groups, copy rope source bits unchanged and restore four padding bytes to zero | physical_storage@1 and full-record encoder contract | Codes, scales, rope and ABI padding jointly define one completed token record. They are not independently replaceable writer outputs. |
| Same file :115-211: earliest input occurrence wins duplicate slot; scan prior rows when N<=256, otherwise ordered reset/atomicMin/winner kernels | [first_occurrence_record_scatter@1](delta/first_occurrence_record_scatter@1/template.yaml) | Owner word occupies reserved four-byte padding after28 scales. Reset precedes every claim; all claims precede ownership reads; only the winner writes the record. Padding restoration must not create a false owner or become visible as a completed cache before the writer grid finishes. |
| `cake_dsv41_fp8_cache_ops.cu:255-274`: warp-parallel earlier-duplicate scan for every append | Same candidate `election=earlier_scan` | The 528-byte FP8 record has no owner scratch padding. One-pass election is a separate resource/applicability binding, not proof that the atomic-padding method is legal for every format. |

The new [proof outline](delta/first_occurrence_record_scatter@1/proof.yaml)
shows why both election methods choose min(input IDs) per slot. This is not
indexed_additive_scatter@1, which sums every occurrence, nor
first_writer_atomic_combine@1, where partial zero initializes before others add.
One record owner avoids torn code/scale combinations, but multiple field stores
are not a transaction for a concurrent reader. Record consumers must follow
completion of the entire writer grid, with no overlapping scratch invocation.

The [logical check](checks/check_record_election.json) enumerates 1365 slot
maps of length0..5 and tests forward/reverse claim order; another witness
checks all120 claim permutations of a duplicate map. Invalid destinations are
excluded and every method returns the same first input. Mixed payload/scale
fields form a record belonging to no input, and a stale owner can suppress a
new invocation's winner. Native warp voting/atomic ordering, quantizer bits,
row-pitch compatibility and consumer publication remain implementation evidence.

Continue Rubin preferred/fallback execution and remaining SM120 compressed
consumer organizations, then move through the other attention families. The
full library inventory remains active; one writer mechanism does not close it.
