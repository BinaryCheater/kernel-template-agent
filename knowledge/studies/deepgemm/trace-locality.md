# Locality: source mapping and queue counterexample

This partial trace follows MegaMoE's call to get_balanced_sm_locality_domains through
`csrc/apis/locality_domain.hpp`, its JIT pointer-chase probe, runtime allocator and
Python localize wrapper in [the pinned source](inputs/sources.json). The MLOPart Python
IPC service, full allocation lifecycle and source tests remain to be reconciled.
No allocation or probe was executed.

`deep_gemm/locality_domain.py:17–25` splits a selected tensor axis into equal domain
parts and moves the domain axis first, materializing a copy. `empty_per_domain` in the
API rounds each domain slice to allocator granularity, so the new leading stride may
include padding. The `localize` transform cannot be replaced by a plain contiguous
reshape. The runtime allocator reserves one virtual span and maps each slice to a
domain-specific handle; pointer classification requires the complete queried range
to fit one mapping. `mlopart.hpp` imports a POSIX file descriptor from the Python
service, imports the CUDA allocation handle and closes the descriptor. The service
and actual locality placement remain unexecuted external/runtime obligations.

Probe configuration (`csrc/jit_kernels/impls/sm100_locality_domain.hpp`) uses4096-byte
chunks with32 lines of128 bytes, eight chunks per SM, a stride7 line cycle and32 timed
hops. Device lane0 indexes chunks by actual SM identifier, follows dependent global
loads and stores integer cycles/hop in uint16. Host takes a median, asks for far/near
ratio at least1.25 and consistent two-SM TPC labels, retries up to five times, then
falls back to alternating TPC labels. The maximum dynamic shared allocation is used
at launch; one CTA per SM is the intended occupancy constraint, not a verified probe
coverage result. No source timing constant is a KTA measurement.

The balancing function has a direct arithmetic counterexample. It finds the larger
label (zero on a tie), selects the first num_tpcs/2 TPCs with that label and flips all
of them. On input TPC domains `[0,1,0,1]`, both zero TPCs are flipped, giving
`[1,1,1,1]`. The required number of flips to reach balance would instead be the
surplus over half; this study records the source rather than editing upstream.

[The replay](checks/moe_quant_locality.py) enumerates340 binary mappings and retains
98 initially balanced mappings made unbalanced. This matters for MegaMoE's shared
scheduler: local queues divide N clusters between domains; below two global waves,
work stealing is disabled. A legal source-level geometry with eight SMs and two tasks
per domain leaves domain0's queue unclaimed if every TPC is relabeled1. Example layer
shapes H1024/I512 have four N clusters per layer, evenly divisible over two domains.
This is a source-equation coverage/progress counterexample, not a demonstrated GPU
hang. With enough work, stealing may recover tasks but does not repair the promised
balance or memory affinity.

Disposition: localized layout and probe/classification path partially traced; mapping
balance claim contradicted by retained arithmetic evidence. No new reusable rule is
proposed from the faulty balancer. Before reusing locality-aware scheduling, require
queue claim coverage independently of this function's name. Continue with MLOPart IPC,
allocator release/reuse, localization descriptor strides and the source tests.

## IPC, mapping lifetime and descriptor closure

`deep_gemm/utils/mlopart.py` starts an isolated Python child with a socketpair and
passes its file descriptor explicitly. The child removes inherited device visibility
and MPS memory-limit settings, starts an MPS control daemon for the requested GPU UUID
in a private pipe directory, requests an MLOPart server and checks exactly two visible
partition devices. An unavailable service produces a warning and disables localized
allocation; the SM mapping query can still use the alternating-TPC fallback. Explicit
localize then fails rather than returning a pretend localized tensor.

Requests contain two signed64 values (bytes,domain). The service creates a pinned
allocation on that domain device, exports a POSIX handle with send_fds, closes the
exported descriptor and releases its own CUDA allocation handle. `csrc/runtime/mlopart.hpp`
imports the received descriptor as a CUDA allocation handle in the caller and closes
the descriptor there. That imported handle belongs to the runtime allocator's mapping.
Closing the service socket waits for the child; EOF exits its request loop and the
finally clause terminates/waits for its MPS daemon. The singleton resets after release,
so a subsequent query can start a new child. `destroy_localizer` preloads the balanced
SM table before stopping the service. Existing imported allocations are intended to
outlive service shutdown; their mapping handles are released by tensor destruction.
The source tests explicitly exercise service restart and retained tensor contents,
but those GPU tests were not executed here.

`csrc/runtime/locality_domain.hpp` records reservations separately from mapped spans.
Tensor release removes the reservation, synchronizes the device, unmaps all subspans,
releases each imported allocation handle and frees virtual address space. A catch-all
suppresses teardown failures after reservation removal, so partial cleanup/resource
leaks are an explicit lifecycle gap. Startup's pipe wait and client.wait lack a source
timeout. No process launch or shutdown claim is inferred from merely reading this path.

`csrc/utils/layout.hpp:142–169` recognizes localization only when the leading dimension
has two domains, each inner slice has contiguous strides and every queried slice lies
in its corresponding allocator mapping. Unlocalized inputs must be contiguous. This
allows a smaller contiguous expert subview but rejects an inner-axis truncation that
leaves a strided slice. Classification is allocation provenance plus strides, not shape
alone. BF16/FP8/FP4 logical shapes are then reconstructed by combining the domain N axes.

`runtime_utils.hpp:296–328` builds `(K,N_per_domain,domain,batch)` tensor-map coordinates.
For localized batched weights, row stride is K, domain stride is padded leading stride,
and expert/batch stride is the contiguous inner slice stride. For ordinary batched
weights, domain stride is N_per_domain*K and batch stride is N*K. The
[address replay](checks/moe_capacity_localized.py) checks32 padded domains and7168
tagged coordinates with asymmetric expert counts and dimensions. Padding cannot be
silently replaced by the product of logical extents. No new Template is introduced:
materialized domain partitioning is an instance of existing storage/relayout mechanisms;
allocation placement and domain affinity require external capabilities.

Final source-study disposition for this group: all registered operations, Python
wrapper, probe, runtime allocator, IPC service and tests have source-backed dispositions.
Balance/queue correctness is contradicted by the retained scalar witness. Actual
placement, process reliability, driver behavior, probe timing, teardown recovery and
end-to-end queue progress remain explicit implementation obligations. The group is
source-traced with gaps; it is not an admitted locality implementation.
