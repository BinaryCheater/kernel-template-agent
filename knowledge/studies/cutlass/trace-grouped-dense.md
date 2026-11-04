# Blackwell dense grouped GEMM

This slice source-extracts the single entry
`examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py`
at the pinned revision. [Binding](inputs/dense-grouped-binding.json) retains its
hash/method spans. Device descriptors, exact retirement/proxy ordering and native
initialization progress remain implementation obligations.

The witness uses FP16 A/B/C, FP32 accumulation, MMA128×128, cluster1×1 and two
persistent workers. Its (M,N,K,L) groups are (128,128,64,1), (256,128,0,1),
(128,256,128,1), producing M/N tile counts[1,2,2]. Both update modes use the same
problem metadata and output domain. K0 owns two output tiles and explicit zero output.

| Source decision | Binding or source branch | Rule field/disposition |
|---|---|---|
| Constructor:101–142; scheduler create calls:823,1096,1224 | use_cached_problem_shapes=True is now a public constructor option, passed to each role's StaticPersistentGroupTileScheduler. The earlier block-scaled callers defaulted false. | Reuse online_group_work_decode@1 cache_shapes=True. The prefix cursor/order mapping is unchanged; this is another instance, not a new decoding rule. Exact cache register/warp lowering remains open. |
| _compute_grid:1777–1812; kernel:773–790 | Physical grid is resident-cluster bounded. Descriptor index is flattened physical CTA index, not current logical group or SM id. Host workspace:2231–2239 allocates three128-byte slots per buffer. | rebind_transfer_descriptors@1/@2 private ownership. Workspace extent must cover every launched CTA and slots must not alias across workers; the hardware residency API is not executed here. |
| _get_tensormap_smem_bytes:1840–1858, __call__:429–435 | SMEM mode allocates three128-byte shared shadows; GMEM mode zero shadow bytes. Both own the same three global descriptor slots. Shadow memory also changes the derived stage budget. | Version1 covers shared staging. New [rebind_transfer_descriptors@2](delta/rebind_transfer_descriptors@2/template.yaml) adds update_site without changing descriptor_count or old domains. Resource change is logical384 versus0 bytes, not a latency/occupancy measurement. |
| Kernel:792–840,1234–1249 | SMEM mode delegates AB initialization to epilogue roles and rendezvous with TMA; GMEM mode initializes AB in TMA and C in epilogue directly in global slots. The comments mention MMA, but selected code uses the epilogue role. | Version2 program covers the actual initialization destination. Named barrier participants are128 epilogue threads plus32 TMA threads. Native copy/fence completion remains an implementation premise. |
| Kernel:856–897 and zero-K else branch | First nonzero or zero-K input work responds to the delegated init handshake. Both paths set tensormap_init_done, even when no AB transfer is issued. | The initial missing-participant interpretation was rejected after inspecting the else branch. Tests bind both branches structurally and account for all160 participants. Removing the zero branch would leave32 missing arrivals; that is a counterexample to the wrong interpretation, not a source defect. Earlier grouped block-scaled SM100 also has this zero-branch handshake. |
| Input publisher:859–897; output:1313–1336 | Changed consumed group rebinding uses AB tuple2 and C tuple1. K0 skips AB updates/transfers but not C updates. Metadata helper:1472–1545 builds A(M,K,1),B(N,K,1),C(M,N,1) from pointer/stride tables with assumed16-byte alignment and batch stride0. | Version1 shared instance or version2 direct_global instance. Same immutable key may skip updates; fixed dtype/layout fields remain compatible across groups. Neither mode removes the old-reader retirement premise. |
| `python/CuTeDSL/cutlass/tensor_utils/tensormap.py:158–186` | Both paths commit/wait source-read bulk groups and sync the publisher warp. Shared path release-copies shadows to global; direct path updates global fields, syncs warp and release-fences. Readers acquire pointers at118–125. | Version2 direct branch requires retirement before the first field mutation; shared branch may stage shadows earlier but must retire before publication. Native input-TMA descriptor retirement is not established solely by a bulk-store read wait. |
| Kernel:1113–1174,1364–1396,1428 | K0 bypasses accumulator readiness/read and writes zero registers; accumulator generations skip it, output generation advances. | Reuse the existing grouped identity/output-staging instance. Logical zero branch and initialization participation are both accounted for; no device run is inferred. |
| Kernel:1410 onward; work metadata:1447–1456 | Store every output subtile through the shared C ring. Scheduler metadata carries nine fields including num_tiles_executed so the ring counts persistent output work rather than restarting per group. | staged_async_output_store@1 supplies source lifetime/completion separation. Role counts and actual stage/layout derivation stay source-bound; warp_specialization@1's four-warp unit still cannot express the one-warp MMA role. |
| Host run:2138–2187; CLI:2621–2625 | AB fp16/bf16; accumulator fp16/fp32 except bf16→fp16; C fp16/bf16/fp32. M/tile/cluster menus follow the source predicates; CLI requires every L1. |12 dtype combinations replayed. The run's alignment guard uses m/n/k/l left by its printing loop, so it checks only the last group. A bad first K7 followed by aligned K8 passes, while reversing the order rejects. Every group's alignment remains a caller/backend binding premise; no new source guard is invented. |

## Versioned extension and evidence

Version1 remains unchanged. Its exact projection is version2 with
update_site=shared_shadow; descriptor_count, constraints, preconditions, scope predicate
and obligations keep their owners. [Migration](delta/rebind_transfer_descriptors@2/migration.md)
records the field map. Counts1..5 have identical shared/global storage amounts under
that projection; direct_global has no shadow allocation. New in-place mutation
introduces a different retirement point and is not justified by an old shared proof.

[Checks](checks/grouped-dense-result.json) evaluate retained pure frontend/resource
predicates, replay the five output tiles in both modes, reuse the existing window32
group decoder, compare version1 fields/effects and retain an early direct-mutation
counterexample. A shared shadow can change while an old published reader remains;
mutating the published descriptor's first field would change that old reader's group.
These are metadata/event/source checks, not native descriptor encoding or runtime
progress tests. No independent admission or new measurement is proposed.
