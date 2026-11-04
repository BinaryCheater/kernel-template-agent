# Persistent GEMM and Stream-K tail waves

When independent output tiles leave an under-filled resident wave, consider
sharing only that tail along K. Preserve the same output/K coordinate space,
then make the partial-result owner and its publication protocol explicit.

Pinned sources under `examples/gemm_tcgen05/` are
`gemm_tcgen5mma_ws_persistent.py` and
`gemm_tcgen5mma_ws_persistent_streamk.py`. The former provides one- and two-CTA
representatives. The latter keeps full waves data-parallel and partitions the
remainder into K-work intervals. This is a design comparison, not a source audit.

| Choice | Concrete source binding | Reuse or delta |
|---|---|---|
| Persistent roles and clocks | Persistent:64–130: TMA warp, TCGEN05 warp, four-warp epilogue reconstruct the same scheduler sequence. `current_iter*Ktiles+k` indexes the input ring; iteration parity selects two TMEM outputs. | `persistent_tile_scheduler@1`, `multi_buffer_rotation@2` and storage-lifetime obligations explain the design. Keep the old stage domain and target-specific participant counts. One TCGEN05 issuer is not a four-warp compute consumer. |
| Two-CTA cooperation | Persistent two-CTA variant and Stream-K cluster_dims2 split M rows across the two CTAs and B columns for cooperative MMA; barriers are cluster-scoped. | Existing cluster/transfer rules supply conditional reuse. The worker unit is a two-CTA cluster, so old single-CTA launch/resource formulas need explicit mapping. |
| Tail-wave selection | Stream-K:16–17,56–72: R=total_tiles mod resident_clusters; complete remaining waves stay output-tile parallel. | A scheduling choice for underfilled tails, not a universal performance optimum. |
| Linear work partition | Stream-K:97–114: if clusters divide evenly among tail tiles with nonempty K splits, keep each split inside one output; otherwise distribute flattened tile*K work with floor/remainder-balanced interval lengths. | `linearized_work_partition@1` supplies tile-major K linearization and the need to combine boundary partials. Its uniform `b*units_per_cta` mapping does not supply the balanced prefix remainder or cooperative worker costs; retain that mapping difference. |
| Partial ownership | Each interval is visited by output segments in reverse tile order, while K runs forward within a segment. A nonfinal K segment publishes a private accumulator; the segment ending at the tile's full K end owns fixup/output. | `reduction_partition@1.assignment=contiguous_ranges` explains partial sums. Reverse segment order publishes trailing boundary work before waiting on leading fixup; it is not reversal of the numerical K loop. |
| Publication and gather | Stream-K:269–356: store private workspace, cluster barrier, release-add completed K count. Final owner acquire-waits for the missing prefix, gathers known peers or scans per-worker tile tags, writes final C, then resets the counter. | Native visibility is a premise. `ordered_global_combine@1` serializes updates through a shared output; `split_combine_kernel@1` uses a second kernel. Neither directly describes this in-kernel final-owner gather. No forced old-field binding is claimed. |
| Workspace reuse | At most one trailing boundary segment per contiguous interval needs publication; host resets tags, final owner resets per-tile counter. Full tiles bypass fixup. | A useful ownership invariant, with initialization and retirement distinct from arithmetic. Counter reset alone does not prove a later launch is safe on an unrelated stream. |

A four-cluster arithmetic example uses7 output tiles,6 K blocks and3 tail tiles.
There are18 tail work units and balanced intervals [0,5),[5,10),[10,14),[14,18).
Their reverse segment order is:

| Worker | Segments in execution order | Finalizer |
|---|---|---|
| 0 | Tile0 K[0,5): publish | None |
| 1 | Tile1 K[0,4): publish; tile0 K[5,6): gather | Tile0 waits for5 prefix blocks |
| 2 | Tile2 K[0,2): publish; tile1 K[4,6): gather | Tile1 waits for4 prefix blocks |
| 3 | Tile2 K[2,6): gather | Tile2 waits for2 prefix blocks |

The remaining4 output tiles form one complete data-parallel wave. Workspace
slices include CTA rank, so each finalizer combines matching M-row shards. This
recovers the consequential scheduling/ownership distinction without adding an
implementation-specific checker. Native lowering/progress and the floating
partial-sum order remain obligations; no timing or GPU run occurred.

Next: compare scale/quantization and grouped/sparse design differences. A paired
rule delta is useful only if the balanced-partition/final-owner choices recur or
need solver-visible fields; the present inventory keeps those gaps explicit.
