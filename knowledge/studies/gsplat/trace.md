# Source decision bindings

| Source | Choice | Rule / obligation |
|---|---|---|
| IntersectTile.cu:214-285 | first_pass iff cum_tiles_per_gauss null; depth=positive finite float bits | counted_overlap_depth_binning@1 count/emit and ordering |
| same:338-371,390-460 | emit start prefix[idx-1], exact same support predicate | count_emit; ellipse and rectangular variants cannot mix count/emit predicates |
| same:454-457 | key=image prefix OR tile<<32 OR depth bits | identity_pairing and field capacity; tile ID distinct from linear image*n_tiles key used in partition clipping |
| same:1090-1175 | paired radix/segmented radix sort | Segmented sorting can omit image bits only when segments already separate images |
| IntersectTileSparse.cu:30-82 | lower_bound sorted key prefix per active tile; terminal offset total | range_recovery; active tile set must contain all emitted tiles |
| RasterizeToPixels3DGSSerialBatchFwd.cu:120-288 | per-pixel done bits, shared batch, CTA-wide completion | Noncommutative alpha recurrence; early termination is threshold approximation, cannot be justified by reduction_partition@1 |

Worked instance: counts2,0,1 -> prefix2,2,3, disjoint output records0/1 and2. Sorted keys restore tile runs while duplicated Gaussian IDs remain separate tile contributions. Binding substitution checks zero-count and final sentinel boundary. No GPU replay performed.
